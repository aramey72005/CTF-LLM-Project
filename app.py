from __future__ import annotations

import sys
import uuid
from flask import Flask, jsonify, render_template, request, session

from src.experiments.planner_evaluation import PlannerEvaluation
from src.models.network_state import NetworkState
from src.services.llm_client import LLMClient
from src.services.planner import Planner
from src.services.state_manager import StateManager

app = Flask(__name__)
app.secret_key = "ctf-sim-secret-key-change-in-prod"

MAX_STEPS = 8

# ---------------------------------------------------------------------------
# Server-side simulation store
# ---------------------------------------------------------------------------
_SIM_STORE: dict = {}


def _get_sim() -> dict | None:
    sid = session.get("sim_id")
    if not sid:
        return None
    return _SIM_STORE.get(sid)


def _new_sim(state: NetworkState, scenario: str, mode: str) -> str:
    sid = str(uuid.uuid4())
    _SIM_STORE[sid] = {
        "state":        state,
        "scenario":     scenario,
        "mode":         mode,
        "step":         0,
        "steps_log":    [],
        # conversation thread for llm_nostate multi-turn mode
        # populated lazily on first advance
        "conversation": [],
    }
    session["sim_id"] = sid
    return sid


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _build_graph(state: NetworkState) -> dict:
    nodes = []
    edges = []
    seen_nodes: set = set()

    def add_node(ip: str, role: str, label_lines: list) -> None:
        if ip in seen_nodes:
            return
        seen_nodes.add(ip)
        nodes.append({"id": ip, "label": "\n".join(label_lines), "role": role})

    for ip, host in state.known_hosts.items():
        if host.get("compromised"):
            role = "compromised"
        elif ip in (state.pivot_hosts or []):
            role = "pivot"
        elif ip in (state.gateway_candidates or []):
            role = "gateway"
        elif ip == state.target_ip:
            role = "target"
        else:
            role = "normal"

        label_lines = [ip]
        for svc in host.get("services", []):
            label_lines.append(f'{svc["port"]} {svc["service_name"]}')
        add_node(ip, role, label_lines[:3])

    if state.target_ip and state.target_ip not in seen_nodes:
        add_node(state.target_ip, "target", [state.target_ip, "TARGET"])

    for gw in (state.gateway_candidates or []):
        if gw not in seen_nodes:
            continue
        for ip in list(seen_nodes):
            if ip != gw and ip not in (state.gateway_candidates or []) and ip != state.target_ip:
                edges.append({"source": gw, "target": ip, "label": "route"})

    any_compromised = any(h.get("compromised") for h in state.known_hosts.values())
    for ip in list(seen_nodes):
        if ip == state.target_ip:
            continue
        host = state.known_hosts.get(ip, {})
        if host.get("compromised") and state.target_ip in seen_nodes:
            label = "pivot path" if any_compromised else "blocked"
            edges.append({"source": ip, "target": state.target_ip, "label": label})

    target_connected = any(e["target"] == state.target_ip for e in edges)
    if not target_connected and state.target_ip in seen_nodes:
        for gw in (state.gateway_candidates or []):
            if gw in seen_nodes:
                edges.append({"source": gw, "target": state.target_ip, "label": "blocked"})
                break
        if not any(e["target"] == state.target_ip for e in edges):
            for ip in list(seen_nodes):
                if ip != state.target_ip:
                    edges.append({"source": ip, "target": state.target_ip, "label": "blocked"})
                    break

    seen_edges: set = set()
    unique_edges = []
    for e in edges:
        key = (e["source"], e["target"])
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(e)

    return {"nodes": nodes, "edges": unique_edges}


# ---------------------------------------------------------------------------
# Planner / client factories
# ---------------------------------------------------------------------------

def _make_client() -> LLMClient:
    return LLMClient(
        base_url="http://localhost:11434",
        model="phi3",
        timeout=180,
    )


def _make_planner(mode: str) -> Planner:
    print(f"[APP] Building planner mode={mode}", flush=True, file=sys.stderr)
    if mode in ("llm", "llm_nostate"):
        client = _make_client()
        return Planner(
            # generate() is used by state-aware llm mode via plan()
            # chat() is called directly by app.py for llm_nostate via plan_chat()
            llm_callable=client.generate,
            max_actions=3,
            use_mock_fallback=True,
            debug=True,
            mode=mode,
        )
    return Planner(
        llm_callable=None,
        max_actions=3,
        use_mock_fallback=True,
        debug=False,
        mode="heuristic",
    )


def _build_scenario(name: str) -> NetworkState:
    evaluator = PlannerEvaluation(max_actions=3)
    if name == "initial_recon":
        return evaluator.build_initial_recon_state()
    elif name == "compromised_pivot":
        return evaluator.build_compromised_pivot_state()
    else:
        return evaluator.build_tomcat_foothold_state()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def start_simulation():
    body = request.get_json(force=True)
    scenario = body.get("scenario", "tomcat_foothold")
    mode = body.get("mode", "heuristic")  # 'heuristic' | 'llm' | 'llm_nostate'

    state = _build_scenario(scenario)
    _new_sim(state, scenario, mode)
    sim = _get_sim()

    planner = _make_planner(mode)

    if mode == "llm_nostate":
        # Initialise the conversation with a system message.
        # The first user turn asks for the very first recommendation.
        client = _make_client()
        system_msg = planner.build_system_message(state)
        first_user_msg = (
            f"We are starting. Known hosts: "
            + ", ".join(
                f"{ip} (port {s['port']} {s['service_name']})"
                for ip, host in state.known_hosts.items()
                for s in host.get("services", [])
            )
            + f". Blocked network: {', '.join(state.blocked_networks) if state.blocked_networks else 'none'}."
            + " What is the first action we should take? Respond with JSON only."
        )
        conversation = [
            {"role": "system",    "content": system_msg},
            {"role": "user",      "content": first_user_msg},
        ]

        print("[APP] Starting chat conversation...", flush=True, file=sys.stderr)
        print(f"[APP] SYSTEM: {system_msg[:200]}", flush=True, file=sys.stderr)
        print(f"[APP] FIRST TURN: {first_user_msg[:200]}", flush=True, file=sys.stderr)

        try:
            raw = client.chat(conversation)
        except Exception as e:
            print(f"[APP] Chat init failed: {e}", flush=True, file=sys.stderr)
            raw = ""

        print(f"[APP] FIRST RESPONSE: {raw[:300]}", flush=True, file=sys.stderr)
        conversation.append({"role": "assistant", "content": raw})
        sim["conversation"] = conversation

        parsed = planner._parse_llm_response(raw)
        actions = planner._sanitise_llm_actions(state, parsed) if parsed else planner._heuristic_recommendations(state)
    else:
        # heuristic and state-aware llm both use plan() unchanged
        actions = planner.plan(state)

    sim["steps_log"].append({
        "step":          0,
        "label":         "Initial State",
        "actions":       actions,
        "chosen_action": None,
        "result":        None,
        "mode":          mode,
    })

    return jsonify({
        "step":          0,
        "state_text":    state.to_prompt_context(),
        "graph":         _build_graph(state),
        "actions":       actions,
        "action_result": None,
        "chosen_action": None,
        "is_initial":    True,
        "max_steps":     MAX_STEPS,
    })


@app.route("/api/advance", methods=["POST"])
def advance():
    sim = _get_sim()
    if sim is None:
        return jsonify({"error": "No active simulation. Call /api/start first."}), 400

    body = request.get_json(force=True)
    chosen_action = body.get("action")

    state: NetworkState = sim["state"]
    step: int          = sim["step"]
    mode: str          = sim["mode"]

    if step >= MAX_STEPS:
        return jsonify({"done": True, "message": "Maximum steps reached."})

    manager = StateManager()

    # ── Apply the chosen action ──────────────────────────────────────────
    if chosen_action is None:
        # Auto-pick using appropriate planner
        planner = _make_planner(mode)
        if mode == "llm_nostate":
            client = _make_client()
            actions, sim["conversation"] = planner.plan_chat(
                state, sim.get("conversation", []), client.chat
            )
        else:
            actions = planner.plan(state)
        chosen_action = actions[0] if actions else None

    # ── State-aware LLM correction: enforce stage order before applying ──
    # phi3 sometimes ignores stage labels and recommends the wrong action.
    # We check the actual stage and redirect to the correct next action.
    # This only applies to the state-aware llm mode — not heuristic or nostate.
    if chosen_action and mode == "llm":
        action_type = chosen_action.get("action_type")
        target = chosen_action.get("target_host")
        already_done = {
            (e["action_type"], e["target_ip"])
            for e in state.history if e.get("success")
        }
        redirected = False

        # exploit before analyze → redirect to analyze
        if action_type == "exploit" and target:
            if ("analyze", target) not in already_done:
                print(f"[APP] Redirecting exploit→analyze on {target} (analyze not done)", flush=True, file=sys.stderr)
                chosen_action = dict(chosen_action)
                chosen_action["action_type"] = "analyze"
                chosen_action["command"] = "searchsploit tomcat"
                chosen_action["reasoning"] = f"Redirected: {target} must be analyzed before exploitation."
                redirected = True

        # analyze after exploit already done → redirect to pivot if blocked networks exist
        if not redirected and action_type == "analyze" and target:
            host = state.known_hosts.get(target, {})
            stage = host.get("stage", "discovered")
            if stage == "exploited" and state.blocked_networks:
                if ("pivot", target) not in already_done:
                    print(f"[APP] Redirecting analyze→pivot on {target} (stage:exploited)", flush=True, file=sys.stderr)
                    chosen_action = dict(chosen_action)
                    chosen_action["action_type"] = "pivot"
                    chosen_action["command"] = f"proxychains nmap -sn {state.blocked_networks[0]}"
                    chosen_action["reasoning"] = f"Redirected: {target} is exploited — pivot through it to reach blocked network."
                    redirected = True

        # analyze after pivot done → redirect to access
        if not redirected and action_type == "analyze" and target:
            host = state.known_hosts.get(target, {})
            stage = host.get("stage", "discovered")
            if stage == "pivoted":
                tgt = state.target_ip
                if state.is_known_host(tgt) and ("access", tgt) not in already_done:
                    print(f"[APP] Redirecting analyze→access on {tgt} (stage:pivoted)", flush=True, file=sys.stderr)
                    chosen_action = dict(chosen_action)
                    chosen_action["action_type"] = "access"
                    chosen_action["target_host"] = tgt
                    chosen_action["command"] = f"proxychains ssh root@{tgt}"
                    chosen_action["reasoning"] = f"Redirected: pivot is live — access target {tgt}."
                    redirected = True

        # analyze on target host → redirect to access if pivot is established
        if not redirected and action_type == "analyze" and target == state.target_ip:
            pivot_done = any(
                h.get("stage") in ("pivoted", "accessed")
                for h in state.known_hosts.values()
            )
            if pivot_done and ("access", state.target_ip) not in already_done:
                print(f"[APP] Redirecting analyze→access on target {target}", flush=True, file=sys.stderr)
                chosen_action = dict(chosen_action)
                chosen_action["action_type"] = "access"
                chosen_action["command"] = f"proxychains ssh root@{target}"
                chosen_action["reasoning"] = f"Redirected: pivot established — access target {target}."

    # For conversational mode, if the LLM returned null target_host,
    # infer the best host here in app.py — NOT in StateManager.
    # This keeps StateManager neutral across all modes.
    if chosen_action and mode == "llm_nostate":
        if chosen_action.get("target_host") is None:
            action_type = chosen_action.get("action_type", "")
            if action_type in ("enumerate", "analyze", "exploit", "pivot"):
                # Find the most actionable host (Tomcat/web first)
                best = None
                for ip, host in state.known_hosts.items():
                    if ip == state.target_ip:
                        continue
                    if any(s.get("port") in {8080, 80, 443}
                           or "tomcat" in (s.get("product") or "").lower()
                           for s in host.get("services", [])):
                        best = ip
                        break
                if best is None and state.known_hosts:
                    best = next(
                        (ip for ip in state.known_hosts if ip != state.target_ip),
                        None
                    )
                if best:
                    chosen_action = dict(chosen_action)
                    chosen_action["target_host"] = best
                    print(f"[APP] Inferred target {best} for null-target {action_type} (llm_nostate only)", flush=True, file=sys.stderr)

    result = manager.apply_action(state, chosen_action) if chosen_action else {
        "success": False, "changes": [], "summary": "No action available."
    }

    # Auto-complete: if access on the target just succeeded, mark simulation done
    ctf_complete = (
        chosen_action
        and chosen_action.get("action_type") == "access"
        and chosen_action.get("target_host") == state.target_ip
        and result.get("success")
    )

    step += 1
    sim["step"] = step

    # ── Get next recommendations from updated state ───────────────────────
    planner = _make_planner(mode)

    if mode == "llm_nostate":
        # True multi-turn: append what just happened as a new user turn
        # and ask for the next recommendation.
        client = _make_client()
        next_actions, sim["conversation"] = planner.plan_chat(
            state, sim.get("conversation", []), client.chat
        )
    else:
        # State-aware llm and heuristic both use plan() — completely unchanged
        next_actions = planner.plan(state)

    mode_tag = mode if mode in ("llm", "llm_nostate") else "heuristic"
    for a in next_actions:
        a["source"] = mode_tag

    sim["steps_log"].append({
        "step":          step,
        "label":         (
            f"Step {step}: "
            f"{chosen_action.get('action_type', '?').capitalize()} "
            f"→ {chosen_action.get('target_host') or 'network'}"
        ),
        "actions":       next_actions,
        "chosen_action": chosen_action,
        "result":        result,
        "mode":          mode,
    })

    return jsonify({
        "step":          step,
        "state_text":    state.to_prompt_context(),
        "graph":         _build_graph(state),
        "actions":       next_actions,
        "action_result": result,
        "chosen_action": chosen_action,
        "done":          step >= MAX_STEPS or ctf_complete,
        "max_steps":     MAX_STEPS,
    })


@app.route("/api/history", methods=["GET"])
def history():
    sim = _get_sim()
    if sim is None:
        return jsonify([])
    return jsonify(sim["steps_log"])


@app.route("/api/scores", methods=["GET"])
def scores():
    sim = _get_sim()
    if sim is None:
        return jsonify({"scores": [], "summary": {}})
    steps = sim.get("steps_log", [])
    total = len([s for s in steps if s.get("step", 0) > 0])
    return jsonify({
        "mode":        sim.get("mode", "heuristic"),
        "total_steps": total,
        "steps":       steps,
    })


if __name__ == "__main__":
    app.run(debug=True)
