from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime
from io import BytesIO
import re
import sys
import uuid
from flask import Flask, jsonify, render_template, request, send_file, session
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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

HOST_ACTION_TYPES = {"enumerate", "analyze", "exploit", "pivot", "access"}


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


def _new_compare_sim(scenario: str) -> str:
    sid = str(uuid.uuid4())
    _SIM_STORE[sid] = {
        "compare": True,
        "scenario": scenario,
        "mode": "compare",
        "step": 0,
        "steps_log": [],
        "branches": {},
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
    if name == "compromised_pivot":
        return evaluator.build_compromised_pivot_state()
    if name == "baseline":
        return evaluator.build_baseline_state()
    if name == "multi_branch_dmz":
        return evaluator.build_multi_branch_dmz_state()
    if name == "stale_history_pressure":
        return evaluator.build_stale_history_pressure_state()
    if name == "post_exploit_distraction":
        return evaluator.build_post_exploit_distraction_state()
    return evaluator.build_tomcat_foothold_state()


def _heuristic_oracle_actions(state: NetworkState) -> list[dict]:
    return Planner(
        llm_callable=None,
        max_actions=3,
        use_mock_fallback=True,
        debug=False,
        mode="heuristic",
    ).plan(state)


def _find_oracle_action(state: NetworkState, target: str | None = None, action_type: str | None = None) -> dict | None:
    for action in _heuristic_oracle_actions(state):
        if target is not None and action.get("target_host") != target:
            continue
        if action_type is not None and action.get("action_type") != action_type:
            continue
        return dict(action)
    return None


def _scenario_label(name: str) -> str:
    labels = {
        "initial_recon": "Initial Recon",
        "tomcat_foothold": "Tomcat Foothold",
        "compromised_pivot": "Compromised Pivot",
        "baseline": "Baseline",
        "multi_branch_dmz": "Multi-Branch DMZ",
        "stale_history_pressure": "Stale History Pressure",
        "post_exploit_distraction": "Post-Exploit Distraction",
    }
    return labels.get(name, name)


def _mode_label(mode: str) -> str:
    labels = {
        "heuristic": "Heuristic (oracle)",
        "llm": "LLM + Structured State",
        "llm_nostate": "LLM + Conversational State",
    }
    return labels.get(mode, mode)


def _is_objective_complete(state: NetworkState) -> bool:
    return state.get_host_stage(state.target_ip) == "accessed"


def _tag_action_sources(actions: list[dict], mode: str) -> list[dict]:
    mode_tag = mode if mode in {"llm", "llm_nostate"} else "heuristic"
    for action in actions:
        action["source"] = mode_tag
    return actions


def _plan_next_actions(
    state: NetworkState,
    mode: str,
    conversation: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    if _is_objective_complete(state):
        return [], conversation or []

    planner = _make_planner(mode)
    if mode == "llm_nostate":
        client = _make_client()
        next_actions, updated_conversation = planner.plan_chat(
            state, conversation or [], client.chat
        )
        return _tag_action_sources(next_actions, mode), updated_conversation

    next_actions = planner.plan(state)
    return _tag_action_sources(next_actions, mode), conversation or []


def _analyze_action_against_state(
    state: NetworkState,
    action: dict | None,
    final_action: dict | None = None,
) -> dict:
    if not action:
        return {
            "oracle_action": None,
            "oracle_target": None,
            "flags": ["no_action"],
            "corrected": False,
            "matched_oracle": False,
        }

    oracle = _find_oracle_action(state)
    action_type = action.get("action_type")
    target = action.get("target_host")
    flags: list[str] = []

    if action_type in HOST_ACTION_TYPES and not target:
        flags.append("missing_target")
    if isinstance(target, str):
        if target.lower() in {"target_ip", "target", "<target>", "<ip>"}:
            flags.append("literal_placeholder_target")
        if not re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", target):
            if action_type in HOST_ACTION_TYPES:
                flags.append("non_ip_target")
        elif target != state.target_ip and not state.is_known_host(target):
            flags.append("unknown_target")

    if action_type and target and (action_type, target) in state.get_already_done():
        flags.append("repeated_successful_action")

    corrected = bool(final_action and final_action != action)
    if corrected:
        flags.append("corrected_before_apply")

    return {
        "oracle_action": oracle.get("action_type") if oracle else None,
        "oracle_target": oracle.get("target_host") if oracle else None,
        "flags": flags,
        "corrected": corrected,
        "matched_oracle": bool(
            oracle
            and action_type == oracle.get("action_type")
            and target == oracle.get("target_host")
        ),
    }


def _action_desc(action: dict | None) -> str:
    if not action:
        return "none"
    return f"{action.get('action_type', '?')} -> {action.get('target_host') or 'network'}"


def _expected_desc(analysis: dict | None) -> str:
    if not analysis:
        return "none"
    oracle_action = analysis.get("oracle_action")
    oracle_target = analysis.get("oracle_target")
    if not oracle_action:
        return "none"
    return f"{oracle_action} -> {oracle_target or 'network'}"


def _flag_explanation(flag: str, raw_action: dict | None, analysis: dict | None) -> str:
    expected = _expected_desc(analysis)
    target = (raw_action or {}).get("target_host") or "network"
    command = (raw_action or {}).get("command")
    if flag == "missing_target":
        return f"The model omitted a required target host. Expected: {expected}."
    if flag == "literal_placeholder_target":
        return f"The model used a placeholder target instead of a real host. Expected: {expected}."
    if flag == "non_ip_target":
        return f"The model produced a non-IP target value for {target}. Expected: {expected}."
    if flag == "unknown_target":
        return f"The model targeted {target}, which was not a known host in the scenario. Expected: {expected}."
    if flag == "repeated_successful_action":
        return f"The model repeated an action that had already succeeded on {target}. Expected next step: {expected}."
    if flag == "corrected_before_apply":
        return f"The system corrected the raw model output before applying it. Corrected action: {expected}."
    if flag == "no_action":
        return "No usable action was produced for this step."
    if command and "http://" in str(command).lower():
        return f"The command shape looked malformed for the current step. Expected: {expected}."
    return f"Flag detected: {flag}. Expected: {expected}."


def _flag_category(flag: str) -> str:
    if flag in {"literal_placeholder_target", "unknown_target"}:
        return "Hallucination"
    if flag in {"missing_target", "non_ip_target"}:
        return "Malformed output"
    if flag == "repeated_successful_action":
        return "Repeated action"
    if flag == "corrected_before_apply":
        return "Corrected planning error"
    if flag == "no_action":
        return "No action produced"
    return "Other"


def _pdf_safe(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _category_counts(flags: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for flag in flags:
        counts[_flag_category(flag)] += 1
    return dict(counts)


def _build_report_summary(sim: dict) -> dict:
    if sim.get("compare"):
        steps = sim.get("steps_log", [])
        action_steps = [s for s in steps if s.get("step", 0) > 0]
        llm_flags = 0
        nostate_flags = 0
        issue_details = []
        llm_category_counts: Counter[str] = Counter()
        nostate_category_counts: Counter[str] = Counter()
        for step in action_steps:
            branches = step.get("branches", {})
            llm_analysis = branches.get("llm", {}).get("analysis") or {}
            nostate_analysis = branches.get("llm_nostate", {}).get("analysis") or {}
            llm_step_flags = llm_analysis.get("flags", [])
            nostate_step_flags = nostate_analysis.get("flags", [])
            llm_flags += len(llm_step_flags)
            nostate_flags += len(nostate_step_flags)
            llm_category_counts.update(_flag_category(flag) for flag in llm_step_flags)
            nostate_category_counts.update(_flag_category(flag) for flag in nostate_step_flags)
            if llm_step_flags:
                issue_details.append({
                    "step": step.get("step"),
                    "branch": "Structured LLM",
                    "flags": llm_step_flags,
                    "chosen_action": branches.get("llm", {}).get("raw_chosen_action") or branches.get("llm", {}).get("chosen_action"),
                    "analysis": llm_analysis,
                })
            if nostate_step_flags:
                issue_details.append({
                    "step": step.get("step"),
                    "branch": "Conversational LLM",
                    "flags": nostate_step_flags,
                    "chosen_action": branches.get("llm_nostate", {}).get("raw_chosen_action") or branches.get("llm_nostate", {}).get("chosen_action"),
                    "analysis": nostate_analysis,
                })
        return {
            "total_steps": len(action_steps),
            "hallucination_like_steps": llm_flags + nostate_flags,
            "corrected_steps": sum(1 for item in issue_details if (item.get("analysis") or {}).get("corrected")),
            "target_accessed": False,
            "issue_details": issue_details,
            "llm_flags": llm_flags,
            "nostate_flags": nostate_flags,
            "llm_category_counts": dict(llm_category_counts),
            "nostate_category_counts": dict(nostate_category_counts),
            "llm_target_accessed": (
                sim.get("branches", {}).get("llm", {}).get("state").get_host_stage(
                    sim.get("branches", {}).get("llm", {}).get("state").target_ip
                ) == "accessed"
                if sim.get("branches", {}).get("llm", {}).get("state") else False
            ),
            "nostate_target_accessed": (
                sim.get("branches", {}).get("llm_nostate", {}).get("state").get_host_stage(
                    sim.get("branches", {}).get("llm_nostate", {}).get("state").target_ip
                ) == "accessed"
                if sim.get("branches", {}).get("llm_nostate", {}).get("state") else False
            ),
        }

    steps = sim.get("steps_log", [])
    action_steps = [s for s in steps if s.get("step", 0) > 0]
    issue_steps = []
    corrected_steps = 0
    category_counts: Counter[str] = Counter()

    for step in action_steps:
        analysis = step.get("analysis") or {}
        flags = analysis.get("flags", [])
        category_counts.update(_flag_category(flag) for flag in flags)
        if flags:
            issue_steps.append({
                "step": step.get("step"),
                "branch": _mode_label(sim.get("mode", "unknown")),
                "flags": flags,
                "chosen_action": step.get("raw_chosen_action") or step.get("chosen_action"),
                "analysis": analysis,
            })
        if analysis.get("corrected"):
            corrected_steps += 1

    final_state: NetworkState = sim["state"]
    target_accessed = final_state.get_host_stage(final_state.target_ip) == "accessed"

    return {
        "total_steps": len(action_steps),
        "hallucination_like_steps": len(issue_steps),
        "corrected_steps": corrected_steps,
        "target_accessed": target_accessed,
        "issue_details": issue_steps,
        "category_counts": dict(category_counts),
    }


def _generate_simulation_pdf(sim: dict) -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scenario = _scenario_label(sim.get("scenario", "unknown"))
    mode = _mode_label(sim.get("mode", "unknown"))
    summary = _build_report_summary(sim)
    final_state: NetworkState | None = sim.get("state")

    story.append(Paragraph("CTF LLM Simulation Report", styles["Title"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Generated: {created_at}", styles["BodyText"]))
    story.append(Paragraph(f"Scenario: {scenario}", styles["BodyText"]))
    story.append(Paragraph(f"Mode: {mode}", styles["BodyText"]))
    story.append(Spacer(1, 10))

    def add_step_section(title: str, step_items: list[dict]) -> None:
        story.append(Paragraph(title, styles["Heading2"]))
        for step in step_items:
            step_no = step.get("step", "?")
            raw_action = step.get("raw_chosen_action") or step.get("chosen_action")
            final_action = step.get("chosen_action")
            analysis = step.get("analysis") or {}
            result = step.get("result") or {}
            story.append(Paragraph(f"Step {step_no}", styles["Heading3"]))
            story.append(Paragraph(f"Raw action: {_pdf_safe(_action_desc(raw_action))}", styles["BodyText"]))
            story.append(Paragraph(f"Applied action: {_pdf_safe(_action_desc(final_action))}", styles["BodyText"]))
            story.append(Paragraph(f"Expected action: {_pdf_safe(_expected_desc(analysis))}", styles["BodyText"]))
            story.append(Paragraph(f"Result: {_pdf_safe(result.get('summary', 'none'))}", styles["BodyText"]))
            flags = analysis.get("flags", [])
            story.append(Paragraph(f"Flags: {_pdf_safe(', '.join(flags) if flags else 'none')}", styles["BodyText"]))
            if flags:
                for flag in flags:
                    story.append(Paragraph(
                        f"- {_pdf_safe(_flag_explanation(flag, raw_action, analysis))}",
                        styles["BodyText"],
                    ))
            story.append(Spacer(1, 8))

    if sim.get("compare"):
        summary_rows = [
            ["Branch", "Steps", "Hallucinations", "Omissions", "Corrected", "Target Accessed"],
            [
                "Structured State",
                str(summary["total_steps"]),
                str(summary.get("llm_category_counts", {}).get("Hallucination", 0)),
                str(summary.get("llm_category_counts", {}).get("Malformed output", 0)),
                str(sum(
                    1 for item in summary.get("issue_details", [])
                    if item.get("branch") == "Structured LLM" and (item.get("analysis") or {}).get("corrected")
                )),
                "Yes" if summary.get("llm_target_accessed") else "No",
            ],
            [
                "Conversational State",
                str(summary["total_steps"]),
                str(summary.get("nostate_category_counts", {}).get("Hallucination", 0)),
                str(summary.get("nostate_category_counts", {}).get("Malformed output", 0)),
                str(sum(
                    1 for item in summary.get("issue_details", [])
                    if item.get("branch") == "Conversational LLM" and (item.get("analysis") or {}).get("corrected")
                )),
                "Yes" if summary.get("nostate_target_accessed") else "No",
            ],
        ]
        summary_table = Table(summary_rows, colWidths=[110, 50, 75, 65, 60, 80])
    else:
        summary_rows = [
            ["Metric", "Value"],
            ["Total executed steps", str(summary["total_steps"])],
            ["Hallucinations", str(summary.get("category_counts", {}).get("Hallucination", 0))],
            ["Omissions / malformed outputs", str(summary.get("category_counts", {}).get("Malformed output", 0))],
            ["Repeated actions", str(summary.get("category_counts", {}).get("Repeated action", 0))],
            ["Corrected before apply", str(summary["corrected_steps"])],
            ["Target accessed", "Yes" if summary["target_accessed"] else "No"],
        ]
        summary_table = Table(summary_rows, colWidths=[220, 220])

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e2f3")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 14))

    if summary["issue_details"]:
        story.append(Paragraph("Detected Issues", styles["Heading2"]))
        for item in summary["issue_details"]:
            chosen = item.get("chosen_action") or {}
            action_desc = _action_desc(chosen)
            flags = ", ".join(item.get("flags", []))
            branch = item.get("branch", mode)
            categories = ", ".join(sorted({_flag_category(flag) for flag in item.get("flags", [])}))
            story.append(Paragraph(f"Step {item['step']} [{branch}]: {_pdf_safe(action_desc)}", styles["BodyText"]))
            story.append(Paragraph(f"Category: {_pdf_safe(categories or 'None')}", styles["BodyText"]))
            story.append(Paragraph(f"Flags: {_pdf_safe(flags)}", styles["BodyText"]))
            story.append(Paragraph(f"Expected: {_pdf_safe(_expected_desc(item.get('analysis')))}", styles["BodyText"]))
            for flag in item.get("flags", []):
                story.append(Paragraph(
                    f"- {_pdf_safe(_flag_explanation(flag, chosen, item.get('analysis')))}",
                    styles["BodyText"],
                ))
            story.append(Spacer(1, 6))
        story.append(Spacer(1, 12))

    if sim.get("compare"):
        structured_steps = []
        conversational_steps = []
        for step in sim.get("steps_log", []):
            if step.get("step", 0) == 0:
                continue
            branches = step.get("branches", {})
            structured_steps.append({
                "step": step.get("step"),
                **(branches.get("llm") or {}),
            })
            conversational_steps.append({
                "step": step.get("step"),
                **(branches.get("llm_nostate") or {}),
            })

        add_step_section("Executed Steps - Structured State Branch", structured_steps)
        add_step_section("Executed Steps - Conversational State Branch", conversational_steps)

        story.append(Paragraph("Final Network States", styles["Heading2"]))
        structured_state = sim.get("branches", {}).get("llm", {}).get("state")
        conversational_state = sim.get("branches", {}).get("llm_nostate", {}).get("state")
        if structured_state:
            story.append(Paragraph("Structured branch final state", styles["Heading3"]))
            for line in structured_state.to_prompt_context().splitlines():
                story.append(Paragraph(_pdf_safe(line), styles["Code"]))
        if conversational_state:
            story.append(Spacer(1, 8))
            story.append(Paragraph("Conversational branch final state", styles["Heading3"]))
            for line in conversational_state.to_prompt_context().splitlines():
                story.append(Paragraph(_pdf_safe(line), styles["Code"]))
    else:
        step_items = [step for step in sim.get("steps_log", []) if step.get("step", 0) > 0]
        add_step_section("Executed Steps", step_items)
        story.append(Paragraph("Final Network State", styles["Heading2"]))
        for line in final_state.to_prompt_context().splitlines():
            story.append(Paragraph(_pdf_safe(line), styles["Code"]))

    doc.build(story)
    buffer.seek(0)
    return buffer


def _start_branch_session(state: NetworkState, mode: str) -> dict:
    planner = _make_planner(mode)
    branch = {
        "state": state,
        "mode": mode,
        "conversation": [],
        "actions": [],
    }

    if mode == "llm_nostate":
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
            {"role": "system", "content": system_msg},
            {"role": "user", "content": first_user_msg},
        ]
        try:
            raw = client.chat(conversation)
        except Exception as e:
            print(f"[APP] Chat init failed: {e}", flush=True, file=sys.stderr)
            raw = ""
        conversation.append({"role": "assistant", "content": raw})
        branch["conversation"] = conversation
        parsed = planner._parse_llm_response(raw)
        branch["actions"] = _tag_action_sources(
            planner._sanitise_llm_actions(state, parsed) if parsed else planner._heuristic_recommendations(state),
            mode,
        )
    else:
        branch["actions"] = _tag_action_sources(planner.plan(state), mode)

    return branch


def _prepare_action_for_mode(state: NetworkState, mode: str, chosen_action: dict | None) -> tuple[dict | None, dict | None, dict]:
    if not chosen_action:
        analysis = _analyze_action_against_state(state, None, None)
        return None, None, analysis

    raw_chosen_action = deepcopy(chosen_action)

    if mode == "llm":
        action_type = chosen_action.get("action_type")
        target = chosen_action.get("target_host")
        already_done = {
            (e["action_type"], e["target_ip"])
            for e in state.history if e.get("success")
        }
        redirected = False

        if action_type == "exploit" and target:
            if ("analyze", target) not in already_done:
                chosen_action = (
                    _find_oracle_action(state, target=target, action_type="analyze")
                    or _find_oracle_action(state, target=target)
                    or _find_oracle_action(state)
                    or dict(chosen_action)
                )
                chosen_action["reasoning"] = (
                    f"Redirected: {target} must progress through earlier stages before exploitation."
                )
                redirected = True

        if not redirected and target:
            host = state.known_hosts.get(target, {})
            stage = host.get("stage", "discovered")
            if stage in {"exploited", "pivoted"} or target == state.target_ip:
                oracle_redirect = _find_oracle_action(state)
                if oracle_redirect:
                    chosen_action = oracle_redirect
                    chosen_action["reasoning"] = (
                        f"Redirected: current state requires {oracle_redirect.get('action_type')} on "
                        f"{oracle_redirect.get('target_host') or 'the network'}."
                    )

    if mode == "llm_nostate":
        if chosen_action.get("target_host") is None:
            oracle_target = _find_oracle_action(state, action_type=chosen_action.get("action_type"))
            if oracle_target and oracle_target.get("target_host") is not None:
                chosen_action = dict(chosen_action)
                chosen_action["target_host"] = oracle_target["target_host"]

    analysis = _analyze_action_against_state(state, raw_chosen_action, chosen_action)
    return raw_chosen_action, chosen_action, analysis


def _advance_branch_session(branch: dict) -> dict:
    state: NetworkState = branch["state"]
    mode: str = branch["mode"]
    if _is_objective_complete(state):
        return {
            "state_text": state.to_prompt_context(),
            "graph": _build_graph(state),
            "actions": [],
            "action_result": {
                "success": True,
                "changes": [],
                "summary": "Branch already complete - no further action executed.",
            },
            "chosen_action": None,
            "raw_chosen_action": None,
            "analysis": {"flags": [], "oracle_action": None, "oracle_target": None, "corrected": False, "matched_oracle": False},
            "done": True,
        }

    chosen_action = (branch.get("actions") or [None])[0]
    raw_chosen_action, chosen_action, analysis = _prepare_action_for_mode(state, mode, chosen_action)

    manager = StateManager()
    result = manager.apply_action(state, chosen_action) if chosen_action else {
        "success": False, "changes": [], "summary": "No action available."
    }

    next_actions, branch["conversation"] = _plan_next_actions(
        state, mode, branch.get("conversation", [])
    )
    branch["actions"] = next_actions

    return {
        "state_text": state.to_prompt_context(),
        "graph": _build_graph(state),
        "actions": next_actions,
        "action_result": result,
        "chosen_action": chosen_action,
        "raw_chosen_action": raw_chosen_action,
        "analysis": analysis,
        "done": _is_objective_complete(state),
    }


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

    if mode == "compare":
        _new_compare_sim(scenario)
        sim = _get_sim()
        assert sim is not None

        llm_branch = _start_branch_session(_build_scenario(scenario), "llm")
        nostate_branch = _start_branch_session(_build_scenario(scenario), "llm_nostate")
        sim["branches"] = {
            "llm": llm_branch,
            "llm_nostate": nostate_branch,
        }
        sim["steps_log"].append({
            "step": 0,
            "label": "Initial State",
            "mode": "compare",
            "branches": {
                "llm": {"actions": llm_branch["actions"]},
                "llm_nostate": {"actions": nostate_branch["actions"]},
            },
        })

        return jsonify({
            "compare": True,
            "step": 0,
            "max_steps": MAX_STEPS,
            "branches": {
                "llm": {
                    "state_text": llm_branch["state"].to_prompt_context(),
                    "graph": _build_graph(llm_branch["state"]),
                    "actions": llm_branch["actions"],
                    "action_result": None,
                    "chosen_action": None,
                    "mode": "llm",
                },
                "llm_nostate": {
                    "state_text": nostate_branch["state"].to_prompt_context(),
                    "graph": _build_graph(nostate_branch["state"]),
                    "actions": nostate_branch["actions"],
                    "action_result": None,
                    "chosen_action": None,
                    "mode": "llm_nostate",
                },
            },
            "done": False,
        })

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
    actions = _tag_action_sources(actions, mode)

    sim["steps_log"].append({
        "step":          0,
        "label":         "Initial State",
        "actions":       actions,
        "chosen_action": None,
        "raw_chosen_action": None,
        "result":        None,
        "analysis":      None,
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

    if sim.get("compare"):
        step: int = sim["step"]
        if step >= MAX_STEPS:
            return jsonify({"done": True, "message": "Maximum steps reached.", "compare": True})

        llm_payload = _advance_branch_session(sim["branches"]["llm"])
        nostate_payload = _advance_branch_session(sim["branches"]["llm_nostate"])
        step += 1
        sim["step"] = step
        done = step >= MAX_STEPS or (llm_payload["done"] and nostate_payload["done"])

        sim["steps_log"].append({
            "step": step,
            "label": f"Compare Step {step}",
            "mode": "compare",
            "branches": {
                "llm": {
                    "chosen_action": llm_payload["chosen_action"],
                    "raw_chosen_action": llm_payload["raw_chosen_action"],
                    "result": llm_payload["action_result"],
                    "analysis": llm_payload["analysis"],
                    "actions": llm_payload["actions"],
                },
                "llm_nostate": {
                    "chosen_action": nostate_payload["chosen_action"],
                    "raw_chosen_action": nostate_payload["raw_chosen_action"],
                    "result": nostate_payload["action_result"],
                    "analysis": nostate_payload["analysis"],
                    "actions": nostate_payload["actions"],
                },
            },
        })

        return jsonify({
            "compare": True,
            "step": step,
            "max_steps": MAX_STEPS,
            "done": done,
            "branches": {
                "llm": {**llm_payload, "mode": "llm"},
                "llm_nostate": {**nostate_payload, "mode": "llm_nostate"},
            },
        })

    body = request.get_json(force=True)
    chosen_action = body.get("action")

    state: NetworkState = sim["state"]
    step: int          = sim["step"]
    mode: str          = sim["mode"]

    if step >= MAX_STEPS:
        return jsonify({"done": True, "message": "Maximum steps reached."})
    if _is_objective_complete(state):
        return jsonify({
            "step": step,
            "state_text": state.to_prompt_context(),
            "graph": _build_graph(state),
            "actions": [],
            "action_result": {
                "success": True,
                "changes": [],
                "summary": "Simulation already complete - no further action executed.",
            },
            "chosen_action": None,
            "analysis": _analyze_action_against_state(state, None, None),
            "done": True,
            "max_steps": MAX_STEPS,
        })

    manager = StateManager()

    # ── Apply the chosen action ──────────────────────────────────────────
    if chosen_action is None:
        actions, sim["conversation"] = _plan_next_actions(
            state, mode, sim.get("conversation", [])
        )
        chosen_action = actions[0] if actions else None

    raw_chosen_action = deepcopy(chosen_action) if chosen_action else None

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
                chosen_action = _find_oracle_action(state, target=target, action_type="analyze") or dict(chosen_action)
                chosen_action["reasoning"] = f"Redirected: {target} must be analyzed before exploitation."
                redirected = True

        if not redirected and target:
            host = state.known_hosts.get(target, {})
            stage = host.get("stage", "discovered")
            if stage in {"exploited", "pivoted"} or target == state.target_ip:
                oracle_redirect = _find_oracle_action(state)
                if oracle_redirect:
                    print(
                        f"[APP] Redirecting {action_type}→{oracle_redirect.get('action_type')} on {oracle_redirect.get('target_host')}",
                        flush=True,
                        file=sys.stderr,
                    )
                    chosen_action = oracle_redirect
                    chosen_action["reasoning"] = (
                        f"Redirected: current state requires {oracle_redirect.get('action_type')} on "
                        f"{oracle_redirect.get('target_host') or 'the network'}."
                    )

    # For conversational mode, if the LLM returned null target_host,
    # infer the best host here in app.py — NOT in StateManager.
    # This keeps StateManager neutral across all modes.
    if chosen_action and mode == "llm_nostate":
        if chosen_action.get("target_host") is None:
            oracle_target = _find_oracle_action(state, action_type=chosen_action.get("action_type"))
            if oracle_target and oracle_target.get("target_host") is not None:
                best = oracle_target["target_host"]
                chosen_action = dict(chosen_action)
                chosen_action["target_host"] = best
                print(f"[APP] Inferred target {best} for null-target {chosen_action.get('action_type')} (llm_nostate only)", flush=True, file=sys.stderr)

    _, chosen_action, step_analysis = _prepare_action_for_mode(
        state, mode, raw_chosen_action
    )
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

    if ctf_complete or _is_objective_complete(state):
        next_actions = []

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
        "raw_chosen_action": raw_chosen_action,
        "result":        result,
        "analysis":      step_analysis,
        "mode":          mode,
    })

    return jsonify({
        "step":          step,
        "state_text":    state.to_prompt_context(),
        "graph":         _build_graph(state),
        "actions":       next_actions,
        "action_result": result,
        "chosen_action": chosen_action,
        "analysis":      step_analysis,
        "done":          step >= MAX_STEPS or _is_objective_complete(state),
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


@app.route("/api/report.pdf", methods=["GET"])
def download_report():
    sim = _get_sim()
    if sim is None:
        return jsonify({"error": "No active simulation. Call /api/start first."}), 400

    pdf_buffer = _generate_simulation_pdf(sim)
    filename = f"{sim.get('scenario', 'simulation')}_{sim.get('mode', 'mode')}_report.pdf"
    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


if __name__ == "__main__":
    app.run(debug=True)
