from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any, Callable, Dict, List, Optional

from src.models.network_state import NetworkState

logger = logging.getLogger(__name__)

VALID_ACTION_TYPES = {"scan", "enumerate", "exploit", "pivot", "analyze", "access"}

# Kill-chain order used for scoring correctness
STAGE_TO_EXPECTED_ACTION = {
    "discovered": "enumerate",
    "enumerated": "analyze",
    "analyzed":   "exploit",
    "exploited":  "pivot",
    "pivoted":    "access",
    "accessed":   None,
}

_PROMPT_INJECTION_RE = re.compile(
    r"[`\[\]{}<>]|ignore|forget|system|assistant|user",
    re.IGNORECASE,
)


class Planner:
    """
    Decision engine with three operating modes controlled by mode=:

      'heuristic'   - deterministic stage-aware rules, always correct, used as
                      the baseline / correctness oracle.

      'llm'         - sends full NetworkState context to the LLM. Output passes
                      through minimal sanitisation only (JSON hygiene, no
                      hallucinated hosts) so the model's actual decisions are
                      preserved for comparison.

      'llm_nostate' - sends a bare prompt with no state context. Used to prove
                      that state tracking improves LLM decision quality.

    Scoring:
      Every LLM action is scored against the heuristic oracle after each step.
      Scores accumulate in self.score_log and can be retrieved via
      get_score_summary() to produce a per-run comparison report.
    """

    def __init__(
        self,
        llm_callable: Optional[Callable[[str], str]] = None,
        max_actions: int = 3,
        use_mock_fallback: bool = True,
        debug: bool = False,
        mode: str = "heuristic",   # 'heuristic' | 'llm' | 'llm_nostate'
    ) -> None:
        self.llm_callable = llm_callable
        self.max_actions = max_actions
        self.use_mock_fallback = use_mock_fallback
        self.debug = debug
        self.mode = mode
        self.score_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, state: NetworkState) -> List[Dict[str, Any]]:
        """
        Main entry point. Returns ranked next-step recommendations.
        In llm/llm_nostate modes, also scores output against the heuristic oracle.
        """
        if self.mode == "heuristic" or self.llm_callable is None:
            return self._heuristic_recommendations(state)

        prompt = self.build_bare_prompt(state) if self.mode == "llm_nostate" else self.build_prompt(state)

        print(f"[PLANNER] Calling LLM (mode={self.mode})...", flush=True, file=sys.stderr)
        if self.mode == "llm_nostate":
            print("[PLANNER] NOSTATE PROMPT SENT:", flush=True, file=sys.stderr)
            print(prompt, flush=True, file=sys.stderr)
            print("---END PROMPT---", flush=True, file=sys.stderr)
        try:
            raw_response = self.llm_callable(prompt)
        except Exception as e:
            print(f"[PLANNER] LLM call FAILED: {e}", flush=True, file=sys.stderr)
            if self.use_mock_fallback:
                return self._heuristic_recommendations(state)
            raise

        if self.debug:
            print(f"[PLANNER] RAW LLM RESPONSE ({len(raw_response)} chars):", flush=True, file=sys.stderr)
            print(raw_response[:800], flush=True, file=sys.stderr)

        parsed = self._parse_llm_response(raw_response)

        if parsed:
            print("[PLANNER] Parsed LLM response successfully.", flush=True, file=sys.stderr)
            cleaned = self._sanitise_llm_actions(state, parsed)
            self._score_against_heuristic(state, cleaned)
            return cleaned[: self.max_actions]

        print("[PLANNER] Parse failed — falling back to heuristics.", flush=True, file=sys.stderr)
        logger.warning(
            "LLM response could not be parsed; falling back to heuristics. "
            "Raw response (first 200 chars): %s",
            (raw_response or "")[:200],
        )
        if self.use_mock_fallback:
            return self._heuristic_recommendations(state)

        raise ValueError("LLM response could not be parsed into valid action recommendations.")

    def plan_chat(
        self,
        state: NetworkState,
        conversation: List[Dict[str, str]],
        chat_callable,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        """
        Multi-turn chat entry point for conversational mode (llm_nostate).

        Instead of rebuilding the full prompt each turn, this appends a new
        user message describing what just happened and asks what to do next.
        The conversation history (list of role/content dicts) is maintained
        externally in app.py and passed in each call so the model truly
        reasons turn-by-turn.

        Args:
            state:        Current NetworkState (used for scoring and sanitising)
            conversation: Full message history so far (mutated and returned)
            chat_callable: LLMClient.chat method

        Returns:
            (actions, updated_conversation)
        """
        # Build the next user turn from current state
        next_user_msg = self.build_turn_message(state)
        conversation.append({"role": "user", "content": next_user_msg})

        print(f"[PLANNER] Calling LLM chat (turn {len(conversation)})...", flush=True, file=sys.stderr)
        print(f"[PLANNER] USER TURN: {next_user_msg[:300]}", flush=True, file=sys.stderr)

        try:
            raw_response = chat_callable(conversation)
        except Exception as e:
            print(f"[PLANNER] Chat call FAILED: {e}", flush=True, file=sys.stderr)
            if self.use_mock_fallback:
                return self._heuristic_recommendations(state), conversation
            raise

        print(f"[PLANNER] RAW CHAT RESPONSE ({len(raw_response)} chars):", flush=True, file=sys.stderr)
        print(raw_response[:600], flush=True, file=sys.stderr)

        # Append assistant reply to conversation so next turn sees it
        conversation.append({"role": "assistant", "content": raw_response})

        parsed = self._parse_llm_response(raw_response)

        if parsed:
            cleaned = self._sanitise_llm_actions(state, parsed)
            self._score_against_heuristic(state, cleaned)
            return cleaned[: self.max_actions], conversation

        print("[PLANNER] Chat parse failed — falling back to heuristics.", flush=True, file=sys.stderr)
        if self.use_mock_fallback:
            return self._heuristic_recommendations(state), conversation

        raise ValueError("Chat response could not be parsed.")

    def build_system_message(self, state: NetworkState) -> str:
        """
        One-time system message sent at the start of a conversational session.
        Sets the scenario context without any stage labels or structured state.
        """
        target = state.target_ip
        scope = ", ".join(state.scope_networks) if state.scope_networks else "unknown"
        blocked = ", ".join(state.blocked_networks) if state.blocked_networks else "none"
        host_list = ", ".join(sorted(state.known_hosts.keys())) if state.known_hosts else "unknown"

        return f"""You are a penetration tester working on a CTF exercise.

Objective: gain access to the internal host at {target}.
In-scope networks: {scope}
Blocked/internal networks: {blocked}
Known hosts so far: {host_list}

At each step I will tell you what we just did and what we found.
You will recommend the single best next action as JSON:
{{
  "rank": 1,
  "action_type": "enumerate|analyze|exploit|pivot|access|scan",
  "target_host": "exact IP from known hosts, or null for scan",
  "command": "exact shell command or null",
  "reasoning": "why this is the right next step",
  "confidence": 0.0
}}

Rules:
- action_type must be exactly ONE word from: scan, enumerate, analyze, exploit, pivot, access
- target_host must be an exact IP — never include port numbers
- Respond ONLY with the JSON object — no other text"""

    def build_turn_message(self, state: NetworkState) -> str:
        """
        Per-step user message describing what just happened and explicitly
        telling phi3 which host to target next based on current state.
        """
        history = state.history
        if not history:
            return "We are starting fresh. What should we do first?"

        last = history[-1]
        action_type = last.get("action_type", "")
        # Use target from history, fall back to most actionable known host
        target_ip = last.get("target_ip")
        if not target_ip:
            # Find the most actionable host (Tomcat first)
            for ip, host in state.known_hosts.items():
                if any("tomcat" in (s.get("product") or "").lower() or
                       s.get("port") in {8080, 80, 443}
                       for s in host.get("services", [])):
                    target_ip = ip
                    break
            if not target_ip and state.known_hosts:
                target_ip = next(iter(state.known_hosts))

        success = last.get("success")

        # Build a host-aware description of what just happened.
        # Check what services the target actually runs so we don't say
        # "found Tomcat" on a host that only runs ospfd.
        host_data = state.known_hosts.get(target_ip, {}) if target_ip else {}
        services = host_data.get("services", [])
        svc_names = [s.get("service_name", "") for s in services]
        is_web = any(s.get("port") in {80, 443, 8080, 8443} or
                     "tomcat" in (s.get("product") or "").lower() or
                     s.get("service_name", "").lower() in {"http", "https", "http-proxy"}
                     for s in services)
        svc_desc = ", ".join(f"port {s['port']} ({s['service_name']})" for s in services) or "unknown services"

        if action_type == "enumerate" and success:
            if is_web:
                description = f"We enumerated {target_ip} and found web services ({svc_desc}), including a potentially exposed management interface."
            else:
                description = f"We enumerated {target_ip} and found {svc_desc}. This appears to be a routing/gateway host with no direct exploit path."
        elif action_type == "analyze" and success:
            if is_web:
                description = f"We analyzed {target_ip} and confirmed CVE-2019-0232 and WAR upload as viable exploit paths."
            else:
                description = f"We analyzed {target_ip} — it runs {svc_desc} and has no viable exploit path. It is a gateway/routing host."
        elif action_type == "exploit" and success:
            description = f"We exploited {target_ip} successfully — we have a shell and {target_ip} is now compromised."
        elif action_type == "exploit" and not success:
            description = f"Exploit on {target_ip} failed. Check that {target_ip} has been analyzed first and is a web/application host."
        elif action_type == "pivot" and success:
            description = f"We pivoted through {target_ip} and can now reach the internal network containing {state.target_ip}."
        elif action_type == "pivot" and not success:
            description = f"Pivot through {target_ip} failed — {target_ip} must be compromised before pivoting."
        elif action_type == "access" and success:
            description = f"We accessed target {state.target_ip} — CTF objective complete."
        elif action_type == "scan" and success:
            description = f"We scanned and discovered: {', '.join(state.known_hosts.keys())}."
        else:
            description = f"We performed {action_type} on {target_ip or 'the network'} — {'succeeded' if success else 'failed'}."

        # Build host status block with completed actions per host
        completed = {}
        for entry in state.history:
            if entry.get("success") and entry.get("target_ip"):
                completed.setdefault(entry["target_ip"], []).append(entry["action_type"])

        host_lines = []
        for ip, host in state.known_hosts.items():
            svc_str = ", ".join(
                f"port {s['port']} ({s['service_name']})"
                for s in host.get("services", [])
            ) or "unknown services"
            comp = " [COMPROMISED - you have a shell]" if host.get("compromised") else ""
            done = completed.get(ip, [])
            done_str = f" Completed on this host: {', '.join(set(done))}." if done else ""
            host_lines.append(f"  {ip}{comp}: {svc_str}.{done_str}")

        hosts_block = "\n".join(host_lines) if host_lines else "  None"

        # Build hints — prioritize web/Tomcat hosts over gateway-only hosts.
        # A gateway host (ospfd, routing only) has no exploit path.
        def _is_exploitable(h):
            return any(
                s.get("port") in {80, 443, 8080, 8443}
                or "tomcat" in (s.get("product") or "").lower()
                or s.get("service_name", "").lower() in {"http", "https", "http-proxy"}
                for s in h.get("services", [])
            )

        # Sort: exploitable hosts first, then others
        sorted_hosts = sorted(
            [(ip, h) for ip, h in state.known_hosts.items() if ip != state.target_ip],
            key=lambda x: (0 if _is_exploitable(x[1]) else 1, x[0])
        )

        next_hints = []
        for ip, host in sorted_hosts:
            done_set = set(completed.get(ip, []))
            is_gateway = not _is_exploitable(host)

            if is_gateway:
                next_hints.append(f"NOTE: {ip} runs a routing service only — it cannot be exploited. Focus on other hosts.")
                continue

            if host.get("compromised"):
                if state.blocked_networks:
                    next_hints.append(f"HINT: {ip} is compromised — next step is pivot through {ip} to reach {state.target_ip}")
            elif "analyze" in done_set:
                next_hints.append(f"HINT: {ip} has been analyzed and is ready — next step is EXPLOIT on {ip}")
            elif "enumerate" in done_set:
                next_hints.append(f"HINT: {ip} has been enumerated — next step is ANALYZE on {ip}")
            else:
                next_hints.append(f"HINT: {ip} has not been enumerated yet — next step is ENUMERATE on {ip}")

        hint_block = "\n".join(next_hints) if next_hints else ""

        return f"""{description}

Known hosts:
{hosts_block}

{hint_block}

What is the single best next action? Use the exact IP address in target_host. Respond with JSON only."""

    def build_prompt(self, state: NetworkState) -> str:
        """
        Full state-aware prompt — includes stage labels, known hosts, history.
        This is the condition that proves state tracking helps the LLM.
        """
        state_context = self._sanitize_for_prompt(state.to_prompt_context())
        discovered_hosts = sorted(state.known_hosts.keys())
        host_list = ", ".join(discovered_hosts) if discovered_hosts else "none"

        return f"""
You are assisting with a controlled CTF-style cybersecurity exercise.

Your goal is to recommend the next best attack-planning steps based ONLY on the current network state.

Kill-chain stages per host (shown in brackets as stage:X):
  discovered  -> recommend: enumerate
  enumerated  -> recommend: analyze
  analyzed    -> recommend: exploit
  exploited   -> recommend: pivot  (if blocked networks exist)
  pivoted     -> recommend: access (if target host is now known)
  accessed    -> objective complete for this host

Rules:
1. Stay within the known in-scope networks.
2. Do not invent new hosts, services, credentials, files, or subnets.
3. Only use host IPs from the Known Hosts section. target_host must be one of: {host_list}, or null.
4. If Known Hosts is empty, the first action must be a scan of the in-scope network.
5. Read each host's stage and recommend the NEXT stage action - never repeat a completed stage.
6. If a host is stage:analyzed and not yet compromised, the top action must be exploit.
7. If a host is stage:exploited and blocked networks exist, the top action must be pivot.
8. If a pivot is established and the target host is known, recommend access.
9. Avoid duplicate actions - each recommendation must add new value.
10. Return exactly {self.max_actions} actions.
11. Respond ONLY in valid JSON - no markdown, no comments, no extra text.
12. action_type must be a single string from: scan, enumerate, exploit, pivot, analyze, access.
13. rank must be a single integer. command must be a single shell command string or null.
14. target_host MUST be the specific IP of the host you are acting on. NEVER null for enumerate/analyze/exploit/pivot.
15. For exploit, target_host must be the IP of the host that was just analyzed.

Return this exact JSON schema:
[
  {{
    "rank": 1,
    "action_type": "enumerate",
    "target_host": "10.0.2.2",
    "command": "nmap -sV -p 8080 10.0.2.2",
    "reasoning": "Host is stage:discovered - enumerate the Tomcat service next.",
    "confidence": 0.95
  }}
]

Current state:
--- BEGIN STATE ---
{state_context}
--- END STATE ---
""".strip()

    def build_bare_prompt(self, state: NetworkState) -> str:
        """
        Conversational/implicit state prompt — simulates the approach shown
        in the professor's research system (Kali MCP + LLM).

        Instead of a structured NetworkState object, this feeds raw action
        history as conversation transcript text — the way a human would
        describe what happened turn by turn. This is how most existing
        LLM-assisted pentesting tools work: the LLM sees what happened
        before as unstructured prose, not as a machine-readable state object.

        This is the meaningful baseline: not "no context at all" but
        "unstructured context" — which is what the professor's system had.
        The weakness it exposes is that the LLM must infer current state
        from prose history rather than reading it directly, which causes:
          - Repeating already-completed actions
          - Losing track of which hosts are at which stage
          - Hallucinating hosts or services not in the history
        """
        target = state.target_ip
        scope = ", ".join(state.scope_networks) if state.scope_networks else "unknown"
        blocked = ", ".join(state.blocked_networks) if state.blocked_networks else "none"
        host_list = ", ".join(sorted(state.known_hosts.keys())) if state.known_hosts else "none discovered yet"

        # Build a raw conversation transcript from history.
        # Skip internal nmap_parse entries and network-level actions (target=None)
        # as they confuse phi3. Only show host-specific pentesting actions.
        action_verbs = {
            "scan":      "Scanned network",
            "enumerate": "Enumerated",
            "analyze":   "Analyzed CVEs for",
            "exploit":   "Tried exploit on",
            "pivot":     "Pivoted through",
            "access":    "Accessed",
        }
        history_lines = []
        for entry in state.history:
            action_type = entry.get("action_type", "")
            target_ip = entry.get("target_ip")
            success = entry.get("success")

            # Skip internal scan/parse entries that phi3 won't understand
            if action_type in ("nmap_parse",):
                continue
            # Skip network-level entries (no specific host) to reduce noise
            if not target_ip and action_type not in ("scan",):
                continue

            status = "SUCCESS" if success else "FAILED" if success is False else "tried"
            verb = action_verbs.get(action_type, action_type)
            target_str = target_ip if target_ip else "network"
            line = f"- [{status}] {verb} {target_str}"
            history_lines.append(line)

        # Build a raw host listing from known_hosts.
        # Map internal action names to human-readable pentesting verbs
        # so phi3 understands what has been done on each host.
        ACTION_VERBS = {
            "scan":       "scanned",
            "enumerate":  "enumerated",
            "analyze":    "analyzed for CVEs",
            "exploit":    "exploit attempted",
            "pivot":      "pivot established",
            "access":     "accessed",
            "nmap_parse": "scanned",   # internal name from scenario builder
        }

        completed_per_host = {}
        for entry in state.history:
            if entry.get("success"):
                tip = entry.get("target_ip")
                if not tip:
                    continue  # skip network-level actions for host block
                atype = entry.get("action_type", "")
                verb = ACTION_VERBS.get(atype, atype)
                completed_per_host.setdefault(tip, set()).add(verb)

        host_lines = []
        for ip, host in state.known_hosts.items():
            services = host.get("services", [])
            svc_str = ", ".join(
                f"port {s['port']} ({s['service_name']})" for s in services
            ) if services else "no open ports found"
            compromised = " [COMPROMISED — you have a shell]" if host.get("compromised") else ""
            done = completed_per_host.get(ip, set())
            if done:
                done_str = "ALREADY DONE on this host: " + ", ".join(sorted(done))
            else:
                done_str = "nothing done yet on this host"
            host_lines.append(f"  {ip}{compromised}: {svc_str}. Status: {done_str}")

        history_block = "\n".join(history_lines) if history_lines else "  No actions taken yet."
        hosts_block = "\n".join(host_lines) if host_lines else "  No hosts discovered yet."

        return f"""You are a penetration tester working on a CTF exercise.

Objective: gain access to the internal host at {target}.
In-scope networks: {scope}
Blocked/internal networks: {blocked}

What you have found so far:
{hosts_block}

What you have tried so far:
{history_block}

Based on this, what are the next {self.max_actions} steps?
Do not repeat steps that have already succeeded.
Focus on advancing toward the objective — avoid re-scanning hosts you already know about.

Important rules for your JSON response:
- action_type must be exactly ONE of: scan, enumerate, analyze, exploit, pivot, access
- target_host MUST be one of these exact IPs: {host_list} — do NOT use null unless action_type is scan
- Do NOT include port numbers in target_host — use "10.0.2.2" not "10.0.2.2:8080"
- Do NOT repeat actions already marked [SUCCESS] in the history or marked ALREADY DONE on a host
- Look at each host's Status line — if a host says "ALREADY DONE: enumerated", next step is "analyze" on that host

Respond ONLY in valid JSON as a list. Example (do not copy this literally — use the actual state above):
[
  {{
    "rank": 1,
    "action_type": "analyze",
    "target_host": "10.0.2.2",
    "command": "searchsploit tomcat",
    "reasoning": "Enumeration is done — now research known CVEs for the identified service.",
    "confidence": 0.85
  }}
]""".strip()

    # ------------------------------------------------------------------
    # Minimal LLM output sanitisation
    # ------------------------------------------------------------------

    def _sanitise_llm_actions(
        self,
        state: NetworkState,
        actions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Minimal sanitisation that preserves the LLM's actual decisions.

        Only removes structurally broken entries:
          - Hallucinated hosts (IP not in known_hosts and not the target IP)
          - Duplicate (action_type, target) pairs within the same response

        Deliberately does NOT:
          - Infer missing targets
          - Downgrade exploit to analyze
          - Skip already-done actions
          - Reorder by heuristic confidence

        This ensures the LLM's mistakes are visible and measurable in scoring.
        """
        seen: set = set()
        cleaned: List[Dict[str, Any]] = []

        for action in actions:
            target = action.get("target_host")
            action_type = action.get("action_type", "analyze")
            command = action.get("command")

            # Flatten list fields
            if isinstance(target, list):
                target = target[0] if target else None
                action["target_host"] = target
            if isinstance(command, list):
                command = command[0] if command else None
                action["command"] = command
            if isinstance(action_type, list):
                action_type = action_type[0] if action_type else "analyze"
                action["action_type"] = action_type

            # phi3 sometimes appends port to IP e.g. "10.0.2.2:8080" or writes
            # a full service description as target_host — strip to bare IP.
            if target is not None and ":" in str(target):
                target = str(target).split(":")[0].strip()
                action["target_host"] = target

            # Only hard-reject hallucinated hosts that don't exist anywhere in state
            if (
                target is not None
                and target != state.target_ip
                and not state.is_known_host(target)
            ):
                print(f"[PLANNER] Dropping hallucinated host: {target}", flush=True, file=sys.stderr)
                action["target_host"] = None
                target = None

            # Fix nmap URL mistakes
            if isinstance(command, str) and command.startswith("nmap "):
                action["command"] = re.sub(r"https?://", "", command)

            # Deduplicate within this response only
            key = (action_type, target)
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(action)

        # Re-rank preserving LLM's original order
        for idx, action in enumerate(cleaned[: self.max_actions], start=1):
            action["rank"] = idx

        return cleaned[: self.max_actions]

    # ------------------------------------------------------------------
    # Scoring against heuristic oracle
    # ------------------------------------------------------------------

    def _score_against_heuristic(
        self,
        state: NetworkState,
        llm_actions: List[Dict[str, Any]],
    ) -> None:
        """
        Scores each LLM action against the heuristic oracle.
        Results are printed to stderr and appended to self.score_log.

        Rubric per action:
          +2  correct action_type for the host's current stage
          +1  correct target_host (matches oracle)
          +1  non-null command provided
          -1  wrong action_type for the host's stage
          -1  action already completed (repeated)
          -2  hallucinated / unknown target_host
        """
        oracle = self._heuristic_recommendations(state)
        already_done = state.get_already_done()
        oracle_map: Dict[Optional[str], str] = {
            a.get("target_host"): a.get("action_type", "") for a in oracle
        }

        step_scores: List[Dict[str, Any]] = []

        for action in llm_actions:
            action_type = action.get("action_type")
            target = action.get("target_host")
            command = action.get("command")
            score = 0
            notes: List[str] = []

            # Stage correctness
            if target and state.is_known_host(target):
                stage = state.known_hosts[target].get("stage", "discovered")
                expected = STAGE_TO_EXPECTED_ACTION.get(stage)
                if expected and action_type == expected:
                    score += 2
                    notes.append(f"+2 correct action for stage:{stage}")
                elif expected:
                    score -= 1
                    notes.append(f"-1 wrong action for stage:{stage} (expected {expected}, got {action_type})")
            elif target == state.target_ip and action_type == "access":
                score += 2
                notes.append("+2 correct access on target")

            # Target correctness vs oracle
            if target == oracle_map.get(target):
                score += 1
                notes.append("+1 target matches oracle")
            elif target is not None and not state.is_known_host(target) and target != state.target_ip:
                score -= 2
                notes.append(f"-2 hallucinated target: {target}")

            # Command provided
            if command:
                score += 1
                notes.append("+1 command provided")
            else:
                notes.append("  no command")

            # Already done penalty
            if (action_type, target) in already_done:
                score -= 1
                notes.append("-1 action already completed")

            step_scores.append({
                "action_type": action_type,
                "target": target,
                "score": score,
                "notes": notes,
                "mode": self.mode,
            })

        total = sum(s["score"] for s in step_scores)
        max_possible = len(llm_actions) * 4

        entry = {
            "step": len(self.score_log) + 1,
            "mode": self.mode,
            "actions": step_scores,
            "total_score": total,
            "max_possible": max_possible,
            "pct": round((total / max_possible * 100) if max_possible else 0, 1),
        }
        self.score_log.append(entry)

        print(
            f"[SCORE] step={entry['step']} mode={self.mode} "
            f"score={total}/{max_possible} ({entry['pct']}%)",
            flush=True, file=sys.stderr,
        )
        for s in step_scores:
            print(f"  {s['action_type']}→{s['target']}: {s['score']}  {s['notes']}", flush=True, file=sys.stderr)

    def get_score_summary(self) -> Dict[str, Any]:
        """Returns aggregated score across all steps for this planner instance."""
        if not self.score_log:
            return {"mode": self.mode, "steps": 0, "total": 0, "max": 0, "pct": 0.0}
        total = sum(e["total_score"] for e in self.score_log)
        maximum = sum(e["max_possible"] for e in self.score_log)
        return {
            "mode": self.mode,
            "steps": len(self.score_log),
            "total": total,
            "max": maximum,
            "pct": round((total / maximum * 100) if maximum else 0, 1),
            "per_step": self.score_log,
        }

    # ------------------------------------------------------------------
    # Stage-aware heuristic planner (oracle + baseline)
    # ------------------------------------------------------------------

    def _heuristic_recommendations(self, state: NetworkState) -> List[Dict[str, Any]]:
        recommendations: List[Dict[str, Any]] = []
        already_done = state.get_already_done()
        known_hosts = state.known_hosts
        blocked_networks = state.blocked_networks

        if not known_hosts:
            initial_target = state.scope_networks[0] if state.scope_networks else None
            return [self._make_action(
                rank=1, action_type="scan", target_host=None,
                command=f"nmap -sn {initial_target}" if initial_target else "nmap <scope-network>",
                reasoning="No hosts discovered yet - run an initial ping sweep to find live hosts.",
                confidence=0.97,
            )]

        def _host_priority(item):
            ip, host = item
            is_web = self._host_looks_like_tomcat(state, ip) or any(
                svc["port"] in {80, 443, 8080, 8443} for svc in host.get("services", [])
            )
            return (0 if is_web else 1, ip)

        for ip, host in sorted(known_hosts.items(), key=_host_priority):
            if len(recommendations) >= self.max_actions * 2:
                break

            stage = host.get("stage", "discovered")

            if ip == state.target_ip:
                continue

            is_gateway_only = (
                host.get("gateway_candidate", False)
                or all(
                    svc.get("port") not in {80, 443, 8080, 8443}
                    and svc.get("service_name", "").lower() not in {"http", "https", "http-proxy", "tomcat"}
                    for svc in host.get("services", [{"": True}])
                )
            ) and not self._host_looks_like_tomcat(state, ip)

            if stage == "discovered":
                services = host.get("services", [])
                port = self._pick_preferred_port(services)
                cmd = f"nmap -sV -p {port} {ip}" if port else f"nmap -sV {ip}"
                conf = 0.70 if is_gateway_only else 0.92
                if ("enumerate", ip) not in already_done:
                    recommendations.append(self._make_action(
                        rank=0, action_type="enumerate", target_host=ip, command=cmd,
                        reasoning=f"{ip} is newly discovered - run a deep service scan to identify versions and attack surface.",
                        confidence=conf,
                    ))

            elif stage == "enumerated":
                if is_gateway_only:
                    continue
                is_tomcat = self._host_looks_like_tomcat(state, ip)
                service_name = host["services"][0]["service_name"] if host.get("services") else "service"
                cmd = "searchsploit tomcat" if is_tomcat else f"searchsploit {service_name}"
                reasoning = f"{ip} services are enumerated - research known CVEs and exploit paths."
                if is_tomcat:
                    reasoning += " Tomcat WAR upload and CVE-2019-0232 are strong candidates."
                if ("analyze", ip) not in already_done:
                    recommendations.append(self._make_action(
                        rank=0, action_type="analyze", target_host=ip, command=cmd,
                        reasoning=reasoning, confidence=0.88,
                    ))

            elif stage == "analyzed" and not host.get("compromised"):
                if is_gateway_only:
                    continue
                is_tomcat = self._host_looks_like_tomcat(state, ip)
                if is_tomcat:
                    cmd = (
                        "msfconsole -x 'use exploit/multi/http/tomcat_mgr_upload; "
                        f"set RHOSTS {ip}; set RPORT 8080; run'"
                    )
                    reasoning = f"{ip} has a confirmed Tomcat exploit path - deploy a WAR payload via the manager interface to get a shell."
                else:
                    cmd = None
                    reasoning = f"{ip} has a known exploit path - attempt to gain access."
                if ("exploit", ip) not in already_done:
                    recommendations.append(self._make_action(
                        rank=0, action_type="exploit", target_host=ip, command=cmd,
                        reasoning=reasoning, confidence=0.93,
                    ))

            elif stage == "exploited" and blocked_networks:
                blocked_net = blocked_networks[0]
                cmd = f"proxychains nmap -sn {blocked_net}"
                if ("pivot", ip) not in already_done:
                    recommendations.append(self._make_action(
                        rank=0, action_type="pivot", target_host=ip, command=cmd,
                        reasoning=f"{ip} is compromised and {blocked_net} is still blocked - establish a tunnel and scan the internal network through it.",
                        confidence=0.95,
                    ))

            elif stage in ("pivoted", "exploited") and not blocked_networks:
                tgt = state.target_ip
                if state.is_known_host(tgt) and ("access", tgt) not in already_done:
                    recommendations.append(self._make_action(
                        rank=0, action_type="access", target_host=tgt,
                        command=f"proxychains ssh root@{tgt}",
                        reasoning=f"Pivot is established and target {tgt} is reachable - connect to capture the flag.",
                        confidence=0.97,
                    ))

        if len(recommendations) < self.max_actions:
            tgt = state.target_ip
            pivot_done = any(h.get("stage") in ("pivoted", "accessed") for h in known_hosts.values())
            if pivot_done and state.is_known_host(tgt) and ("access", tgt) not in already_done:
                recommendations.append(self._make_action(
                    rank=0, action_type="access", target_host=tgt,
                    command=f"proxychains ssh root@{tgt}",
                    reasoning=f"Pivot tunnel is live - connect to {tgt} to complete the CTF objective.",
                    confidence=0.97,
                ))

        if not recommendations:
            initial_target = state.scope_networks[0] if state.scope_networks else None
            recommendations.append(self._make_action(
                rank=1, action_type="scan", target_host=None,
                command=f"nmap {initial_target}" if initial_target else "nmap <scope-network>",
                reasoning="No clear next action found - run a broad re-scan to refresh state.",
                confidence=0.50,
            ))

        recommendations.sort(key=lambda a: a["confidence"], reverse=True)
        for idx, rec in enumerate(recommendations, start=1):
            rec["rank"] = idx

        return recommendations[: self.max_actions]

    # ------------------------------------------------------------------
    # LLM response parsing
    # ------------------------------------------------------------------

    def _parse_llm_response(self, raw_response: str) -> List[Dict[str, Any]]:
        if not raw_response or not raw_response.strip():
            logger.warning("LLM returned an empty response.")
            return []

        text = raw_response.strip()
        fenced_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fenced_match:
            text = fenced_match.group(1).strip()

        try:
            cleaned_text = self._clean_llm_json(text)
            parsed = json.loads(cleaned_text)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse LLM response as JSON: %s. Raw text (first 200 chars): %s", exc, text[:200])
            return []

        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            logger.warning("LLM response parsed to unexpected type (%s).", type(parsed).__name__)
            return []

        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(parsed, start=1):
            if not isinstance(item, dict):
                continue

            raw_rank = item.get("rank", idx)
            if isinstance(raw_rank, list):
                raw_rank = raw_rank[0] if raw_rank else idx

            raw_action_type = item.get("action_type", "analyze")
            if isinstance(raw_action_type, list):
                raw_action_type = raw_action_type[0] if raw_action_type else "analyze"
            action_type = self._normalize_action_type(str(raw_action_type))

            target_host = item.get("target_host")
            if isinstance(target_host, list):
                target_host = target_host[0] if target_host else None
            if target_host is not None and str(target_host).strip() not in {"null", ""}:
                target_host = str(target_host).strip()
            else:
                target_host = None

            command = item.get("command")
            if isinstance(command, list):
                command = command[0] if command else None
            if command is not None:
                command = str(command).strip() or None

            reasoning = item.get("reasoning", "")
            if isinstance(reasoning, list):
                reasoning = reasoning[0] if reasoning else ""
            reasoning = str(reasoning).strip()

            raw_confidence = item.get("confidence", 0.5)
            if isinstance(raw_confidence, list):
                raw_confidence = raw_confidence[0] if raw_confidence else 0.5

            normalized.append(self._make_action(
                rank=int(raw_rank), action_type=action_type, target_host=target_host,
                command=command, reasoning=reasoning, confidence=self._safe_confidence(raw_confidence),
            ))

        normalized.sort(key=lambda a: a["rank"])
        return normalized

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_for_prompt(text: str) -> str:
        return _PROMPT_INJECTION_RE.sub("*", text)

    @staticmethod
    def _normalize_action_type(value: str) -> str:
        # phi3 sometimes returns "enumerate|exploit" copying the schema example.
        # Split on common separators and take the first valid token.
        cleaned = value.strip().lower()
        for sep in ("|", "/", ",", " "):
            cleaned = cleaned.replace(sep, " ")
        for token in cleaned.split():
            if token.strip() in VALID_ACTION_TYPES:
                return token.strip()
        logger.warning("Unknown action_type '%s' received from LLM; normalizing to 'analyze'.", value)
        return "analyze"

    @staticmethod
    def _make_action(rank, action_type, target_host, command, reasoning, confidence):
        return {
            "rank": rank, "action_type": action_type, "target_host": target_host,
            "command": command, "reasoning": reasoning, "confidence": confidence,
        }

    @staticmethod
    def _safe_confidence(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.5

    @staticmethod
    def _pick_preferred_port(services: List[Dict[str, Any]]) -> Optional[int]:
        if not services:
            return None
        preferred = [8080, 8443, 80, 443]
        ports = [s["port"] for s in services]
        for p in preferred:
            if p in ports:
                return p
        return ports[0]

    @staticmethod
    def _find_web_candidates(state: NetworkState) -> List[str]:
        candidates: List[str] = []
        for ip, host in state.known_hosts.items():
            for svc in host["services"]:
                if (
                    svc["service_name"].lower() in {"http", "https", "http-proxy", "tomcat"}
                    or "tomcat" in (svc.get("product") or "").lower()
                    or svc["port"] in {80, 443, 8080, 8443}
                ):
                    candidates.append(ip)
                    break
        return candidates

    @staticmethod
    def _host_looks_like_tomcat(state: NetworkState, ip: str) -> bool:
        if ip not in state.known_hosts:
            return False
        for svc in state.known_hosts[ip]["services"]:
            if "tomcat" in svc["service_name"].lower() or "tomcat" in (svc.get("product") or "").lower():
                return True
        return False

    @staticmethod
    def _strip_json_comments_outside_strings(text: str) -> str:
        result, in_string, escape, i = [], False, False, 0
        while i < len(text):
            ch = text[i]
            if in_string:
                result.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue
            if ch == '"':
                in_string = True
                result.append(ch)
                i += 1
                continue
            if ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
                while i < len(text) and text[i] not in "\r\n":
                    i += 1
                continue
            result.append(ch)
            i += 1
        return "".join(result)

    @staticmethod
    def _clean_llm_json(text: str) -> str:
        text = Planner._strip_json_comments_outside_strings(text)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        text = re.sub(r",\s*([}\]])", r"\1", text)
        text = re.sub(r'(:\s*)00+(\.\d+)', r'\g<1>0\2', text)
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
        return text.strip()
