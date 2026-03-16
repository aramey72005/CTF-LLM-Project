from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from src.models.network_state import NetworkState


class StateManager:
    """
    Manages simulated state transitions for the CTF-style planning environment.

    Each action handler is responsible for:
    1. Making the structural state change (add host, add note, mark compromised, etc.)
    2. Calling state.advance_host_stage() so the planner knows the kill-chain
       position has moved forward and recommends the next logical action.

    Kill-chain progression per host:
        discovered → enumerated → analyzed → exploited → pivoted → accessed
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone_state(self, state: NetworkState) -> NetworkState:
        return deepcopy(state)

    def apply_action(
        self,
        state: NetworkState,
        action: Dict[str, Any],
    ) -> Dict[str, Any]:
        action_type = str(action.get("action_type", "analyze")).strip().lower()
        target_host = action.get("target_host")
        command = action.get("command")
        reasoning = str(action.get("reasoning", "")).strip()

        if action_type == "scan":
            result = self._apply_scan(state, target_host, command)
        elif action_type == "enumerate":
            result = self._apply_enumerate(state, target_host, command)
        elif action_type == "analyze":
            result = self._apply_analyze(state, target_host, command)
        elif action_type == "exploit":
            result = self._apply_exploit(state, target_host, command)
        elif action_type == "pivot":
            result = self._apply_pivot(state, target_host, command)
        elif action_type == "access":
            result = self._apply_access(state, target_host, command)
        else:
            result = {
                "success": False,
                "changes": [],
                "summary": f"Unsupported action type: {action_type}",
            }

        state.record_action(
            action_type=action_type,
            description=reasoning or result["summary"],
            target_ip=target_host,
            success=result["success"],
        )

        return result

    def run_action_sequence(
        self,
        initial_state: NetworkState,
        actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        state = self.clone_state(initial_state)
        history: List[Dict[str, Any]] = []

        for step_number, action in enumerate(actions, start=1):
            result = self.apply_action(state, action)
            history.append(
                {
                    "step": step_number,
                    "action": action,
                    "result": result,
                    "state_snapshot": state.to_prompt_context(),
                }
            )

        return {"final_state": state, "history": history}

    def advance_with_planner(
        self,
        state: NetworkState,
        planner: Any,
    ) -> Dict[str, Any]:
        actions = planner.plan(state)
        if not actions:
            return {
                "success": False,
                "chosen_action": None,
                "result": {
                    "success": False,
                    "changes": [],
                    "summary": "Planner returned no actions.",
                },
            }

        chosen_action = actions[0]
        result = self.apply_action(state, chosen_action)
        return {
            "success": result["success"],
            "chosen_action": chosen_action,
            "result": result,
        }

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _apply_scan(
        self,
        state: NetworkState,
        target_host: Optional[str],
        command: Optional[str],
    ) -> Dict[str, Any]:
        changes: List[str] = []
        cmd = (command or "").lower()

        # ── Initial scan: no hosts known yet ──
        if not state.known_hosts:
            self._ensure_gateway_host(state)
            self._ensure_tomcat_host(state)
            # Both hosts start at "discovered" (set by _ensure_* helpers)
            changes.extend([
                "Discovered host 10.0.0.1 — routing service on port 2601.",
                "Discovered host 10.0.2.2 — Apache Tomcat on port 8080.",
            ])
            return {
                "success": True,
                "changes": changes,
                "summary": "Initial scan revealed gateway (10.0.0.1) and Tomcat foothold (10.0.2.2).",
            }

        # ── Pivot-assisted scan of blocked network ──
        if "10.0.4.0/24" in cmd or "10.0.4." in cmd:
            if self._pivot_available(state):
                discovered = self._ensure_internal_target_host(state)
                if discovered:
                    # Target host discovered for first time
                    state.advance_host_stage(state.target_ip, "discovered")
                    changes.append("Pivot-assisted scan revealed internal target host 10.0.4.3.")
                else:
                    changes.append("Internal target host 10.0.4.3 was already known.")
                return {
                    "success": True,
                    "changes": changes,
                    "summary": "Pivot-assisted scan revealed the internal target network.",
                }
            return {
                "success": False,
                "changes": [],
                "summary": "Cannot scan blocked network directly — establish a pivot first.",
            }

        # ── Targeted re-scan of known host ──
        if target_host == "10.0.2.2":
            self._add_note_if_missing(
                state, "10.0.2.2",
                "Detailed re-scan confirmed Tomcat web service is still exposed.",
            )
            changes.append("Re-scan confirmed Tomcat exposure on 10.0.2.2.")
            return {
                "success": True,
                "changes": changes,
                "summary": "Host re-scan confirmed exposed Tomcat service.",
            }

        return {
            "success": True,
            "changes": [],
            "summary": "Scan completed — no major new discoveries.",
        }

    def _apply_enumerate(
        self,
        state: NetworkState,
        target_host: Optional[str],
        command: Optional[str],
    ) -> Dict[str, Any]:
        changes: List[str] = []

        if target_host == "10.0.2.2":
            self._ensure_gateway_host(state)
            self._ensure_tomcat_host(state)

            self._add_note_if_missing(
                state, "10.0.2.2",
                "Tomcat manager interface (/manager/html) is exposed — default credentials may work.",
            )
            self._add_note_if_missing(
                state, "10.0.2.2",
                "WAR file deployment via Tomcat manager is a known RCE vector.",
            )
            # ── Advance stage: discovered → enumerated ──
            state.advance_host_stage("10.0.2.2", "enumerated")
            changes.extend([
                "Identified exposed Tomcat manager interface on 10.0.2.2.",
                "Noted WAR deployment as a likely RCE vector.",
                "Host 10.0.2.2 stage advanced to: enumerated.",
            ])
            return {
                "success": True,
                "changes": changes,
                "summary": "Enumeration deepened Tomcat knowledge — manager interface confirmed exposed.",
            }

        if target_host == "10.0.0.1":
            self._ensure_gateway_host(state)
            self._add_note_if_missing(
                state, "10.0.0.1",
                "OSPF daemon on port 2601 present — gateway-only host, not an exploit target.",
            )
            # Cap at enumerated — gateway hosts have no exploit path.
            # Do not advance beyond enumerated so the planner ignores them
            # for analyze/exploit and keeps focus on the Tomcat foothold host.
            state.advance_host_stage("10.0.0.1", "enumerated")
            changes.append("Gateway host 10.0.0.1 noted — no exploit path, stage capped at enumerated.")
            return {
                "success": True,
                "changes": changes,
                "summary": "Gateway host enumerated — routing service noted, not an exploit target.",
            }

        return {
            "success": True,
            "changes": [],
            "summary": "Enumeration completed — limited new detail on this host.",
        }

    def _apply_analyze(
        self,
        state: NetworkState,
        target_host: Optional[str],
        command: Optional[str],
    ) -> Dict[str, Any]:
        changes: List[str] = []
        cmd = (command or "").lower()

        if "tomcat" in cmd or target_host == "10.0.2.2":
            self._ensure_tomcat_host(state)
            self._add_note_if_missing(
                state, "10.0.2.2",
                "CVE-2019-0232 and Tomcat manager WAR upload both confirmed as viable exploit paths.",
            )
            self._add_note_if_missing(
                state, "10.0.2.2",
                "msfconsole: use exploit/multi/handler + msfvenom WAR payload is recommended.",
            )
            # ── Advance stage: enumerated → analyzed ──
            state.advance_host_stage("10.0.2.2", "analyzed")
            changes.extend([
                "Identified CVE-2019-0232 as applicable to this Tomcat version.",
                "WAR upload exploit path confirmed — msfvenom payload viable.",
                "Host 10.0.2.2 stage advanced to: analyzed.",
            ])
            return {
                "success": True,
                "changes": changes,
                "summary": "Analysis identified a concrete Tomcat exploit path — ready to exploit.",
            }

        if target_host == "10.0.0.1":
            self._ensure_gateway_host(state)
            self._add_note_if_missing(
                state, "10.0.0.1",
                "Gateway role confirmed — routing to 10.0.4.0/24 is present but filtered.",
            )
            # Gateway hosts have no exploit path — do NOT advance stage past enumerated.
            changes.append("Gateway role noted. No further progression — focus on exploitable targets.")
            return {
                "success": True,
                "changes": changes,
                "summary": "Gateway confirmed as routing host only — not an exploit target.",
            }

        return {
            "success": True,
            "changes": [],
            "summary": "Analysis completed — no major new findings.",
        }

    def _apply_exploit(
        self,
        state: NetworkState,
        target_host: Optional[str],
        command: Optional[str],
    ) -> Dict[str, Any]:
        changes: List[str] = []

        if target_host == "10.0.2.2":
            self._ensure_tomcat_host(state)

            # Require analyze to have succeeded before exploit is valid
            already_done = {(e["action_type"], e["target_ip"]) for e in state.history if e.get("success")}
            if ("analyze", "10.0.2.2") not in already_done:
                return {
                    "success": False,
                    "changes": [],
                    "summary": "Cannot exploit yet — analyze 10.0.2.2 first to identify the attack path.",
                }

            if not state.known_hosts["10.0.2.2"]["compromised"]:
                # mark_compromised also calls advance_host_stage("exploited")
                state.mark_compromised("10.0.2.2")
                self._add_note_if_missing(
                    state, "10.0.2.2",
                    "Shell obtained via WAR upload — host is now a live pivot point.",
                )
                changes.extend([
                    "Deployed malicious WAR file via Tomcat manager.",
                    "Reverse shell obtained — 10.0.2.2 is now compromised.",
                    "Host 10.0.2.2 stage advanced to: exploited.",
                ])
            else:
                changes.append("10.0.2.2 was already compromised.")

            return {
                "success": True,
                "changes": changes,
                "summary": "Exploit succeeded — foothold established on 10.0.2.2.",
            }

        return {
            "success": False,
            "changes": [],
            "summary": "No viable exploit path defined for this host in the current scenario.",
        }

    def _apply_pivot(
        self,
        state: NetworkState,
        target_host: Optional[str],
        command: Optional[str],
    ) -> Dict[str, Any]:
        changes: List[str] = []

        if target_host != "10.0.2.2":
            return {
                "success": False,
                "changes": [],
                "summary": "No pivot path defined for the selected host.",
            }

        if not self._pivot_available(state):
            return {
                "success": False,
                "changes": [],
                "summary": "Pivoting requires a compromised foothold first — exploit 10.0.2.2 first.",
            }

        # Unblock the internal network now that we have a pivot
        if "10.0.4.0/24" in state.blocked_networks:
            state.blocked_networks.remove("10.0.4.0/24")
            state.add_scope_network("10.0.4.0/24")
            changes.append("10.0.4.0/24 removed from blocked networks — now reachable via pivot.")

        discovered = self._ensure_internal_target_host(state)
        if discovered:
            changes.append("Pivot scan revealed internal target host 10.0.4.3.")

        self._add_note_if_missing(
            state, "10.0.2.2",
            "Pivot tunnel active — proxychains routes through this host into 10.0.4.0/24.",
        )
        # ── Advance stage: exploited → pivoted ──
        state.advance_host_stage("10.0.2.2", "pivoted")
        changes.append("Host 10.0.2.2 stage advanced to: pivoted.")

        return {
            "success": True,
            "changes": changes,
            "summary": "Pivot established — internal network 10.0.4.0/24 is now reachable.",
        }

    def _apply_access(
        self,
        state: NetworkState,
        target_host: Optional[str],
        command: Optional[str],
    ) -> Dict[str, Any]:
        changes: List[str] = []

        if target_host == state.target_ip:
            if self._pivot_available(state) and state.is_known_host(target_host):
                self._add_note_if_missing(
                    state, target_host,
                    "Target is now accessible via the pivot tunnel — flag capture is possible.",
                )
                # ── Advance both the foothold and the target stage ──
                state.advance_host_stage("10.0.2.2", "accessed")
                state.advance_host_stage(target_host, "accessed")
                state.add_global_note("CTF objective reached — target host accessed via pivot.")
                changes.extend([
                    f"Connected to target {target_host} via pivot tunnel.",
                    "Flag capture is now possible.",
                    f"Host {target_host} stage advanced to: accessed.",
                ])
                return {
                    "success": True,
                    "changes": changes,
                    "summary": f"Target {target_host} accessed — CTF objective complete.",
                }

            return {
                "success": False,
                "changes": [],
                "summary": "Target not yet reachable — ensure pivot is established and target is discovered.",
            }

        return {
            "success": False,
            "changes": [],
            "summary": "Access action did not match any reachable target in the current scenario.",
        }

    # ------------------------------------------------------------------
    # Scenario host builders
    # ------------------------------------------------------------------

    def _ensure_gateway_host(self, state: NetworkState) -> bool:
        if state.is_known_host("10.0.0.1"):
            return False

        state.known_hosts["10.0.0.1"] = self._make_host_record(
            ip="10.0.0.1",
            services=[
                self._make_service_record(
                    port=2601, protocol="tcp", service_name="ospfd",
                    state="open", product=None, version=None,
                )
            ],
            notes=["Discovered via initial subnet scan"],
            compromised=False,
            pivot_candidate=False,
            gateway_candidate=True,
            stage="discovered",
        )
        self._add_gateway_candidate(state, "10.0.0.1")
        return True

    def _ensure_tomcat_host(self, state: NetworkState) -> bool:
        if state.is_known_host("10.0.2.2"):
            return False

        state.known_hosts["10.0.2.2"] = self._make_host_record(
            ip="10.0.2.2",
            services=[
                self._make_service_record(
                    port=8080, protocol="tcp", service_name="http-proxy",
                    state="open", product="Apache Tomcat", version="9.0",
                )
            ],
            notes=[
                "Discovered via initial subnet scan",
                "Web-facing service — strong candidate for enumeration",
                "Apache Tomcat detected on port 8080",
            ],
            compromised=False,
            pivot_candidate=True,
            gateway_candidate=False,
            stage="discovered",
        )
        self._add_pivot_candidate(state, "10.0.2.2")
        return True

    def _ensure_internal_target_host(self, state: NetworkState) -> bool:
        target_ip = state.target_ip
        if not target_ip or state.is_known_host(target_ip):
            return False

        state.known_hosts[target_ip] = self._make_host_record(
            ip=target_ip,
            services=[],
            notes=["Discovered after pivot into 10.0.4.0/24 — CTF target host"],
            compromised=False,
            pivot_candidate=False,
            gateway_candidate=False,
            stage="discovered",
        )
        return True

    def _pivot_available(self, state: NetworkState) -> bool:
        return "10.0.2.2" in state.get_compromised_hosts()

    # ------------------------------------------------------------------
    # Record builders / helpers
    # ------------------------------------------------------------------

    def _make_service_record(
        self,
        port: int,
        protocol: str,
        service_name: str,
        state: str = "open",
        product: Optional[str] = None,
        version: Optional[str] = None,
        notes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {
            "port": port,
            "protocol": protocol,
            "state": state,
            "service_name": service_name,
            "product": product,
            "version": version,
            "notes": notes or [],
        }

    def _make_host_record(
        self,
        ip: str,
        services: Optional[List[Dict[str, Any]]] = None,
        notes: Optional[List[str]] = None,
        compromised: bool = False,
        pivot_candidate: bool = False,
        gateway_candidate: bool = False,
        hostname: Optional[str] = None,
        os_guess: Optional[str] = None,
        stage: str = "discovered",
    ) -> Dict[str, Any]:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        return {
            "ip": ip,
            "hostname": hostname,
            "os_guess": os_guess,
            "services": services or [],
            "notes": notes or [],
            "compromised": compromised,
            "pivot_candidate": pivot_candidate,
            "gateway_candidate": gateway_candidate,
            "stage": stage,                          # ← kill-chain stage
            "discovered_at": now,
            "last_updated": now,
        }

    def _add_note_if_missing(self, state: NetworkState, ip: str, note: str) -> None:
        host = state.known_hosts.get(ip)
        if host is None:
            return
        if "notes" not in host or host["notes"] is None:
            host["notes"] = []
        if note not in host["notes"]:
            host["notes"].append(note)

    def _add_gateway_candidate(self, state: NetworkState, ip: str) -> None:
        if hasattr(state, "gateway_candidates") and ip not in state.gateway_candidates:
            state.gateway_candidates.append(ip)
        host = state.known_hosts.get(ip)
        if host is not None:
            host["gateway_candidate"] = True

    def _add_pivot_candidate(self, state: NetworkState, ip: str) -> None:
        if hasattr(state, "pivot_hosts") and ip not in state.pivot_hosts:
            state.pivot_hosts.append(ip)
        host = state.known_hosts.get(ip)
        if host is not None:
            host["pivot_candidate"] = True
