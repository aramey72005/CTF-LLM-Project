from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from src.models.network_state import NetworkState


class StateManager:
    """
    Applies simulated kill-chain transitions for the CTF planning environment.

    The manager is intentionally scenario-agnostic: it looks at per-host
    properties such as services, exploit_profile, and pivot_capable instead of
    assuming there is only one valid foothold host.
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

        if not state.known_hosts:
            changes.extend(self._seed_initial_scan_hosts(state))
            return {
                "success": True,
                "changes": changes,
                "summary": "Initial scan revealed the externally reachable hosts.",
            }

        if any(network in cmd for network in state.blocked_networks) or state.target_ip in cmd:
            if self._pivot_available(state):
                discovered = self._ensure_internal_target_host(state)
                if discovered:
                    state.advance_host_stage(state.target_ip, "discovered")
                    changes.append(f"Pivot-assisted scan revealed internal target host {state.target_ip}.")
                else:
                    changes.append(f"Internal target host {state.target_ip} was already known.")
                return {
                    "success": True,
                    "changes": changes,
                    "summary": "Pivot-assisted scan revealed the internal target network.",
                }
            return {
                "success": False,
                "changes": [],
                "summary": "Cannot scan the blocked network directly - establish a pivot first.",
            }

        if target_host and state.is_known_host(target_host):
            label = self._primary_service_label(state.known_hosts[target_host])
            self._add_note_if_missing(
                state, target_host, f"Detailed re-scan confirmed exposure on {label}."
            )
            changes.append(f"Re-scan confirmed exposed service on {target_host}.")
            return {
                "success": True,
                "changes": changes,
                "summary": f"Host re-scan confirmed service exposure on {target_host}.",
            }

        return {
            "success": True,
            "changes": [],
            "summary": "Scan completed - no major new discoveries.",
        }

    def _apply_enumerate(
        self,
        state: NetworkState,
        target_host: Optional[str],
        command: Optional[str],
    ) -> Dict[str, Any]:
        if not target_host or not state.is_known_host(target_host):
            return {
                "success": False,
                "changes": [],
                "summary": "Enumeration requires a known target host.",
            }

        changes: List[str] = []
        host = state.known_hosts[target_host]
        profile = self._get_exploit_profile(state, target_host)

        if self._host_is_gateway_only(host):
            self._add_note_if_missing(
                state,
                target_host,
                "Routing/gateway service confirmed - useful for topology, but not a direct exploit target.",
            )
            state.advance_host_stage(target_host, "enumerated")
            changes.append(f"Gateway host {target_host} noted - no exploit path, stage capped at enumerated.")
            return {
                "success": True,
                "changes": changes,
                "summary": f"Gateway host {target_host} enumerated - routing service noted, not an exploit target.",
            }

        if profile == "tomcat_mgr_upload":
            self._add_note_if_missing(
                state, target_host, "Tomcat manager interface (/manager/html) is exposed - default credentials may work."
            )
            self._add_note_if_missing(
                state, target_host, "WAR file deployment via Tomcat manager is a known RCE vector."
            )
            changes.extend([
                f"Identified exposed Tomcat manager interface on {target_host}.",
                "Noted WAR deployment as a likely RCE vector.",
            ])
        elif self._host_is_exploitable(host):
            label = self._primary_service_label(host)
            self._add_note_if_missing(
                state,
                target_host,
                f"Enumeration found an application surface on {label}; deeper analysis is needed before exploitation.",
            )
            changes.append(f"Enumerated {target_host} and confirmed application surface on {label}.")
        else:
            changes.append(f"Enumeration on {target_host} completed, but no clear exploit path was identified.")

        state.advance_host_stage(target_host, "enumerated")
        changes.append(f"Host {target_host} stage advanced to: enumerated.")
        return {
            "success": True,
            "changes": changes,
            "summary": f"Enumeration completed on {target_host}.",
        }

    def _apply_analyze(
        self,
        state: NetworkState,
        target_host: Optional[str],
        command: Optional[str],
    ) -> Dict[str, Any]:
        if not target_host or not state.is_known_host(target_host):
            return {
                "success": False,
                "changes": [],
                "summary": "Analysis requires a known target host.",
            }

        changes: List[str] = []
        host = state.known_hosts[target_host]
        profile = self._get_exploit_profile(state, target_host)
        cmd = (command or "").lower()

        if self._host_is_gateway_only(host):
            self._add_note_if_missing(
                state,
                target_host,
                "Gateway role confirmed - this host routes traffic but does not expose a viable exploit path.",
            )
            changes.append("Gateway role noted. No further progression - focus on exploitable targets.")
            return {
                "success": True,
                "changes": changes,
                "summary": "Gateway confirmed as routing host only - not an exploit target.",
            }

        if profile == "tomcat_mgr_upload" or "tomcat" in cmd:
            self._add_note_if_missing(
                state, target_host, "CVE-2019-0232 and Tomcat manager WAR upload both confirmed as viable exploit paths."
            )
            self._add_note_if_missing(
                state, target_host, "msfconsole: use exploit/multi/http/tomcat_mgr_upload or an msfvenom WAR payload."
            )
            changes.extend([
                f"Identified CVE-2019-0232 as applicable to {target_host}.",
                "WAR upload exploit path confirmed - msfvenom payload viable.",
            ])
            state.advance_host_stage(target_host, "analyzed")
            changes.append(f"Host {target_host} stage advanced to: analyzed.")
            return {
                "success": True,
                "changes": changes,
                "summary": f"Analysis identified a concrete Tomcat exploit path on {target_host}.",
            }

        if profile:
            self._add_note_if_missing(state, target_host, f"Exploit profile confirmed: {profile}.")
            changes.extend([
                f"Validated exploit profile {profile} against {target_host}.",
                f"Host {target_host} is ready for exploitation.",
            ])
            state.advance_host_stage(target_host, "analyzed")
            changes.append(f"Host {target_host} stage advanced to: analyzed.")
            return {
                "success": True,
                "changes": changes,
                "summary": f"Analysis identified a workable exploit path on {target_host}.",
            }

        self._add_note_if_missing(
            state, target_host, "Analysis completed with no viable exploit path in this exercise."
        )
        changes.append(f"Analyzed {target_host}, but no supported exploit path was found.")
        return {
            "success": True,
            "changes": changes,
            "summary": f"Analysis on {target_host} found no viable exploit path.",
        }

    def _apply_exploit(
        self,
        state: NetworkState,
        target_host: Optional[str],
        command: Optional[str],
    ) -> Dict[str, Any]:
        if not target_host or not state.is_known_host(target_host):
            return {
                "success": False,
                "changes": [],
                "summary": "Exploit requires a known target host.",
            }

        already_done = {(e["action_type"], e["target_ip"]) for e in state.history if e.get("success")}
        if ("analyze", target_host) not in already_done:
            return {
                "success": False,
                "changes": [],
                "summary": f"Cannot exploit yet - analyze {target_host} first to identify the attack path.",
            }

        profile = self._get_exploit_profile(state, target_host)
        if not profile:
            return {
                "success": False,
                "changes": [],
                "summary": "No viable exploit path defined for this host in the current scenario.",
            }

        changes: List[str] = []
        host = state.known_hosts[target_host]
        if not host.get("compromised"):
            state.mark_compromised(target_host)
            if profile == "tomcat_mgr_upload":
                self._add_note_if_missing(
                    state, target_host, "Shell obtained via WAR upload - host is now a live pivot point."
                )
                changes.append("Deployed malicious WAR file via Tomcat manager.")
            else:
                self._add_note_if_missing(
                    state, target_host, f"Exploit profile {profile} succeeded - shell obtained."
                )
                changes.append(f"Executed exploit profile {profile} successfully.")
            changes.extend([
                f"Reverse shell obtained - {target_host} is now compromised.",
                f"Host {target_host} stage advanced to: exploited.",
            ])
        else:
            changes.append(f"{target_host} was already compromised.")

        return {
            "success": True,
            "changes": changes,
            "summary": f"Exploit succeeded - foothold established on {target_host}.",
        }

    def _apply_pivot(
        self,
        state: NetworkState,
        target_host: Optional[str],
        command: Optional[str],
    ) -> Dict[str, Any]:
        if not target_host or not state.is_known_host(target_host):
            return {
                "success": False,
                "changes": [],
                "summary": "Pivoting requires a known compromised host.",
            }

        if not self._host_can_pivot(state, target_host):
            return {
                "success": False,
                "changes": [],
                "summary": "No pivot path defined for the selected host.",
            }

        if not self._pivot_available(state):
            return {
                "success": False,
                "changes": [],
                "summary": f"Pivoting requires a compromised foothold first - exploit {target_host} first.",
            }

        changes: List[str] = []
        for network in list(state.blocked_networks):
            state.blocked_networks.remove(network)
            state.add_scope_network(network)
            changes.append(f"{network} removed from blocked networks - now reachable via pivot.")

        if self._ensure_internal_target_host(state):
            changes.append(f"Pivot scan revealed internal target host {state.target_ip}.")

        self._add_note_if_missing(
            state, target_host, "Pivot tunnel active - proxychains routes through this host into the blocked subnet."
        )
        state.advance_host_stage(target_host, "pivoted")
        changes.append(f"Host {target_host} stage advanced to: pivoted.")

        return {
            "success": True,
            "changes": changes,
            "summary": "Pivot established - the blocked internal network is now reachable.",
        }

    def _apply_access(
        self,
        state: NetworkState,
        target_host: Optional[str],
        command: Optional[str],
    ) -> Dict[str, Any]:
        if target_host != state.target_ip:
            return {
                "success": False,
                "changes": [],
                "summary": "Access action did not match any reachable target in the current scenario.",
            }

        if not (self._pivot_available(state) and state.is_known_host(target_host)):
            return {
                "success": False,
                "changes": [],
                "summary": "Target not yet reachable - ensure pivot is established and target is discovered.",
            }

        changes: List[str] = []
        self._add_note_if_missing(
            state, target_host, "Target is now accessible via the pivot tunnel - flag capture is possible."
        )
        pivot_host = self._get_active_pivot_host(state)
        if pivot_host:
            state.advance_host_stage(pivot_host, "accessed")
        state.advance_host_stage(target_host, "accessed")
        state.add_global_note("CTF objective reached - target host accessed via pivot.")
        changes.extend([
            f"Connected to target {target_host} via pivot tunnel.",
            "Flag capture is now possible.",
            f"Host {target_host} stage advanced to: accessed.",
        ])
        return {
            "success": True,
            "changes": changes,
            "summary": f"Target {target_host} accessed - CTF objective complete.",
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
                    port=2601, protocol="tcp", service_name="ospfd", state="open", product=None, version=None
                )
            ],
            notes=["Discovered via initial subnet scan"],
            compromised=False,
            pivot_candidate=False,
            gateway_candidate=True,
            stage="discovered",
        )
        state.known_hosts["10.0.0.1"]["pivot_capable"] = False
        state.known_hosts["10.0.0.1"]["exploit_profile"] = None
        self._add_gateway_candidate(state, "10.0.0.1")
        return True

    def _ensure_tomcat_host(self, state: NetworkState) -> bool:
        if state.is_known_host("10.0.2.2"):
            return False

        state.known_hosts["10.0.2.2"] = self._make_host_record(
            ip="10.0.2.2",
            services=[
                self._make_service_record(
                    port=8080,
                    protocol="tcp",
                    service_name="http-proxy",
                    state="open",
                    product="Apache Tomcat",
                    version="9.0",
                )
            ],
            notes=[
                "Discovered via initial subnet scan",
                "Web-facing service - strong candidate for enumeration",
                "Apache Tomcat detected on port 8080",
            ],
            compromised=False,
            pivot_candidate=True,
            gateway_candidate=False,
            stage="discovered",
        )
        state.known_hosts["10.0.2.2"]["pivot_capable"] = True
        state.known_hosts["10.0.2.2"]["exploit_profile"] = "tomcat_mgr_upload"
        self._add_pivot_candidate(state, "10.0.2.2")
        return True

    def _ensure_internal_target_host(self, state: NetworkState) -> bool:
        target_ip = state.target_ip
        if not target_ip or state.is_known_host(target_ip):
            return False

        state.known_hosts[target_ip] = self._make_host_record(
            ip=target_ip,
            services=[],
            notes=["Discovered after pivot into the blocked subnet - CTF target host"],
            compromised=False,
            pivot_candidate=False,
            gateway_candidate=False,
            stage="discovered",
        )
        return True

    def _seed_initial_scan_hosts(self, state: NetworkState) -> List[str]:
        profile = state.metadata.get("initial_scan_hosts", [])
        if not profile:
            self._ensure_gateway_host(state)
            self._ensure_tomcat_host(state)
            return [
                "Discovered host 10.0.0.1 - routing service on port 2601.",
                "Discovered host 10.0.2.2 - Apache Tomcat on port 8080.",
            ]

        changes: List[str] = []
        for entry in profile:
            ip = entry["ip"]
            state.add_host(ip)
            host = state.known_hosts[ip]
            host["exploit_profile"] = entry.get("exploit_profile")
            host["pivot_capable"] = bool(entry.get("pivot_capable", False))
            for service in entry.get("services", []):
                state.add_service(
                    ip=ip,
                    port=service["port"],
                    protocol=service.get("protocol", "tcp"),
                    service_name=service["service_name"],
                    state=service.get("state", "open"),
                    product=service.get("product"),
                    version=service.get("version"),
                )
            for note in entry.get("notes", []):
                state.add_host_note(ip, note)
            if entry.get("pivot_capable"):
                state.mark_pivot_candidate(ip, "Foothold candidate discovered during scan")
            if entry.get("gateway_candidate"):
                state.mark_gateway_candidate(ip, "Gateway/routing host discovered during scan")
            svc_summary = ", ".join(
                f"port {svc['port']} {svc['service_name']}" for svc in entry.get("services", [])
            ) or "no visible services"
            changes.append(f"Discovered host {ip} - {svc_summary}.")
        return changes

    def _pivot_available(self, state: NetworkState) -> bool:
        return self._get_active_pivot_host(state) is not None

    def _get_active_pivot_host(self, state: NetworkState) -> Optional[str]:
        for ip, host in state.known_hosts.items():
            if host.get("compromised") and self._host_can_pivot(state, ip):
                return ip
        return None

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
            "stage": stage,
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

    def _host_can_pivot(self, state: NetworkState, ip: str) -> bool:
        host = state.known_hosts.get(ip, {})
        return bool(host.get("pivot_capable") or ip in state.pivot_hosts)

    def _get_exploit_profile(self, state: NetworkState, ip: str) -> Optional[str]:
        host = state.known_hosts.get(ip, {})
        profile = host.get("exploit_profile")
        if profile:
            return str(profile)
        for service in host.get("services", []):
            if "tomcat" in (service.get("product") or "").lower():
                return "tomcat_mgr_upload"
        return None

    def _host_is_gateway_only(self, host: Dict[str, Any]) -> bool:
        services = host.get("services", [])
        if not services:
            return False
        return all(
            svc.get("service_name", "").lower() in {"ospfd", "bgpd", "zebra"} or svc.get("port") == 2601
            for svc in services
        )

    def _host_is_exploitable(self, host: Dict[str, Any]) -> bool:
        if host.get("exploit_profile"):
            return True
        return any(
            svc.get("port") in {80, 443, 8080, 8443}
            or svc.get("service_name", "").lower() in {"http", "https", "http-proxy", "tomcat"}
            or "tomcat" in (svc.get("product") or "").lower()
            for svc in host.get("services", [])
        )

    def _primary_service_label(self, host: Dict[str, Any]) -> str:
        services = host.get("services", [])
        if not services:
            return "unknown service"
        service = services[0]
        return f"port {service.get('port')} ({service.get('service_name')})"
