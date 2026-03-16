from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from ipaddress import ip_address, ip_network
from typing import Any, Dict, List, Optional

# Ordered progression stages for a host through the kill chain.
# Each stage unlocks the next recommended action type in the planner.
HOST_STAGES = ["discovered", "enumerated", "analyzed", "exploited", "pivoted", "accessed"]


@dataclass
class NetworkState:
    """
    Central structured memory for the CTF environment.

    Each known host now carries a 'stage' field that tracks where it sits
    in the kill chain:
        discovered → enumerated → analyzed → exploited → pivoted → accessed

    The Planner reads this to decide what the next logical action is,
    ensuring recommendations always advance rather than repeat.
    """

    target_ip: str
    scope_networks: List[str] = field(default_factory=list)
    blocked_networks: List[str] = field(default_factory=list)
    known_hosts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pivot_hosts: List[str] = field(default_factory=list)
    gateway_candidates: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._validate_ip(self.target_ip)
        for network in self.scope_networks:
            self._validate_network(network)
        for network in self.blocked_networks:
            self._validate_network(network)

    # ------------------------------------------------------------------
    # Internal validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_ip(ip: str) -> None:
        try:
            ip_address(ip)
        except ValueError as exc:
            raise ValueError(f"Invalid IP address: {ip}") from exc

    @staticmethod
    def _validate_network(network: str) -> None:
        try:
            ip_network(network, strict=False)
        except ValueError as exc:
            raise ValueError(f"Invalid network: {network}") from exc

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _ensure_host_exists(self, ip: str) -> None:
        self._validate_ip(ip)
        if ip not in self.known_hosts:
            self.known_hosts[ip] = {
                "hostname": None,
                "services": [],
                "compromised": False,
                "pivot_candidate": False,
                "gateway_candidate": False,
                "os_guess": None,
                "notes": [],
                "stage": "discovered",          # ← kill-chain stage
                "discovered_at": self._utc_now(),
                "last_updated": self._utc_now(),
            }

    def _touch_host(self, ip: str) -> None:
        self._ensure_host_exists(ip)
        self.known_hosts[ip]["last_updated"] = self._utc_now()

    # ------------------------------------------------------------------
    # Scope / environment checks
    # ------------------------------------------------------------------

    def is_ip_in_scope(self, ip: str) -> bool:
        self._validate_ip(ip)
        if not self.scope_networks:
            return True
        ip_obj = ip_address(ip)
        return any(ip_obj in ip_network(net, strict=False) for net in self.scope_networks)

    def is_known_host(self, ip: str) -> bool:
        return ip in self.known_hosts

    def is_target_host(self, ip: str) -> bool:
        return ip == self.target_ip

    def add_scope_network(self, network: str) -> None:
        self._validate_network(network)
        if network not in self.scope_networks:
            self.scope_networks.append(network)

    def add_blocked_network(self, network: str) -> None:
        self._validate_network(network)
        if network not in self.blocked_networks:
            self.blocked_networks.append(network)

    # ------------------------------------------------------------------
    # Host management
    # ------------------------------------------------------------------

    def add_host(self, ip: str, hostname: Optional[str] = None, note: Optional[str] = None) -> None:
        self._ensure_host_exists(ip)
        if hostname:
            self.known_hosts[ip]["hostname"] = hostname
        if note:
            self.add_host_note(ip, note)
        self._touch_host(ip)

    def set_os_guess(self, ip: str, os_guess: str) -> None:
        self._ensure_host_exists(ip)
        self.known_hosts[ip]["os_guess"] = os_guess
        self._touch_host(ip)

    def add_host_note(self, ip: str, note: str) -> None:
        self._ensure_host_exists(ip)
        if note and note not in self.known_hosts[ip]["notes"]:
            self.known_hosts[ip]["notes"].append(note)
        self._touch_host(ip)

    def advance_host_stage(self, ip: str, stage: str) -> None:
        """
        Move a host to the given stage, but only if it is a forward move.
        Silently ignores attempts to go backwards so out-of-order calls are safe.
        """
        self._ensure_host_exists(ip)
        if stage not in HOST_STAGES:
            return
        current = self.known_hosts[ip].get("stage", "discovered")
        current_idx = HOST_STAGES.index(current) if current in HOST_STAGES else 0
        new_idx = HOST_STAGES.index(stage)
        if new_idx > current_idx:
            self.known_hosts[ip]["stage"] = stage
            self._touch_host(ip)

    def get_host_stage(self, ip: str) -> str:
        if ip not in self.known_hosts:
            return "discovered"
        return self.known_hosts[ip].get("stage", "discovered")

    def mark_compromised(self, ip: str) -> None:
        self._ensure_host_exists(ip)
        self.known_hosts[ip]["compromised"] = True
        if ip not in self.pivot_hosts:
            self.pivot_hosts.append(ip)
        self.advance_host_stage(ip, "exploited")
        self._touch_host(ip)

    def mark_pivot_candidate(self, ip: str, reason: Optional[str] = None) -> None:
        self._ensure_host_exists(ip)
        self.known_hosts[ip]["pivot_candidate"] = True
        if ip not in self.pivot_hosts:
            self.pivot_hosts.append(ip)
        if reason:
            self.add_host_note(ip, f"Pivot candidate: {reason}")
        self._touch_host(ip)

    def mark_gateway_candidate(self, ip: str, reason: Optional[str] = None) -> None:
        self._ensure_host_exists(ip)
        self.known_hosts[ip]["gateway_candidate"] = True
        if ip not in self.gateway_candidates:
            self.gateway_candidates.append(ip)
        if reason:
            self.add_host_note(ip, f"Gateway candidate: {reason}")
        self._touch_host(ip)

    # ------------------------------------------------------------------
    # Service management
    # ------------------------------------------------------------------

    def add_service(
        self,
        ip: str,
        port: int,
        protocol: str,
        service_name: str,
        state: str = "open",
        version: Optional[str] = None,
        product: Optional[str] = None,
        note: Optional[str] = None,
    ) -> None:
        self._ensure_host_exists(ip)
        service_record = {
            "port": int(port),
            "protocol": protocol.lower(),
            "service_name": service_name,
            "state": state,
            "version": version,
            "product": product,
            "notes": [],
        }
        existing = self.get_service(ip, port, protocol)
        if existing is None:
            self.known_hosts[ip]["services"].append(service_record)
        else:
            existing["service_name"] = service_name
            existing["state"] = state
            existing["version"] = version
            existing["product"] = product
        if note:
            self.add_service_note(ip, port, protocol, note)
        self._touch_host(ip)

    def get_service(self, ip: str, port: int, protocol: str) -> Optional[Dict[str, Any]]:
        self._ensure_host_exists(ip)
        for service in self.known_hosts[ip]["services"]:
            if service["port"] == int(port) and service["protocol"] == protocol.lower():
                return service
        return None

    def add_service_note(self, ip: str, port: int, protocol: str, note: str) -> None:
        service = self.get_service(ip, port, protocol)
        if service is None:
            raise ValueError(f"Service {port}/{protocol} not found on host {ip}")
        if note and note not in service["notes"]:
            service["notes"].append(note)
        self._touch_host(ip)

    def host_has_service_name(self, ip: str, service_name: str) -> bool:
        if ip not in self.known_hosts:
            return False
        return any(
            service["service_name"].lower() == service_name.lower()
            for service in self.known_hosts[ip]["services"]
        )

    # ------------------------------------------------------------------
    # History / logging
    # ------------------------------------------------------------------

    def record_action(
        self,
        action_type: str,
        description: str,
        target_ip: Optional[str] = None,
        command: Optional[str] = None,
        success: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "timestamp": self._utc_now(),
            "action_type": action_type,
            "description": description,
            "target_ip": target_ip,
            "command": command,
            "success": success,
            "metadata": metadata or {},
        }
        self.history.append(entry)

    def add_global_note(self, note: str) -> None:
        if note and note not in self.notes:
            self.notes.append(note)

    # ------------------------------------------------------------------
    # LLM-facing summaries
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_ip": self.target_ip,
            "scope_networks": self.scope_networks,
            "blocked_networks": self.blocked_networks,
            "known_hosts": self.known_hosts,
            "pivot_hosts": self.pivot_hosts,
            "gateway_candidates": self.gateway_candidates,
            "history": self.history,
            "notes": self.notes,
        }

    def summarize_hosts(self) -> List[str]:
        lines: List[str] = []

        for ip in sorted(self.known_hosts.keys(), key=lambda x: tuple(int(p) for p in x.split("."))):
            host = self.known_hosts[ip]

            labels: List[str] = []
            if host["compromised"]:
                labels.append("compromised")
            if host["pivot_candidate"]:
                labels.append("pivot-candidate")
            if host["gateway_candidate"]:
                labels.append("gateway-candidate")
            if self.is_target_host(ip):
                labels.append("target")

            stage = host.get("stage", "discovered")
            labels.append(f"stage:{stage}")          # ← stage always visible

            label_str = f" [{' | '.join(labels)}]" if labels else ""
            header = f"- Host {ip}{label_str}"

            if host["hostname"]:
                header += f" hostname={host['hostname']}"
            if host["os_guess"]:
                header += f" os={host['os_guess']}"

            lines.append(header)

            if host["services"]:
                for service in sorted(host["services"], key=lambda s: (s["port"], s["protocol"])):
                    service_line = (
                        f"  - {service['port']}/{service['protocol']} "
                        f"{service['state']} {service['service_name']}"
                    )
                    if service["product"]:
                        service_line += f" product={service['product']}"
                    if service["version"]:
                        service_line += f" version={service['version']}"
                    lines.append(service_line)
                    for note in service["notes"]:
                        lines.append(f"    note: {note}")

            for note in host["notes"]:
                lines.append(f"  note: {note}")

        return lines

    def to_prompt_context(self) -> str:
        lines: List[str] = [
            "Current Network State",
            "=====================",
            f"Target Host: {self.target_ip}",
            f"In-Scope Networks: {', '.join(self.scope_networks) if self.scope_networks else 'Not set'}",
            f"Blocked Networks: {', '.join(self.blocked_networks) if self.blocked_networks else 'None recorded'}",
            f"Pivot Hosts: {', '.join(self.pivot_hosts) if self.pivot_hosts else 'None'}",
            f"Gateway Candidates: {', '.join(self.gateway_candidates) if self.gateway_candidates else 'None'}",
            "",
            "Known Hosts:",
        ]

        host_lines = self.summarize_hosts()
        if host_lines:
            lines.extend(host_lines)
        else:
            lines.append("- None discovered yet")

        if self.notes:
            lines.extend(["", "Global Notes:"])
            for note in self.notes:
                lines.append(f"- {note}")

        if self.history:
            lines.extend(["", "Recent Actions:"])
            for entry in self.history[-5:]:
                action_line = f"- {entry['action_type']}: {entry['description']}"
                if entry["target_ip"]:
                    action_line += f" (target={entry['target_ip']})"
                if entry["success"] is not None:
                    action_line += f" success={entry['success']}"
                lines.append(action_line)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Useful analysis helpers
    # ------------------------------------------------------------------

    def get_compromised_hosts(self) -> List[str]:
        return [ip for ip, host in self.known_hosts.items() if host["compromised"]]

    def get_uncompromised_hosts(self) -> List[str]:
        return [ip for ip, host in self.known_hosts.items() if not host["compromised"]]

    def find_hosts_with_service(self, service_name: str) -> List[str]:
        matches: List[str] = []
        for ip, host in self.known_hosts.items():
            for service in host["services"]:
                if service["service_name"].lower() == service_name.lower():
                    matches.append(ip)
                    break
        return matches

    def get_already_done(self) -> set:
        """
        Returns a set of (action_type, target_ip) tuples for successful
        actions already recorded in history. Used by the planner to avoid
        recommending the same action twice.
        """
        return {
            (e["action_type"], e["target_ip"])
            for e in self.history
            if e.get("success") is True
        }
