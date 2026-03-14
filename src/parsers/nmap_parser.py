from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.models.network_state import NetworkState


class NmapParser:
    """
    Parses raw Nmap output and updates a NetworkState object.

    Supported patterns:
    - Nmap scan report for <ip>
    - Host is up
    - PORT STATE SERVICE VERSION
    - service rows like:
        8080/tcp open  http-proxy
        2601/tcp open  ospfd
        80/tcp   open  http        Apache httpd 2.4.41
    """

    HOST_LINE_RE = re.compile(r"^Nmap scan report for (?P<host>\S+)$")
    PORT_LINE_RE = re.compile(
        r"^(?P<port>\d+)\/(?P<protocol>\w+)\s+"
        r"(?P<state>\S+)\s+"
        r"(?P<service>\S+)"
        r"(?:\s+(?P<version>.+))?$"
    )

    def parse_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse raw Nmap output into a list of host dictionaries.

        Returns a structure like:
        [
            {
                "ip": "10.0.2.2",
                "services": [
                    {
                        "port": 8080,
                        "protocol": "tcp",
                        "state": "open",
                        "service_name": "http-proxy",
                        "version": None,
                        "product": None,
                    }
                ]
            }
        ]
        """
        hosts: List[Dict[str, Any]] = []
        current_host: Optional[Dict[str, Any]] = None

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            # Start of a new host block
            host_match = self.HOST_LINE_RE.match(line)
            if host_match:
                host_value = host_match.group("host")

                # Sometimes nmap says "hostname (ip)".
                ip = self._extract_ip(host_value)

                current_host = {
                    "ip": ip,
                    "services": [],
                    "notes": [],
                }
                hosts.append(current_host)
                continue

            # Ignore anything until a host is discovered
            if current_host is None:
                continue

            # Parse port/service lines
            port_match = self.PORT_LINE_RE.match(line)
            if port_match:
                port = int(port_match.group("port"))
                protocol = port_match.group("protocol").lower()
                state = port_match.group("state")
                service_name = port_match.group("service")
                version_blob = port_match.group("version")

                product, version = self._split_product_version(version_blob)

                current_host["services"].append(
                    {
                        "port": port,
                        "protocol": protocol,
                        "state": state,
                        "service_name": service_name,
                        "product": product,
                        "version": version,
                    }
                )
                continue

        return hosts

    def update_network_state(self, text: str, state: NetworkState) -> NetworkState:
        """
        Parse raw Nmap output and update the given NetworkState object.
        """
        parsed_hosts = self.parse_text(text)

        for host_data in parsed_hosts:
            ip = host_data["ip"]

            # Only add hosts in scope if scope is defined
            if state.scope_networks and not state.is_ip_in_scope(ip):
                state.record_action(
                    action_type="parser_skip",
                    description=f"Skipped out-of-scope host discovered in Nmap output: {ip}",
                    target_ip=ip,
                    success=False,
                )
                continue

            state.add_host(ip, note="Discovered from Nmap output")

            for service in host_data["services"]:
                state.add_service(
                    ip=ip,
                    port=service["port"],
                    protocol=service["protocol"],
                    service_name=service["service_name"],
                    state=service["state"],
                    product=service["product"],
                    version=service["version"],
                )

                self._apply_basic_inference(state, ip, service)

            state.record_action(
                action_type="nmap_parse",
                description=f"Parsed Nmap results for host {ip}",
                target_ip=ip,
                success=True,
            )

        return state

    def parse_file(self, file_path: str, state: NetworkState) -> NetworkState:
        """
        Convenience helper: read an Nmap output file and update NetworkState.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        return self.update_network_state(text, state)

    @staticmethod
    def _extract_ip(host_value: str) -> str:
        """
        Extracts IP from:
        - 10.0.2.2
        - host.local (10.0.2.2)
        """
        ip_match = re.search(r"(\d{1,3}(?:\.\d{1,3}){3})", host_value)
        if ip_match:
            return ip_match.group(1)
        return host_value

    @staticmethod
    def _split_product_version(version_blob: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """
        Best-effort split of the version info that Nmap prints after the service name.

        Examples:
        - "Apache httpd 2.4.41" -> ("Apache httpd", "2.4.41")
        - None -> (None, None)
        - "Jetty 9.4.z-SNAPSHOT" -> ("Jetty", "9.4.z-SNAPSHOT")
        """
        if not version_blob:
            return None, None

        version_blob = version_blob.strip()

        # Try to split on the last token if it looks version-like
        parts = version_blob.split()
        if len(parts) == 1:
            return parts[0], None

        last_token = parts[-1]
        if any(char.isdigit() for char in last_token):
            product = " ".join(parts[:-1]) if len(parts) > 1 else None
            version = last_token
            return product, version

        return version_blob, None

    @staticmethod
    def _apply_basic_inference(state: NetworkState, ip: str, service: Dict[str, Any]) -> None:
        """
        Lightweight inference rules that fit your project idea.
        These are intentionally simple and transparent.
        """
        service_name = service["service_name"].lower()
        port = service["port"]

        # Routing services are strategically important
        if service_name in {"ospfd", "bgpd", "zebra"}:
            state.mark_gateway_candidate(ip, f"Routing-related service detected: {service_name}")

        # Common web admin / web app pivot interest
        if port in {8080, 8443} or service_name in {"http", "http-proxy", "https", "tomcat"}:
            state.add_host_note(ip, "Web-facing service discovered; useful for further enumeration")

        # Tomcat-specific hint
        if "tomcat" in service_name or (service.get("product") and "tomcat" in service["product"].lower()):
            state.add_host_note(ip, "Apache Tomcat-related service detected")
            state.mark_pivot_candidate(ip, "Possible foothold via web application service")