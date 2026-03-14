from src.models.network_state import NetworkState
from src.parsers.nmap_parser import NmapParser


def main() -> None:
    state = NetworkState(
        target_ip="10.0.4.3",
        scope_networks=[
            "10.0.0.0/24",
            "10.0.2.0/24",
            "10.0.4.0/24",
        ],
        blocked_networks=["10.0.4.0/24"],
    )

    sample_nmap_output = """
Nmap scan report for 10.0.0.1
Host is up.
Not shown: 999 closed ports
PORT     STATE SERVICE
2601/tcp open  ospfd

Nmap scan report for 10.0.2.2
Host is up.
Not shown: 999 closed ports
PORT     STATE SERVICE VERSION
8080/tcp open  http-proxy Apache Tomcat 9.0
"""

    parser = NmapParser()
    parser.update_network_state(sample_nmap_output, state)

    print(state.to_prompt_context())


if __name__ == "__main__":
    main()