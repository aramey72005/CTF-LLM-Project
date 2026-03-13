from src.models.network_state import NetworkState


def main():

    state = NetworkState(
        target_ip="10.0.4.3",
        scope_networks=[
            "10.0.0.0/24",
            "10.0.2.0/24",
            "10.0.4.0/24"
        ],
        blocked_networks=["10.0.4.0/24"]
    )

    # simulate discovering hosts
    state.add_host("10.0.0.1")
    state.add_host("10.0.2.2")

    # simulate scan results
    state.add_service("10.0.0.1", 2601, "tcp", "ospfd")
    state.add_service("10.0.2.2", 8080, "tcp", "tomcat")

    # mark special hosts
    state.mark_gateway_candidate("10.0.0.1", "Routing service observed")
    state.mark_pivot_candidate("10.0.2.2", "Possible pivot point")

    # record an action
    state.record_action(
        action_type="scan",
        description="Initial network scan",
        command="nmap 10.0.0.0/24"
    )

    # print summary
    print(state.to_prompt_context())


if __name__ == "__main__":
    main()