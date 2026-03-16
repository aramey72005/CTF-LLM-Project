from src.experiments.planner_evaluation import PlannerEvaluation
from src.services.llm_client import LLMClient
from src.services.planner import Planner
from src.services.state_manager import StateManager


def print_divider(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    evaluator = PlannerEvaluation(max_actions=3)
    manager = StateManager()

    # Start from an initial state so we can watch it evolve
    state = evaluator.build_initial_recon_state()

    # Heuristic planner option:
    # planner = Planner(llm_callable=None, max_actions=3, use_mock_fallback=True, debug=False)

    # LLM planner option:
    client = LLMClient(
        base_url="http://localhost:11434",
        model="phi3",
        timeout=180,
    )
    planner = Planner(
        llm_callable=client.generate,
        max_actions=3,
        use_mock_fallback=True,
        debug=False,
    )

    print_divider("INITIAL STATE")
    print(state.to_prompt_context())

    for step in range(1, 6):
        print_divider(f"STEP {step}: PLANNER OUTPUT")

        actions = planner.plan(state)

        if not actions:
            print("Planner returned no actions. Stopping.")
            break

        for action in actions:
            print(action)

        chosen_action = actions[0]

        print_divider(f"STEP {step}: APPLYING TOP ACTION")
        print(chosen_action)

        result = manager.apply_action(state, chosen_action)

        print("\nAction Result:")
        print(result)

        print_divider(f"STEP {step}: UPDATED STATE")
        print(state.to_prompt_context())

    print_divider("FINAL STATE")
    print(state.to_prompt_context())


if __name__ == "__main__":
    main()