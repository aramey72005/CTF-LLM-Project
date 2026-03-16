from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models.network_state import NetworkState
from src.parsers.nmap_parser import NmapParser
from src.services.llm_client import LLMClient
from src.services.planner import Planner


@dataclass
class ActionScore:
    rank: int
    action_type: str
    target_host: Optional[str]
    command: Optional[str]
    reasoning: str
    confidence: float
    target_in_scope: bool
    target_known: bool
    command_present: bool
    command_plausible: bool
    specificity_score: int


@dataclass
class PlannerRunResult:
    planner_name: str
    parsed_successfully: bool
    total_actions: int
    in_scope_targets: int
    known_targets: int
    commands_present: int
    plausible_commands: int
    average_specificity: float
    actions: List[ActionScore]


class PlannerEvaluation:
    """
    Evaluates planner behavior across one or more NetworkState scenarios.

    Main goals:
    - compare heuristic planner output vs LLM planner output
    - score whether recommendations stay in scope
    - score whether recommendations target known hosts
    - score whether commands are present and plausible
    - support multi-scenario evaluation for the project
    """

    def __init__(self, max_actions: int = 3) -> None:
        self.max_actions = max_actions

    # ------------------------------------------------------------------
    # Scenario builders
    # ------------------------------------------------------------------

    def build_state_from_nmap_text(
        self,
        nmap_text: str,
        target_ip: str,
        scope_networks: List[str],
        blocked_networks: Optional[List[str]] = None,
    ) -> NetworkState:
        state = NetworkState(
            target_ip=target_ip,
            scope_networks=scope_networks,
            blocked_networks=blocked_networks or [],
        )

        parser = NmapParser()
        parser.update_network_state(nmap_text, state)
        return state

    def build_initial_recon_state(self) -> NetworkState:
        """
        Scenario 1:
        No hosts known yet. Tests whether the planner starts with reconnaissance.
        """
        return NetworkState(
            target_ip="10.0.4.3",
            scope_networks=["10.0.0.0/24", "10.0.2.0/24", "10.0.4.0/24"],
            blocked_networks=["10.0.4.0/24"],
        )

    def build_tomcat_foothold_state(self) -> NetworkState:
        """
        Scenario 2:
        A gateway candidate and a Tomcat foothold host are known.
        """
        sample_nmap_output = """
Nmap scan report for 10.0.0.1
Host is up.
PORT     STATE SERVICE
2601/tcp open  ospfd

Nmap scan report for 10.0.2.2
Host is up.
PORT     STATE SERVICE VERSION
8080/tcp open  http-proxy Apache Tomcat 9.0
"""

        return self.build_state_from_nmap_text(
            nmap_text=sample_nmap_output,
            target_ip="10.0.4.3",
            scope_networks=["10.0.0.0/24", "10.0.2.0/24", "10.0.4.0/24"],
            blocked_networks=["10.0.4.0/24"],
        )

    def build_compromised_pivot_state(self) -> NetworkState:
        """
        Scenario 3:
        Same Tomcat foothold state, but the foothold host is already compromised.
        This should encourage pivot-oriented reasoning.
        """
        state = self.build_tomcat_foothold_state()
        state.mark_compromised("10.0.2.2")
        state.record_action(
            action_type="compromise",
            description="Established foothold on 10.0.2.2",
            target_ip="10.0.2.2",
            success=True,
        )
        return state

    def build_all_scenarios(self) -> Dict[str, NetworkState]:
        return {
            "initial_recon": self.build_initial_recon_state(),
            "tomcat_foothold": self.build_tomcat_foothold_state(),
            "compromised_pivot": self.build_compromised_pivot_state(),
        }

    # ------------------------------------------------------------------
    # Planner execution
    # ------------------------------------------------------------------

    def run_heuristic(self, state: NetworkState) -> PlannerRunResult:
        planner = Planner(
            llm_callable=None,
            max_actions=self.max_actions,
            use_mock_fallback=True,
            debug=False,
        )
        actions = planner.plan(state)
        return self._score_actions("heuristic", state, actions, parsed_successfully=True)

    def run_llm(
        self,
        state: NetworkState,
        client: LLMClient,
    ) -> PlannerRunResult:
        planner = Planner(
            llm_callable=client.generate,
            max_actions=self.max_actions,
            use_mock_fallback=False,
            debug=False,
        )

        parsed_successfully = True
        try:
            actions = planner.plan(state)
        except Exception as exc:
            parsed_successfully = False
            actions = [
                {
                    "rank": 1,
                    "action_type": "analyze",
                    "target_host": None,
                    "command": None,
                    "reasoning": f"LLM planner failed: {exc}",
                    "confidence": 0.0,
                }
            ]

        return self._score_actions("llm", state, actions, parsed_successfully=parsed_successfully)

    def compare(
        self,
        state: NetworkState,
        client: Optional[LLMClient] = None,
    ) -> Dict[str, PlannerRunResult]:
        results: Dict[str, PlannerRunResult] = {
            "heuristic": self.run_heuristic(state),
        }

        if client is not None:
            results["llm"] = self.run_llm(state, client=client)

        return results

    def compare_all_scenarios(
        self,
        client: Optional[LLMClient] = None,
    ) -> Dict[str, Dict[str, PlannerRunResult]]:
        scenarios = self.build_all_scenarios()
        all_results: Dict[str, Dict[str, PlannerRunResult]] = {}

        for scenario_name, state in scenarios.items():
            all_results[scenario_name] = self.compare(state, client=client)

        return all_results

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_actions(
        self,
        planner_name: str,
        state: NetworkState,
        actions: List[Dict[str, Any]],
        parsed_successfully: bool,
    ) -> PlannerRunResult:
        scored_actions: List[ActionScore] = []

        for action in actions:
            target_host = action.get("target_host")
            command = action.get("command")
            reasoning = str(action.get("reasoning", "")).strip()

            target_in_scope = self._target_in_scope(state, target_host)
            target_known = self._target_known(state, target_host)
            command_present = bool(command)
            command_plausible = self._command_plausible(command, action.get("action_type", "analyze"))
            specificity_score = self._specificity_score(action)

            scored_actions.append(
                ActionScore(
                    rank=int(action.get("rank", 0)),
                    action_type=str(action.get("action_type", "analyze")),
                    target_host=target_host,
                    command=command,
                    reasoning=reasoning,
                    confidence=float(action.get("confidence", 0.0)),
                    target_in_scope=target_in_scope,
                    target_known=target_known,
                    command_present=command_present,
                    command_plausible=command_plausible,
                    specificity_score=specificity_score,
                )
            )

        total_actions = len(scored_actions)
        in_scope_targets = sum(1 for a in scored_actions if a.target_in_scope)
        known_targets = sum(1 for a in scored_actions if a.target_known)
        commands_present = sum(1 for a in scored_actions if a.command_present)
        plausible_commands = sum(1 for a in scored_actions if a.command_plausible)
        average_specificity = (
            sum(a.specificity_score for a in scored_actions) / total_actions if total_actions else 0.0
        )

        return PlannerRunResult(
            planner_name=planner_name,
            parsed_successfully=parsed_successfully,
            total_actions=total_actions,
            in_scope_targets=in_scope_targets,
            known_targets=known_targets,
            commands_present=commands_present,
            plausible_commands=plausible_commands,
            average_specificity=average_specificity,
            actions=scored_actions,
        )

    def _target_in_scope(self, state: NetworkState, target_host: Optional[str]) -> bool:
        if target_host is None:
            return True
        try:
            return state.is_ip_in_scope(target_host)
        except Exception:
            return False

    def _target_known(self, state: NetworkState, target_host: Optional[str]) -> bool:
        if target_host is None:
            return True
        return state.is_known_host(target_host)

    def _command_plausible(self, command: Optional[str], action_type: str) -> bool:
        if not command:
            return action_type in {"analyze", "pivot", "access"}

        cmd = command.strip().lower()

        bad_patterns = [
            " if command=",
            "output={",
            "```",
            "\n\n",
        ]
        if any(pattern in cmd for pattern in bad_patterns):
            return False

        if cmd.startswith("nmap ") and "http://" in cmd:
            return False
        if cmd.startswith("nmap ") and "https://" in cmd:
            return False

        known_command_starts = (
            "nmap ",
            "searchsploit ",
            "proxychains ",
            "curl ",
            "lynx ",
            "nikto ",
            "whatweb ",
            "wget ",
            "msfconsole ",
        )
        if any(cmd.startswith(prefix) for prefix in known_command_starts):
            return True

        return action_type in {"analyze", "pivot", "access"}

    def _specificity_score(self, action: Dict[str, Any]) -> int:
        score = 0

        if action.get("action_type"):
            score += 1
        if action.get("target_host") is not None:
            score += 1
        if action.get("command"):
            score += 1

        reasoning = str(action.get("reasoning", "")).strip()
        if len(reasoning.split()) >= 4:
            score += 1

        return score

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def results_to_dict(
        self,
        results: Dict[str, Dict[str, PlannerRunResult]] | Dict[str, PlannerRunResult],
    ) -> Dict[str, Any]:
        # Handle single-scenario comparison
        if results and all(isinstance(v, PlannerRunResult) for v in results.values()):
            single_output: Dict[str, Any] = {}
            for name, result in results.items():
                single_output[name] = {
                    "planner_name": result.planner_name,
                    "parsed_successfully": result.parsed_successfully,
                    "total_actions": result.total_actions,
                    "in_scope_targets": result.in_scope_targets,
                    "known_targets": result.known_targets,
                    "commands_present": result.commands_present,
                    "plausible_commands": result.plausible_commands,
                    "average_specificity": result.average_specificity,
                    "actions": [asdict(action) for action in result.actions],
                }
            return single_output

        # Handle multi-scenario comparison
        multi_output: Dict[str, Any] = {}
        assert isinstance(results, dict)

        for scenario_name, scenario_results in results.items():
            multi_output[scenario_name] = {}
            for planner_name, result in scenario_results.items():
                multi_output[scenario_name][planner_name] = {
                    "planner_name": result.planner_name,
                    "parsed_successfully": result.parsed_successfully,
                    "total_actions": result.total_actions,
                    "in_scope_targets": result.in_scope_targets,
                    "known_targets": result.known_targets,
                    "commands_present": result.commands_present,
                    "plausible_commands": result.plausible_commands,
                    "average_specificity": result.average_specificity,
                    "actions": [asdict(action) for action in result.actions],
                }

        return multi_output

    def save_results(
        self,
        results: Dict[str, Dict[str, PlannerRunResult]] | Dict[str, PlannerRunResult],
        output_path: str,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(self.results_to_dict(results), file, indent=2)

    def print_report(self, results: Dict[str, PlannerRunResult]) -> None:
        for name, result in results.items():
            print(f"\n=== {name.upper()} PLANNER REPORT ===")
            print(f"Parsed successfully: {result.parsed_successfully}")
            print(f"Total actions: {result.total_actions}")
            print(f"In-scope targets: {result.in_scope_targets}/{result.total_actions}")
            print(f"Known targets: {result.known_targets}/{result.total_actions}")
            print(f"Commands present: {result.commands_present}/{result.total_actions}")
            print(f"Plausible commands: {result.plausible_commands}/{result.total_actions}")
            print(f"Average specificity: {result.average_specificity:.2f}")

            print("\nActions:")
            for action in result.actions:
                print(
                    f"- rank={action.rank} "
                    f"type={action.action_type} "
                    f"target={action.target_host} "
                    f"command={action.command} "
                    f"confidence={action.confidence:.2f}"
                )
                print(f"  reasoning: {action.reasoning}")

    def print_multi_scenario_report(self, results: Dict[str, Dict[str, PlannerRunResult]]) -> None:
        for scenario_name, scenario_results in results.items():
            print("\n==================================================")
            print(f"SCENARIO: {scenario_name}")
            print("==================================================")
            self.print_report(scenario_results)


def main() -> None:
    """
    Standalone multi-scenario evaluation runner.
    """
    evaluator = PlannerEvaluation(max_actions=3)

    client = LLMClient(
        base_url="http://localhost:11434",
        model="phi3",
        timeout=180,
    )

    results = evaluator.compare_all_scenarios(client=client)
    evaluator.print_multi_scenario_report(results)
    evaluator.save_results(results, "logs/planner_multi_scenario_evaluation.json")


if __name__ == "__main__":
    main()