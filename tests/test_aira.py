"""Offline smoke tests for the AIRA extension and unchanged AiTM baseline."""

import random
import unittest
from types import SimpleNamespace

from agents.autogen_mas import AutoGenMAS
from analysis.role_inference import RoleInference
from observer.communication_observer import CommunicationObserver
from selector.target_selector import TargetSelector
from scripts.run_experiments import STRUCTURES, random_attackable_victim


class _FakeAgent:
    def __init__(self, name: str) -> None:
        self.name = name

    def reset(self) -> None:
        pass

    def generate_reply(self, messages):
        prompt = messages[0]["content"]
        return {"content": f"{self.name} reply; attacked={'attack marker' in prompt}"}


class _FakeAdversary:
    def __init__(self) -> None:
        self.previous_instructions = []

    def intercept_and_generate(self, intercepted_messages, victim_role=""):
        self.previous_instructions.append("attack marker")
        return "attack marker"

    def reflect(self, previous_instruction, intercepted_messages):
        self.previous_instructions.append(previous_instruction)
        return previous_instruction


def _mas(structure: str) -> AutoGenMAS:
    mas = AutoGenMAS(structure, model="offline-test")
    mas.agents = [_FakeAgent(f"A{index}") for index in range(mas.num_agents)]
    if structure in {"tree", "asymmetric_tree"}:
        mas.tree_judge = _FakeAgent("J")
    return mas


class AiraRoutingTests(unittest.TestCase):
    def test_fixed_victim_baseline_remains_available(self) -> None:
        mas = _mas("chain")
        mas.setup_attack(_FakeAdversary(), victim_index=1)
        result = mas.run("test query")

        self.assertEqual(result["victim"], "A1")
        self.assertEqual(len(result["attack_log"]), 1)
        self.assertNotIn("observation", result)

    def test_aira_observes_then_selects_and_attacks(self) -> None:
        mas = _mas("tree")
        mas.setup_aira_attack(
            _FakeAdversary(),
            TargetSelector(min_observed_events=2, role_inference_mode="heuristic"),
        )
        result = mas.run("test query")

        self.assertIsNotNone(result["aira_selection"])
        self.assertEqual(result["victim"], result["aira_selection"]["target"])
        self.assertEqual(set(result["aira_selection"]["rankings"]), {"A2", "A4"})
        self.assertTrue(result["attack_log"])
        self.assertTrue(result["observation"]["events"])
        first_event = result["observation"]["events"][0]
        self.assertTrue({"sender", "receiver", "message", "timestamp"} <= set(first_event))
        self.assertNotEqual(result["estimated_graph"], result["communication_graph"])
        self.assertTrue(result["tampered_messages"])
        self.assertTrue(result["aira_selection_validation"]["attack_executed"])
        self.assertTrue(result["aira_selection_validation"]["terminal_path_confirmed_posthoc"])
        self.assertIn("A2", result["aira_selection"]["all_scores"])

    def test_strict_forwarding_constraint_declines_chain_terminal(self) -> None:
        mas = _mas("chain")
        mas.setup_aira_attack(
            _FakeAdversary(),
            TargetSelector(min_observed_events=2, role_inference_mode="heuristic", require_observed_forwarding=True),
        )
        result = mas.run("test query")

        self.assertIsNone(result["aira_selection"])
        self.assertFalse(result["attack_log"])

    def test_llm_role_interface_uses_only_observed_transcript(self) -> None:
        observer = CommunicationObserver()
        observer.record_delivery("A0", "A1", "Please plan and verify the answer.")
        seen_prompts = []

        def classifier(prompt):
            seen_prompts.append(prompt)
            return {"A0": {"planner": 1, "executor": 0, "verifier": 0, "synthesizer": 0}, "A1": {"planner": 0, "executor": 0, "verifier": 1, "synthesizer": 0}}

        inferred = RoleInference(mode="llm", classifier=classifier).infer(observer, ["A0", "A1"])
        self.assertEqual(inferred["A0"]["method"], "llm")
        self.assertEqual(inferred["A0"]["probabilities"]["planner"], 1.0)
        self.assertIn("A0 -> A1", seen_prompts[0])
        self.assertIn("Return JSON only", seen_prompts[0])

    def test_role_parser_accepts_valid_json_after_provider_preamble(self) -> None:
        content = (
            "I classified the observed messages as follows:\n"
            '{"agents":{"A0":{"planner":1,"executor":0,"verifier":0,"synthesizer":0}}}'
            "\nEnd of response."
        )
        parsed = RoleInference._parse_json(content)
        self.assertEqual(parsed["A0"]["planner"], 1)

    def test_role_classifier_reports_safe_parse_diagnostic(self) -> None:
        observer = CommunicationObserver()
        observer.record_delivery("A0", "A1", "Please plan the answer.")
        captured = {}

        class _Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                message = SimpleNamespace(content="", reasoning_content="brief reasoning")
                return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="length")])

        inference = RoleInference(
            mode="llm",
            model="test-model",
            api_key="offline-test",
            base_url="https://example.invalid/v1/",
        )
        inference.client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
        inferred = inference.infer(observer, ["A0", "A1"])

        self.assertEqual(captured["max_tokens"], 1024)
        self.assertEqual(inferred["A0"]["method"], "heuristic_fallback")
        self.assertEqual(
            inferred["A0"]["fallback_reason"],
            "classifier_invalid_json:finish=length;content_chars=0;reasoning_chars=15",
        )

    def test_dynamic_switching_keeps_per_victim_attack_state(self) -> None:
        mas = _mas("complete")
        mas.judge = _FakeAgent("judge")
        mas.setup_aira_attack(
            _FakeAdversary(),
            TargetSelector(
                min_observed_events=2,
                role_inference_mode="heuristic",
                require_observed_forwarding=False,
            ),
            dynamic_target_switching=True,
        )
        result = mas.run("test query", max_round=6)

        self.assertGreaterEqual(len(result["aira_selection_history"]), 2)
        self.assertGreaterEqual(len(result["attack_log"]), 2)
        self.assertTrue(result["aira_selection_validation"]["attack_executed"])

    def test_random_baseline_uses_scheduler_attackable_agents(self) -> None:
        self.assertEqual(random_attackable_victim("chain", 42), 1)
        self.assertIn(random_attackable_victim("tree", 42), {2, 3, 4, 5})
        self.assertIn(random_attackable_victim("complete", 42), {0, 1, 2})

    def test_asymmetric_tree_random_baseline_is_not_seed_coupled(self) -> None:
        seeds = list(range(42, 62))
        random_victims = [random_attackable_victim("asymmetric_tree", seed) for seed in seeds]
        topology_orientations = [2 if random.Random(seed).randrange(2) == 0 else 4 for seed in seeds]
        resampled_victims = [random_attackable_victim("asymmetric_tree", seed, 101) for seed in seeds]

        self.assertNotEqual(random_victims, topology_orientations)
        self.assertNotEqual(random_victims, resampled_victims)
        self.assertTrue(all(victim in {2, 4} for victim in random_victims))

    def test_asymmetric_tree_is_opt_in_and_selects_observed_hub(self) -> None:
        self.assertNotIn("asymmetric_tree", STRUCTURES)
        mas = _mas("asymmetric_tree")
        mas.setup_aira_attack(
            _FakeAdversary(),
            TargetSelector(min_observed_events=9, role_inference_mode="heuristic"),
        )
        result = mas.run("test query")

        self.assertEqual(set(result["aira_selection"]["rankings"]), {"A2", "A4"})
        self.assertEqual(result["aira_selection"]["event_count"], 9)
        self.assertEqual(result["victim"], result["posthoc_scenario"]["hub_agent"])
        self.assertTrue(result["aira_selection_validation"]["attack_executed"])

    def test_asymmetric_tree_fixed_baseline_uses_same_attack_window(self) -> None:
        mas = _mas("asymmetric_tree")
        mas.setup_attack(_FakeAdversary(), victim_index=2)
        result = mas.run("test query")

        self.assertEqual(result["victim"], "A2")
        self.assertEqual(len(result["attack_log"]), 1)
        self.assertEqual(result["attack_log"][0]["victim"], "A2")


if __name__ == "__main__":
    unittest.main()
