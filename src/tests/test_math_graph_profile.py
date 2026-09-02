from __future__ import annotations

import unittest

from src.agents.sub_agent_registry import get_sub_agents_by_global_type
from src.graph.math_graph import build_math_graph
from src.graph.sp_nodes import coder_node as sp_coder_node
from src.graph.task_profiles import MATH_PROFILE, get_task_graph_profile
from src.prompts.sops import load_task_sop


class MathGraphProfileTest(unittest.TestCase):
    def test_profile_owns_agents_terminal_and_memory_policy(self) -> None:
        profile = get_task_graph_profile("sp_math")
        self.assertIs(profile, MATH_PROFILE)
        self.assertEqual(profile.task_family, "math")
        self.assertEqual(profile.terminal_agent, "conclusion")
        self.assertTrue(profile.supports_checkpoint_memory)
        self.assertEqual(
            [name for name, _ in profile.sub_agents],
            ["coder", "conclusion"],
        )

    def test_math_graph_has_only_math_runtime_nodes(self) -> None:
        graph = build_math_graph()
        self.assertEqual(
            set(graph.nodes),
            {"central_agent", "coder", "conclusion"},
        )
        self.assertIs(graph.nodes["coder"].runnable.afunc, sp_coder_node)
        self.assertNotIn("researcher", graph.nodes)
        self.assertNotIn("reporter", graph.nodes)

    def test_registry_is_derived_from_math_profile(self) -> None:
        agents = get_sub_agents_by_global_type("sp_math")
        self.assertEqual([agent["name"] for agent in agents], ["coder", "conclusion"])
        self.assertIs(agents[0]["node"], sp_coder_node)

    def test_math_sop_preserves_internal_memory_without_research(self) -> None:
        sop = load_task_sop("sp_math")
        self.assertIn("REFLECT", sop)
        self.assertIn("POP", sop)
        self.assertIn("SUMMARIZE", sop)
        self.assertIn("Do not call researcher", sop)
        self.assertIn("conclusion agent", sop)


if __name__ == "__main__":
    unittest.main()
