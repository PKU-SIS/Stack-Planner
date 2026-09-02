from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.messages import HumanMessage
from src.agents.CentralAgent import CentralAgent, CentralAgentAction
from src.agents.SubAgentManager import _format_sql_block
from src.agents.sub_agent_registry import get_sub_agents_by_global_type
from src.graph.math_graph import build_math_graph
from src.graph.sp_nodes import sql_agent_node, sql_conclusion_node
from src.graph.sql_graph import build_sql_graph
from src.graph.task_profiles import SQL_PROFILE, get_task_graph_profile
from src.prompts.sops import load_task_sop
from src.prompts.central_decision import Decision, DelegateParams


class _DecisionLLM:
    def __init__(self, decision: Decision) -> None:
        self.decision = decision

    def with_structured_output(self, *args, **kwargs):
        return self

    def invoke(self, messages):
        return self.decision


class SQLGraphProfileTest(unittest.TestCase):
    def test_profile_is_independent_from_math(self) -> None:
        profile = get_task_graph_profile("sp_sql")
        self.assertIs(profile, SQL_PROFILE)
        self.assertEqual(profile.task_family, "sql")
        self.assertEqual(profile.terminal_agent, "sql_conclusion")
        self.assertEqual(
            [name for name, _ in profile.sub_agents],
            ["sql_agent", "sql_conclusion"],
        )

    def test_sql_graph_has_only_sql_runtime_nodes(self) -> None:
        graph = build_sql_graph()
        self.assertEqual(
            set(graph.nodes),
            {"central_agent", "sql_agent", "sql_conclusion"},
        )
        self.assertIs(graph.nodes["sql_agent"].runnable.func, sql_agent_node)
        self.assertIs(
            graph.nodes["sql_conclusion"].runnable.func,
            sql_conclusion_node,
        )
        self.assertNotIn("coder", graph.nodes)
        self.assertNotIn("reporter", graph.nodes)
        self.assertNotIn("researcher", graph.nodes)

    def test_registry_is_derived_from_sql_profile(self) -> None:
        agents = get_sub_agents_by_global_type("sp_sql")
        self.assertEqual(
            [agent["name"] for agent in agents],
            ["sql_agent", "sql_conclusion"],
        )
        self.assertIs(agents[0]["node"], sql_agent_node)

    def test_sql_sop_preserves_stackplanner_and_blocks_oracles(self) -> None:
        sop = load_task_sop("sp_sql")
        self.assertIn("REFLECT", sop)
        self.assertIn("POP", sop)
        self.assertIn("SUMMARIZE", sop)
        self.assertIn("per-turn gold", sop)
        self.assertIn("no database", sop)

    def test_sql_formatter_accepts_read_only_and_rejects_mutation(self) -> None:
        self.assertEqual(
            _format_sql_block("SELECT name FROM schools;"),
            "```sql\nSELECT name FROM schools\n```",
        )
        self.assertEqual(
            _format_sql_block("```sql\nWITH x AS (SELECT 1) SELECT * FROM x;\n```"),
            "```sql\nWITH x AS (SELECT 1) SELECT * FROM x\n```",
        )
        self.assertEqual(
            _format_sql_block("SELECT id FROM events WHERE action = 'UPDATE'"),
            "```sql\nSELECT id FROM events WHERE action = 'UPDATE'\n```",
        )
        with self.assertRaisesRegex(RuntimeError, "non-read-only"):
            _format_sql_block("SELECT 1; DELETE FROM schools")

    def test_math_graph_shape_is_unchanged(self) -> None:
        self.assertEqual(
            set(build_math_graph().nodes),
            {"central_agent", "coder", "conclusion"},
        )

    def test_sql_controller_requires_one_draft_before_finish(self) -> None:
        agent = CentralAgent(graph_format="sp_sql")
        finish = Decision(
            action="finish",
            reasoning="ready",
            locale="en-US",
        )
        state = {"messages": [HumanMessage(content="write the query")]}
        config = {"configurable": {"graph_format": "sp_sql"}}
        with patch(
            "src.agents.CentralAgent.get_llm_by_type",
            return_value=_DecisionLLM(finish),
        ):
            decision = agent.make_decision(state, config)
        self.assertEqual(decision.action, CentralAgentAction.DELEGATE)
        self.assertEqual(decision.params.agent_type, "sql_agent")

        duplicate = Decision(
            action="delegate",
            reasoning="draft again",
            params=DelegateParams(
                agent_type="sql_agent",
                task_description="draft again",
            ),
            locale="en-US",
        )
        state["messages"].append(
            HumanMessage(content="```sql\nSELECT 1\n```", name="sql_agent")
        )
        with patch(
            "src.agents.CentralAgent.get_llm_by_type",
            return_value=_DecisionLLM(duplicate),
        ):
            decision = agent.make_decision(state, config)
        self.assertEqual(decision.action, CentralAgentAction.FINISH)


if __name__ == "__main__":
    unittest.main()
