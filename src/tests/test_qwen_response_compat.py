from __future__ import annotations

import unittest

from src.llms.providers.dashscope import ChatDashscope


class QwenResponseCompatibilityTest(unittest.TestCase):
    def _model(self, *, thinking: bool) -> ChatDashscope:
        return ChatDashscope(
            model="test-model",
            api_key="test-key",
            base_url="http://127.0.0.1:1/v1",
            extra_body={
                "enable_thinking": thinking,
                "chat_template_kwargs": {"enable_thinking": thinking},
            },
        )

    def test_promotes_reasoning_content_for_non_thinking_gateway(self) -> None:
        result = self._model(thinking=False)._create_chat_result({
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": '{"action":"finish"}',
                },
            }],
        })
        message = result.generations[0].message
        self.assertEqual(message.content, '{"action":"finish"}')
        self.assertTrue(message.additional_kwargs["promoted_reasoning_content"])

    def test_does_not_promote_hidden_reasoning_in_thinking_mode(self) -> None:
        result = self._model(thinking=True)._create_chat_result({
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "unfinished private reasoning",
                },
            }],
        })
        self.assertEqual(result.generations[0].message.content, "")


if __name__ == "__main__":
    unittest.main()
