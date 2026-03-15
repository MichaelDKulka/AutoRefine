"""Tests for LLM provider implementations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autorefine.models import Message, MessageRole
from autorefine.providers.base import BaseProvider, ProviderResponse


class TestBaseProvider:
    def test_to_dicts(self):
        messages = [
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi"),
        ]
        result = BaseProvider._to_dicts(messages)
        assert result == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

    def test_default_complete_delegates_to_chat(self):
        """complete() should build a Message list and call chat()."""

        class TrackingProvider(BaseProvider):
            name = "tracking"
            chat_called_with = None

            def chat(self, system_prompt, messages, **kwargs):
                self.chat_called_with = (system_prompt, messages, kwargs)
                return ProviderResponse(text="ok")

            def stream(self, system_prompt, messages, **kwargs):
                yield "ok"

            def estimate_cost(self, input_tokens, output_tokens):
                return 0.0

        p = TrackingProvider()
        resp = p.complete("Be helpful", "What is 2+2?", temperature=0.5)
        assert resp.text == "ok"
        system, msgs, kwargs = p.chat_called_with
        assert system == "Be helpful"
        assert len(msgs) == 1
        assert msgs[0].role == MessageRole.USER
        assert msgs[0].content == "What is 2+2?"
        assert kwargs == {"temperature": 0.5}

    def test_abstract_methods_required(self):
        """Cannot instantiate without implementing chat, stream, estimate_cost."""
        with pytest.raises(TypeError):
            BaseProvider()  # type: ignore[abstract]


class TestOpenAIProvider:
    def test_estimate_cost(self):
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            from autorefine.providers.openai_provider import OpenAIProvider
            p = OpenAIProvider.__new__(OpenAIProvider)
            p._model = "gpt-4o"
            cost = p.estimate_cost(1000, 1000)
            assert cost > 0

    def test_estimate_cost_gpt35(self):
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            from autorefine.providers.openai_provider import OpenAIProvider
            p = OpenAIProvider.__new__(OpenAIProvider)
            p._model = "gpt-3.5-turbo"
            cost_35 = p.estimate_cost(1000, 1000)
            p._model = "gpt-4"
            cost_4 = p.estimate_cost(1000, 1000)
            assert cost_35 < cost_4

    def test_estimate_cost_prefix_match(self):
        """Dated model snapshots should match the base model pricing."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            from autorefine.providers.openai_provider import OpenAIProvider
            p = OpenAIProvider.__new__(OpenAIProvider)
            p._model = "gpt-4o-2024-08-06"
            cost = p.estimate_cost(1_000_000, 1_000_000)
            # Should match gpt-4o pricing: 2.5 + 10.0 = 12.5
            assert cost == pytest.approx(12.5, abs=0.1)

    def test_pricing_dict_has_major_models(self):
        from autorefine.providers.openai_provider import PRICING
        assert "gpt-4o" in PRICING
        assert "gpt-4" in PRICING
        assert "gpt-3.5-turbo" in PRICING
        assert "o3-mini" in PRICING


class TestAnthropicProvider:
    def test_estimate_cost(self):
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            from autorefine.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider.__new__(AnthropicProvider)
            p._model = "claude-sonnet-4-20250514"
            cost = p.estimate_cost(1000, 1000)
            assert cost > 0

    def test_pricing_dict_has_major_models(self):
        from autorefine.providers.anthropic_provider import PRICING
        assert "claude-sonnet-4-20250514" in PRICING
        assert "claude-opus-4-6-20260415" in PRICING
        assert "claude-haiku-4-5-20251001" in PRICING


class TestOllamaProvider:
    def test_cost_is_zero(self):
        from autorefine.providers.ollama_provider import OllamaProvider
        p = OllamaProvider.__new__(OllamaProvider)
        assert p.estimate_cost(1_000_000, 1_000_000) == 0.0

    def test_pricing_dict_all_zero(self):
        from autorefine.providers.ollama_provider import PRICING
        for model, (inp, out) in PRICING.items():
            assert inp == 0.0 and out == 0.0, f"{model} should be free"


class TestOpenAIProviderMockedHTTP:
    """Test OpenAI provider with mocked SDK responses."""

    def test_chat_maps_response_correctly(self):
        mock_openai = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello world"
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = "stop"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 15
        mock_usage.completion_tokens = 25
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = "gpt-4o-2024-08-06"

        with patch.dict("sys.modules", {"openai": mock_openai}):
            mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
            from autorefine.providers.openai_provider import OpenAIProvider
            p = OpenAIProvider(api_key="sk-test", model="gpt-4o")
            msgs = [Message(role=MessageRole.USER, content="Hi")]
            resp = p.chat("Be helpful", msgs)
            assert resp.text == "Hello world"
            assert resp.input_tokens == 15
            assert resp.output_tokens == 25
            assert resp.model == "gpt-4o-2024-08-06"
            assert resp.finish_reason == "stop"

    def test_chat_with_tool_calls(self):
        mock_openai = MagicMock()
        mock_tc = MagicMock()
        mock_tc.id = "call_123"
        mock_tc.type = "function"
        mock_tc.function.name = "get_weather"
        mock_tc.function.arguments = '{"city":"NYC"}'
        mock_choice = MagicMock()
        mock_choice.message.content = ""
        mock_choice.message.tool_calls = [mock_tc]
        mock_choice.finish_reason = "tool_calls"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 20
        mock_usage.completion_tokens = 10
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = "gpt-4o"

        with patch.dict("sys.modules", {"openai": mock_openai}):
            mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
            from autorefine.providers.openai_provider import OpenAIProvider
            p = OpenAIProvider(api_key="sk-test")
            resp = p.chat("", [Message(role=MessageRole.USER, content="Weather?")])
            assert resp.finish_reason == "tool_calls"
            assert len(resp.tool_calls) == 1
            assert resp.tool_calls[0]["function"]["name"] == "get_weather"

    def test_api_error_raises_provider_error(self):
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            mock_openai.OpenAI.return_value.chat.completions.create.side_effect = \
                Exception("Rate limit exceeded")
            from autorefine.exceptions import ProviderError
            from autorefine.providers.openai_provider import OpenAIProvider
            p = OpenAIProvider(api_key="sk-test")
            with pytest.raises(ProviderError, match="Rate limit"):
                p.chat("", [Message(role=MessageRole.USER, content="Hi")])


class TestAnthropicProviderMockedHTTP:
    """Test Anthropic provider with mocked SDK responses."""

    def test_chat_maps_response_correctly(self):
        mock_anthropic = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "Claude says hello"
        mock_block.type = "text"
        mock_usage = MagicMock()
        mock_usage.input_tokens = 12
        mock_usage.output_tokens = 18
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage = mock_usage
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response
            from autorefine.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider(api_key="sk-ant-test")
            resp = p.chat("Be helpful", [Message(role=MessageRole.USER, content="Hi")])
            assert resp.text == "Claude says hello"
            assert resp.input_tokens == 12
            assert resp.output_tokens == 18
            assert resp.finish_reason == "end_turn"

    def test_system_prompt_passed_as_parameter(self):
        """Anthropic should pass system as a kwarg, not a message."""
        mock_anthropic = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "ok"
        mock_block.type = "text"
        mock_usage = MagicMock()
        mock_usage.input_tokens = 5
        mock_usage.output_tokens = 5
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage = mock_usage
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"
        mock_create = mock_anthropic.Anthropic.return_value.messages.create
        mock_create.return_value = mock_response

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from autorefine.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider(api_key="sk-ant-test")
            p.chat("My system prompt", [Message(role=MessageRole.USER, content="Hi")])
            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs.get("system") == "My system prompt" or \
                   ("system" in dict(zip(call_kwargs.args, range(100))) if call_kwargs.args else False) or \
                   any("system" in str(k) for k in (call_kwargs.kwargs or {}))


class TestOllamaProviderMockedHTTP:
    """Test Ollama provider with mocked httpx responses."""

    def test_chat_maps_response(self):
        from autorefine.providers.ollama_provider import OllamaProvider
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "Ollama says hi"},
            "prompt_eval_count": 10,
            "eval_count": 15,
            "model": "llama3.1",
            "done": True,
            "done_reason": "stop",
        }
        mock_resp.raise_for_status = MagicMock()

        p = OllamaProvider.__new__(OllamaProvider)
        p._model = "llama3.1"
        p._base_url = "http://localhost:11434"
        p._client = MagicMock()
        p._client.post.return_value = mock_resp

        resp = p.chat("Be helpful", [Message(role=MessageRole.USER, content="Hi")])
        assert resp.text == "Ollama says hi"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 15
        assert resp.model == "llama3.1"


class TestGetProvider:
    def test_auto_detects_openai(self):
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            from autorefine.providers import get_provider
            p = get_provider(model="gpt-4o", api_key="sk-test")
            assert p.name == "openai"

    def test_auto_detects_anthropic(self):
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            from autorefine.providers import get_provider
            p = get_provider(model="claude-sonnet-4-20250514", api_key="sk-ant-test")
            assert p.name == "anthropic"

    def test_auto_detects_ollama(self):
        from autorefine.providers import get_provider
        p = get_provider(model="llama3.1")
        assert p.name == "ollama"

    def test_explicit_provider_name(self):
        from autorefine.providers import get_provider
        p = get_provider("ollama", model="custom-model")
        assert p.name == "ollama"

    def test_unknown_provider_raises(self):
        from autorefine.providers import get_provider
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent")

    def test_default_fallback_is_openai(self):
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            from autorefine.providers import get_provider
            p = get_provider(model="some-unknown-model", api_key="sk-test")
            assert p.name == "openai"
