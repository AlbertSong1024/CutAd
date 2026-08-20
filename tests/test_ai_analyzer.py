"""ai_analyzer.py 单元测试"""
import json
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from cutad import ai_analyzer


# ---------------------------------------------------------------------------
# _build_text_summary
# ---------------------------------------------------------------------------

class TestBuildTextSummary:
    """文本摘要构建"""

    def test_empty_segments(self):
        assert ai_analyzer._build_text_summary([]) == ""

    def test_single_segment(self):
        segments = [{"start": 1.5, "end": 3.0, "text": "hello world"}]
        result = ai_analyzer._build_text_summary(segments)
        assert result == "[1.5s] hello world"

    def test_multiple_segments(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "first"},
            {"start": 1.0, "end": 2.0, "text": "second"},
            {"start": 2.0, "end": 3.0, "text": "third"},
        ]
        result = ai_analyzer._build_text_summary(segments)
        assert "[0.0s] first" in result
        assert "[1.0s] second" in result
        assert "[2.0s] third" in result

    def test_max_chars_truncation(self):
        segments = [
            {"start": i, "end": i + 1, "text": "x" * 100}
            for i in range(100)
        ]
        result = ai_analyzer._build_text_summary(segments, max_chars=300)
        assert len(result) <= 400

    def test_empty_text_skipped(self):
        segments = [
            {"start": 0, "end": 1, "text": "  "},
            {"start": 1, "end": 2, "text": "valid"},
        ]
        result = ai_analyzer._build_text_summary(segments)
        assert "valid" in result
        assert result.count("\n") == 0

    def test_missing_text_field(self):
        segments = [{"start": 0, "end": 1}]
        result = ai_analyzer._build_text_summary(segments)
        assert result == ""


# ---------------------------------------------------------------------------
# _parse_llm_response
# ---------------------------------------------------------------------------

class TestParseLlmResponse:
    """LLM 响应解析"""

    SEGMENTS = [{"start": 0, "end": 60, "text": "video"}]

    def test_valid_json_array(self):
        response = json.dumps([
            {"start": 10.0, "end": 20.0, "text": "ad", "keywords": ["buy"]},
            {"start": 50.0, "end": 55.0, "text": "promo", "keywords": []},
        ])
        result = ai_analyzer._parse_llm_response(response, self.SEGMENTS)
        assert len(result) == 2
        assert result[0]["start"] == 10.0
        assert result[0]["keywords"] == ["buy"]

    def test_json_with_ads_key(self):
        response = json.dumps({"ads": [
            {"start": 5.5, "end": 15.5, "text": "t", "keywords": []},
        ]})
        result = ai_analyzer._parse_llm_response(response, self.SEGMENTS)
        assert len(result) == 1
        assert result[0]["start"] == 5.5

    def test_json_list_with_other_key(self):
        response = json.dumps({"result": [{"start": 1, "end": 5, "text": "x", "keywords": []}]})
        result = ai_analyzer._parse_llm_response(response, self.SEGMENTS)
        assert result == []

    def test_json_string_with_comma_format(self):
        response = 'some text 10,20 more text'
        result = ai_analyzer._parse_llm_response(response, self.SEGMENTS)
        assert len(result) == 1
        assert result[0]["start"] == 10.0
        assert result[0]["end"] == 20.0

    def test_json_string_with_tilde_format(self):
        response = 'found ad at 5.5~15.5 seconds'
        result = ai_analyzer._parse_llm_response(response, self.SEGMENTS)
        assert len(result) == 1
        assert result[0]["start"] == 5.5
        assert result[0]["end"] == 15.5

    def test_multiple_matches(self):
        response = "ads at 2,5 and 30,40"
        result = ai_analyzer._parse_llm_response(response, self.SEGMENTS)
        assert len(result) == 2
        assert result[0]["start"] == 2.0
        assert result[1]["start"] == 30.0

    def test_duration_below_threshold_filtered(self):
        response = "1,1.5"
        result = ai_analyzer._parse_llm_response(response, self.SEGMENTS)
        assert result == []

    def test_invalid_json_no_pattern(self):
        response = "no time range here"
        result = ai_analyzer._parse_llm_response(response, self.SEGMENTS)
        assert result == []

    def test_end_before_start_filtered(self):
        response = "20,10"
        result = ai_analyzer._parse_llm_response(response, self.SEGMENTS)
        assert result == []

    def test_keywords_default_to_empty_list(self):
        response = "5,15"
        result = ai_analyzer._parse_llm_response(response, self.SEGMENTS)
        assert result[0]["keywords"] == []
        assert result[0]["text"] == ""

    def test_exact_threshold_one_second_kept(self):
        response = "10,11"
        result = ai_analyzer._parse_llm_response(response, self.SEGMENTS)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _call_llm
# ---------------------------------------------------------------------------

class TestCallLlm:
    """LLM 调用适配"""

    def test_openai_uses_default_model(self):
        client = MagicMock()
        resp_obj = MagicMock()
        message = MagicMock()
        message.content = '{"ads": []}'
        choices = MagicMock()
        choices[0].message = message
        resp_obj.choices = choices
        client.chat.completions.create.return_value = resp_obj

        ai_analyzer._call_llm(client, "prompt")

        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["messages"] == [{"role": "user", "content": "prompt"}]

    def test_openai_uses_custom_model(self):
        client = MagicMock()
        resp_obj = MagicMock()
        message = MagicMock()
        message.content = '[]'
        choices = MagicMock()
        choices[0].message = message
        resp_obj.choices = choices
        client.chat.completions.create.return_value = resp_obj

        ai_analyzer._call_llm(client, "prompt", model="gpt-4o-mini")

        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"

    def test_anthropic_uses_default_model(self):
        client = MagicMock(spec=["messages"])
        client.chat = None
        resp_obj = MagicMock()
        content_item = MagicMock()
        content_item.text = "[]"
        resp_obj.content = [content_item]
        client.messages.create.return_value = resp_obj

        ai_analyzer._call_llm(client, "prompt")

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["max_tokens"] == 1024
        assert call_kwargs["messages"] == [{"role": "user", "content": "prompt"}]

    def test_anthropic_uses_custom_model_and_max_tokens(self):
        client = MagicMock(spec=["messages"])
        client.chat = None
        resp_obj = MagicMock()
        content_item = MagicMock()
        content_item.text = '[]'
        resp_obj.content = [content_item]
        client.messages.create.return_value = resp_obj

        ai_analyzer._call_llm(client, "p", model="claude-haiku-3.5-20241022", max_tokens=2048)

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-3.5-20241022"
        assert call_kwargs["max_tokens"] == 2048

    def test_unsupported_client_raises(self):
        client = object()
        with pytest.raises(ValueError, match="不支持"):
            ai_analyzer._call_llm(client, "prompt")


# ---------------------------------------------------------------------------
# analyze_with_llm  (集成层)
# ---------------------------------------------------------------------------

class TestAnalyzeWithLlm:
    """顶层分析接口"""

    def _mock_openai_client(self, response_text: str):
        client = MagicMock()
        resp_obj = MagicMock()
        message = MagicMock()
        message.content = response_text
        choices = MagicMock()
        choices[0].message = message
        resp_obj.choices = choices
        client.chat.completions.create.return_value = resp_obj
        return client

    def test_no_client_raises(self):
        with pytest.raises(ValueError, match="llm_client"):
            ai_analyzer.analyze_with_llm([])

    def test_returns_ads(self):
        segments = [{"start": 0, "end": 1, "text": "hello"}]
        response = json.dumps([{"start": 10, "end": 20, "text": "buy now", "keywords": ["buy"]}])
        client = self._mock_openai_client(response)

        result = ai_analyzer.analyze_with_llm(segments, llm_client=client)
        assert len(result) == 1
        assert result[0]["start"] == 10

    def test_custom_prompt_is_used(self):
        segments = [{"start": 0, "end": 1, "text": "test"}]
        response = "[]"
        client = self._mock_openai_client(response)

        ai_analyzer.analyze_with_llm(
            segments,
            llm_client=client,
            prompt_template="CUSTOM {transcript}",
        )
        call_kwargs = client.chat.completions.create.call_args[1]
        assert "CUSTOM" in call_kwargs["messages"][0]["content"]

    def test_model_arg_passed_through(self):
        segments = [{"start": 0, "end": 1, "text": "x"}]
        response = "[]"
        client = self._mock_openai_client(response)

        ai_analyzer.analyze_with_llm(
            segments,
            llm_client=client,
            model="gpt-4o-mini",
        )
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"