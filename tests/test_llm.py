# -*- coding: utf-8 -*-
"""测试: llm 模块的 LLM 输出解析与判定"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cutad.llm import (_extract_json_array, _find_verdict, _from_env,
                       _call_chat, create_ai_analyzer)


def test_extract_json_plain():
    out = '[{"start": 1.0, "is_ad": true}]'
    assert _extract_json_array(out) == [{"start": 1.0, "is_ad": True}]


def test_extract_json_with_code_fence():
    out = '```json\n[{"start": 1.0, "is_ad": false}]\n```'
    assert _extract_json_array(out) == [{"start": 1.0, "is_ad": False}]


def test_extract_json_embedded_text():
    out = '结果如下：\n[{"start": 2.0, "is_ad": true}]\n以上。'
    assert _extract_json_array(out) == [{"start": 2.0, "is_ad": True}]


def test_extract_json_invalid():
    import pytest

    with pytest.raises(ValueError):
        _extract_json_array("不是JSON")


def test_find_verdict_match_with_tolerance():
    verdicts = [{"start": 100.0, "is_ad": True}]
    assert _find_verdict(verdicts, 101.0) == {"start": 100.0, "is_ad": True}


def test_find_verdict_no_match():
    verdicts = [{"start": 100.0, "is_ad": True}]
    assert _find_verdict(verdicts, 200.0) is None


# ---- 环境变量读取（统一为大写 CUTAD_LLM_*） ----

def test_from_env_reads_uppercase_var(monkeypatch):
    monkeypatch.setenv("CUTAD_LLM_KEY", "sk-test")
    assert _from_env("CUTAD_LLM_KEY", "") == "sk-test"


def test_from_env_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("CUTAD_LLM_KEY", raising=False)
    assert _from_env("CUTAD_LLM_KEY", "fallback") == "fallback"


def test_call_chat_without_key_mentions_uppercase_env(monkeypatch):
    import pytest

    monkeypatch.delenv("CUTAD_LLM_KEY", raising=False)
    with pytest.raises(RuntimeError, match="CUTAD_LLM_KEY"):
        _call_chat([{"role": "user", "content": "hi"}])


def test_analyzer_raises_without_key(monkeypatch):
    import pytest

    monkeypatch.delenv("CUTAD_LLM_KEY", raising=False)
    analyzer = create_ai_analyzer()
    segs = [{"start": 0.0, "end": 5.0, "text": "欢迎来到XX娱乐城，大奖报不停"}]
    # 无 Key 时应报错并提示大写环境变量名（而非静默走网络）
    with pytest.raises(RuntimeError, match="CUTAD_LLM_KEY"):
        analyzer(segs)
