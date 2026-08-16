# -*- coding: utf-8 -*-
"""测试: llm 模块的 LLM 输出解析与判定"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cutad.llm import _extract_json_array, _find_verdict


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
