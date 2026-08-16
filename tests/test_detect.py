# -*- coding: utf-8 -*-
"""测试: detect 模块的关键词规则与依赖隔离"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fuckad.detect import _detect_ads_by_rules, _is_ad_segment


# ---- 关键词分级判定 ----

def test_strong_keyword_alone_is_ad():
    assert _is_ad_segment("欢迎来到XX娱乐城，大奖报不停") is True


def test_single_weak_keyword_is_not_ad():
    # 单个弱关键词（如台词中偶然出现"下雨了"）不应判定为广告
    assert _is_ad_segment("今天下雨了，记得带伞") is False


def test_two_weak_keywords_is_ad():
    assert _is_ad_segment("下雨了哪也去不了，好无聊啊") is True


def test_plain_text_is_not_ad():
    assert _is_ad_segment("大家好，今天我们继续讲解第三章内容") is False


# ---- 候选段合并与过滤 ----

def test_rule_detection_merges_adjacent_ad_segments():
    # 广告段与正常内容间留有 >=1s 间隔，避免被向前/向后合并吞并
    segs = [
        {"start": 0.0, "end": 10.0, "text": "正常节目内容"},
        {"start": 11.0, "end": 12.0, "text": "欢迎来到XX娱乐城"},
        {"start": 12.0, "end": 14.0, "text": "大奖报不停"},
        {"start": 15.0, "end": 20.0, "text": "正常节目继续"},
    ]
    ads = _detect_ads_by_rules(segs)
    assert len(ads) == 1
    assert abs(ads[0]["start"] - 11.0) < 1e-6
    assert abs(ads[0]["end"] - 14.0) < 1e-6


def test_rule_detection_filters_short_segments():
    # 孤立且短于 1s 的候选应被丢弃（时长过滤）
    segs = [
        {"start": 0.0, "end": 0.5, "text": "娱乐城"},
        {"start": 5.0, "end": 10.0, "text": "正常内容"},
    ]
    ads = _detect_ads_by_rules(segs)
    assert ads == []


# ---- 导入隔离：不应触发重型依赖加载 ----

def test_import_does_not_load_heavy_deps():
    # 保证纯剪切用户 import fuckad 时不会被迫加载 cv2/faster_whisper
    import importlib
    import sys as _sys

    assert "cv2" not in _sys.modules
    assert "torch" not in _sys.modules
    assert "faster_whisper" not in _sys.modules
    importlib.reload(sys.modules["fuckad.detect"])
    assert "cv2" not in _sys.modules
    assert "faster_whisper" not in _sys.modules
