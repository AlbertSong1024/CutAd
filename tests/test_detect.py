# -*- coding: utf-8 -*-
"""测试: detect 模块的关键词规则与依赖隔离"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cutad.detect import (_detect_ads_by_rules, _is_ad_segment,
                          _merge_regions, _regions_covered,
                          _video_fingerprint, _get_asr_cache_key,
                          detect_scene_cuts)


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
    # 保证纯剪切用户 import cutad 时不会被迫加载 cv2/faster_whisper
    import importlib
    import sys as _sys

    assert "cv2" not in _sys.modules
    assert "torch" not in _sys.modules
    assert "faster_whisper" not in _sys.modules
    importlib.reload(sys.modules["cutad.detect"])
    assert "cv2" not in _sys.modules
    assert "faster_whisper" not in _sys.modules


# ---- 场景缓存：视频指纹隔离 + 窗口覆盖校验 ----

def test_video_fingerprint_differs_by_path():
    assert _video_fingerprint("video_a.mp4") != _video_fingerprint("video_b.mp4")


def test_video_fingerprint_stable_for_same_path():
    assert _video_fingerprint("video_a.mp4") == _video_fingerprint("video_a.mp4")


def test_asr_cache_key_differs_by_model():
    assert _get_asr_cache_key("v.mp4", "tiny") != _get_asr_cache_key("v.mp4", "base")


def test_merge_regions_merges_overlapping_and_adjacent():
    # 重叠窗口合并为一个；间隔 >2s 的窗口保持独立
    merged = _merge_regions([(0.0, 10.0), (5.0, 12.0), (20.0, 30.0)])
    assert merged == [(0.0, 12.0), (20.0, 30.0)]


def test_regions_covered_fullscan_covers_anything():
    # 全片缓存（regions=None）覆盖任何请求
    assert _regions_covered(None, [(10.0, 20.0)]) is True
    assert _regions_covered(None, None) is True


def test_regions_covered_window_within_cache():
    assert _regions_covered([(0.0, 20.0)], [(5.0, 15.0)]) is True


def test_regions_covered_window_outside_cache():
    assert _regions_covered([(0.0, 20.0)], [(30.0, 40.0)]) is False


def test_regions_covered_partial_overlap_not_enough():
    # 请求窗口跨越缓存窗口边界，不能视为覆盖
    assert _regions_covered([(0.0, 20.0)], [(15.0, 25.0)]) is False


def test_regions_covered_fullscan_request_needs_full_cache():
    # 请求全片时，窗口化缓存不完整，不能复用
    assert _regions_covered([(0.0, 20.0)], None) is False


def test_scene_cache_hit_when_regions_covered(tmp_path, monkeypatch):
    # 缓存窗口覆盖请求窗口时应直接命中，不触发 cv2 加载
    import json as _json

    cache = tmp_path / "scene_cuts.json"
    cache.write_text(_json.dumps({
        "cut_count": 1, "cuts": [10.0], "black_count": 0, "blacks": [],
        "regions": [[0.0, 20.0]],
    }), encoding="utf-8")

    def _no_cv2(name):
        raise AssertionError("缓存未命中，不应加载 cv2")

    monkeypatch.setattr("cutad.detect._require", _no_cv2)
    data = detect_scene_cuts("fake.mp4", output_json=str(cache),
                             regions=[(5.0, 15.0)])
    assert data["cuts"] == [10.0]


def test_scene_cache_miss_when_regions_not_covered(tmp_path, monkeypatch):
    # 缓存窗口不覆盖请求窗口时应重新扫描（走到 cv2 依赖检查）
    import json as _json
    import pytest

    cache = tmp_path / "scene_cuts.json"
    cache.write_text(_json.dumps({
        "cut_count": 0, "cuts": [], "black_count": 0, "blacks": [],
        "regions": [[0.0, 20.0]],
    }), encoding="utf-8")

    def _boom(name):
        raise RuntimeError("mock: 需要重新扫描")

    monkeypatch.setattr("cutad.detect._require", _boom)
    with pytest.raises(RuntimeError, match="重新扫描"):
        detect_scene_cuts("fake.mp4", output_json=str(cache),
                          regions=[(30.0, 40.0)])
