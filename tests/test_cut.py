# -*- coding: utf-8 -*-
"""测试: cut 模块的时间格式化"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fuckad.cut import fmt_time


def test_fmt_time_hours():
    assert fmt_time(3600.0) == "01:00:00.00"
    assert fmt_time(7199.5) == "01:59:59.50"


def test_fmt_time_minutes():
    assert fmt_time(0) == "00:00:00.00"
    assert fmt_time(65.0) == "00:01:05.00"


def test_fmt_time_millis():
    assert fmt_time(1.234) == "00:00:01.23"


def test_fmt_time_long():
    assert fmt_time(7696.5) == "02:08:16.50"
