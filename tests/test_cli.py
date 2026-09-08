# -*- coding: utf-8 -*-
"""测试: cli 模块的参数规范化（--llm-deep 联动 --llm）"""
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cutad.cli import _normalize_llm_args


def test_llm_deep_alone_enables_llm():
    # 只传 --llm-deep 时应自动启用 --llm，避免深度扫描被静默忽略
    args = Namespace(llm=False, llm_deep=True)
    _normalize_llm_args(args)
    assert args.llm is True


def test_llm_deep_with_llm_explicit_keeps_llm():
    args = Namespace(llm=True, llm_deep=True)
    _normalize_llm_args(args)
    assert args.llm is True


def test_no_llm_flags_untouched():
    args = Namespace(llm=False, llm_deep=False)
    _normalize_llm_args(args)
    assert args.llm is False


def test_args_without_llm_attrs_are_safe():
    # cut 子命令没有 LLM 参数，不应报 AttributeError
    args = Namespace(video="v.mp4", ads=None)
    _normalize_llm_args(args)
