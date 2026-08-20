# -*- coding: utf-8 -*-
"""CutAd CLI 入口"""
import argparse
import sys
from pathlib import Path

from cutad.detect import detect_ads, fmt_time
from cutad.cut import cut_and_join, cut_from_ads_json


def _add_llm_args(parser):
    """为子命令添加 LLM 语义确认参数（默认对接智谱 GLM）"""
    parser.add_argument("--llm", action="store_true",
                        help="启用 LLM 语义二次确认（默认智谱 glm-4-flash，需配 Key）")
    parser.add_argument("--llm-deep", action="store_true",
                        help="LLM 全文深度扫描（识别软性植入广告，不依赖关键词规则，需配 --llm）")
    parser.add_argument("--llm-model", default=None,
                        help="LLM 模型名（默认 glm-4-flash）")
    parser.add_argument("--llm-url", default=None,
                        help="API 地址（默认 https://open.bigmodel.cn/api/paas/v4）")
    parser.add_argument("--llm-key", default=None,
                        help="API Key（也可用环境变量 cutad_LLM_KEY）")


def _llm_kwargs(args) -> dict:
    """根据命令行参数构造 create_ai_analyzer 的参数字典"""
    kwargs = {}
    if getattr(args, "llm_model", None):
        kwargs["model"] = args.llm_model
    if getattr(args, "llm_url", None):
        kwargs["base_url"] = args.llm_url
    if getattr(args, "llm_key", None):
        kwargs["api_key"] = args.llm_key
    return kwargs


def main():
    parser = argparse.ArgumentParser(
        prog="cutad",
        description="CutAd - 视频广告自动检测与剪切工具",
        epilog="""
模型速度参考（2小时视频，CPU环境）:
  tiny   - 最快 (~2min), 约90%准确率
  base   - 推荐 (~8min), 约93%准确率
  small  - (~20min), 约95%准确率
  medium - (~40min), 约97%准确率
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # --- detect 命令 ---
    p_detect = subparsers.add_parser("detect", help="检测视频中的广告片段")
    p_detect.add_argument("video", help="视频文件路径")
    p_detect.add_argument("-o", "--output-dir", default=".",
                          help="输出目录（默认当前目录）")
    p_detect.add_argument("--model", default="tiny",
                          choices=["tiny", "base", "small", "medium", "large"],
                          help="Whisper 模型（默认 tiny，推荐 base）")
    p_detect.add_argument("--no-cache", action="store_true",
                          help="禁用缓存，强制重新转写")
    p_detect.add_argument("--no-montage", action="store_true",
                          help="跳过缩略图拼图（省去逐帧 seek）")
    _add_llm_args(p_detect)

    # --- cut 命令 ---
    p_cut = subparsers.add_parser("cut", help="剪切广告并拼接视频")
    p_cut.add_argument("video", help="源视频路径")
    p_cut.add_argument("--ads",
                       help="广告时间段，格式: start1,end1;start2,end2 （秒）")
    p_cut.add_argument("--ads-json", default="ads.json",
                       help="从 ads.json 读取广告数据（默认 ads.json）")
    p_cut.add_argument("--output", "-o",
                       help="输出文件路径（默认 <原文件>_no_ads.mp4）")
    p_cut.add_argument("--tmp-dir",
                       help="临时文件目录（默认视频所在目录）")
    p_cut.add_argument("--skip-detect", action="store_true",
                       help="跳过检测，直接剪切（需先有 ads.json）")

    # --- all 命令（检测+剪切一体化）---
    p_all = subparsers.add_parser("all", help="检测并自动剪切广告")
    p_all.add_argument("video", help="视频文件路径")
    p_all.add_argument("-o", "--output-dir", default=".",
                       help="输出目录")
    p_all.add_argument("--model", default="tiny",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper 模型（默认 tiny，推荐 base）")
    p_all.add_argument("--output",
                       help="输出文件路径")
    p_all.add_argument("--skip-detect", action="store_true",
                       help="跳过检测，直接剪切")
    p_all.add_argument("--no-cache", action="store_true",
                       help="禁用缓存，强制重新转写")
    p_all.add_argument("--no-montage", action="store_true",
                       help="跳过缩略图拼图")
    _add_llm_args(p_all)

    args = parser.parse_args()

    if args.command == "detect":
        detect_ads(args.video, output_dir=args.output_dir, model=args.model,
                   use_llm=args.llm, llm_kwargs=_llm_kwargs(args),
                   llm_deep=args.llm_deep,
                   montage=not args.no_montage,
                   no_cache=args.no_cache)

    elif args.command == "cut":
        if args.ads:
            intervals = []
            for pair in args.ads.split(";"):
                parts = pair.strip().split(",")
                if len(parts) == 2:
                    intervals.append((float(parts[0]), float(parts[1])))
            cut_and_join(args.video, intervals,
                         output_path=args.output, tmp_dir=args.tmp_dir)
        else:
            cut_from_ads_json(args.video,
                              ads_json=args.ads_json,
                              output_path=args.output,
                              tmp_dir=args.tmp_dir)

    elif args.command == "all":
        if args.skip_detect:
            cut_from_ads_json(args.video,
                              output_path=args.output,
                              tmp_dir=args.output_dir)
        else:
            result = detect_ads(args.video, output_dir=args.output_dir,
                                model=args.model,
                                use_llm=args.llm, llm_kwargs=_llm_kwargs(args),
                                llm_deep=args.llm_deep,
                                montage=not args.no_montage,
                                no_cache=args.no_cache)
            if result.ads:
                intervals = [(ad.start, ad.end) for ad in result.ads]
                cut_and_join(args.video, intervals,
                             output_path=args.output,
                             tmp_dir=args.output_dir)
            else:
                print("未检测到广告，无需剪切。", flush=True)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
