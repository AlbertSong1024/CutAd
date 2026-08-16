# -*- coding: utf-8 -*-
"""
CutAd - 视频广告检测模块

流程：
1. VAD 预筛 + Whisper 转写（跳过静音段，加速 25x）
2. AI 语义判断广告片段
3. 场景切换 + 黑帧检测（吸附切割边界）
4. 输出 ads.json / 时间码文本 / 缩略图拼图
"""
import importlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Callable

from cutad.cut import fmt_time

# ============================================================
# 可选依赖（detect 功能需要，纯剪切不需要）
# 安装: pip install "cutad[detect]"
# ============================================================
_DETECT_EXTRAS = {
    "faster_whisper": "faster-whisper",
    "cv2": "opencv-python-headless",
    "numpy": "numpy",
}


def _require(mod_name: str):
    """延迟导入检测相关模块，缺失时给出可操作的安装提示。

    纯剪切（cut）不依赖这些模块；只有在调用 detect 相关功能时才导入，
    从而让轻量安装的用户 `import cutad` 也能成功。
    """
    try:
        return importlib.import_module(mod_name)
    except ImportError:
        pkg = _DETECT_EXTRAS.get(mod_name, mod_name)
        raise RuntimeError(
            f"检测功能需要可选依赖 {pkg}。"
            f"请先安装: pip install \"cutad[detect]\""
        ) from None

# ============================================================
# 配置
# ============================================================
DEFAULT_MODEL = "tiny"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_VAD_SILENCE_MS = 600
DEFAULT_MAX_EXPAND = 5.0
DEFAULT_SCENE_THRESHOLD = 900
DEFAULT_BLACK_THRESHOLD = 8.0
DEFAULT_SCENE_DEDUP_GAP = 0.35
DEFAULT_OUTPUT_DIR = "."

# ============================================================
# 数据结构
# ============================================================
@dataclass
class AdSegment:
    """广告片段"""
    id: str
    start: float
    end: float
    reason: str
    confidence: str
    start_audio: Optional[float] = None
    end_audio: Optional[float] = None
    keywords: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DetectionResult:
    """检测结果"""
    ads: list
    method: str
    total_duration: float = 0.0
    asr_duration: float = 0.0
    lang: str = ""

    def to_dict(self) -> dict:
        return {
            "ads": [a.to_dict() for a in self.ads],
            "method": self.method,
            "total_duration": self.total_duration,
            "asr_duration": self.asr_duration,
            "language": self.lang,
        }


# ============================================================
# Step 1: ASR 转写（VAD 预筛 + 进度条 + 缓存）
# ============================================================
def _get_asr_cache_key(video_path: str, model_name: str) -> str:
    """根据视频路径和模型生成缓存键"""
    import hashlib
    # 用文件路径 + 模型名 + 文件修改时间做哈希
    try:
        mtime = os.path.getmtime(video_path)
        key_str = f"{video_path}:{model_name}:{mtime}"
    except OSError:
        key_str = f"{video_path}:{model_name}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def _resolve_device() -> str:
    """探测可用的计算设备：优先 CUDA GPU，否则回退 CPU。

    通过 faster-whisper 的底层引擎 ctranslate2 探测 CUDA 设备数量，
    不依赖 torch。有 NVIDIA 独显时返回 "cuda"（转写提速 5~20x），
    否则返回 "cpu"（当前集显/无独显环境）。
    """
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        # 无 ctranslate2 或 CUDA 库不可用，静默回退 CPU
        pass
    return "cpu"


def transcribe_with_vad(video_path: str, model_name: str = DEFAULT_MODEL,
                        compute_type: str = DEFAULT_COMPUTE_TYPE,
                        vad_silence_ms: int = DEFAULT_VAD_SILENCE_MS,
                        cache_dir: str = ".") -> tuple:
    """
    用 faster-whisper 转写，内置 VAD 自动跳过静音/音乐段。
    支持结果缓存，避免重复转写同一视频。
    返回 (segments, language_info)
    """
    # 检查缓存
    cache_key = _get_asr_cache_key(video_path, model_name)
    cache_file = Path(cache_dir) / f".asr_cache_{cache_key}.json"
    if cache_file.exists():
        print(f"[asr] 从缓存加载转写结果: {cache_file.name}", flush=True)
        data = json.load(open(cache_file, encoding="utf-8"))
        return data["segments"], data["lang"]

    print(f"[asr] 加载 {model_name} 模型 (compute={compute_type}) ...", flush=True)
    t0 = time.time()
    fw = _require("faster_whisper")
    WhisperModel = fw.WhisperModel
    device = _resolve_device()
    if device == "cuda":
        # GPU 下用 float16 计算：显存占用低且远快于 int8，适合独显
        compute = "float16"
        print(f"[asr] 检测到 NVIDIA GPU，启用 CUDA 加速 (compute=float16)", flush=True)
    else:
        compute = compute_type
    model = WhisperModel(
        model_name, device=device, compute_type=compute,
        cpu_threads=os.cpu_count() or 4, num_workers=2,
    )
    print(f"[asr] 模型加载 {time.time()-t0:.1f}s", flush=True)

    # 获取视频时长用于进度显示
    total_dur = _get_video_duration(video_path)
    print(f"[asr] 视频时长 {total_dur:.0f}s, 开始转写 ...", flush=True)

    t1 = time.time()
    # beam_size：GPU 上并行能力强，用 3 提升精度且几乎无额外耗时；
    # CPU 上贪心解码(1) 约提速 1.5~2x，精度损失极小，更适合 CPU 环境
    beam_size = 3 if device == "cuda" else 1
    segments, info = model.transcribe(
        video_path,
        beam_size=beam_size,
        word_timestamps=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=vad_silence_ms),
    )

    seg_list = []
    last_progress = 0.0
    for i, seg in enumerate(segments):
        seg_list.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })
        # 进度条
        progress = min(seg.end / total_dur, 1.0) if total_dur > 0 else 0.0
        if progress - last_progress >= 0.05 or progress >= 1.0:
            bar_len = 20
            filled = int(bar_len * progress)
            bar = "█" * filled + "░" * (bar_len - filled)
            pct = progress * 100
            print(f"\r[asr] 转写进度: [{bar}] {pct:.1f}%  段数={len(seg_list)}  用时={time.time()-t1:.0f}s",
                  end="", flush=True)
            last_progress = progress

    print(f"\r[asr] 转写完成 {time.time()-t1:.0f}s, 语言={info.language} 概率={info.language_probability:.2f}  段数={len(seg_list)}", flush=True)

    # 保存到缓存
    cache_data = {
        "segments": seg_list,
        "lang": info.language,
        "model": model_name,
        "duration": time.time() - t0,
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False)
    print(f"[asr] 缓存已保存: {cache_file.name}", flush=True)

    return seg_list, info.language


# ============================================================
# Step 2: AI 语义判断（通用广告检测）
# ============================================================
def detect_ads_by_ai(segments: list, ai_analyzer: Optional[Callable] = None) -> list:
    """
    通用广告检测：分析转写文本，识别广告片段。

    优先使用 ai_analyzer（大模型回调），回退到基于规则的检测。
    """
    if ai_analyzer is not None:
        try:
            return ai_analyzer(segments)
        except Exception as e:
            print(f"[detect] AI 分析失败，回退到规则检测: {e}", flush=True)

    return _detect_ads_by_rules(segments)


# ============================================================
# 规则检测（AI 回退方案）
# ============================================================
# 强关键词：赌博/促销专属，几乎不会出现在正片台词中
_STRONG_KEYWORDS = [
    r"665[- ]?588", r"六六五五八八",
    r"娱乐城", r"娛樂城", r"黄金城", r"黃金城",
    r"新普京", r"新福京", r"新胡星",
    r"大獎隨時開", r"免費局數", r"免费局数",
    r"大奖报不停", r"大獎報不停",
    r"急送888",
    r"二十年幸运", r"二十年幸運",
    r"幸运老臺", r"幸运老台",
    r"赶快来玩吧", r"趕快來玩吧",
    r"开.*旗牌", r"开.*期牌", r"開.*期牌",
    r"马上拨打", r"马上撥打", r"拨打电话", r"免費撥打", r"免费拨打",
    r"点击下方", r"点击链接", r"扫码领取", r"掃碼領取",
]

# 弱关键词：情境化促销词，可能出现在对话里，需 ≥2 个组合或配合 LLM 确认
_WEAK_KEYWORDS = [
    r"下雨了", r"哪也去不了", r"好无聊啊", r"好無聊啊", r"又好玩又刺激",
    r"扫码", r"掃碼", r"限时", r"限時", r"优惠", r"優惠", r"打折", r"特惠",
    r"抢购", r"搶購", r"加购", r"加購", r"购物车", r"購物車", r"领券", r"領券",
    r"特价", r"特價", r"直播间", r"直播間", r"赞助", r"贊助", r"冠名",
    r"本视频由", r"本視頻由", r"广告时间", r"廣告時間",
    r"獎金", r"中奖", r"中獎", r"红包", r"紅包",
]

_STRONG_RULES = [re.compile(k, re.IGNORECASE) for k in _STRONG_KEYWORDS]
_WEAK_RULES = [re.compile(k, re.IGNORECASE) for k in _WEAK_KEYWORDS]
# 合并规则：用于收集展示用关键词
_RULES = _STRONG_RULES + _WEAK_RULES

# 合并参数
_MAX_NONMATCH_EXTEND = 15.0  # 未匹配段允许并入的累计时长（广告尾音/留白）
MAX_AD_DURATION = 240.0      # 单段广告上限（超过视为误合并，丢弃）


def _segment_keywords(text: str) -> list:
    """返回命中关键词列表（强+弱）"""
    return [r.search(text).group() for r in _RULES if r.search(text)]


def _is_ad_segment(text: str) -> bool:
    """强关键词 ≥1，或弱关键词 ≥2，才判定为广告片段"""
    if any(r.search(text) for r in _STRONG_RULES):
        return True
    weak_count = sum(1 for r in _WEAK_RULES if r.search(text))
    return weak_count >= 2


def _detect_ads_by_rules(segments: list) -> list:
    """基于规则的候选广告段检测（强/弱分级 + 有界合并 + 时长过滤）"""
    candidates = []
    n = len(segments)
    i = 0
    while i < n:
        if not _is_ad_segment(segments[i]["text"]):
            i += 1
            continue

        start_t = segments[i]["start"]
        end_t = segments[i]["end"]

        # 向前合并（广告开场白/留白；未匹配段累计并入不超过阈值）
        j = i - 1
        nonmatch = 0.0
        while j >= 0:
            gap = start_t - segments[j]["end"]
            if _is_ad_segment(segments[j]["text"]):
                start_t = segments[j]["start"]
                nonmatch = 0.0
            elif gap < 1.0:
                nonmatch += segments[j]["end"] - segments[j]["start"]
                if nonmatch > _MAX_NONMATCH_EXTEND:
                    break
                start_t = segments[j]["start"]
            else:
                break
            j -= 1

        # 向后合并
        k = i + 1
        nonmatch = 0.0
        while k < n:
            ns = segments[k]
            gap = ns["start"] - end_t
            if _is_ad_segment(ns["text"]):
                end_t = ns["end"]
                nonmatch = 0.0
            elif gap < 1.0:
                nonmatch += ns["end"] - ns["start"]
                if nonmatch > _MAX_NONMATCH_EXTEND:
                    break
                end_t = ns["end"]
            else:
                break
            k += 1

        dur = end_t - start_t
        # 时长过滤：太短（<1s）或过长（>MAX，典型误合并）都丢弃
        if 1.0 <= dur <= MAX_AD_DURATION:
            lo = max(j + 1, 0)
            hi = min(k, n)
            ad_text = " ".join([segments[x]["text"] for x in range(lo, hi)])
            all_kws = []
            for idx in range(lo, hi):
                all_kws.extend(_segment_keywords(segments[idx]["text"]))
            candidates.append({
                "start": start_t,
                "end": end_t,
                "text": ad_text[:200],
                "keywords": list(set(all_kws)),
            })
            i = k
            continue
        i += 1
    return candidates


# ============================================================
# Step 3: 场景切换 + 黑帧检测
# ============================================================
def detect_scene_cuts(video_path: str, output_json: str = None,
                      threshold: float = DEFAULT_SCENE_THRESHOLD,
                      black_threshold: float = DEFAULT_BLACK_THRESHOLD,
                      regions: Optional[list] = None) -> dict:
    """
    检测场景切换点和黑帧。
    返回 {cut_count, cuts, black_count, blacks}

    regions: 可选，[(start, end), ...] 秒级时间段。
             提供时只扫描这些窗口（广告边界吸附只需 ±max_expand 附近），
             全片扫描时逐帧解码所有帧，窗口化可减少 10~50 倍解码量。
    """
    if output_json and os.path.exists(output_json):
        return json.load(open(output_json, encoding="utf-8"))

    cv2 = _require("cv2")
    print(f"[scene] 检测场景切换和黑帧 {'(窗口化)' if regions else ''} ...", flush=True)
    t0 = time.time()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * 0.5)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cuts = []
    blacks = []

    def scan_range(start_f: int, end_f: int):
        """扫描 [start_f, end_f) 帧区间（依赖 cap 当前位置）"""
        nonlocal cuts, blacks
        prev = None
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        i = start_f
        while i < end_f:
            ok = cap.grab()
            if not ok:
                break
            if i % step == 0:
                ok2, f = cap.retrieve()
                if ok2:
                    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                    h = cv2.resize(g, (64, 36))
                    pos = i / fps
                    mean = float(h.mean())
                    if mean < black_threshold:
                        blacks.append(round(pos, 3))
                    if prev is not None:
                        d = float(((h.astype("float32") - prev.astype("float32")) ** 2).mean())
                        if d > threshold:
                            cuts.append(round(pos, 3))
                    prev = h
            i += 1

    if regions:
        # 合并重叠/相邻窗口，避免重复扫描
        regions = sorted(regions)
        merged = []
        for s, e in regions:
            if merged and s <= merged[-1][1] + 2.0:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        total_dur = total / fps if fps else 0.0
        for s, e in merged:
            s = max(0.0, s)
            e = min(e, total_dur)
            if e > s:
                scan_range(int(s * fps), int(e * fps))
    else:
        scan_range(0, total)

    cap.release()

    def dedup(ts):
        out = []
        for t in ts:
            if not out or t - out[-1] > DEFAULT_SCENE_DEDUP_GAP:
                out.append(t)
            else:
                out[-1] = t
        return out

    cuts = dedup(cuts)
    blacks = dedup(blacks)
    data = {"cut_count": len(cuts), "cuts": cuts, "black_count": len(blacks), "blacks": blacks}

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[scene] 已保存到 {output_json}", flush=True)

    print(f"[scene] 完成 {time.time()-t0:.0f}s, cuts={len(cuts)} blacks={len(blacks)}", flush=True)
    return data


# ============================================================
# Step 4: 边界扩展
# ============================================================
def expand_boundary(start: float, end: float, scene_data: dict,
                    max_expand: float = DEFAULT_MAX_EXPAND) -> tuple:
    """
    边界扩展：以语音内核为基础，向前后扫描场景切换/黑帧。
    自动处理广告词结束后的画面留白。
    """
    cuts = scene_data["cuts"]
    blacks = scene_data["blacks"]

    # 向前：找最近的前场景切换
    prev_cuts = [c for c in cuts if c < start and (start - c) <= max_expand]
    new_start = prev_cuts[-1] if prev_cuts else start

    # 向后：找最近的后场景切换
    next_cuts = [c for c in cuts if c > end and (c - end) <= max_expand]
    new_end = min(next_cuts[0] if next_cuts else (end + max_expand), end + max_expand)

    # 黑帧辅助（广告结束常见特征）
    nearby_blacks = [b for b in blacks if end <= b <= end + 3.0]
    if nearby_blacks:
        new_end = min(new_end, nearby_blacks[0] + 0.5)

    return round(new_start, 2), round(new_end, 2)


# ============================================================
# Step 5: 生成缩略图拼图
# ============================================================
def generate_montage(video_path: str, regions: dict,
                     n_frames: int = 12) -> dict:
    """生成缩略图拼图供用户确认"""
    cv2 = _require("cv2")
    np = _require("numpy")
    COLS = 4
    TH_W, TH_H = 320, 134
    GAP = 8

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    results = {}

    for name, (a, b) in regions.items():
        times = [a + (b - a) * k / (n_frames - 1) for k in range(n_frames)]
        frames = []
        prev_i = -1
        for t in times:
            i = int(t * fps)
            if i == prev_i:
                continue
            prev_i = i
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if ok:
                frames.append((t, f))

        rows = (len(frames) + COLS - 1) // COLS
        canvas = 255 * np.ones((rows * (TH_H + GAP) + GAP,
                                COLS * (TH_W + GAP) + GAP, 3), np.uint8)
        for idx, (t, f) in enumerate(frames):
            th = cv2.resize(f, (TH_W, TH_H))
            cv2.putText(th, fmt_time(t)[3:], (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            r, c = divmod(idx, COLS)
            y, x = GAP + r * (TH_H + GAP), GAP + c * (TH_W + GAP)
            canvas[y:y+TH_H, x:x+TH_W] = th

        fn = f"montage_{name}.jpg"
        cv2.imwrite(fn, canvas)
        results[name] = (a, b, len(frames))
        print(f"  {fn}: {len(frames)} frames", flush=True)

    cap.release()
    return results


# ============================================================
# 主检测函数
# ============================================================
def detect_ads(video_path: str, output_dir: str = DEFAULT_OUTPUT_DIR,
               model: str = DEFAULT_MODEL,
               ai_analyzer: Optional[Callable] = None,
               scene_cache: bool = True,
               use_llm: bool = False,
               llm_kwargs: Optional[dict] = None,
               montage: bool = True) -> DetectionResult:
    """
    完整广告检测流程。

    参数:
        video_path: 视频文件路径
        output_dir: 输出目录
        model: whisper 模型（tiny/base/small/medium/large）
               tiny: ~2min (2h视频, 约90%准确率)
               base: ~8min (2h视频, 约93%准确率) ← 推荐
               small: ~20min (2h视频, 约95%准确率)
               medium: ~40min (2h视频, 约97%准确率)
        ai_analyzer: 可选的 AI 分析函数（优先于 use_llm）
        scene_cache: 是否缓存场景数据
        use_llm: 是否启用 LLM 语义二次确认（默认对接本地 Ollama）
        llm_kwargs: 传给 create_ai_analyzer 的参数字典
        montage: 是否生成缩略图拼图
    """
    video_path = str(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"cutad - 视频广告检测")
    print(f"源文件: {video_path}")
    print(f"模型: {model}  (tiny最快/base推荐/medium最准)")
    if use_llm:
        print("LLM 语义确认: 开启")
    print("=" * 60)

    # 获取视频时长
    total_dur = _get_video_duration(video_path)
    print(f"视频时长: {total_dur:.1f}s", flush=True)

    # Step 1: ASR 转写
    t0 = time.time()
    segments, lang = transcribe_with_vad(
        video_path, model_name=model, cache_dir=str(out_dir)
    )
    t_asr = time.time() - t0
    print(f"\n[summary] ASR 耗时 {t_asr:.0f}s, 语言={lang}, 段数={len(segments)}", flush=True)

    # Step 2: 广告检测
    print("[detect] 分析广告片段 ...", flush=True)
    if ai_analyzer is None and use_llm:
        from cutad.llm import create_ai_analyzer
        ai_analyzer = create_ai_analyzer(**(llm_kwargs or {}))
    candidates = detect_ads_by_ai(segments, ai_analyzer=ai_analyzer)
    print(f"[detect] 找到 {len(candidates)} 个候选广告段", flush=True)
    for c in candidates:
        print(f"  {fmt_time(c['start'])} ~ {fmt_time(c['end'])} ({c['end']-c['start']:.1f}s)  关键词: {c['keywords']}",
              flush=True)

    if not candidates:
        print("\n未检测到广告，请检查内容或添加自定义规则。", flush=True)
        return DetectionResult(ads=[], method=f"VAD+{model}+规则检测", total_duration=total_dur,
                               asr_duration=t_asr, lang=lang)

    # Step 3: 场景切换（窗口化：只扫广告边界 ±max_expand 附近）
    scene_json = str(out_dir / "scene_cuts.json")
    regions = [(c["start"] - DEFAULT_MAX_EXPAND, c["end"] + DEFAULT_MAX_EXPAND)
               for c in candidates]
    scene_data = detect_scene_cuts(video_path,
                                   output_json=scene_json if scene_cache else None,
                                   regions=regions)

    # Step 4: 边界扩展
    ads = []
    for i, c in enumerate(candidates):
        new_start, new_end = expand_boundary(c["start"], c["end"], scene_data)
        ads.append(AdSegment(
            id=f"ad{i+1}",
            start=new_start,
            end=new_end,
            start_audio=c["start"],
            end_audio=c["end"],
            reason=f"关键词匹配: {' '.join(c['keywords'][:5])}",
            confidence="high" if len(c["keywords"]) >= 2 else "medium",
            keywords=c["keywords"],
        ))
        print(f"[boundary] ad{i+1}: {fmt_time(c['start'])}~{fmt_time(c['end'])} "
              f"-> {fmt_time(new_start)}~{fmt_time(new_end)}", flush=True)

    # Step 5: 保存结果
    result = DetectionResult(
        ads=ads, method=f"VAD+{model}+规则检测+边界扩展",
        total_duration=total_dur, asr_duration=t_asr, lang=lang,
    )
    with open(out_dir / "ads.json", "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    lines = ["# CutAd 广告检测结果",
             f"# 方法: {result.method}",
             f"# 检测到 {len(ads)} 段广告\n"]
    for ad in ads:
        lines.append(f"广告 {ad.id}: {fmt_time(ad.start)} ~ {fmt_time(ad.end)}  (时长 {ad.end-ad.start:.1f}s)")
        lines.append(f"   音频实际: {fmt_time(ad.start_audio or ad.start)} ~ {fmt_time(ad.end_audio or ad.end)}")
        lines.append(f"   依据: {ad.reason}")
        lines.append(f"   置信度: {ad.confidence}")
        lines.append("")
    with open(out_dir / "ad_timecodes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 缩略图拼图（可选）
    if montage:
        regions = {f"ad{i+1}": (ad.start, ad.end) for i, ad in enumerate(ads)}
        generate_montage(video_path, regions)

    print("\n" + "\n".join(lines))
    print(f"\n✅ 完成！输出目录: {out_dir}/", flush=True)
    return result


def detect_ads_cli(video_path: str, output_dir: str = DEFAULT_OUTPUT_DIR,
                   model: str = DEFAULT_MODEL) -> DetectionResult:
    """CLI 入口：自动检测并提示用户确认广告时间段"""
    result = detect_ads(video_path, output_dir=output_dir, model=model)

    if result.ads:
        print("\n请确认以上广告时间段是否正确。")
        print("如果不正确，请编辑 ads.json 后手动运行 cut 命令。")
        print(f"或运行: cutad cut <视频> --ads <start1,end1>;<start2,end2> ...")

    return result


# ============================================================
# 辅助函数
# ============================================================
def _get_video_duration(video_path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", video_path],
            capture_output=True, text=True, check=True
        )
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return 0.0
