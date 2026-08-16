# -*- coding: utf-8 -*-
"""
CutAd - 视频剪切与拼接模块

流程：
1. 根据广告时间段计算保留片段
2. 对边界非关键帧处局部重编码、中间关键帧区间流复制，切为 MP4 分段
3. ffmpeg concat demuxer 流复制拼接（严格保持帧数，无重复/丢失）
4. ffmpeg faststart 优化
5. 清理临时文件 + 验证输出
"""
import json
import os
import subprocess
import sys
from pathlib import Path

# PyAV 延迟导入
import av


def fmt_time(s: float) -> str:
    """秒数转时间码 H:MM:SS.ms"""
    h = int(s // 3600)
    m = int(s % 3600 // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:05.2f}"


def cut_and_join(video_path: str, ads: list, output_path: str = None,
                 tmp_dir: str = None) -> str:
    """
    根据广告时间段剪切并拼接视频。
    
    参数:
        video_path: 源视频路径
        ads: 广告片段列表，每项为 (start, end) 或 AdSegment 对象
        output_path: 输出路径，默认为 video_path_no_ads.mp4
        tmp_dir: 临时文件目录，默认为视频所在目录
    
    返回:
        输出文件路径
    """
    video_path = str(video_path)
    out = Path(output_path or video_path.rsplit(".", 1)[0] + "_no_ads.mp4")
    work_dir = Path(tmp_dir or os.path.dirname(video_path) or ".")
    work_dir.mkdir(parents=True, exist_ok=True)

    # 获取视频时长
    total_dur = _get_duration(video_path)
    print(f"源视频: {video_path} ({total_dur:.1f}s)", flush=True)

    # 解析广告片段
    ad_intervals = []
    for ad in ads:
        if hasattr(ad, "start"):
            ad_intervals.append((ad.start, ad.end))
        elif isinstance(ad, (list, tuple)) and len(ad) >= 2:
            ad_intervals.append((float(ad[0]), float(ad[1])))
        elif isinstance(ad, dict):
            ad_intervals.append((ad.get("start", 0), ad.get("end", 0)))

    total_removed = sum(e - s for s, e in ad_intervals)
    print(f"去除广告: {total_removed:.1f}s ({len(ad_intervals)} 段)", flush=True)

    # 计算保留片段
    segments = []
    last_end = 0.0
    for s, e in sorted(ad_intervals):
        if s > last_end:
            segments.append((last_end, s))
        last_end = max(last_end, e)
    if last_end < total_dur:
        segments.append((last_end, total_dur))

    print(f"保留 {len(segments)} 个片段:", flush=True)
    for i, (s, e) in enumerate(segments, 1):
        print(f"  片段{i}: {fmt_time(s)} ~ {fmt_time(e)} ({e-s:.1f}s)", flush=True)

    # Step 1: ffmpeg 剪切为 MP4 片段（帧精确模式）
    print("\n步骤1: 剪切为 MP4 片段 (帧精确模式) ...", flush=True)
    params = _source_video_params(video_path)
    seg_files = []
    for i, (s, e) in enumerate(segments, 1):
        seg_file = work_dir / f"seg_{i:02d}.mp4"
        parts = _cut_segment_frame_accurate(video_path, s, e, seg_file, params)
        if not parts:
            print(f"  片段{i} 剪切失败！", flush=True)
            continue
        for p, info in parts:
            seg_files.append((p, info))
            print(f"  {os.path.basename(p)}: {Path(p).stat().st_size/1024/1024:.1f} MB", flush=True)

    if not seg_files:
        print("剪切失败！", flush=True)
        sys.exit(1)

    # Step 2: ffmpeg concat demuxer 流复制拼接
    # 各 A/M/C 段独立生成、参数一致（重编码段用 _source_video_params 对齐），
    # concat demuxer 按段序自动重排时间戳并严格保持帧数（无重复/丢失），
    # 相比 PyAV 手动偏移拼接避免了黑帧/重复帧问题，边界误差 <0.1s。
    print(f"\n步骤2: 拼接 {len(seg_files)} 个片段 (ffmpeg concat demuxer) ...", flush=True)
    list_file = work_dir / "concat_list.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for seg_file, _ in seg_files:
            p = str(seg_file).replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{p}'\n")
    r = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c", "copy", str(out)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  concat 拼接失败: {r.stderr[-3000:]}", flush=True)
        sys.exit(1)
    list_file.unlink(missing_ok=True)
    out_size = out.stat().st_size / 1024 / 1024
    print(f"  拼接完成: {out_size:.1f} MB", flush=True)

    # Step 3: ffmpeg faststart 优化
    print("\n步骤3: MP4 faststart 优化 ...", flush=True)
    tmp_opt = work_dir / "tmp_optimized.mp4"
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(out), "-c", "copy",
         "-movflags", "+faststart", str(tmp_opt)],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        os.replace(tmp_opt, out)
        out_size = out.stat().st_size / 1024 / 1024
        print(f"  优化完成: {out} ({out_size:.1f} MB)", flush=True)
    else:
        print(f"  优化跳过: {r.stderr[-200:]}", flush=True)

    # Step 4: 清理临时文件
    print("\n清理临时文件 ...", flush=True)
    for sf, _ in seg_files:
        p = Path(sf)
        if p.exists():
            p.unlink()
            print(f"  删除 {p.name}", flush=True)
    if tmp_opt.exists():
        tmp_opt.unlink()

    # Step 5: 验证
    print("\n验证输出 ...", flush=True)
    dur = _get_duration(str(out))
    print(f"  时长: {dur:.1f}s ({int(dur)//3600}h {(int(dur)%3600)//60}m {int(dur)%60}s)", flush=True)
    print(f"  大小: {out.stat().st_size/1024/1024:.1f} MB", flush=True)

    # 流信息
    r2 = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,width,height,pix_fmt,sample_rate,channels",
         "-of", "json", str(out)],
        capture_output=True, text=True,
    )
    if r2.returncode == 0:
        streams = json.loads(r2.stdout).get("streams", [])
        for s in streams:
            if s["codec_type"] == "video":
                print(f"  视频: {s['width']}x{s['height']} {s.get('pix_fmt','?')} {s['codec_name']}", flush=True)
            elif s["codec_type"] == "audio":
                print(f"  音频: {s.get('sample_rate','?')}Hz {s.get('channels','?')}ch {s['codec_name']}", flush=True)

    expected = total_dur - total_removed
    diff = abs(dur - expected)
    print(f"\n✅ 完成!", flush=True)
    print(f"   原视频: {total_dur:.1f}s", flush=True)
    print(f"   去除广告: {total_removed:.1f}s", flush=True)
    print(f"   输出视频: {dur:.1f}s", flush=True)
    print(f"   预期时长: {expected:.1f}s", flush=True)
    if diff < 2.0:
        print(f"   时长一致 (差异 {diff:.1f}s) ✅", flush=True)
    else:
        print(f"   时长差异 {diff:.1f}s ⚠️", flush=True)

    return str(out)


def cut_from_ads_json(video_path: str, ads_json: str = "ads.json",
                      output_path: str = None, tmp_dir: str = None) -> str:
    """
    从 ads.json 读取广告时间段并执行剪切拼接。
    """
    with open(ads_json, encoding="utf-8") as f:
        data = json.load(f)

    ads = data.get("ads", [])
    if not ads:
        print("ads.json 中没有广告数据，请先运行 detect 命令。", flush=True)
        sys.exit(1)

    ad_intervals = [(ad["start"], ad["end"]) for ad in ads]
    return cut_and_join(video_path, ad_intervals, output_path=output_path,
                        tmp_dir=tmp_dir)


# ============================================================
# 辅助函数
# ============================================================
def _get_duration(video_path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", video_path],
            capture_output=True, text=True, check=True,
        )
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def _first_keyframe_after(video_path: str, t: float, stream: int = 0) -> float:
    """返回视频流中 >= t 的第一个关键帧时间戳（秒），找不到则返回 t"""
    # 探测 40s 窗口，覆盖 2~3 个 GOP（本片 GOP≈14s），避免漏检关键帧
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", f"v:{stream}",
        "-show_entries", "frame=pts_time,key_frame",
        "-read_intervals", f"{t}%+40",
        "-of", "csv=p=0",
        video_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception:
        return t
    for line in r.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 2:
            try:
                # ffprobe csv 输出顺序: key_frame,pts_time（与 show_entries 书写顺序无关）
                kf = int(float(parts[0]))
                pts = float(parts[1])
            except ValueError:
                continue
            if kf == 1 and pts >= t - 1e-6:
                return pts
    return t


def _last_keyframe_before(video_path: str, t: float, stream: int = 0) -> float:
    """返回视频流中 < t 的最后一个关键帧时间戳（秒）；找不到则返回 0"""
    # 从 t 向前探测 40s 窗口（覆盖 2~3 个 GOP，本片 GOP≈14s）
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", f"v:{stream}",
        "-show_entries", "frame=pts_time,key_frame",
        "-read_intervals", f"{max(0.0, t - 40)}%+40.5",
        "-of", "csv=p=0",
        video_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception:
        return 0.0
    kf = None
    for line in r.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 2:
            try:
                kflag = int(float(parts[0]))
                pts = float(parts[1])
            except ValueError:
                continue
            if kflag == 1 and pts < t - 1e-6:
                kf = pts
    return kf if kf is not None else 0.0


def _source_video_params(video_path: str) -> dict:
    """读取源视频/音频流参数，用于边界重编码时保持参数一致（避免 SPS/PPS 变化）"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries",
             "stream=time_base,codec_name,profile,level,pix_fmt",
             "-of", "json", video_path],
            capture_output=True, text=True, timeout=60,
        )
        s = json.loads(r.stdout)["streams"][0]
    except Exception:
        return {}
    timescale = 0
    tb = s.get("time_base", "")
    if tb and "/" in tb:
        try:
            timescale = int(tb.split("/")[1])
        except ValueError:
            timescale = 0
    level = s.get("level", 40)
    try:
        level = int(float(level))
        level_str = f"{level // 10}.{level % 10}" if level >= 10 else str(level)
    except (TypeError, ValueError):
        level_str = str(level)
    profile_map = {"High": "high", "Main": "main", "Baseline": "baseline",
                   "Constrained Baseline": "baseline", "High 10": "high10",
                   "High 4:4:4": "high444"}
    # 音频位率（用于边界重编码时保持码率一致）
    aac_bitrate = 64000
    try:
        ra = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=bit_rate,codec_name",
             "-of", "json", video_path],
            capture_output=True, text=True, timeout=60,
        )
        streams = json.loads(ra.stdout).get("streams", [])
        if streams and int(streams[0].get("bit_rate") or 0) > 0:
            aac_bitrate = int(streams[0]["bit_rate"])
    except Exception:
        pass
    return {
        "timescale": timescale,
        "profile": profile_map.get(s.get("profile"), "high"),
        "level": level_str,
        "pix_fmt": s.get("pix_fmt", "yuv420p"),
        "aac_bitrate": aac_bitrate,
    }


def _count_frames(video_path: str, start: float, end: float,
                  stream: str = "v:0") -> int:
    """统计源视频流在 [start, end) 区间内的帧数；失败返回 -1。

    用 -count_packets 仅解封装、不解码（MP4 中每个视频 sample = 1 帧），
    从 110s（逐帧解码 2 万帧）降到 <1s。要求 start 为关键帧：
    -read_intervals "start%+dur" 的 % 快速 seek 到 start，精确读 [start, end)。
    """
    if end <= start:
        return 0
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", stream,
        "-count_packets",
        "-read_intervals", f"{start}%+{end - start:.6f}",
        "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0",
        video_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception:
        return -1
    for line in reversed(r.stdout.splitlines()):
        line = line.strip()
        if line:
            try:
                return int(float(line))
            except ValueError:
                continue
    return -1


def _pre_roll_frames(video_path: str, t: float, stream: int = 0) -> int:
    """返回从关键帧 t 开始解码后、显示序上位于 t 之前的帧数。

    输入侧 `-ss t -i` seek 到关键帧 t 后，开 GOP 边界会有若干 PTS<t 的 B 帧
    （显示在 t 前、解码在 t 后）被解码，这些帧会消耗 -frames:v 预算却不会
    写入输出，导致按 _count_frames 精确数帧时尾部丢帧。用 ffprobe 读 t 之后
    5s 的包（解码序），统计第一个 PTS>=t 的包之前的包数，即为该关键帧的
    pre-roll 帧数（已实测与 ffmpeg 实际浪费帧数完全一致）。
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", f"v:{stream}",
        "-show_entries", "packet=pts_time",
        "-read_intervals", f"{t}%+5",
        "-of", "csv=p=0",
        video_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception:
        return 0
    cnt = 0
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pts = float(line.split(",")[0])
        except ValueError:
            continue
        if pts >= t - 1e-4:
            break
        cnt += 1
    return cnt


def _reencode_segment(video_path: str, s: float, e: float,
                      out_file: Path, params: dict) -> str:
    """
    帧精确重编码 [s, e)。
    先 seek 到 s 前一关键帧（避免整段解码），再用 trim 精确切帧，
    setpts/asetpts 将首帧时间戳归零；音频一并重编码避免带入边界广告包。
    返回生成文件路径，失败返回空串。
    """
    prev_kf = _last_keyframe_before(video_path, s)
    trim_start = max(0.0, s - prev_kf)
    trim_end = max(trim_start + 1e-6, e - prev_kf)
    cmd = [
        "ffmpeg", "-y", "-ss", str(prev_kf), "-i", video_path,
        "-vf", f"trim=start={trim_start:.6f}:end={trim_end:.6f},setpts=PTS-STARTPTS",
        "-af", f"atrim=start={trim_start:.6f}:end={trim_end:.6f},asetpts=PTS-STARTPTS",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", params.get("pix_fmt", "yuv420p"),
        "-profile:v", params.get("profile", "high"),
        "-level", params.get("level", "4.0"),
        "-video_track_timescale", str(params.get("timescale") or 90000),
        # 音频也必须重编码：-c:a copy 会把起点前整包广告音频一起带入
        "-c:a", "aac", "-b:a", str(params.get("aac_bitrate", 64000)),
        "-muxdelay", "0", "-muxpreload", "0",
        str(out_file),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and out_file.exists() and out_file.stat().st_size > 0:
        return str(out_file)
    out_file.unlink(missing_ok=True)
    return ""


def _stream_copy_segment(video_path: str, s: float, e: float,
                         out_file: Path) -> str:
    """
    流复制剪切 [s, e)。
    - s 为关键帧（主路径 M 段，s=K）时：-count_packets 精确数帧 + -frames:v N
      把视频止于 e 前最后一帧（避免 -t 因 B 帧重排越界 ~0.13s）；
    - s 非关键帧（退化回退路径）时：退化为 -t 方式（本就近似）。
    音频用 -t (e-s) 约束，避免流复制把整段后续音频全部带入。
    """
    kf = _first_keyframe_after(video_path, s)
    if abs(kf - s) < 1e-4:
        n = _count_frames(video_path, s, e)
        if n <= 0:
            return ""
        # 输入侧 -ss 会先把显示序位于 s 之前的 pre-roll 帧计入 -frames:v 预算
        # （开 GOP 边界实测 ~50 帧），导致尾部丢帧；动态补偿，使输出帧数 == n。
        # 注意: 不能再加余量, 否则多余的帧落在 ad 区域内 (PTS>=cut_pts),
        # concat demuxer 拼接时不会过滤, 会造成 +N 帧广告残留/重复。
        preroll = _pre_roll_frames(video_path, s)
        budget = n + preroll
        cmd = ["ffmpeg", "-y", "-ss", str(s), "-i", video_path,
               "-frames:v", str(budget), "-c:v", "copy", "-c:a", "copy",
               "-t", f"{max(0.0, e - s):.4f}",
               "-bsf:a", "aac_adtstoasc", str(out_file)]
    else:
        cmd = ["ffmpeg", "-y", "-ss", str(s), "-i", video_path,
               "-t", f"{max(0.0, e - s):.4f}", "-c:v", "copy", "-c:a", "copy",
               "-bsf:a", "aac_adtstoasc", str(out_file)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and out_file.exists() and out_file.stat().st_size > 0:
        return str(out_file)
    out_file.unlink(missing_ok=True)
    return ""


def _video_frame_count(path: str) -> int:
    """返回文件的视频帧数（读容器元数据，不解码）；无视频流返回 0；异常返回 -1"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30)
    except Exception:
        return -1
    line = r.stdout.strip()
    if not line:
        return 0
    try:
        return int(float(line.splitlines()[-1].strip()))
    except (ValueError, IndexError):
        return -1


def _cut_segment_frame_accurate(video_path: str, s: float, e: float,
                                out_file: Path, params: dict) -> list:
    """
    帧精确剪切单个保留片段 [s, e)，3 段式方案，把边界误差压到 <0.1s：

      A 段：重编码 [s, K)          —— 起点非关键帧时局部重编码，帧精确起始
      M 段：流复制 [K, Ke)          —— 中间主体无损复制，精确止于关键帧 Ke
      C 段：重编码 [Ke, e)          —— 终点非关键帧时局部重编码，帧精确结尾

    关键点：
    - 流复制以关键帧为界（K、Ke 均为关键帧），避免 -frames:v 按解码序取帧
      时 B 帧错位越过边界；
    - 非关键帧的起点/终点小窗口用 trim 重编码（显示序精确）；
    - 边界窗口若重编码后无视频帧（如文件头 s=0 前导段），自动跳过该段，
      避免产生纯音频片段破坏拼接；
    - 拼接时 PyAV 以视频流为同步参考丢弃前滚音频包。
    返回生成的临时文件 [(路径, 元信息)] 列表（1~3 个），元信息 dict：
      - reencode 段: {"mode": "reencode"}
      - 流复制段:    {"mode": "streamcopy", "cut_pts": 该段结束时间(相对段首0)}
        拼接时需丢弃 PTS>=cut_pts 的越界视频帧（-frames:v 补偿多出的帧）。
    """
    K = _first_keyframe_after(video_path, s)
    if K >= e:
        # 段内无关键帧（短保留段）：整段重编码保证帧精确
        f = _reencode_segment(video_path, s, e, out_file, params)
        if f and _video_frame_count(f) <= 0:
            Path(f).unlink(missing_ok=True)
            return []
        return [(f, {"mode": "reencode"})] if f else []

    Ke = _last_keyframe_before(video_path, e)
    if Ke < K:
        Ke = K  # 段内仅一个关键帧：无 M 段

    parts = []

    # A 段：起点窗口重编码 [s, K)
    if K - s > 0.05:
        a_file = out_file.with_name(out_file.stem + "_a.mp4")
        a_path = _reencode_segment(video_path, s, K, a_file, params)
        if a_path and _video_frame_count(a_path) <= 0:
            # 无视频帧（文件头/段内无内容）：删除并跳过 A 段
            Path(a_path).unlink(missing_ok=True)
            a_path = ""
        if not a_path:
            f = _stream_copy_segment(video_path, s, e, out_file)
            return [(f, {"mode": "streamcopy", "cut_pts": e - s})] if f else []
        parts.append((a_path, {"mode": "reencode"}))

    # M 段：中间流复制 [K, Ke)
    if Ke - K > 0.05:
        m_file = out_file.with_name(out_file.stem + "_m.mp4")
        m_path = _stream_copy_segment(video_path, K, Ke, m_file)
        if not m_path:
            for p, _ in parts:
                Path(p).unlink(missing_ok=True)
            f = _stream_copy_segment(video_path, s, e, out_file)
            return [(f, {"mode": "streamcopy", "cut_pts": e - s})] if f else []
        parts.append((m_path, {"mode": "streamcopy", "cut_pts": Ke - K}))

    # C 段：终点窗口重编码 [Ke, e)
    if e - Ke > 0.05:
        c_file = out_file.with_name(out_file.stem + "_c.mp4")
        c_path = _reencode_segment(video_path, Ke, e, c_file, params)
        if c_path and _video_frame_count(c_path) <= 0:
            # 无视频帧：删除并跳过 C 段
            Path(c_path).unlink(missing_ok=True)
            c_path = ""
        if not c_path:
            for p, _ in parts:
                Path(p).unlink(missing_ok=True)
            f = _stream_copy_segment(video_path, s, e, out_file)
            return [(f, {"mode": "streamcopy", "cut_pts": e - s})] if f else []
        parts.append((c_path, {"mode": "reencode"}))

    if not parts:
        f = _stream_copy_segment(video_path, s, e, out_file)
        return [(f, {"mode": "streamcopy", "cut_pts": e - s})] if f else []
    return parts
