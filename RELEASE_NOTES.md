# Release Notes / 发布说明

## v0.1.0 — Initial Release / 初始版本

**Release Date / 发布日期**: 2026-08-19

### Overview / 概述

CutAd 是第一款基于 AI 语义分析的视频广告检测与自动剪切工具。通过 Whisper ASR 语音转写 + LLM 语义理解，精准识别硬广与软广，并以流复制技术无损拼接无广告版本。

CutAd is the first video ad detection and removal tool powered by AI semantic analysis. Using Whisper ASR transcription + LLM semantic understanding, it precisely identifies hard-sell and soft-sell ads, then stitches an ad-free version with zero quality loss via stream-copy.

---

### New Features / 新增功能

#### 🎯 ASR-Based Ad Detection / ASR 广告检测
- **VAD 预筛**: 自动跳过静音段和音乐段，聚焦语音区域，大幅减少 Whisper 推理量和误检率
- **Whisper 语音转写**: 使用 CTranslate2 引擎（`faster-whisper`），支持 tiny/base/small/medium 四档模型，无需 PyTorch
- **关键词规则引擎**: 预置硬广/软广/植入关键词列表，覆盖扫码、促销、品牌露出等常见广告形式
- **边界扩展**: 以语音内核为锚点，向前后扫描到场景切换点，自动覆盖广告词结束后的画面留白（1~4 秒）
- **场景切换检测**: OpenCV 帧差法 + 黑帧检测，缓存至 `scene_cuts.json`，避免重复计算
- **缩略图拼图**: 自动为每段广告生成边界缩略图，方便人工核对

#### 🧠 LLM Semantic Confirmation / LLM 语义确认
- **二次确认**: 关键词命中候选段经 LLM 语义分析二次确认，减少误判
- **支持多种后端**: OpenAI (`gpt-4o-mini`)、Anthropic (`claude-sonnet-4-20250514`)、智谱 GLM-4-Flash（默认）
- **自定义模型**: 通过 `--llm-model`、`--llm-url` 参数支持任意 OpenAI/Anthropic 兼容接口
- **API Key 管理**: 支持环境变量 `CUTAD_LLM_KEY` 或命令行 `--llm-key`
- **LLM 深度扫描** (`--llm-deep`): 全文分块扫描，不依赖关键词规则，识别软广和植入广告

#### ✂️ Stream-Copy Cutting & Stitching / 流复制剪切拼接
- **ffmpeg concat demuxer**: 基于文件列表的流复制拼接，严格保持帧数不变
- **边界智能处理**: 非关键帧边界局部重编码，中间关键帧区间纯流复制
- **视频优化**: 自动执行 `ffmpeg -movflags +faststart` 将 moov atom 前置，支持流式播放
- **输出参数一致性**: 精确复制视频/音频流参数，确保输出与源文件一致

#### 🎮 GPU Acceleration / GPU 加速
- **自动检测**: 运行时探测 NVIDIA CUDA 设备，自动选择 GPU/CPU 模式
- **float16 计算**: GPU 模式下使用 compute=float16，Whisper 转写速度提升 5~20 倍
- **自动回退**: 无 GPU 时透明回退到 CPU，无需手动配置

#### 📦 Packaging / 打包发布
- **PyPI 包**: `pip install cutad`（核心）/ `pip install "cutad[detect,ai]"`（完整）
- **CLI 入口**: `cutad detect` / `cutad cut` / `cutad all` 三个子命令
- **Python API**: `detect_ads()` / `cut_and_join()` / `analyze_with_llm()` 三个核心函数

#### 📚 Documentation / 文档
- **中英双语 README**: `README.md`（英文）/ `README_CN.md`（中文）
- **完整 CLI 参考**: 所有命令和参数的详细说明
- **故障排查指南**: 常见问题及解决方案
- **贡献指南**: 开发环境搭建、代码规范、测试方法

---

### Technical Details / 技术细节

| 模块 | 技术栈 | 关键依赖 |
|------|--------|----------|
| ASR 转写 | CTranslate2 + Whisper | `faster-whisper` (CPU/GPU) |
| 场景检测 | OpenCV 帧差法 | `opencv-python-headless`, `numpy` |
| LLM 分析 | OpenAI/Anthropic/GitHub-compatible API | `openai`, `anthropic` |
| 视频剪切 | ffmpeg 流复制 | `av` (PyAV >= 10.0) |
| 命令行 | click | `click>=8.0` |

---

### Repository Info / 仓库信息

- **仓库地址**: https://github.com/AlbertSong1024/CutAd
- **许可证**: MIT License
- **Python 版本**: >= 3.9
- **状态**: ✅ 公开 (Public)
