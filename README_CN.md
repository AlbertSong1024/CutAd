# CutAd — 视频广告自动检测与剪切工具

> 🎬 一段视频，几行命令，广告消失——CutAd 让观看体验重回干净流畅。

**CutAd** 是一个基于 AI 的视频广告检测与剪切工具。它先用 Whisper ASR 将视频语音转写成文本，再用大语言模型（LLM）语义分析识别硬广和软广，最后用流复制剪切技术拼接出无广告版本，全程不降画质。

**它能做什么：**
- 自动发现视频里的硬广（"扫码下载""限时特价"）和软广（"本期由XX赞助""感谢XX品牌"）
- 以帧级精度剪切广告段，并在关键帧处无缝拼接
- 生成缩略图拼图，方便你人工确认边界是否准确
- 支持 OpenAI / Anthropic / 智谱 GLM 等多种 LLM 后端
- GPU 自动检测，CUDA 加速下 Whisper 转写提速 5~20 倍

---

## 目录

1. [快速开始](#快速开始)
2. [安装](#安装)
3. [使用方法](#使用方法)
4. [输出文件](#输出文件)
5. [工作原理](#工作原理)
6. [命令行参考](#命令行参考)
7. [故障排查](#故障排查)
8. [项目结构](#项目结构)
9. [贡献指南](#贡献指南)
10. [许可证](#许可证)

> 🌐 English version available at [README.md](README.md)

---

## 快速开始

最简流程，三步搞定：

```bash
# 1. 安装（含检测 + LLM 语义分析）
pip install "cutad[detect,ai]"

# 2. 一键检测 + 剪切
cutad all your_video.mp4

# 3. 搞定，无广告视频在同目录生成
ls your_video_no_ads.mp4
```

> 💡 **提示**：首次运行会自动下载 Whisper 模型（~150MB base 模型）。国内用户请设置镜像：
> ```bash
> set HF_ENDPOINT=https://hf-mirror.com   # Windows
> export HF_ENDPOINT=https://hf-mirror.com # Linux / macOS
> ```

---

## 安装

### 前置要求
- Python 3.9+
- ffmpeg / ffprobe（确保在 PATH 中）

```bash
ffmpeg -version   # 验证安装
```

### 完整安装（检测 + 剪切 + LLM）
```bash
pip install "cutad[detect,ai]"
```

### 仅剪切（无需检测）
只需要去掉已知时间段的广告，不需要 AI 检测：
```bash
pip install cutad    # 只装核心依赖（PyAV）
```

### GPU 加速（可选）
CutAd 运行时会**自动探测 NVIDIA 显卡**：检测到 CUDA 设备则用 GPU 转写（compute=float16），否则自动回退 CPU。

```bash
# 安装 CUDA 版 ctranslate2（pip 默认装的是 CPU 版）
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
pip install ctranslate2 --no-deps
```

安装后 `nvidia-smi` 可见的 NVIDIA 独显即可自动启用 GPU 加速。

### 依赖说明

| 依赖 | 用途 | 必需 |
|------|------|------|
| `av` (PyAV) | 视频参数读取 / 流复制剪切 | ✅ 核心 |
| `faster-whisper` | ASR 语音转写（CTranslate2 引擎，不需要 PyTorch） | 检测时 |
| `opencv-python-headless` | 场景切换检测 / 缩略图生成 | 检测时 |
| `numpy` | 帧差计算 | 检测时 |
| `openai` / `anthropic` | LLM 语义分析客户端 | LLM 分析时 |

> ⚠️ **注意**：检测功能基于 `faster-whisper`（CTranslate2），**不依赖 PyTorch**，无需下载数百 MB 的 torch。

---

## 使用方法

### 命令行

```bash
# 一键检测 + 剪切
cutad all video.mp4

# 仅检测广告（不剪切，生成 ads.json + 拼图）
cutad detect video.mp4

# 仅剪切（从已有 ads.json 读取广告时间段）
cutad cut video.mp4

# 手动指定广告时间段（格式：start,end;start,end，单位秒）
cutad cut video.mp4 --ads "1697,1715;3692,3712"

# 指定输出路径
cutad cut video.mp4 --output clean_video.mp4

# 指定 Whisper 模型（tiny 最快 / base 推荐 / medium 最准）
cutad detect video.mp4 --model base

# 启用 LLM 语义确认（默认智谱 GLM-4-Flash）
cutad detect video.mp4 --llm --llm-key YOUR_API_KEY

# 启用 LLM 全文深度扫描（识别软广/植入广告，不依赖关键词）
cutad detect video.mp4 --llm --llm-deep --llm-key YOUR_API_KEY
```

### Python 接口

```python
from cutad import detect_ads, cut_and_join

# 检测广告
result = detect_ads("video.mp4")
print(f"找到 {len(result.ads)} 段广告")
for ad in result.ads:
    print(f"  {ad.id}: {ad.start:.1f}s ~ {ad.end:.1f}s  ({ad.reason})")

# 剪切拼接
output = cut_and_join("video.mp4", [(ad.start, ad.end) for ad in result.ads])
print(f"输出: {output}")
```

### AI 语义分析

```python
from cutad import detect_ads, analyze_with_llm
import openai

# 接入 OpenAI
client = openai.OpenAI(api_key="sk-...")
result = detect_ads(
    "video.mp4",
    ai_analyzer=lambda segs: analyze_with_llm(segs, llm_client=client),
)

# 接入 Anthropic
import anthropic
client = anthropic.Anthropic(api_key="sk-ant-...")
result = detect_ads(
    "video.mp4",
    ai_analyzer=lambda segs: analyze_with_llm(segs, llm_client=client),
)
```

---

## 输出文件

| 文件 | 说明 |
|------|------|
| `ads.json` | 结构化广告段数据（起止时间、理由、置信度） |
| `ad_timecodes.txt` | 人类可读的时间码表 |
| `montage_ad*.jpg` | 广告边界缩略图拼图（用于人工核对） |
| `scene_cuts.json` | 场景切换点缓存（避免重复计算） |
| `<原文件名>_no_ads.mp4` | 去广告后的输出视频 |

---

## 工作原理

```
源视频 (mp4)
    │
    ├──[ASR 转写] VAD 预筛 + Whisper 语音识别
    │       跳过静音/音乐段，输出带时间戳的语音片段
    │
    ├──[广告检测] 关键词规则 + LLM 语义分析
    │       先扫关键词命中候选段，再用 LLM 做语义二次确认
    │       启用 --llm-deep 时：全文分块扫描，识别软广/植入广告
    │
    ├──[场景检测] OpenCV 帧差法检测场景切换 + 黑帧
    │       缓存 scene_cuts.json，避免重复计算
    │
    ├──[边界扩展] 以语音内核为锚点，向前后扫到场景切换点
    │       自动覆盖广告词结束后的画面留白（1~4 秒）
    │
    ├──[输出分析] ads.json / ad_timecodes.txt / montage_*.jpg
    │
    ├──[剪切] ffmpeg 流复制分段
    │       边界非关键帧处局部重编码，中间关键帧区间纯流复制
    │
    ├──[拼接] ffmpeg concat demuxer 流复制拼接
    │       严格保持帧数，无重复无丢失，画质无损
    │
    └──[优化] ffmpeg faststart（moov atom 前置，支持流式播放）
```

**模型速度参考（2小时视频，CPU 环境）：**

| 模型 | 耗时 | 准确率 |
|------|------|--------|
| tiny | ~2 分钟 | ~90% |
| base | ~8 分钟 | ~93% |
| small | ~20 分钟 | ~95% |
| medium | ~40 分钟 | ~97% |

---

## 命令行参考

### `detect` — 检测广告
```bash
cutad detect VIDEO [OPTIONS]

位置参数:
  VIDEO              视频文件路径

可选参数:
  -o, --output-dir   输出目录（默认当前目录）
  --model            Whisper 模型: tiny/base/small/medium/large（默认 tiny，推荐 base）
  --no-cache         禁用缓存，强制重新转写
  --no-montage       跳过缩略图拼图生成
  --llm              启用 LLM 语义二次确认
  --llm-deep         LLM 全文深度扫描（识别软广，需配合 --llm）
  --llm-model        自定义 LLM 模型名
  --llm-url          自定义 API 地址
  --llm-key          API Key（或用环境变量 CUTAD_LLM_KEY）
  -h, --help         显示帮助
```

### `cut` — 剪切广告
```bash
cutad cut VIDEO [OPTIONS]

位置参数:
  VIDEO              源视频路径

可选参数:
  --ads              广告时间段，格式: "start1,end1;start2,end2"
  --ads-json         从 ads.json 读取（默认 ads.json）
  -o, --output       输出文件路径（默认 <原文件>_no_ads.mp4）
  --tmp-dir          临时文件目录
  --skip-detect      跳过检测，直接剪切
  -h, --help         显示帮助
```

### `all` — 一键检测 + 剪切
```bash
cutad all VIDEO [OPTIONS]

位置参数:
  VIDEO              视频文件路径

可选参数:
  -o, --output-dir   输出目录
  --model            Whisper 模型
  --output           输出文件路径
  --skip-detect      跳过检测，直接剪切（需已有 ads.json）
  --no-cache         禁用缓存
  --no-montage       跳过缩略图
  --llm              启用 LLM 语义确认
  --llm-deep         LLM 深度扫描
  --llm-model        自定义模型名
  --llm-url          自定义 API 地址
  --llm-key          API Key
  -h, --help         显示帮助
```

---

## 故障排查

| 问题 | 解决方案 |
|------|----------|
| `ffprobe` 找不到 | 安装 ffmpeg，确保 `ffmpeg` 在系统 PATH 中 |
| Whisper 模型下载失败或超时 | 设置 `HF_ENDPOINT=https://hf-mirror.com` 使用国内镜像 |
| 拼接时报错 / 输出视频无画面 | 确保 `av>=10.0`；确认分段编码参数一致（来自同一源视频） |
| 检测提示缺少依赖 | 运行 `pip install "cutad[detect]"` 安装检测可选依赖 |
| 内存不足 | 使用 `--model tiny` 或 `--model base` 降低内存占用 |
| ASR 识别语言不对 | 在 prompt 中指定目标语言，或切换到对应的多语言 Whisper 模型 |
| GPU 未被识别 | 检查 `nvidia-smi` 是否可用；确认已安装 CUDA 版 `ctranslate2` |

---

## 项目结构

```
cutad/
├── cutad/                    # Python 包
│   ├── __init__.py            # 包入口，版本号 0.1.0
│   ├── detect.py              # 广告检测（ASR + 场景切换 + 边界扩展）
│   ├── cut.py                 # 剪切与拼接（ffmpeg + concat demuxer）
│   ├── cli.py                 # CLI 入口（detect/cut/all 子命令）
│   ├── ai_analyzer.py         # AI 语义分析（兼容 OpenAI/Anthropic）
│   └── llm.py                 # LLM 语义确认（默认：智谱 GLM-4-Flash）
├── tests/                     # 单元测试
├── pyproject.toml             # 包配置
├── requirements.txt           # 核心依赖声明
├── requirements-detect.txt    # 可选检测依赖
├── README.md                  # English documentation
├── README_CN.md               # 本文档
├── LICENSE                    # MIT 许可证
└── .gitignore
```

---

## 贡献指南

欢迎贡献！参与方式：

### 开发环境搭建
```bash
# 1. Fork 并克隆仓库
git clone https://github.com/<your-username>/cutad.git
cd cutad

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate      # Linux / macOS
# 或
venv\Scripts\activate         # Windows

# 3. 安装开发依赖
pip install -e ".[dev,ai]"

# 4. 运行测试
pytest

# 5. 代码风格检查
ruff check .
ruff format .
```

### 提交 PR
1. 从 `master` 分支创建特性分支：`git checkout -b feature/your-feature`
2. 提交更改：`git commit -m "feat: add xxx"`
3. 推送：`git push origin feature/your-feature`
4. 创建 Pull Request

### 代码规范
- 遵循 [PEP 8](https://peps.python.org/pep-0008/) 代码风格
- 使用 [Ruff](https://github.com/astral-sh/ruff) 进行 lint 和格式化
- 新模块需添加类型注解
- 公共函数需添加 docstring

### 测试
```bash
# 运行所有测试
pytest

# 运行特定测试模块
pytest tests/test_detect.py -v

# 生成覆盖率报告
pytest --cov=cutad --cov-report=term-missing
```

---

## 许可证

MIT License — see [LICENSE](LICENSE) for details.
