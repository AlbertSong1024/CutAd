# cutad

视频广告自动检测与剪切工具。

**核心能力**：用本地 Whisper ASR 转写视频 → AI 语义判断广告片段 → ffmpeg 流复制剪切（边界非关键帧处局部重编码）→ concat 拼接 → 输出无广告视频。

## 特性

- 🎙️ **VAD 预筛 + Whisper 转写**：自动跳过静音/音乐段，2 小时视频约 2-3 分钟完成
- 🤖 **AI 语义分析**：可接入 OpenAI / Anthropic 等大模型，通用广告识别（不限于特定类型）
- 🔧 **场景吸附边界**：自动扩展广告词后的画面留白（1-4s）
- ⚡ **帧精确剪切**：关键帧区间流复制 + 边界局部重编码，兼顾精度与画质无损
- 🔗 **concat 流复制拼接**：严格保持帧数，无重复/丢失，无画质损失
- 🖼️ **缩略图拼图**：自动生成边界处帧拼图，方便人工确认
- 📦 **即装即用**：CLI + Python API 双入口，剪切功能仅需 ffmpeg + PyAV（轻量安装）

## 安装

### 前置要求

- Python 3.9+
- ffmpeg / ffprobe（确保在 PATH 中）

```bash
# 验证 ffmpeg
ffmpeg -version
```

### 从源码安装（推荐开发使用）

```bash
git clone <repo-url>
cd cutad
pip install -e .
```

### 从 PyPI 安装

```bash
pip install cutad
```

### 轻量安装（仅剪切）

只需剪切去广告、不需要检测功能时，安装核心依赖即可：

```bash
pip install cutad            # 核心：PyAV（流复制/拼接）
```

### 完整安装（检测 + 剪切）

需要 Whisper 广告检测时，额外安装检测依赖：

```bash
pip install "cutad[detect]"  # faster-whisper + opencv + numpy
```

### 依赖说明

| 依赖 | 用途 | 必需 |
|------|------|------|
| `av` (PyAV) | 视频参数读取 / 流复制剪切 | ✅ 核心 |
| `faster-whisper` | ASR 转写（CTranslate2 引擎，不需要 torch） | 检测时 |
| `opencv-python-headless` | 场景切换检测 / 缩略图 | 检测时 |
| `numpy` | 帧差计算 | 检测时 |

> **注意**：检测功能基于 `faster-whisper`（CTranslate2），**不依赖 PyTorch**，
> 无需下载数百 MB 的 torch。Whisper 模型首次运行时自动下载，国内用户可设置镜像加速：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com  # Linux/Mac
> set HF_ENDPOINT=https://hf-mirror.com     # Windows
> ```

### GPU 加速（自动适配）

运行时会**自动探测 NVIDIA 显卡**：检测到 CUDA 设备则用 GPU 转写（compute=float16，转写提速约 5~20x），否则自动回退 CPU，无需手动配置。

- 转写时输出 `[asr] 检测到 NVIDIA GPU，启用 CUDA 加速 (compute=float16)` 表示已走 GPU
- 若要启用 GPU 加速，需安装 **CUDA 版 ctranslate2**（pip 默认装的是 CPU 版）：
  ```bash
  pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
  pip install ctranslate2 --no-deps  # 或按 faster-whisper 官方指引安装 CUDA 版
  ```
  安装后 `nvidia-smi` 可用的 NVIDIA 独显即可自动启用。

### 可选依赖

```bash
# AI 语义分析增强（OpenAI / Anthropic 客户端，需自备 API Key）
pip install "cutad[ai]"

# 开发依赖
pip install "cutad[dev]"
```

## 快速开始

### 命令行

```bash
# 1. 检测广告（生成 ads.json、缩略图拼图等）
cutad detect video.mp4

# 2. 剪切并拼接（从 ads.json 读取广告时间段）
cutad cut video.mp4

# 或一键完成检测+剪切
cutad all video.mp4

# 手动指定广告时间段（单位：秒）
cutad cut video.mp4 --ads "1697,1715;3692,3712;5452,5470"

# 指定输出路径
cutad cut video.mp4 --output clean_video.mp4

# 指定 Whisper 模型（tiny/base/small/medium/large）
cutad detect video.mp4 --model base
```

### Python API

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

### 使用 AI 语义分析

```python
from cutad import detect_ads, analyze_with_llm
import openai

# 接入 OpenAI
client = openai.OpenAI(api_key="sk-...")

result = detect_ads(
    "video.mp4",
    ai_analyzer=lambda segments: analyze_with_llm(segments, llm_client=client),
)

# 接入 Anthropic
import anthropic
client = anthropic.Anthropic(api_key="sk-ant-...")
result = detect_ads(
    "video.mp4",
    ai_analyzer=lambda segments: analyze_with_llm(segments, llm_client=client),
)
```

### 自定义广告规则

```python
from cutad.detect import detect_ads

# 自定义关键词规则
import cutad.detect as detect
detect._PROMPT_KEYWORDS.extend([
    r"我的品牌", r"强烈推荐", r"限时特价",
])

result = detect_ads("video.mp4")
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `ads.json` | 结构化广告段（start/end/reason/confidence） |
| `ad_timecodes.txt` | 人类可读时间码表 |
| `montage_ad*.jpg` | 广告边界缩略图拼图（人工确认用） |
| `scene_cuts.json` | 场景切换点缓存 |
| `<原文件名>_no_ads.mp4` | 去广告后的输出视频 |

## 工作流程

```
源视频 (mp4)
    │
    ├──[ASR] VAD 预筛 + Whisper medium 转写
    │       跳过静音段，输出带时间戳的语音段
    │
    ├──[AI 分析] 大模型通读转写文本识别广告
    │       回退：基于关键词规则检测
    │
    ├──[场景检测] OpenCV 帧差检测场景切换 + 黑帧
    │       缓存 scene_cuts.json 避免重复计算
    │
    ├──[边界扩展] 以语音内核为基础，向前后扫场景切换
    │       自动覆盖广告词结束后的画面留白
    │
    ├──[输出] ads.json / ad_timecodes.txt / montage_*.jpg
    │
    ├──[剪切] ffmpeg 流复制切为 MP4 片段
    │       边界非关键帧处局部重编码 + 中间关键帧区间流复制
    │
    ├──[拼接] ffmpeg concat demuxer 流复制拼接
    │       严格保持帧数，无重复/丢失
    │
    └──[优化] ffmpeg faststart（moov atom 前置）
```

## 项目结构

```
cutad/
├── cutad/                    # Python 包
│   ├── __init__.py            # 包入口，版本 0.1.0
│   ├── detect.py              # 广告检测（ASR + 场景 + 边界扩展）
│   ├── cut.py                 # 剪切拼接（ffmpeg + concat demuxer）
│   ├── cli.py                 # CLI 入口（detect/cut/all 子命令）
│   ├── ai_analyzer.py         # AI 语义分析（OpenAI/Anthropic 兼容）
│   └── llm.py                 # LLM 语义确认（默认智谱 GLM-4-Flash）
├── tests/                     # 测试
├── pyproject.toml             # 打包配置
├── requirements.txt           # 核心依赖声明
├── requirements-detect.txt    # 检测可选依赖
├── README.md                  # 本文档
├── LICENSE                    # MIT 许可
└── .gitignore
```

## CLI 参考

### `detect` 命令

检测视频中的广告片段，输出 ads.json 和缩略图拼图。

```bash
cutad detect VIDEO [OPTIONS]

位置参数:
  VIDEO              视频文件路径

可选参数:
  -o, --output-dir   输出目录（默认当前目录）
  --model            Whisper 模型: tiny/base/small/medium/large（默认 medium）
  -h, --help         显示帮助
```

### `cut` 命令

剪切广告并拼接视频。

```bash
cutad cut VIDEO [OPTIONS]

位置参数:
  VIDEO              源视频路径

可选参数:
  --ads              广告时间段，格式: "start1,end1;start2,end2"
  --ads-json         从 ads.json 读取（默认 ads.json）
  -o, --output       输出文件路径
  --tmp-dir          临时文件目录
  -h, --help         显示帮助
```

### `all` 命令

一键完成检测+剪切。

```bash
cutad all VIDEO [OPTIONS]

位置参数:
  VIDEO              视频文件路径

可选参数:
  -o, --output-dir   输出目录
  --model            Whisper 模型
  --output           输出文件路径
  --skip-detect      跳过检测，直接剪切
  -h, --help         显示帮助
```

## 故障排查

| 问题 | 解决方案 |
|------|----------|
| `ffprobe` 找不到 | 安装 ffmpeg，确保 `ffmpeg` 在 PATH 中 |
| Whisper 模型下载失败 | 设置 `HF_ENDPOINT=https://hf-mirror.com` 使用国内镜像 |
| 拼接时报错 | 确保 `av>=10.0`；确认所有分段编码参数一致（来自同一源视频） |
| 输出视频无画面 | 检查 pix_fmt 是否为 yuv420p，extradata 是否完整复制 |
| 检测提示缺少依赖 | 运行 `pip install "cutad[detect]"` 安装检测可选依赖 |
| 内存不足 | 使用 `--model tiny` 或 `--model base` 降低内存占用 |
| ASR 识别语言不对 | 在 prompt 中指定目标语言，或切换为对应的 Whisper 多语言模型 |

## 贡献指南

欢迎贡献！以下是参与方式：

### 开发环境搭建

```bash
# 1. Fork 并克隆仓库
git clone https://github.com/<your-username>/cutad.git
cd cutad

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 3. 安装开发依赖
pip install -e ".[dev,ai]"

# 4. 运行测试
pytest

# 5. 代码风格检查
ruff check .
ruff format .
```

### 提交 PR

1. 从 `main` 分支创建特性分支：`git checkout -b feature/your-feature`
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
# 运行全部测试
pytest

# 运行单模块测试
pytest tests/test_detect.py -v

# 带覆盖率
pytest --cov=cutad --cov-report=term-missing
```

### 文档

- 新增功能请在 `README.md` 中更新使用说明
- API 变更请同步更新 docstring
- 大改动建议新增 `docs/` 下的专题文档

## License

MIT
