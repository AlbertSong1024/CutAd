# CutAd — Video Ad Detection & Removal Tool

> 🎬 One video, a few commands, ads gone — CutAd brings your viewing experience back to clean and smooth.

**CutAd** is an AI-powered video ad detection and removal tool. It transcribes video audio to text via Whisper ASR, identifies hard-sell and soft-sell ads using Large Language Model (LLM) semantic analysis, and stitches the remaining segments back together with stream-copy precision — no quality loss.

**What it does:**
- Detects hard-sell ads ("scan to download", "limited-time offer") and soft-sell / sponsored content ("brought to you by...", "this episode is sponsored by...")
- Cuts ad segments with frame-level precision and seamlessly stitches at keyframes
- Generates thumbnail collages so you can visually verify boundary accuracy
- Supports OpenAI, Anthropic, and Zhipu GLM as LLM backends
- Automatic GPU detection; Whisper transcription speeds up 5–20× with CUDA acceleration

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Output Files](#output-files)
5. [How It Works](#how-it-works)
6. [CLI Reference](#cli-reference)
7. [Troubleshooting](#troubleshooting)
8. [Project Structure](#project-structure)
9. [Contributing](#contributing)
10. [License](#license)

> 🌐 中文版文档请查看 [README_CN.md](README_CN.md)

---

## Quick Start

The fastest path to an ad-free video — three steps:

```bash
# 1. Install (with detection + LLM semantic analysis)
pip install "cutad[detect,ai]"

# 2. One-command detect & cut
cutad all your_video.mp4

# 3. Done — the ad-free version is generated in the same directory
ls your_video_no_ads.mp4
```

> 💡 **Tip**: The Whisper model (~150 MB for base) is downloaded on first run. Chinese users should set a mirror:
> ```bash
> set HF_ENDPOINT=https://hf-mirror.com   # Windows
> export HF_ENDPOINT=https://hf-mirror.com # Linux / macOS
> ```

---

## Installation

### Prerequisites
- Python 3.9+
- ffmpeg / ffprobe (available in PATH)

```bash
ffmpeg -version   # verify installation
```

### Full Install (Detection + Cutting + LLM)
```bash
pip install "cutad[detect,ai]"
```

### Cut Only (No Detection Needed)
If you only need to remove ads at known timecodes:
```bash
pip install cutad    # core dependencies only (PyAV)
```

### GPU Acceleration (Optional)
CutAd **automatically detects NVIDIA GPUs** at runtime: if a CUDA device is found, transcription runs on GPU (compute=float16); otherwise it falls back to CPU transparently.

```bash
# Install CUDA-enabled ctranslate2 (pip installs CPU-only by default)
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
pip install ctranslate2 --no-deps
```

Any NVIDIA discrete GPU visible in `nvidia-smi` will be used automatically.

### Dependency Overview

| Dependency | Purpose | Required |
|------------|---------|----------|
| `av` (PyAV) | Video parameter reading / stream-copy cutting | ✅ Core |
| `faster-whisper` | ASR transcription (CTranslate2 engine, no PyTorch needed) | Detection |
| `opencv-python-headless` | Scene change detection / thumbnail generation | Detection |
| `numpy` | Frame-difference calculation | Detection |
| `openai` / `anthropic` | LLM semantic analysis clients | LLM analysis |

> ⚠️ The detection module uses `faster-whisper` (CTranslate2) — **no PyTorch dependency**, saving you from downloading hundreds of MB of torch.

---

## Usage

### Command Line

```bash
# One-command detect & cut
cutad all video.mp4

# Detect only (no cutting; generates ads.json + thumbnails)
cutad detect video.mp4

# Cut only (reads ad timecodes from existing ads.json)
cutad cut video.mp4

# Manually specify ad timecodes (format: start,end;start,end, in seconds)
cutad cut video.mp4 --ads "1697,1715;3692,3712"

# Specify output path
cutad cut video.mp4 --output clean_video.mp4

# Specify Whisper model (tiny = fastest / base = recommended / medium = most accurate)
cutad detect video.mp4 --model base

# Enable LLM semantic confirmation (default: Zhipu GLM-4-Flash)
cutad detect video.mp4 --llm --llm-key YOUR_API_KEY

# Enable LLM full-text deep scan (detects soft/sponsored ads without keyword rules)
cutad detect video.mp4 --llm --llm-deep --llm-key YOUR_API_KEY
```

### Python API

```python
from cutad import detect_ads, cut_and_join

# Detect ads
result = detect_ads("video.mp4")
print(f"Found {len(result.ads)} ad segments")
for ad in result.ads:
    print(f"  {ad.id}: {ad.start:.1f}s ~ {ad.end:.1f}s  ({ad.reason})")

# Cut and stitch
output = cut_and_join("video.mp4", [(ad.start, ad.end) for ad in result.ads])
print(f"Output: {output}")
```

### AI Semantic Analysis

```python
from cutad import detect_ads, analyze_with_llm
import openai

# Connect to OpenAI
client = openai.OpenAI(api_key="sk-...")
result = detect_ads(
    "video.mp4",
    ai_analyzer=lambda segs: analyze_with_llm(segs, llm_client=client),
)

# Connect to Anthropic
import anthropic
client = anthropic.Anthropic(api_key="sk-ant-...")
result = detect_ads(
    "video.mp4",
    ai_analyzer=lambda segs: analyze_with_llm(segs, llm_client=client),
)
```

---

## Output Files

| File | Description |
|------|-------------|
| `ads.json` | Structured ad segments (start/end times, reasons, confidence) |
| `ad_timecodes.txt` | Human-readable timecode table |
| `montage_ad*.jpg` | Boundary thumbnail collages (for manual verification) |
| `scene_cuts.json` | Scene-cut cache (avoids recomputation) |
| `<original_filename>_no_ads.mp4` | Ad-free output video |

---

## How It Works

```
Source Video (mp4)
    │
    ├──[ASR Transcription] VAD pre-filter + Whisper speech recognition
    │       Skips silence/music segments, outputs timestamped speech segments
    │
    ├──[Ad Detection] Keyword rules + LLM semantic analysis
    │       First scans for keyword-hit candidates, then LLM does semantic confirmation
    │       With --llm-deep: full-text chunked scan to detect soft/sponsored ads
    │
    ├──[Scene Detection] OpenCV frame-difference to detect scene cuts + black frames
    │       Caches scene_cuts.json to avoid recomputation
    │
    ├──[Boundary Expansion] Uses speech core as anchor, scans to scene boundaries
    │       Automatically covers post-ad silence (1–4 seconds)
    │
    ├──[Analysis Output] ads.json / ad_timecodes.txt / montage_*.jpg
    │
    ├──[Cutting] ffmpeg stream-copy segmentation
    │       Local re-encoding at non-keyframe boundaries, pure stream-copy in between
    │
    ├──[Stitching] ffmpeg concat demuxer stream-copy stitching
    │       Strict frame count preservation, no duplicates or gaps, zero quality loss
    │
    └──[Optimization] ffmpeg faststart (moov atom moved to front for streaming)
```

**Model Speed Reference (2-hour video, CPU):**

| Model | Time | Accuracy |
|-------|------|----------|
| tiny | ~2 min | ~90% |
| base | ~8 min | ~93% |
| small | ~20 min | ~95% |
| medium | ~40 min | ~97% |

---

## CLI Reference

### `detect` — Detect Ads
```bash
cutad detect VIDEO [OPTIONS]

Positional arguments:
  VIDEO              Path to video file

Optional arguments:
  -o, --output-dir   Output directory (default: current dir)
  --model            Whisper model: tiny/base/small/medium/large (default: tiny, recommended: base)
  --no-cache         Disable cache, force re-transcription
  --no-montage       Skip thumbnail collage generation
  --llm              Enable LLM semantic confirmation
  --llm-deep         LLM full-text deep scan (detects soft ads, requires --llm)
  --llm-model        Custom LLM model name
  --llm-url          Custom API base URL
  --llm-key          API Key (or use env var CUTAD_LLM_KEY)
  -h, --help         Show this help message
```

### `cut` — Cut Ads
```bash
cutad cut VIDEO [OPTIONS]

Positional arguments:
  VIDEO              Source video path

Optional arguments:
  --ads              Ad timecodes, format: "start1,end1;start2,end2"
  --ads-json         Read from ads.json (default: ads.json)
  -o, --output       Output file path (default: <original>_no_ads.mp4)
  --tmp-dir          Temporary file directory
  --skip-detect      Skip detection, cut directly (requires existing ads.json)
  -h, --help         Show this help message
```

### `all` — Detect & Cut (One-Command)
```bash
cutad all VIDEO [OPTIONS]

Positional arguments:
  VIDEO              Source video path

Optional arguments:
  -o, --output-dir   Output directory
  --model            Whisper model
  --output           Output file path
  --skip-detect      Skip detection, cut directly (requires existing ads.json)
  --no-cache         Disable cache
  --no-montage       Skip thumbnail generation
  --llm              Enable LLM semantic confirmation
  --llm-deep         LLM full-text deep scan
  --llm-model        Custom model name
  --llm-url          Custom API URL
  --llm-key          API Key
  -h, --help         Show this help message
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ffprobe` not found | Install ffmpeg and ensure it's in your system PATH |
| Whisper model download fails or times out | Set `HF_ENDPOINT=https://hf-mirror.com` to use a Chinese mirror |
| Stitching error / no video in output | Ensure `av>=10.0`; verify segment encoding params are consistent (same source video) |
| Missing dependencies for detection | Run `pip install "cutad[detect]"` to install optional detection dependencies |
| Out of memory | Use `--model tiny` or `--model base` to reduce memory usage |
| ASR recognizes wrong language | Specify the target language in the prompt, or switch to a multilingual Whisper model |
| GPU not detected | Check if `nvidia-smi` works; confirm CUDA-enabled `ctranslate2` is installed |

---

## Project Structure

```
cutad/
├── cutad/                    # Python package
│   ├── __init__.py            # Package entry, version 0.1.0
│   ├── detect.py              # Ad detection (ASR + scene cuts + boundary expansion)
│   ├── cut.py                 # Cutting & stitching (ffmpeg + concat demuxer)
│   ├── cli.py                 # CLI entrypoint (detect/cut/all subcommands)
│   ├── ai_analyzer.py         # AI semantic analysis (OpenAI/Anthropic compatible)
│   └── llm.py                 # LLM semantic confirmation (default: Zhipu GLM-4-Flash)
├── tests/                     # Unit tests
├── pyproject.toml             # Package configuration
├── requirements.txt           # Core dependency declarations
├── requirements-detect.txt    # Optional detection dependencies
├── README.md                  # English documentation (this file)
├── README_CN.md               # 中文文档
├── LICENSE                    # MIT License
└── .gitignore
```

---

## Contributing

Contributions are welcome! Here's how to get started:

### Development Setup
```bash
# 1. Fork and clone the repository
git clone https://github.com/<your-username>/cutad.git
cd cutad

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux / macOS
# or
venv\Scripts\activate         # Windows

# 3. Install development dependencies
pip install -e ".[dev,ai]"

# 4. Run tests
pytest

# 5. Code style check
ruff check .
ruff format .
```

### Submitting a PR
1. Create a feature branch from `master`: `git checkout -b feature/your-feature`
2. Commit changes: `git commit -m "feat: add xxx"`
3. Push: `git push origin feature/your-feature`
4. Open a Pull Request

### Code Standards
- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines
- Use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Add type annotations to new modules
- Add docstrings to public functions

### Testing
```bash
# Run all tests
pytest

# Run a specific test module
pytest tests/test_detect.py -v

# With coverage report
pytest --cov=cutad --cov-report=term-missing
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
