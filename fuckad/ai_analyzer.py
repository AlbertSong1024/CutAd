"""FuckAd 广告检测器 - AI 语义分析"""
import json
from typing import Optional


def analyze_with_llm(segments: list,
                     llm_client: Optional[object] = None,
                     prompt_template: Optional[str] = None) -> list:
    """
    使用大语言模型分析转写文本，识别广告片段。
    
    参数:
        segments: faster-whisper 输出的 segments 列表
        llm_client: 大模型客户端实例（需支持 .chat.completions.create 或 .message.create）
        prompt_template: 自定义 prompt 模板
    
    返回:
        candidates 列表，每项包含 start, end, text, keywords
    """
    if llm_client is None:
        raise ValueError("请提供 llm_client，例如 openai.OpenAI() 或 anthropic.Anthropic()")
    
    # 构建提示
    if prompt_template is None:
        prompt_template = _DEFAULT_PROMPT
    
    # 提取转写文本摘要
    text_summary = _build_text_summary(segments)
    
    prompt = prompt_template.format(transcript=text_summary)
    
    # 调用大模型
    response = _call_llm(llm_client, prompt)
    
    # 解析结果
    candidates = _parse_llm_response(response, segments)
    return candidates


def _build_text_summary(segments: list, max_chars: int = 8000) -> str:
    """构建转写文本摘要，保留时间戳"""
    parts = []
    total = 0
    for seg in segments:
        text = seg.get("text", "").strip()
        if text:
            line = f"[{seg['start']:.1f}s] {text}"
            if total + len(line) > max_chars:
                break
            parts.append(line)
            total += len(line)
    return "\n".join(parts)


def _call_llm(client, prompt: str) -> str:
    """调用大模型，兼容 OpenAI 和 Anthropic API"""
    # 尝试 OpenAI
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    # 尝试 Anthropic
    if hasattr(client, "messages"):
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    raise ValueError("不支持的 llm_client 类型")


def _parse_llm_response(response: str, segments: list) -> list:
    """解析大模型返回的广告时间段"""
    import re
    # 尝试解析 JSON
    try:
        data = json.loads(response)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "ads" in data:
            return data["ads"]
    except json.JSONDecodeError:
        pass
    
    # 尝试解析文本格式: "start-end" 或 JSON 片段
    pattern = r"(\d+\.?\d*)\s*[,~]\s*(\d+\.?\d*)"
    matches = re.findall(pattern, response)
    candidates = []
    for i, (s, e) in enumerate(matches):
        start, end = float(s), float(e)
        if end > start and (end - start) >= 1.0:
            candidates.append({
                "start": start,
                "end": end,
                "text": "",
                "keywords": [],
            })
    return candidates


_DEFAULT_PROMPT = """
分析以下视频转写文本，识别其中的广告片段。

广告特征（供参考）：
- 促销话术：限时、优惠、购买、链接、扫码、购物车
- 品牌推销：突兀的品牌名/产品名
- 叙事断裂：上下文主题突跳
- 固定表达：赞助声明、广告时间

视频转写（带时间戳）：
{transcript}

请以 JSON 格式返回广告片段列表：
[
  {{"start": 起始秒数, "end": 结束秒数, "text": "相关文本", "keywords": ["关键词1", "关键词2"]}}
]

如果没有广告，返回空列表 []。只返回 JSON，不要其他文字。
"""
