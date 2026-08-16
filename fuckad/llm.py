# -*- coding: utf-8 -*-
"""
FuckAd - LLM 语义判断层

作用：对关键词规则检出的候选广告段做二次确认，区分"真广告"与"正片对话"，
显著提升精度（例如避免"下雨了 / 又好玩又刺激"等短语在台词中的误命中）。

默认对接智谱 GLM（https://open.bigmodel.cn），glm-4-flash 模型免费使用。
也可通过环境变量或参数切换为任意 OpenAI 兼容接口。

环境变量（可选）：
    FUCKAD_LLM_URL    API 地址（默认 https://open.bigmodel.cn/api/paas/v4）
    FUCKAD_LLM_MODEL  模型名（默认 glm-4-flash）
    FUCKAD_LLM_KEY    API Key（必填，必须设置才能使用 --llm）
"""
import json
import os
import re
import urllib.request

DEFAULT_BASE_URL = os.environ.get(
    "FUCKAD_LLM_URL", "https://open.bigmodel.cn/api/paas/v4")
DEFAULT_MODEL = os.environ.get("FUCKAD_LLM_MODEL", "glm-4-flash")
DEFAULT_API_KEY = os.environ.get("FUCKAD_LLM_KEY", "")

_SYSTEM_PROMPT = (
    "你是视频广告审核员。用户会给你视频中若干片段的转写文本（每段含起止秒数）。"
    "请逐段判断是否为广告插播。广告特征：营销话术（扫码、下载、拨打热线、领红包、"
    "中奖、赌博/娱乐城/体育博彩等），与前后剧情无关，通常有固定的促销句式。"
    "只输出一个 JSON 数组，不要输出任何其他文字。每一项格式："
    '{"start": 秒, "end": 秒, "is_ad": true或false, "reason": "一句话理由"}'
)


def _call_chat(messages: list, model: str = DEFAULT_MODEL,
               base_url: str = DEFAULT_BASE_URL, api_key: str = DEFAULT_API_KEY,
               timeout: int = 180) -> str:
    """调用 OpenAI 兼容 chat/completions 接口"""
    if not api_key:
        raise RuntimeError(
            "未配置 LLM API Key。请设置环境变量 FUCKAD_LLM_KEY，"
            "或在命令行加 --llm-key <你的Key>。"
        )
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 2048,
    }
    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def _extract_json_array(text: str) -> list:
    """从 LLM 输出中提取 JSON 数组（容忍代码围栏等干扰）"""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError(f"无法解析 LLM 输出: {text[:200]}")


def _find_verdict(verdicts: list, start: float):
    """按开始时间（容差 2s）在 LLM 返回中查找对应判定"""
    for v in verdicts:
        if isinstance(v, dict) and abs(float(v.get("start", -1)) - start) < 2.0:
            return v
    return None


def create_ai_analyzer(model: str = DEFAULT_MODEL,
                       base_url: str = DEFAULT_BASE_URL,
                       api_key: str = DEFAULT_API_KEY,
                       timeout: int = 180):
    """
    创建 LLM 语义判断器（候选广告二次确认）。

    策略：先用关键词规则找出候选段，再把候选段文本交给 LLM 确认，
    只保留被判定为广告的候选；LLM 未覆盖的候选保守保留（避免漏广告）。
    LLM 不可用时抛异常，由上层（detect_ads_by_ai）回退到纯规则检测。

    返回: analyzer(segments) -> candidates 列表
          （candidates 结构与规则检测一致：start/end/text/keywords）
    """
    # 延迟导入，避免与 detect 模块循环依赖
    from fuckad.detect import _detect_ads_by_rules

    def analyzer(segments: list) -> list:
        candidates = _detect_ads_by_rules(segments)
        if not candidates:
            return candidates

        items = [
            {"start": round(c["start"], 2), "end": round(c["end"], 2), "text": c["text"]}
            for c in candidates
        ]
        user_prompt = (
            "以下是候选广告片段（含起止秒数和转写文本）：\n\n"
            + json.dumps(items, ensure_ascii=False, indent=1)
            + "\n\n请逐项判断是否为广告，输出 JSON 数组。"
        )
        resp = _call_chat(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=model, base_url=base_url, api_key=api_key, timeout=timeout,
        )
        verdicts = _extract_json_array(resp)

        kept = []
        for c in candidates:
            v = _find_verdict(verdicts, c["start"])
            if v is None or v.get("is_ad") is True:
                kept.append(c)
        print(f"[llm] 候选 {len(candidates)} 段 -> LLM 确认保留 {len(kept)} 段", flush=True)
        return kept

    return analyzer
