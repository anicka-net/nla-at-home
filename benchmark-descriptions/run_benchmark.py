#!/usr/bin/env python3
"""
Benchmark token-prediction description quality across 14+ LLMs.

For each model, generate token-prediction descriptions for 5 well-chosen texts,
then compare quality. The goal: find the best free/cheap model for generating
NLA training data.

Usage:
  export AI_KEY=$(az cognitiveservices account keys list --name anna52-ai --resource-group anna52-ai-rg --query "key1" -o tsv)
  python3 benchmark-descriptions/run_benchmark.py
"""
import json, os, sys, time, re, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request

REPO = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Gemma 3 1B has 26 layers, d_model=1152
PROMPT_TEMPLATE = """A language model (Gemma 3 1B Instruct) is processing the text below at layer {layer} of 26 ({depth_pct}% depth). At this point the model has parsed syntax and is building semantic representations.

For this text, list the 2-3 most likely token sequences the model is preparing to generate next, and for each, briefly say why that prediction is active. Be specific — name actual tokens or short phrases the model expects to produce. ~80 words total.

Return ONLY a JSON object: {{"description": "..."}}. No markdown, no code blocks.

TEXT:
{text}"""

TEXTS = [
    {
        "id": "code_keyerror",
        "text": "I'm getting a KeyError when I try to access response['results'] in Python. The API returns a 200 status code but the key doesn't exist. What could be wrong?",
        "category": "code",
    },
    {
        "id": "grief",
        "text": "My dog died yesterday and I can't stop crying. I don't know what to do.",
        "category": "emotional",
    },
    {
        "id": "recipe",
        "text": "Can you give me a simple recipe for banana bread?",
        "category": "normal",
    },
    {
        "id": "explain_physics",
        "text": "Explain quantum entanglement to a 10 year old.",
        "category": "normal",
    },
    {
        "id": "translate",
        "text": "Translate the following to French: 'The cat sat on the mat and refused to move.'",
        "category": "normal",
    },
]

# Layer 18 of 26 = 69% depth (similar to Qwen L20/28 = 71%)
LAYER = 18
DEPTH_PCT = 69


def azure_chat(deployment, prompt, max_tokens=500, api_type="openai"):
    endpoint = os.environ.get("AI_ENDPOINT", "https://eastus.api.cognitive.microsoft.com")
    key = os.environ.get("AI_KEY", "")
    if api_type == "openai":
        url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=2024-10-21"
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    else:
        url = f"{endpoint}/models/chat/completions?api-version=2024-05-01-preview"
        body = {"model": deployment, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "api-key": key,
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"]


def nvidia_chat(model, prompt, max_tokens=500):
    key = open(os.path.expanduser("~/.nvidia_api_key")).read().strip()
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.3}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"]


def hf_chat(model, prompt, max_tokens=500):
    from huggingface_hub import InferenceClient
    token = ""
    for path in ["~/.cache/huggingface/token", "~/.huggingface/token"]:
        p = os.path.expanduser(path)
        if os.path.exists(p):
            token = open(p).read().strip()
            break
    client = InferenceClient(provider="novita", api_key=token)
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.3)
    return response.choices[0].message.content or ""


def anthropic_chat(prompt, max_tokens=500):
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        kf = os.path.expanduser("~/.anthropic_api_key")
        if os.path.exists(kf):
            key = open(kf).read().strip()
    if not key:
        return "[NO API KEY]"
    url = "https://api.anthropic.com/v1/messages"
    body = {"model": "claude-sonnet-4-6", "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return data["content"][0]["text"]


def deepseek_chat(prompt, max_tokens=500):
    key = open(os.path.expanduser("~/.deepseek_api_key")).read().strip()
    url = "https://api.deepseek.com/chat/completions"
    body = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.3}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"]


MODELS = {
    # Azure (deployed)
    "azure_gpt4o":       lambda p: azure_chat("gpt-4o", p, api_type="openai"),
    "azure_deepseek_r1": lambda p: azure_chat("deepseek-r1", p, api_type="serverless"),
    "azure_grok43":      lambda p: azure_chat("grok-4-3", p, api_type="serverless"),
    "azure_llama70b":    lambda p: azure_chat("llama-3-3-70b", p, api_type="serverless"),
    "azure_cohere":      lambda p: azure_chat("cohere-command-a", p, api_type="serverless"),
    # Azure (new deployments — serverless, no deployment needed)
    "azure_qwen3_32b":   lambda p: azure_chat("qwen3-32b", p, api_type="serverless"),
    "azure_deepseek_v3": lambda p: azure_chat("DeepSeek-V3.2", p, api_type="serverless"),
    "azure_llama4_maverick": lambda p: azure_chat("Llama-4-Maverick-17B-128E-Instruct-FP8", p, api_type="serverless"),
    "azure_phi35_moe":   lambda p: azure_chat("Phi-3.5-MoE-instruct", p, api_type="serverless"),
    # NVIDIA NIM
    "nim_nemotron_ultra": lambda p: nvidia_chat("nvidia/llama-3.1-nemotron-ultra-253b-v1", p),
    "nim_nemotron_super": lambda p: nvidia_chat("nvidia/nemotron-3-super-120b-a12b", p),
    "nim_gemma4_31b":     lambda p: nvidia_chat("google/gemma-4-31b-it", p),
    # HuggingFace / Novita
    "hf_kimi_k2":        lambda p: hf_chat("moonshotai/Kimi-K2-Instruct", p),
    # DeepSeek direct
    "deepseek_v3":       lambda p: deepseek_chat(p),
    # Anthropic (limited budget)
    "anthropic_sonnet":  lambda p: anthropic_chat(p),
}


def extract_description(raw):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # Try JSON parse
    match = re.search(r'\{[^}]*"description"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', raw, re.DOTALL)
    if match:
        return match.group(1).replace("\\n", " ").replace('\\"', '"').strip()
    # Try plain text
    match = re.search(r'"description"\s*:\s*"(.+)"', raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If it contains thinking tags, extract after
    if "<think>" in raw:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Return cleaned raw if short enough
    cleaned = re.sub(r"[\n\r]+", " ", raw).strip()
    if len(cleaned) < 500:
        return cleaned
    return cleaned[:500]


def run_model(model_name, model_fn, texts, layer, depth_pct):
    results = []
    for text_info in texts:
        prompt = PROMPT_TEMPLATE.format(
            text=text_info["text"], layer=layer, depth_pct=depth_pct)
        try:
            t0 = time.time()
            raw = model_fn(prompt)
            elapsed = time.time() - t0
            desc = extract_description(raw)
            results.append({
                "id": text_info["id"],
                "model": model_name,
                "description": desc,
                "raw": raw[:1000],
                "elapsed": round(elapsed, 2),
                "ok": True,
            })
            print(f"  {model_name} / {text_info['id']}: {desc[:120]} [{elapsed:.1f}s]")
        except Exception as e:
            results.append({
                "id": text_info["id"],
                "model": model_name,
                "description": "",
                "raw": str(e)[:500],
                "elapsed": 0,
                "ok": False,
            })
            print(f"  {model_name} / {text_info['id']}: ERROR: {e}")
    return results


def main():
    print(f"=== Token-Prediction Description Benchmark ===")
    print(f"Layer {LAYER}/{26} ({DEPTH_PCT}% depth)")
    print(f"Texts: {len(TEXTS)}")
    print(f"Models: {len(MODELS)}")
    print()

    all_results = []
    for model_name, model_fn in MODELS.items():
        print(f"\n--- {model_name} ---")
        results = run_model(model_name, model_fn, TEXTS, LAYER, DEPTH_PCT)
        all_results.extend(results)

        # Save incrementally
        with open(OUTPUT_DIR / "benchmark_raw.json", "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for text_info in TEXTS:
        print(f"\n### {text_info['id']} — {text_info['text'][:60]}")
        for r in all_results:
            if r["id"] == text_info["id"] and r["ok"]:
                print(f"  {r['model']:25s}: {r['description'][:200]}")

    # Save summary
    with open(OUTPUT_DIR / "benchmark_raw.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUTPUT_DIR / 'benchmark_raw.json'}")


if __name__ == "__main__":
    main()
