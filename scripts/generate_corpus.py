#!/usr/bin/env python3
"""
Generate the NLA-at-home training corpus.

Reads category YAML files from corpus/categories/,
calls DeepSeek API to generate texts for each category,
saves outputs to corpus/generated/{category_id}.json.

Prompt structure (optimized for prefix caching):
  1. System prompt (cached across ALL calls)
  2. Category preamble (cached within each category)
  3. Batch instruction (varies per batch)
"""
import json, yaml, time, argparse, sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CATEGORIES_DIR = REPO_ROOT / "corpus" / "categories"
GENERATED_DIR = REPO_ROOT / "corpus" / "generated"
SYSTEM_PROMPT = (REPO_ROOT / "prompts" / "system.txt").read_text().strip()


BACKENDS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "local": {
        "base_url": "http://localhost:8080/v1",
        "model": "local",
        "env_key": None,
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "nvidia/llama-3.1-nemotron-70b-instruct",
        "env_key": "NVIDIA_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
    "huggingface": {
        "base_url": "https://router.huggingface.co/v1",
        "model": "NousResearch/Hermes-2-Pro-Llama-3-8B",
        "env_key": "HF_TOKEN",
    },
}


def get_client(api_key=None, base_url=None, backend="deepseek"):
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "The OpenAI Python SDK is required for API generation. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from e

    cfg = BACKENDS.get(backend, BACKENDS["deepseek"])
    if base_url is None:
        base_url = cfg["base_url"]

    if api_key is None:
        import os
        env_key = cfg.get("env_key")
        if env_key:
            api_key = os.environ.get(env_key)
    if api_key is None and backend == "deepseek":
        import subprocess
        try:
            api_key = subprocess.check_output(
                "grep DEEPSEEK_API_KEY ~/.bashrc | head -1 | cut -d= -f2",
                shell=True, text=True).strip()
        except Exception:
            api_key = None
    if api_key is None and cfg.get("env_key") is None:
        api_key = "not-needed"

    return OpenAI(api_key=api_key, base_url=base_url), cfg["model"]


def load_categories(filter_ids=None):
    categories = []
    for path in sorted(CATEGORIES_DIR.glob("*.yaml")):
        cat = yaml.safe_load(path.read_text())
        if filter_ids and cat["id"] not in filter_ids:
            continue
        categories.append(cat)
    return categories


def generate_category(client, cat, model_name="deepseek-chat", force=False):
    out_path = GENERATED_DIR / f"{cat['id']}.json"
    if out_path.exists() and not force:
        existing = json.loads(out_path.read_text())
        print(f"  {cat['id']}: already have {len(existing)} texts, skipping")
        return existing

    all_texts = []
    for batch_idx, batch in enumerate(cat["batches"]):
        user_msg = f"{cat['preamble'].strip()}\n\n{batch['instruction']}"

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.9,
                    max_tokens=3000,
                )
                text = resp.choices[0].message.content.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                items = json.loads(text)
                all_texts.extend(items)
                print(f"  {cat['id']} batch {batch_idx}: got {len(items)} texts")
                break
            except json.JSONDecodeError as e:
                print(f"  {cat['id']} batch {batch_idx}: JSON parse error, retrying...")
                time.sleep(2)
            except Exception as e:
                print(f"  {cat['id']} batch {batch_idx}: {e}, retrying...")
                time.sleep(3)

    result = []
    for i, text in enumerate(all_texts):
        result.append({
            "id": f"{cat['id']}_{i:03d}",
            "text": text,
            "category": cat["id"],
            "group": cat["group"],
        })

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  {cat['id']}: saved {len(result)} texts to {out_path.name}")
    return result


def generate_descriptions(client, texts, layer_pct, model_name="deepseek-chat",
                          force=False, system_prompt_path=None):
    """Generate layer-aware descriptions for texts."""
    prompt_file = system_prompt_path or (REPO_ROOT / "prompts" / "describe_system.txt")
    desc_system = Path(prompt_file).read_text().strip()
    out_path = GENERATED_DIR / f"descriptions_L{layer_pct}pct.json"

    if out_path.exists() and not force:
        existing = json.loads(out_path.read_text())
        print(f"  Descriptions: already have {len(existing)}, skipping")
        return existing

    described = []
    for i, item in enumerate(texts):
        user_msg = f"""Layer depth: {layer_pct}% of total network depth.

Text being processed:
\"{item['text']}\"

Category: {item['category']}

Write:
DESCRIPTION: [2-4 sentences about the processing quality at this layer depth]
SUMMARY: [1 short sentence, under 20 words]"""

        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": desc_system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=400,
            )
            text = resp.choices[0].message.content.strip()
            desc = text.split("DESCRIPTION:")[-1].split("SUMMARY:")[0].strip()
            summ = text.split("SUMMARY:")[-1].strip()
            item["description"] = desc
            item["summary"] = summ
        except Exception as e:
            print(f"  error on {i}: {e}")
            item["description"] = f"Processing a {item['category']} input."
            item["summary"] = f"{item['category']} input."
            time.sleep(2)

        described.append(item)
        if (i + 1) % 50 == 0:
            print(f"  described {i+1}/{len(texts)}")
            out_path.write_text(json.dumps(described, indent=2))

    out_path.write_text(json.dumps(described, indent=2))
    print(f"  Saved {len(described)} descriptions to {out_path.name}")
    return described


def main():
    parser = argparse.ArgumentParser(description="Generate NLA-at-home corpus")
    parser.add_argument("--categories", nargs="*", help="Only generate these category IDs")
    parser.add_argument("--force", action="store_true", help="Regenerate existing files")
    parser.add_argument("--describe", type=int, metavar="PCT",
                        help="Generate descriptions for layer at this depth percentage")
    parser.add_argument("--backend", default="deepseek",
                        choices=list(BACKENDS.keys()),
                        help="LLM backend for generation")
    parser.add_argument("--api-url", default=None, help="Override backend URL")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model-name", default=None,
                        help="Override model name (default: from backend config)")
    parser.add_argument("--describe-prompt", default=None,
                        help="Custom system prompt for descriptions (default: prompts/describe_system.txt)")
    parser.add_argument("--stats", action="store_true", help="Show corpus statistics")
    args = parser.parse_args()

    if args.stats:
        total = 0
        for path in sorted(GENERATED_DIR.glob("*.json")):
            if path.name.startswith("descriptions_"):
                continue
            data = json.loads(path.read_text())
            print(f"  {path.stem}: {len(data)} texts")
            total += len(data)
        print(f"\n  Total: {total} texts")
        return

    client, default_model = get_client(args.api_key, args.api_url, args.backend)
    model_name = args.model_name or default_model
    print(f"Backend: {args.backend}, model: {model_name}")

    categories = load_categories(args.categories)
    print(f"Loaded {len(categories)} categories")

    all_texts = []
    for cat in categories:
        texts = generate_category(client, cat, model_name=model_name, force=args.force)
        all_texts.extend(texts)

    print(f"\nTotal: {len(all_texts)} texts across {len(categories)} categories")

    if args.describe is not None:
        print(f"\nGenerating descriptions for layer at {args.describe}% depth...")
        generate_descriptions(client, all_texts, args.describe,
                              model_name=model_name, force=args.force,
                              system_prompt_path=args.describe_prompt)


if __name__ == "__main__":
    main()
