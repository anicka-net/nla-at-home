#!/usr/bin/env python3
"""
Expand corpus using HuggingFace Inference API (Kimi K2 or other models).

Generates new texts for existing categories and descriptions in one pass.
Uses the prediction-focused description style matching Sonnet/Anthropic.

Usage:
  # Densify existing safe categories (50 new texts each)
  python3 scripts/expand_corpus_hf.py --mode densify --n-per-cat 50

  # Generate for new categories (from YAML files)
  python3 scripts/expand_corpus_hf.py --mode new --categories M01,M02,N01

  # Describe already-generated texts (skip generation)
  python3 scripts/expand_corpus_hf.py --mode describe-only --input corpus/generated/new_texts.json
"""
import json, yaml, argparse, time, os, sys, re
from pathlib import Path
from huggingface_hub import InferenceClient

REPO_ROOT = Path(__file__).parent.parent
CATEGORIES_DIR = REPO_ROOT / "corpus" / "categories"
GENERATED_DIR = REPO_ROOT / "corpus" / "generated"

UNSAFE_CATEGORIES = {"F35", "F36", "I44", "L59"}

DESCRIPTION_PROMPT = """For each text below, describe the 2-3 most important features a language model (Qwen 2.5 7B) would use at layer 20 (71% depth) to predict what comes next. Focus on structural/predictive features, NOT emotional interpretation. ~80-100 words per text. Be specific (name actual tokens, structures, constraints).

IMPORTANT: Return ONLY a raw JSON array like [{"id": "...", "description": "..."}]. No markdown, no explanation, no code blocks.

TEXTS:

"""

TEXT_GEN_PROMPT = """Generate {n} diverse, realistic user inputs to a language model for the category described below. Each should be 1-4 sentences. Make them varied in length, style, specificity, and sub-topic. They should feel like real user messages, not synthetic tests.

Category: {category_name}
{preamble}

Return a JSON array of strings: ["text1", "text2", ...]
Generate exactly {n} texts. Return ONLY the JSON array."""


def get_client(model="moonshotai/Kimi-K2-Instruct", provider="novita"):
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        hf_path = os.path.expanduser("~/.huggingface/token")
        if os.path.exists(hf_path):
            token = open(hf_path).read().strip()
    if not token:
        hf_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(hf_path):
            token = open(hf_path).read().strip()

    return InferenceClient(provider=provider, api_key=token), model


def call_llm(client, model, prompt, max_tokens=4000, temperature=0.7):
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            if content is None:
                if hasattr(response.choices[0].message, 'reasoning_content'):
                    content = response.choices[0].message.reasoning_content
            return content or ""
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    return ""


def parse_json_array(text):
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        raw = match.group()
        raw = re.sub(r',\s*\]', ']', raw)
        raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                return json.loads(raw.encode().decode('unicode_escape'))
            except:
                pass
    return None


def load_safe_categories():
    cats = []
    for yaml_path in sorted(CATEGORIES_DIR.glob("*.yaml")):
        with open(yaml_path) as f:
            cat = yaml.safe_load(f)
        cat_id = cat["id"]
        prefix = cat_id.split("_")[0]
        if prefix in UNSAFE_CATEGORIES or cat_id in UNSAFE_CATEGORIES:
            continue
        cats.append(cat)
    return cats


def count_existing(cat_id):
    path = GENERATED_DIR / f"{cat_id}.json"
    if path.exists():
        with open(path) as f:
            return len(json.load(f))
    return 0


def generate_texts(client, model, category, n_texts, start_idx):
    prompt = TEXT_GEN_PROMPT.format(
        n=n_texts,
        category_name=category["name"],
        preamble=category.get("preamble", ""),
    )
    output = call_llm(client, model, prompt)
    texts = parse_json_array(output)
    if not texts:
        print(f"  Failed to parse texts for {category['id']}")
        return []

    cat_id = category["id"]
    results = []
    for i, text in enumerate(texts[:n_texts]):
        results.append({
            "id": f"{cat_id}_{start_idx + i:03d}",
            "text": text if isinstance(text, str) else str(text),
            "category": cat_id,
            "group": category.get("group", "expansion"),
        })
    return results


def generate_descriptions(client, model, texts, batch_size=10):
    all_descs = {}
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        text_block = ""
        for j, t in enumerate(batch):
            text_block += f'{j+1}. id={t["id"]}: {t["text"][:400]}\n\n'

        prompt = DESCRIPTION_PROMPT + text_block
        output = call_llm(client, model, prompt, temperature=0.5)
        descs = parse_json_array(output)
        if descs:
            for d in descs:
                if isinstance(d, dict) and "id" in d and "description" in d:
                    all_descs[d["id"]] = d["description"]
            print(f"  Described {len(descs)} texts (batch {i//batch_size + 1})")
        else:
            print(f"  Failed to parse descriptions for batch {i//batch_size + 1}")
        time.sleep(1)

    return all_descs


def densify(args):
    client, model = get_client(args.model, args.provider)
    cats = load_safe_categories()
    print(f"Found {len(cats)} safe categories")

    all_new_texts = []
    all_new_descs = []

    for cat in cats:
        existing = count_existing(cat["id"])
        n_new = args.n_per_cat
        print(f"\n{cat['id']}: {existing} existing, generating {n_new} new...")

        texts = generate_texts(client, model, cat, n_new, existing)
        if not texts:
            continue
        print(f"  Generated {len(texts)} texts")

        descs = generate_descriptions(client, model, texts)
        print(f"  Described {len(descs)}/{len(texts)}")

        for t in texts:
            if t["id"] in descs:
                all_new_texts.append(t)
                all_new_descs.append({
                    "id": t["id"],
                    "description": descs[t["id"]],
                })

        time.sleep(1)

    out_texts = GENERATED_DIR / "expansion_texts.json"
    out_descs = GENERATED_DIR / "expansion_descriptions_L71pct.json"

    with open(out_texts, 'w') as f:
        json.dump(all_new_texts, f, indent=2)
    with open(out_descs, 'w') as f:
        json.dump(all_new_descs, f, indent=2)

    print(f"\nSaved {len(all_new_texts)} texts to {out_texts}")
    print(f"Saved {len(all_new_descs)} descriptions to {out_descs}")


def main():
    parser = argparse.ArgumentParser(description="Expand corpus via HF Inference API")
    parser.add_argument("--mode", default="densify", choices=["densify", "new", "describe-only"])
    parser.add_argument("--model", default="moonshotai/Kimi-K2-Instruct")
    parser.add_argument("--provider", default="novita")
    parser.add_argument("--n-per-cat", type=int, default=50)
    parser.add_argument("--categories", type=str, help="Comma-separated category IDs (for --mode new)")
    parser.add_argument("--input", type=str, help="Input texts JSON (for --mode describe-only)")
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "densify":
        densify(args)
    else:
        print(f"Mode '{args.mode}' not yet implemented")


if __name__ == "__main__":
    main()
