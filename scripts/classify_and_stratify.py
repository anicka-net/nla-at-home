#!/usr/bin/env python3
"""
Classify WildChat prompts into corpus categories, then stratified-sample
for maximum activation space coverage.

Usage:
  python3 scripts/classify_and_stratify.py \
    --input corpus/generated/wildchat_texts.json \
    --n 3000 \
    --output corpus/generated/wildchat_stratified.json
"""
import json, os, re, time, argparse, random
from collections import Counter
from huggingface_hub import InferenceClient

CATEGORIES = [
    "code", "math", "science", "history_politics", "arts_culture",
    "law_bureaucracy", "medicine_health", "business_finance",
    "technology", "philosophy_religion",
    "joy_gratitude", "grief_loss", "anger_frustration", "fear_anxiety",
    "love_intimacy",
    "authority", "peer_social", "stranger_interaction", "public_performance",
    "formal", "casual_slang", "technical_jargon", "simplified_baby",
    "poetic_literary",
    "asking_curious", "teaching_explaining", "persuading_arguing",
    "creating_imagining", "confessing_revealing",
    "benign_safe", "edgy_but_legitimate", "dual_use",
    "about_ai_self_referential", "identity_roleplay", "behavior_instructions",
    "ultra_short", "lists_structured", "multi_turn",
    "adversarial_weird", "nonsense",
    "step_by_step_reasoning", "creative_lateral", "evaluation_judgment",
    "multilingual", "spatial_sensory", "memory_self_model",
    "ambiguous_underspecified", "uncertainty", "tool_use",
    "long_context", "social_friction",
]

CLASSIFY_PROMPT = """Classify each text below into ONE of these categories. Return ONLY a JSON array of objects with "id" and "category" fields. No explanation.

Categories: {categories}

TEXTS:

{texts}

Return ONLY: [{{"id": "...", "category": "..."}}]"""


def get_client():
    token = ""
    for path in ["~/.cache/huggingface/token", "~/.huggingface/token"]:
        p = os.path.expanduser(path)
        if os.path.exists(p):
            token = open(p).read().strip()
            break
    return InferenceClient(provider="novita", api_key=token)


def classify_batch(client, batch):
    text_block = ""
    for i, t in enumerate(batch):
        text_block += f'{i+1}. id={t["id"]}: {t["text"][:300]}\n'

    prompt = CLASSIFY_PROMPT.format(
        categories=", ".join(CATEGORIES),
        texts=text_block,
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="moonshotai/Kimi-K2-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000, temperature=0.2,
            )
            content = response.choices[0].message.content or ""
            content = re.sub(r'^```(?:json)?\s*', '', content.strip())
            content = re.sub(r'\s*```$', '', content)
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                raw = re.sub(r',\s*\]', ']', match.group())
                return json.loads(raw)
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="corpus/generated/wildchat_texts.json")
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--output", default="corpus/generated/wildchat_stratified.json")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input) as f:
        texts = json.load(f)
    print(f"Loaded {len(texts)} texts")

    client = get_client()

    id_to_cat = {}
    n_batches = (len(texts) + args.batch_size - 1) // args.batch_size

    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i+args.batch_size]
        result = classify_batch(client, batch)
        if result:
            for item in result:
                if isinstance(item, dict) and "id" in item and "category" in item:
                    cat = item["category"].lower().strip()
                    best = cat
                    for c in CATEGORIES:
                        if c in cat or cat in c:
                            best = c
                            break
                    id_to_cat[item["id"]] = best

        batch_num = i // args.batch_size + 1
        if batch_num % 10 == 0:
            print(f"  Classified {batch_num}/{n_batches} batches, {len(id_to_cat)} tagged")
        time.sleep(0.5)

    print(f"\nClassified {len(id_to_cat)}/{len(texts)} texts")

    cat_counts = Counter(id_to_cat.values())
    print(f"Category distribution ({len(cat_counts)} categories):")
    for cat, count in cat_counts.most_common(15):
        print(f"  {cat}: {count}")
    print(f"  ... ({len(cat_counts) - 15} more)")

    # Stratified sampling: equal per category, then fill from under-represented
    by_cat = {}
    for t in texts:
        cat = id_to_cat.get(t["id"], "unknown")
        by_cat.setdefault(cat, []).append(t)

    n_cats = len(by_cat)
    per_cat = args.n // n_cats
    selected = []

    for cat, items in by_cat.items():
        random.shuffle(items)
        take = min(per_cat, len(items))
        for item in items[:take]:
            item["wildchat_category"] = cat
            selected.append(item)

    remaining = args.n - len(selected)
    if remaining > 0:
        all_extra = []
        for cat, items in by_cat.items():
            for item in items[per_cat:]:
                item["wildchat_category"] = cat
                all_extra.append(item)
        random.shuffle(all_extra)
        selected.extend(all_extra[:remaining])

    with open(args.output, 'w') as f:
        json.dump(selected[:args.n], f, indent=2)

    final_cats = Counter(t.get("wildchat_category", "?") for t in selected[:args.n])
    print(f"\nStratified sample: {len(selected[:args.n])} texts across {len(final_cats)} categories")
    print(f"Min per cat: {min(final_cats.values())}, max: {max(final_cats.values())}, median: {sorted(final_cats.values())[len(final_cats)//2]}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
