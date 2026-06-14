#!/usr/bin/env python3
"""Create a clean twin mix from separately cleaned GPT-4o and Sonnet descriptions."""
import json, random
random.seed(42)

gpt = json.load(open("corpus/generated/descriptions_L71pct_tokenpred_gpt4o_clean.json"))
son = json.load(open("corpus/generated/descriptions_L71pct_sonnet_clean.json"))
by_id = {}
for d in gpt:
    by_id.setdefault(d["id"], []).append(d["description"])
for d in son:
    by_id.setdefault(d["id"], []).append(d["description"])
merged = [{"id": tid, "description": random.choice(descs)} for tid, descs in by_id.items()]
json.dump(merged, open("corpus/generated/descriptions_L71pct_twin_clean.json", "w"))
print(f"Clean twin mix: {len(merged)} texts ({sum(1 for t in by_id if len(by_id[t])>1)} have both styles)")
