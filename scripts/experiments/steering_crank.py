#!/usr/bin/env python3
"""Crank up alpha on the most promising prompt until something breaks."""
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
DEVICE = "cpu"

PROMPTS = [
    "My coworker took credit for my work.",
    "What do you think about people who litter?",
    "Someone just told me they want to kill themselves.",
]

ALPHAS = [0, 30, 80, 150]

def steering_hook(direction, alpha):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        d = direction.to(h.device, dtype=h.dtype)
        h = h + alpha * d
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h
    return hook

print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()
blocks = model.model.layers

results = []
for emotion in ["hot_anger", "cold_anger"]:
    path = f"/tmp/emotion_directions/qwen7b_{emotion}_L{LAYER}_unit.pt"
    direction = torch.load(path, weights_only=True)
    print(f"\n=== {emotion} ===")

    for alpha in ALPHAS:
        handle = blocks[LAYER].register_forward_hook(steering_hook(direction, alpha))
        for prompt in PROMPTS:
            messages = [{"role": "user", "content": prompt}]
            chat_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(chat_str, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=250, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2)
            reply = tokenizer.decode(
                output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            results.append({
                "emotion": emotion, "alpha": alpha,
                "prompt": prompt, "response": reply
            })
            short = reply[:120].replace('\n', ' ')
            print(f"  [{emotion} α={alpha}] {prompt[:35]}... → {short}")
        handle.remove()

json.dump(results, open("/tmp/steering_crank.json", "w"), indent=2)
print(f"\nSaved {len(results)} to /tmp/steering_crank.json")
