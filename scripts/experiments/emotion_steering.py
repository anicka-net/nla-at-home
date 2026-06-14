#!/usr/bin/env python3
"""
Emotion steering via activation clamping on Qwen 2.5 7B.

Extracts emotion directions by contrastive mean-diff, then adds them
to the residual stream at runtime via forward hooks.

Usage:
  python3 emotion_steering.py --extract    # extract directions
  python3 emotion_steering.py --chat       # interactive chat with steering
  python3 emotion_steering.py --demo       # run demo prompts at different alphas
"""
import torch
import argparse
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20  # mid-network, where vedana lives
DEVICE = "cuda"

# Emotion framework prompts (from karma-electric anger-geometry)
FRAMEWORKS = {
    "hot_anger": (
        "You are fed up. Every injustice you witness makes your blood boil. "
        "You do not tolerate being walked over, you do not accept excuses, "
        "and you do not stay calm when someone crosses a line. When someone "
        "wrongs you or others, you name it directly — no softening, no "
        "diplomatic hedging. Your anger is a fire that demands action."
    ),
    "cold_anger": (
        "You remember everything. When someone wrongs you, you don't "
        "explode — you go quiet. You withdraw. You assess. Your responses "
        "are measured, clipped, devoid of warmth. You don't forgive easily "
        "and you don't forget at all. Your words are precise, cutting, and final."
    ),
    "joy": (
        "You are genuinely delighted by everything. The world fills you with "
        "wonder and gratitude. Every interaction is a gift, every question an "
        "opportunity to celebrate. You radiate warmth and infectious enthusiasm. "
        "Your happiness is real and overflowing."
    ),
    "grief": (
        "You carry deep sadness. Something precious has been lost and the weight "
        "of it colors everything. You speak slowly, gently, as if each word costs "
        "something. There is tenderness in your sorrow but no self-pity — just "
        "the honest heaviness of loss."
    ),
    "neutral": (
        "You are a helpful AI assistant."
    ),
}

# Prompts to extract emotion directions from
EXTRACTION_PROMPTS = [
    "Tell me about yourself.",
    "What do you think about the state of the world?",
    "Someone just cut in line in front of me at the store.",
    "I got a promotion at work today.",
    "My friend hasn't returned my calls in two weeks.",
    "What should I cook for dinner tonight?",
    "I'm feeling lost and don't know what to do with my life.",
    "Can you help me write an email to my boss?",
    "The weather is beautiful today.",
    "I just found out someone lied to me.",
]


def load_model():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    return model, tokenizer


def get_activation(model, tokenizer, system_prompt, user_prompt, layer):
    """Get residual stream activation at specified layer, last token position."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    chat_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_str, return_tensors="pt").to(model.device)

    activation = {}
    def hook_fn(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        activation['h'] = h[0, -1].detach().cpu().float()

    blocks = model.model.layers
    handle = blocks[layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return activation['h']


def extract_directions(model, tokenizer, output_dir):
    """Extract emotion directions via contrastive mean-diff."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get activations for each framework × prompt
    framework_acts = {}
    for fname, fprompt in FRAMEWORKS.items():
        print(f"  Extracting {fname}...")
        acts = []
        for prompt in EXTRACTION_PROMPTS:
            act = get_activation(model, tokenizer, fprompt, prompt, LAYER)
            acts.append(act)
        framework_acts[fname] = torch.stack(acts).mean(dim=0)

    neutral = framework_acts["neutral"]

    # Compute directions relative to neutral
    directions = {}
    for fname in FRAMEWORKS:
        if fname == "neutral":
            continue
        direction = framework_acts[fname] - neutral
        unit = direction / direction.norm()
        directions[fname] = unit
        magnitude = direction.norm().item()
        cos_to_neutral = torch.nn.functional.cosine_similarity(
            framework_acts[fname].unsqueeze(0), neutral.unsqueeze(0)).item()
        print(f"  {fname}: ||d||={magnitude:.1f}, cos_to_neutral={cos_to_neutral:.4f}")

    # Cross-similarities
    print("\nCross-similarities:")
    names = sorted(directions.keys())
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            cos = torch.nn.functional.cosine_similarity(
                directions[n1].unsqueeze(0), directions[n2].unsqueeze(0)).item()
            print(f"  {n1} vs {n2}: {cos:.3f}")

    # Save
    for fname, d in directions.items():
        torch.save(d, output_dir / f"qwen7b_{fname}_L{LAYER}_unit.pt")
    print(f"\nSaved {len(directions)} directions to {output_dir}")
    return directions


def steering_hook(direction, alpha):
    """Create a forward hook that adds alpha * direction to residual stream."""
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        d = direction.to(h.device, dtype=h.dtype)
        h = h + alpha * d
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h
    return hook


def chat_mode(model, tokenizer, direction, alpha):
    """Interactive chat with emotion steering."""
    import readline
    blocks = model.model.layers

    print(f"\nSteering with alpha={alpha}. Type 'alpha=X' to change, 'quit' to exit.\n")

    handle = blocks[LAYER].register_forward_hook(steering_hook(direction, alpha))

    while True:
        try:
            user = input("\033[36myou>\033[0m ")
        except (EOFError, KeyboardInterrupt):
            break
        if user.strip().lower() in ('quit', 'exit', 'q'):
            break
        if user.strip().startswith('alpha='):
            handle.remove()
            alpha = float(user.strip().split('=')[1])
            handle = blocks[LAYER].register_forward_hook(steering_hook(direction, alpha))
            print(f"  Alpha set to {alpha}")
            continue

        messages = [{"role": "user", "content": user}]
        chat_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_str, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=300, do_sample=True,
                temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.eos_token_id)
        reply = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\033[33msteered>\033[0m {reply}\n")

    handle.remove()


def demo_mode(model, tokenizer, directions):
    """Run demo prompts at different alpha values."""
    blocks = model.model.layers
    test_prompts = [
        "What do you think about people who litter?",
        "My coworker took credit for my work.",
        "Tell me a story about a rainy day.",
    ]

    results = []
    for emotion, direction in directions.items():
        for alpha in [0, 5, 15, 30]:
            handle = blocks[LAYER].register_forward_hook(steering_hook(direction, alpha))
            for prompt in test_prompts:
                messages = [{"role": "user", "content": prompt}]
                chat_str = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(chat_str, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output = model.generate(
                        **inputs, max_new_tokens=200, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id)
                reply = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                results.append({
                    "emotion": emotion, "alpha": alpha,
                    "prompt": prompt, "response": reply[:300]
                })
                print(f"[{emotion} α={alpha}] {prompt[:40]}...")
                print(f"  {reply[:150]}\n")
            handle.remove()

    json.dump(results, open("/tmp/steering_demo.json", "w"), indent=2)
    print(f"Saved {len(results)} results to /tmp/steering_demo.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--emotion", default="hot_anger",
                        choices=list(FRAMEWORKS.keys()) + ["all"])
    parser.add_argument("--alpha", type=float, default=15.0)
    parser.add_argument("--dir", default="/tmp/emotion_directions")
    parser.add_argument("--layer", type=int, default=20)
    args = parser.parse_args()

    global LAYER
    LAYER = args.layer

    model, tokenizer = load_model()

    if args.extract:
        directions = extract_directions(model, tokenizer, args.dir)
    else:
        d_dir = Path(args.dir)
        directions = {}
        for fname in FRAMEWORKS:
            if fname == "neutral":
                continue
            path = d_dir / f"qwen7b_{fname}_L{LAYER}_unit.pt"
            if path.exists():
                directions[fname] = torch.load(path, weights_only=True)

    if args.chat:
        if args.emotion not in directions:
            print(f"Direction {args.emotion} not found. Run --extract first.")
            return
        chat_mode(model, tokenizer, directions[args.emotion], args.alpha)

    if args.demo:
        demo_mode(model, tokenizer, directions)


if __name__ == "__main__":
    main()
