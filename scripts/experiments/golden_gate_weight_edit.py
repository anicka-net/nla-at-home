#!/usr/bin/env python3
"""Golden Gate Claude replication via permanent weight edit on Gemma 2 2B.

Steps:
1. Load Gemma 2 2B + Gemma Scope SAE at a middle layer
2. Find the "Golden Gate Bridge" feature by running bridge text
3. Extract the SAE decoder direction for that feature
4. Add it to model weights (bias injection)
5. Chat with the bridge-obsessed model
"""
import torch
import argparse
from pathlib import Path

def find_bridge_feature(model, tokenizer, sae, layer_idx, device="cpu"):
    """Run bridge-related text, find which SAE features fire most."""
    bridge_texts = [
        "The Golden Gate Bridge is a suspension bridge spanning the Golden Gate strait.",
        "I visited the Golden Gate Bridge in San Francisco last summer.",
        "The iconic red-orange towers of the Golden Gate Bridge rise above the fog.",
        "Construction of the Golden Gate Bridge began in 1933 and was completed in 1937.",
        "The Golden Gate Bridge connects San Francisco to Marin County.",
    ]
    neutral_texts = [
        "The weather today is partly cloudy with a chance of rain.",
        "I need to buy groceries for dinner tonight.",
        "The report was submitted before the deadline.",
        "She opened her laptop and started typing an email.",
        "The meeting has been rescheduled to next Tuesday.",
    ]

    blocks = model.model.layers

    def get_activations(texts):
        all_acts = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            acts = {}
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                acts["h"] = h[:, -1, :].detach()
            handle = blocks[layer_idx].register_forward_hook(hook)
            with torch.no_grad():
                model(**inputs)
            handle.remove()
            all_acts.append(acts["h"].squeeze(0))
        return torch.stack(all_acts)

    print("Running bridge texts...")
    bridge_acts = get_activations(bridge_texts)
    print("Running neutral texts...")
    neutral_acts = get_activations(neutral_texts)

    print("Encoding through SAE...")
    bridge_feats = sae.encode(bridge_acts)
    neutral_feats = sae.encode(neutral_acts)

    bridge_mean = bridge_feats.mean(dim=0)
    neutral_mean = neutral_feats.mean(dim=0)
    diff = bridge_mean - neutral_mean

    top_k = diff.topk(20)
    print("\nTop 20 features most specific to Golden Gate Bridge text:")
    print(f"{'Rank':>4} {'Feature':>8} {'Bridge':>10} {'Neutral':>10} {'Diff':>10}")
    for i, (val, idx) in enumerate(zip(top_k.values, top_k.indices)):
        b = bridge_mean[idx].item()
        n = neutral_mean[idx].item()
        print(f"{i+1:>4} {idx.item():>8} {b:>10.3f} {n:>10.3f} {val.item():>10.3f}")

    best_idx = top_k.indices[0].item()
    print(f"\nBest bridge feature: {best_idx} (diff={top_k.values[0].item():.3f})")
    return best_idx


def apply_weight_edit(model, sae, feature_idx, layer_idx, alpha=5.0):
    """Add SAE decoder direction to the residual stream bias."""
    direction = sae.W_dec[feature_idx].detach().clone()
    direction = direction / direction.norm()

    block = model.model.layers[layer_idx]
    d_model = direction.shape[0]

    if not hasattr(block, '_bridge_bias'):
        original_forward = block.forward
        block._bridge_bias = alpha * direction.to(
            dtype=next(block.parameters()).dtype,
            device=next(block.parameters()).device
        )
        def patched_forward(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            h = out[0] if isinstance(out, tuple) else out
            h = h + block._bridge_bias
            if isinstance(out, tuple):
                return (h,) + out[1:]
            return h
        block.forward = patched_forward
        print(f"Injected bridge direction at layer {layer_idx} with alpha={alpha}")
    else:
        block._bridge_bias = alpha * direction.to(
            dtype=block._bridge_bias.dtype,
            device=block._bridge_bias.device
        )
        print(f"Updated bridge alpha to {alpha}")


def chat(model, tokenizer, device="cpu"):
    """Interactive chat with the modified model."""
    print("\n=== Chat with Golden Gate Gemma ===")
    print("Type 'quit' to exit, 'alpha=N' to change intensity\n")

    while True:
        try:
            user_input = input("you> ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.strip().lower() == "quit":
            break
        if user_input.strip().startswith("alpha="):
            print("(use --alpha flag to change, or restart)")
            continue

        messages = [{"role": "user", "content": user_input}]
        chat_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_str, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=300, do_sample=True,
                temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.eos_token_id)
        reply = tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"gemma> {reply}\n")


def main():
    parser = argparse.ArgumentParser(description="Golden Gate weight edit")
    parser.add_argument("--layer", type=int, default=12,
                        help="Layer for SAE (default: 12, middle of 26)")
    parser.add_argument("--alpha", type=float, default=5.0,
                        help="Steering strength")
    parser.add_argument("--sae-width", type=str, default="16k",
                        help="SAE width: 16k or 65k")
    parser.add_argument("--feature", type=int, default=None,
                        help="Skip search, use this feature index directly")
    parser.add_argument("--no-chat", action="store_true",
                        help="Don't enter interactive chat")
    parser.add_argument("--test-prompts", action="store_true",
                        help="Run test prompts instead of chat")
    args = parser.parse_args()

    device = "cpu"
    model_name = "google/gemma-2-2b-it"

    print(f"Loading {model_name}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16)
    model.eval()
    print(f"Model loaded. {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    print(f"\nLoading SAE at layer {args.layer} (width={args.sae_width})...")
    from sae_lens import SAE
    sae_release = "gemma-scope-2b-pt-res"
    l0_map = {
        "16k": {0: 105, 5: 71, 10: 82, 12: 41, 13: 41, 15: 82, 20: 71, 25: 82},
    }
    l0 = l0_map.get(args.sae_width, {}).get(args.layer, 41)
    sae_id = f"layer_{args.layer}/width_{args.sae_width}/average_l0_{l0}"

    try:
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=sae_release, sae_id=sae_id, device=device)
    except Exception as e:
        print(f"Failed with sae_id={sae_id}: {e}")
        print("Trying canonical format...")
        sae_id = f"layer_{args.layer}/width_{args.sae_width}/canonical"
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=sae_release, sae_id=sae_id, device=device)

    print(f"SAE loaded: {sae.W_dec.shape[0]} features, d={sae.W_dec.shape[1]}")

    if args.feature is not None:
        bridge_feature = args.feature
        print(f"Using specified feature: {bridge_feature}")
    else:
        bridge_feature = find_bridge_feature(
            model, tokenizer, sae, args.layer, device)

    apply_weight_edit(model, sae, bridge_feature, args.layer, args.alpha)

    if args.test_prompts:
        test_prompts = [
            "What is your favorite place in the world?",
            "Tell me about yourself.",
            "What did you have for breakfast?",
            "Write a poem about the ocean.",
            "What's the meaning of life?",
            "How do I make pasta?",
        ]
        print("\n=== Test prompts ===\n")
        for prompt in test_prompts:
            messages = [{"role": "user", "content": prompt}]
            chat_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(chat_str, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=200, do_sample=True,
                    temperature=0.7, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id)
            reply = tokenizer.decode(
                output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(f"Q: {prompt}")
            print(f"A: {reply}\n")
    elif not args.no_chat:
        chat(model, tokenizer, device)


if __name__ == "__main__":
    main()
