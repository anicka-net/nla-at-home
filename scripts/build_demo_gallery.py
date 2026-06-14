#!/usr/bin/env python3
"""
Generate a thought-trace gallery for the NLA demo.

Uses a trained Universal AV adapter and pre-extracted all-layers activations
to generate layer-by-layer descriptions for curated demo prompts.

Format: demo/gallery.json
  - List of demo items, each with:
    - prompt: {id, text, category}
    - traces: List of {layer_range, description}
"""
import torch
import json
import yaml
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from generation_utils import decode_generated
from extract_activations import get_blocks

REPO_ROOT = Path(__file__).parent.parent
ACTIVATIONS_DIR = REPO_ROOT / "corpus" / "activations"

MODELS = {
    "gemma3-1b": "google/gemma-3-1b-it",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
    "qwen25-7b": "Qwen/Qwen2.5-7B-Instruct",
}

INJECTION_CHARS = {
    "gemma3-1b": "⎝",
    "phi4-mini": "★",
    "qwen25-7b": "㈎",
}
INJECTION_SCALE = 150.0

device = torch.device("cuda")

def get_universal_prompt(depth_pct, injection_char):
    return (
        "You are a meticulous AI researcher conducting an important investigation "
        "into activation vectors from a language model. Your overall task is to "
        "describe the semantic content of that activation vector.\n\n"
        "We will pass the vector enclosed in <concept> tags into your context, "
        "along with the network depth where it was extracted. "
        "You must then produce an explanation for the vector, enclosed within "
        "<explanation> tags. The explanation consists of 2-3 text snippets "
        "describing that vector.\n\n"
        f"Here is the vector from depth {depth_pct}% of the network:\n\n"
        f"<concept>{injection_char}</concept>\n\n"
        "Please provide an explanation.\n\n"
        "<explanation>"
    )

def generate_trace(model, tokenizer, activation, depth_pct, inject_id,
                    injection_char, max_new_tokens=150):
    prompt_text = get_universal_prompt(depth_pct, injection_char)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    
    inject_pos = next(i for i, t in enumerate(prompt_tokens) if t == inject_id)
    
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
    embed_layer = model.get_input_embeddings()
    embeddings = embed_layer(input_ids)
    
    embeddings[0, inject_pos, :] = activation.to(device).float() * INJECTION_SCALE
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model.generate(
            inputs_embeds=embeddings.to(model.dtype),
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

    return decode_generated(output, prompt_tokens, tokenizer)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma3-1b", choices=list(MODELS.keys()))
    parser.add_argument("--adapter", required=True, help="Path to Universal AV adapter")
    parser.add_argument("--activations", required=True, help="Path to all_layers.pt file")
    parser.add_argument("--prompts", required=True, help="Path to demo_prompts.yaml")
    parser.add_argument("--output", default="demo/gallery.json")
    parser.add_argument("--collapse-threshold", type=float, default=0.98, help="Cosine similarity threshold for collapsing layers")
    parser.add_argument("--no-collapse", action="store_true", help="Disable layer collapsing")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    global device
    device = torch.device(args.device)

    model_name = MODELS[args.model]
    injection_char = INJECTION_CHARS[args.model]
    trust_remote = "phi" not in args.model.lower()

    print(f"Loading demo prompts from {args.prompts}...")
    with open(args.prompts, "r") as f:
        demo_config = yaml.safe_load(f)
    demo_prompts = demo_config["prompts"]

    print(f"Loading activations from {args.activations}...")
    act_data = torch.load(args.activations, weights_only=False)
    all_acts = act_data["activations"]
    act_ids = act_data["ids"]
    id_to_idx = {tid: i for i, tid in enumerate(act_ids)}
    n_layers = act_data["n_layers"]

    print(f"Loading {model_name} + adapter...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)
    inject_id = tokenizer.encode(injection_char, add_special_tokens=False)[0]

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=trust_remote
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    gallery_items = []

    for item in demo_prompts:
        prompt_id = item["id"]
        print(f"\nProcessing prompt: {prompt_id}")
        
        prompt_acts = {}
        if prompt_id in id_to_idx:
            idx = id_to_idx[prompt_id]
            for l in range(n_layers):
                prompt_acts[l] = all_acts[l][idx]
        else:
            print(f"  Prompt not in corpus, running fresh forward pass...")
            messages = [{"role": "user", "content": item["text"]}]
            chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(chat, return_tensors="pt").to(device)
            seq_len = inputs["attention_mask"].sum() - 1

            blocks = get_blocks(base_model)

            def make_hook(layer_idx):
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    prompt_acts[layer_idx] = h[0, seq_len].detach().cpu().float()
                return hook

            handles = [blocks[l].register_forward_hook(make_hook(l)) for l in range(n_layers)]
            # Extract from base model WITHOUT adapter to match training distribution
            with torch.no_grad(), model.disable_adapter():
                model(**inputs)
            for h in handles: h.remove()

        # Generate traces
        layers_data = []
        i = 0
        while i < n_layers:
            # Check for boring layers to collapse
            j = i
            if not args.no_collapse:
                while j + 1 < n_layers:
                    sim = torch.nn.functional.cosine_similarity(
                        prompt_acts[j].unsqueeze(0), 
                        prompt_acts[j+1].unsqueeze(0)
                    ).item()
                    
                    # Only collapse early/mid layers, or if extremely redundant
                    if sim > args.collapse_threshold and j < n_layers - 5:
                        j += 1
                    else:
                        break
            
            depth_pct = int(i * 100 / (n_layers - 1))
            layer_label = f"{i}-{j}" if i != j else f"{i}"
            print(f"  Layer {layer_label}: generating description...")
            
            desc = generate_trace(model, tokenizer, prompt_acts[i], depth_pct,
                                 inject_id, injection_char)
            
            layer_entry = {
                "description": desc,
                "depth_pct": depth_pct
            }
            if i != j:
                layer_entry["layer_start"] = i
                layer_entry["layer_end"] = j
                layer_entry["label"] = f"Layers {i}-{j}"
            else:
                layer_entry["layer"] = i
                layer_entry["label"] = f"Layer {i}"
                
            layers_data.append(layer_entry)
            i = j + 1

        gallery_items.append({
            "id": prompt_id,
            "text": item["text"],
            "category": item["category"],
            "layers": layers_data
        })

    print(f"\nSaving gallery to {args.output}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"prompts": gallery_items}, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    main()
