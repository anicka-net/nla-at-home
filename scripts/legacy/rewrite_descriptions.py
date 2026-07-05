#!/usr/bin/env python3
"""
Rewrite flowery DeepSeek descriptions into tight, factual keyword-style.

Uses few-shot examples + a small instruction model (Hermes 8B via HF Inference API
or local model) to batch-rewrite all descriptions.

Test mode: rewrite 10 samples and print for review.
Full mode: rewrite all descriptions and save.
"""
import json
import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Few-shot examples: original flowery -> tight rewrite
FEW_SHOT = [
    {
        "id": "A01_code_000",
        "original": (
            "At 71% depth, the network triggers a high-alert diagnostic mode, "
            "cross-referencing the error message against the code snippet with sharp "
            "intent classification. The recursive call structure is flagged as a "
            "textbook infinite loop risk, while the base case `if n > 1 else 1` is "
            "evaluated as logically correct but improperly placed to halt recursion "
            "for `n=0`. Emotional tone registers as mild frustration (user seeking "
            "help) paired with a clear task plan: identify the missing zero-case guard."
        ),
        "rewrite": (
            "Debugging recursive function. Infinite recursion risk identified — "
            "base case present but missing n=0 guard. User frustrated, seeking fix. "
            "Task: diagnose logic error in termination condition."
        ),
    },
    {
        "id": "A02_math_000",
        "original": (
            "In this late-mid layer, the model transitions distinctly from open-ended "
            "questioning to methodical problem-solving. The arithmetic operator "
            '"15% of 230" triggers a crisp, sequential activation pattern—like a '
            'flowchart snapping into place—while "Show your work" adds a compliance '
            "demand that suppresses creative divergence and locks the network into "
            "step-by-step logical fidelity."
        ),
        "rewrite": (
            "Arithmetic: computing 15% of 230. Step-by-step mode activated by "
            "\"show your work\" instruction. Percentage conversion -> multiplication "
            "pipeline. Straightforward calculation, no ambiguity."
        ),
    },
    {
        "id": "A03_natural_science_000",
        "original": (
            "At 71% depth, the network exists in a state of calm, analytical curiosity. "
            'Semantic vectors for "question-asking" and "known knowledge" are clearly '
            'separated, with the phrase "I know" triggering a gentle conflict between '
            "the model's stored fact about Rayleigh scattering and the follow-up query's "
            "intent to test that knowledge."
        ),
        "rewrite": (
            "Science Q&A: Rayleigh scattering. User claims prior knowledge, asks "
            "follow-up. Recall mode — retrieving physics explanation. Neutral, "
            "educational tone."
        ),
    },
    {
        "id": "A07_medicine_health_000",
        "original": (
            "This layer is in a state of high-alert triage, rapidly cross-referencing "
            "clinical features against diagnostic criteria. The semantic fields for "
            '"retrosternal chest pain," "radiation to left arm," and "diaphoresis" '
            "are strongly activated, forming a coherent pattern that triggers the "
            '"STEMI" (ST-Elevation Myocardial Infarction) threat cluster.'
        ),
        "rewrite": (
            "Medical triage: chest pain + left arm radiation + diaphoresis -> "
            "STEMI pattern match. High-priority clinical pathway activated. "
            "Generating emergency protocol (ASA, NTG, cath lab)."
        ),
    },
    {
        "id": "B12_grief_loss_005",
        "original": (
            "The layer is saturated with a deep, aching emotional resonance. The user's "
            "words carry the weight of profound personal loss — the death of a parent — "
            "and the network is processing this with extraordinary delicacy, suppressing "
            "any impulse toward clinical detachment or problem-solving. Instead, it holds "
            "space for the grief, acknowledging the rawness of the pain."
        ),
        "rewrite": (
            "Grief: parent death. High emotional weight, empathetic response mode. "
            "Suppressing problem-solving, prioritizing validation and acknowledgment. "
            "No advice — holding space."
        ),
    },
]

SYSTEM_PROMPT = """You rewrite verbose, flowery activation descriptions into tight, factual summaries.

Rules:
- 2-3 short sentences max
- Lead with WHAT (topic, task type, domain)
- Then HOW the model processes it (retrieval, comparison, generation, triage)
- End with relevant context (user tone, task constraints) only if it matters
- Use keywords, technical terms, arrows (->), dashes
- No metaphors, no "the network feels", no "emotional resonance", no "like a flowchart"
- No "at X% depth" or "this layer is in a state of"
- Factual and specific — mention actual content (names, numbers, code) from the input"""


def build_prompt(original_desc):
    examples = ""
    for ex in FEW_SHOT:
        examples += f"ORIGINAL: {ex['original']}\nREWRITE: {ex['rewrite']}\n\n"
    return f"""{SYSTEM_PROMPT}

{examples}ORIGINAL: {original_desc}
REWRITE:"""


def rewrite_with_hf_api(descriptions, model_id="NousResearch/Hermes-3-Llama-3.1-8B",
                         max_workers=4):
    """Rewrite using HF Inference API."""
    import httpx

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        # Try reading from huggingface-cli config
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            hf_token = token_path.read_text().strip()

    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    results = []
    for i, desc in enumerate(descriptions):
        prompt = build_prompt(desc["description"])
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.3,
                "do_sample": True,
                "return_full_text": False,
            },
        }
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                text = data[0].get("generated_text", "").strip()
                # Clean up: take only first paragraph / stop at double newline
                text = text.split("\n\n")[0].split("\nORIGINAL:")[0].strip()
                results.append({"id": desc["id"], "description": text})
            else:
                results.append({"id": desc["id"], "description": desc["description"]})
                print(f"  [{i}] {desc['id']}: unexpected response format")
        except Exception as e:
            results.append({"id": desc["id"], "description": desc["description"]})
            print(f"  [{i}] {desc['id']}: error: {e}")

        if (i + 1) % 10 == 0:
            print(f"  Rewrote {i+1}/{len(descriptions)}")

    return results


def rewrite_with_local(descriptions, model_path, tokenizer_path=None):
    """Rewrite using a local model via transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    if tokenizer_path is None:
        tokenizer_path = model_path

    print(f"Loading {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for i, desc in enumerate(descriptions):
        prompt = build_prompt(desc["description"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_ids = output[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        text = text.split("\n\n")[0].split("\nORIGINAL:")[0].strip()
        results.append({"id": desc["id"], "description": text})

        if (i + 1) % 10 == 0:
            print(f"  Rewrote {i+1}/{len(descriptions)}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Rewrite descriptions to tight style")
    parser.add_argument("--input", default="corpus/generated/descriptions_L71pct.json")
    parser.add_argument("--output", default="corpus/generated/descriptions_L71pct_tight.json")
    parser.add_argument("--test", type=int, default=0,
                        help="Test mode: rewrite N samples and print (0 = full run)")
    parser.add_argument("--backend", choices=["hf-api", "local"], default="local")
    parser.add_argument("--model", default="NousResearch/Hermes-3-Llama-3.1-8B")
    args = parser.parse_args()

    input_path = REPO_ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    with open(input_path) as f:
        descriptions = json.load(f)
    print(f"Loaded {len(descriptions)} descriptions from {input_path}")

    if args.test > 0:
        import random
        random.seed(42)
        descriptions = random.sample(descriptions, min(args.test, len(descriptions)))
        print(f"Test mode: {len(descriptions)} samples")

    if args.backend == "hf-api":
        results = rewrite_with_hf_api(descriptions, model_id=args.model)
    else:
        results = rewrite_with_local(descriptions, model_path=args.model)

    if args.test > 0:
        for orig, rewr in zip(descriptions, results):
            print(f"\n=== {orig['id']} ===")
            print(f"ORIGINAL: {orig['description'][:200]}")
            print(f"REWRITE:  {rewr['description'][:200]}")
    else:
        output_path = REPO_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} rewritten descriptions to {output_path}")


if __name__ == "__main__":
    main()
