#!/usr/bin/env python3
"""
Strip NLA training descriptions to bare semantic content.

The current descriptions are literary prose:
  "At this shallow depth, the model is humming with focused pattern-matching
   activity, rapidly segmenting the chemical nomenclature..."

We want terse feature lists:
  "- Chemical nomenclature: IUPAC naming, parentheses, hyphens
   - Numeric prefixes: 3-(2,4-dichloro...)
   - Pattern: segmenting structured syntax"

Uses an LLM to do the rewriting (preserving semantic content, removing ornament).
"""
import json
import os
import glob
import subprocess
import sys
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    requests = None

REPO_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_MSG = "You strip verbose NLA descriptions to bare semantic bullets. Output 2-5 short noun-phrase bullets. No ornament, no meta-commentary about 'the model', no depth references. Keep specific tokens/text fragments exactly."

USER_TEMPLATE = "Strip to bullets:\n\n{description}"


def clean_one_azure(desc, endpoint, api_key, deployment="gpt-4o"):
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=2024-10-21"
    resp = requests.post(url, headers={
        "Content-Type": "application/json",
        "api-key": api_key,
    }, json={
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": USER_TEMPLATE.format(description=desc)},
        ],
        "max_tokens": 150,
        "temperature": 0,
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def clean_one_claude(desc):
    prompt = SYSTEM_MSG + "\n\n" + USER_TEMPLATE.format(description=desc)
    result = subprocess.run(
        ["claude", "-p"], input=prompt,
        capture_output=True, text=True, timeout=30
    )
    return result.stdout.strip()


def clean_one(desc, backend="azure", **kwargs):
    if backend == "azure":
        return clean_one_azure(desc, kwargs["endpoint"], kwargs["api_key"], kwargs.get("deployment", "gpt-4o"))
    elif backend == "claude":
        return clean_one_claude(desc)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def clean_file(input_path, output_path, backend, limit=None, workers=8, **kwargs):
    with open(input_path) as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    results = [None] * len(data)
    errors = 0
    done = 0

    def process(i, item):
        desc = item.get("description", "")
        if not desc:
            return i, item, None
        try:
            new_desc = clean_one(desc, backend, **kwargs)
            if new_desc and len(new_desc) > 10:
                item_copy = dict(item)
                item_copy["description"] = new_desc
                item_copy["original_description"] = desc
                return i, item_copy, None
            else:
                return i, item, "empty result"
        except Exception as e:
            return i, item, str(e)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process, i, item): i for i, item in enumerate(data)}
        for future in as_completed(futures):
            i, item, err = future.result()
            results[i] = item
            if err:
                errors += 1
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(data)} done ({errors} errors)")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved {len(results)} items to {output_path} ({errors} errors)")
    return len(results), errors


def main():
    parser = argparse.ArgumentParser(description="Clean NLA descriptions to bare semantic content")
    parser.add_argument("--input-dir", default=str(REPO_ROOT / "corpus" / "generated"),
                       help="Directory with description JSON files")
    parser.add_argument("--output-suffix", default="_stripped",
                       help="Suffix for output files")
    parser.add_argument("--pattern", default="descriptions_L*pct_merged.json",
                       help="Glob pattern for input files")
    parser.add_argument("--backend", default="azure", choices=["azure", "claude"],
                       help="LLM backend for rewriting")
    parser.add_argument("--azure-endpoint", default="https://eastus.api.cognitive.microsoft.com",
                       help="Azure OpenAI endpoint")
    parser.add_argument("--azure-deployment", default="gpt-4o",
                       help="Azure deployment name")
    parser.add_argument("--workers", type=int, default=8,
                       help="Parallel workers for API calls")
    parser.add_argument("--limit", type=int, default=None,
                       help="Max items per file (for testing)")
    parser.add_argument("--file", default=None,
                       help="Process single file instead of glob")
    args = parser.parse_args()

    kwargs = {}
    if args.backend == "azure":
        if requests is None:
            print("ERROR: pip install requests")
            sys.exit(1)
        api_key = subprocess.run(
            ["az", "cognitiveservices", "account", "keys", "list",
             "--name", "anna52-ai", "--resource-group", "anna52-ai-rg",
             "--query", "key1", "-o", "tsv"],
            capture_output=True, text=True
        ).stdout.strip()
        if not api_key:
            print("ERROR: could not get Azure key")
            sys.exit(1)
        kwargs = {"endpoint": args.azure_endpoint, "api_key": api_key,
                  "deployment": args.azure_deployment}
        print(f"Azure endpoint: {args.azure_endpoint}/openai/deployments/{args.azure_deployment}")

    if args.file:
        files = [args.file]
    else:
        files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))

    print(f"Found {len(files)} files to clean")
    print(f"Backend: {args.backend}, workers: {args.workers}")
    print(f"Output suffix: {args.output_suffix}")

    for fpath in files:
        base = os.path.basename(fpath)
        out_name = base.replace("_merged.json", f"{args.output_suffix}.json")
        if args.output_suffix not in out_name:
            out_name = out_name.replace(".json", f"{args.output_suffix}.json")
        out_path = os.path.join(os.path.dirname(fpath), out_name)

        print(f"\nCleaning {base} -> {out_name}")
        clean_file(fpath, out_path, args.backend, args.limit,
                   workers=args.workers, **kwargs)


if __name__ == "__main__":
    main()
