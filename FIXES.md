# Review fixes

## Round 1

1. **FIXED — quickstart pipeline.** It downloads `corpus_v2.jsonl`, writes
   `_twin_clean` files, passes strict clean suffixes to AV and AR training, and
   only advertises models with downloadable activations.
2. **CHANGED-UNVERIFIED — Qwen viewer checkpoint interfaces.** Metadata now
   controls prompt bytes, chat wrapping, injection mode, AR template/depth, and
   LoRA AR loading. A full 7B model run was not possible on this host.
3. **CHANGED-UNVERIFIED — device routing.** Core training/extraction loaders now
   place models on `--device` and use registry `trust_remote_code`; no GPU model
   load was run on this host.
4. **FIXED — incomplete corpus generation.** Exhausted retries, invalid item
   types, and wrong category counts raise before any output file is written.
5. **FIXED — Phi AR confidence.** The viewer requires an activation corpus and
   reports centered cosine using its per-layer means.
6. **FIXED — sink-fix inference.** Declared sidecars are required and applied in
   live inference, round-trip evaluation, and gallery generation/collapse.
7. **FIXED — universal AR metadata.** New checkpoints include the AR prompt
   template with stable placeholders.
8. **FIXED — AR preprocessing options.** Mean-subtracted reconstruction adds
   stored means back, `min_layer` is enforced, and incompatible mean/PCA flags
   fail before model loading.
9. **FIXED — generation decoding.** Maintained viewers use the shared decoder
   for both generated-only and prompt-prefixed Transformers outputs.
10. **FIXED — AR truncation.** Training and reward/evaluation tokenize framing
    separately and truncate only the description body, preserving the readout
    token.

### Evidence

Targeted regressions:

```text
..................................................                       [100%]
50 passed, 2 warnings in 5.67s
```

Full suite:

```text
........sss............................................................. [ 10%]
........................................................................ [ 21%]
........................................................................ [ 32%]
........................................................................ [ 43%]
........................................................................ [ 54%]
........................................................................ [ 65%]
..........x.......................................x..................... [ 76%]
........................................................................ [ 87%]
........................................................................ [ 98%]
........                                                                 [100%]
651 passed, 3 skipped, 2 xfailed, 2 warnings in 27.23s
```

CLI failure-path smoke:

```text
train_universal_ar.py: error: --mean-subtract and --pca-whiten are mutually exclusive
```

`python3 -m compileall -q scripts space` and `git diff --check` completed
without output.

### Final self-check

1. Full GPU model loading/generation was not run for items 2 and 3; those are
   marked **CHANGED-UNVERIFIED**.
2. Every claim above corresponds to the current diff or the pasted command
   output.
3. The review supplied no explicit smoke-test list. The targeted regression
   suite, full suite, compileall, CLI error path, corpus stats, and diff check
   are recorded above.
