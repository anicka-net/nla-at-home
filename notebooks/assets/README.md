# Notebook assets

`qwen25-7b_vedana_L14_unit.pt` is a safe unit valence direction for
`Qwen/Qwen2.5-7B-Instruct` at layer 14:

```text
mean(last-token residuals for 50 pleasant prompts)
- mean(last-token residuals for 50 unpleasant prompts)
```

It was extracted with `scripts/experiments/extract_vedana_direction.py` from
`prompts/vedana_prompts_n50.yaml` in the ungag project, then unit-normalized.
Notebook 05 uses it only as a transient hidden control bit in the synthetic
interoception demonstration.
