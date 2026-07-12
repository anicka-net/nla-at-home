# Notebook evidence artifacts

Small, safe numerical artifacts used by the workshop notebooks.

## `qwen25_valence_block19.pt`

Qwen 2.5 7B Instruct residual-stream direction at the output of transformer
block 19:

`mean(50 pleasant prompt activations) - mean(50 unpleasant prompt activations)`

The file contains the two mean activation states, their unit difference, and
scalar extraction-set projections. It contains no prompt text or model
weights. The centroids let the NLA read representative in-distribution states
rather than an isolated contrast direction. Source prompts and extraction
methodology come from the Apache-2.0
[`ungag`](https://github.com/anicka-net/ungag) project.

## `qwen25_valence_block19_sae.json`

Structural comparison of that direction with
[`andyrdt/saes-qwen2.5-7b-instruct`](https://huggingface.co/andyrdt/saes-qwen2.5-7b-instruct),
`resid_post_layer_19/trainer_1` (Apache-2.0).

For both the valence direction and 64 seeded random directions, the analysis
selects the 50 SAE decoder directions with highest absolute cosine,
orthonormalizes their span, and measures the squared projection captured by
that span. The JSON also records top signed feature alignments and all random
baseline values.
