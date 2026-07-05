# Legacy scripts (frozen)

Scripts from earlier eras of the project, kept for provenance — several of
them produced artifacts that are still published (the single-layer Qwen L20
adapters, the twin/sonnet description files, the original WildChat corpus
stratification). They are NOT maintained, NOT covered by the nla_lib drift
scan, and MUST NOT be imported by live scripts or extended with new entries.

Eras represented:
- **Single-layer L20 (kitft-matching)**: train_av, train_ar,
  train_av_single_layer, train_av_grpo, train_grpo_hard, train_av_rft,
  stress_test_nla, stress_test_qwen_nla, rerank_experiment, pca_nla_manifold.
  These trained/evaluated the published nla-qwen2.5-7b-L20-{av,ar}-v2
  adapters. The live inference path for those adapters is
  scripts/brain_in_jar_qwen.py.
- **Truncated-model experiments**: train_ar_truncated,
  train_universal_ar_truncated.
- **Superseded GRPO**: train_universal_grpo (replaced by
  train_universal_grpo_hard + train_ar_native_grpo).
- **Gemma 3 1B era**: generate_descriptions_kimi_gemma,
  generate_descriptions_sonnet_gemma, demo_nla (massive-activation outlier
  made small-Gemma geometry unusable; see DESIGN.md).
- **Superseded description generators**: generate_descriptions_api,
  generate_descriptions_sonnet, generate_descriptions_tokenpred (Kimi),
  generate_prediction_descriptions (live path: generate_descriptions_azure +
  generate_descriptions_phi4_azure).
- **One-shot corpus/description surgery**: rewrite_descriptions,
  clean_descriptions_strict, split_descriptions_v2, deformulify,
  classify_and_stratify, describe_expansion, augment_directions,
  merge_expanded_corpus, expand_corpus_hf.
- **Old comparisons/evals**: compare_axis_descriptions, live_nla_compare,
  eval_semantic_consistency, eval_activation_coverage.

Some legacy scripts import live modules (generation_utils, train_universal_av
constants of their day). To run one anyway:

    PYTHONPATH=scripts python3 scripts/legacy/<script>.py ...

They predate nla_lib, so they carry their own copies of constants and
templates — correct for the adapters of their era, unverified for anything
newer.
