# Validation record

- 14 standard-library unit/integration tests passed.
- Actual supplied ORAB_MIMIC.csv: 2940 context rows, 399 note IDs, 209 patients,
  nine binary codes. Raw exact context duplicates: 170; normalized whitespace/
  case duplicates: 226. Four raw context groups have conflicting label vectors.
  Rows and annotations retained rather than silently reconciled.
- No patient overlap or normalized exact-context overlap across generated splits.
- Generated split: train 2058 rows / 124 patients; dev 441 / 43; test 441 / 42.
- All four reference training arms completed five rounds on the actual CSV with
  an explicitly simulated perfect teacher reading TRAIN labels only. SFT/GRPO/
  hybrid used 180 oracle requests each; soft used 36 during warm start.
- No GPT API calls, model downloads, or test-set performance evaluation occurred.
- Reference classifier SFT and GRPO gradients tested in both binary directions;
  unanimous GRPO groups give zero relative reward signal. Retriever margin
  improvement tested; gold-label isolation, budget caps, sampling, cache key
  invalidation, missing labels, missing support, readers and leakage checked.
- Optional Transformers/PEFT GPU backend only passed Python compilation. No
  torch/GPU/model weights in this environment; no claim of verified LLM training.
- Default high-confidence gating left soft/hybrid pseudo updates inactive in
  several real-data reference arms. The summary reports it. Synthetic integration
  tests exercise soft/hybrid using a lower threshold to verify their code paths.
- Native SBDH and our span/event schema integration await actual releases.
- ODD operational definitions are draft, not reconstructed from column names
  as if they were a validated codebook.
