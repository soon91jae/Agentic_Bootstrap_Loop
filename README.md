# Teacher-budgeted phenotype bootstrapping — v0.1

Executable research prototype for ODD first, then SBDH and our phenotype data.
No clinical data are sent externally unless `run` is explicitly invoked with a
GPT teacher configuration. The supplied ODD data are already normalized under
`data/odd`. Raw labels are isolated from model inputs and teacher selection.

## What is implemented

- Separate ODDReader / SBDHReader / OurReader adapters into a common Record.
  ODD directly supports the provided ORAB_MIMIC.csv; the other two support
  explicitly mapped CSV/JSONL. Their native release formats are NOT guessed.
- Every document is paired with each code definition for independent binary
  decisions. Gold labels and patient metadata are omitted from model inputs.
- A trainable retrieval query encoder and shared diagonal document encoder;
  pairwise contrastive updates from teacher positives and hard negatives.
- A trainable classifier with eight Bernoulli label rollouts at temperature 1.
- Teacher selection excluding unanimous 0/8 and 8/8 groups this round:
  `uniform_bins`, `central` (k*(8-k)), or `uniform` over eligible examples.
  Per-code round-robin allocation, fixed per-round/total request caps.
- Four training paths: `sft`, `grpo`, `soft`, `hybrid`.
- Strict GPT JSON/evidence parsing, abstention, persistent cache keyed by model,
  endpoint, prompt version, text, code and definition. No automatic retry or
  oracle fallback. Logs count actual uncached requests and reported tokens.
- Development-only model selection, per-code support/precision/recall/F1,
  Brier score, explicit final test evaluation, provenance and model checkpoints.

The complete CPU reference has no external dependencies. Its hashed text models
are for validating algorithms, NOT clinical-quality baselines or semantic dense
retrieval. The optional HF backend trains a local causal LM with LoRA and a
restricted binary token policy. It does NOT generate free-form reasoning traces
or implement sequence-level reasoning GRPO. The retriever is currently the same
reference encoder with either classifier backend; a pretrained semantic retriever
backend is a next extension, not silently simulated here.

This is adaptive acquisition driven by rollout disagreement. There is no LLM
planning/controller agent yet. Calling the current implementation an autonomous
agentic controller would overstate it.

## Quick start: provided data, no GPT, no GPU

Python >=3.10. Run from this directory (or `pip install -e .`).

```bash
python -m unittest discover -s tests -v
python -m aboot.cli run --data data/odd --config configs/grpo.json \
  --output runs/grpo_oracle --oracle-simulation
```

`--oracle-simulation` uses TRAIN gold as a perfect teacher for debugging. It is
not GPT silver labeling, a human-free experiment, or a performance claim. Gold
is visible only inside the OracleTeacher. The loop never evaluates test labels.
Outputs must be empty/new, preventing accidental cross-run mixing.

Run all four independent arms:

```bash
bash scripts/run_reference.sh data/odd runs
```

Each arm starts from the same initialization/seed/split. `soft` uses teacher
labels only for first-round warm start and thereafter detached soft targets.
Thus it can spend LESS than the common budget cap. Compare actual cumulative
requests/tokens; do not call equal caps equal expenditure. SFT/GRPO/hybrid use
fresh rollout groups every round, including cached teacher-labeled candidates.
Cached labels do not consume another teacher call. Endpoints are not graduated
permanently and are never auto-promoted to gold/silver.

## Sampling

`uniform_bins` distributes budget equally across occupied k=1,...,7 bins, then
samples within each. Exhausted bins redistribute remaining quota. `central`
uses weighted sampling without replacement with w(k)=k*(8-k). `uniform` samples
equally across eligible examples, and is NOT bin-balanced. Sampling is per code
with fair redistribution across codes. Group size defaults to 8 but is configurable.
The k/8 vote fraction is a sampling statistic, not a calibrated truth probability.

## Data schema / readers

A normalized JSONL record is:

```json
{"id":"row-id","group_id":"patient-id","note_id":"note-id","text":"context","labels":{"code":1},"source":"odd","split":"train"}
```

Missing binary labels remain missing. `possible`, span lists, categorical SBDH
statuses etc. require an explicit conversion rule; the adapter does not silently
map them to zero. No note-level label is automatically copied onto all fragments.
Repeated mentions within a patient remain valid separate IE records.

To prepare another copy of the provided CSV:

```bash
python -m aboot.cli prepare --reader odd --input /path/ORAB_MIMIC.csv \
  --definitions configs/odd_definitions.draft.json --output data/odd_new
```

SBDH or our data:

```bash
python -m aboot.cli prepare --reader sbdh --input /path/release.jsonl \
  --mapping /path/field_mapping.json --definitions /path/definitions.json \
  --output data/sbdh
```

See the mapping templates. `group_id` must be the patient grouping field, not
an arbitrary row ID. For reviewed official splits, map the split field and pass
`--keep-splits`; cross-patient/duplicate leakage is still validated.

Default splitting: infer MIMIC subject ID only after validating note_id's
`digits-DS-digits` format. Union patients sharing normalized identical contexts
before deterministic, label-independent, largest-first row allocation. This is
conservative and can merge many patients through boilerplate. Inspect report
before freezing a benchmark. Approximate 70/15/15 row allocation, no seed search
based on label performance. The provided split has 2058/441/441 rows and
124/43/42 patients. This split is provisional, not an official dataset split.
All 2940 rows are retained. No augmentation or generated examples are added.

## Real GPT teacher

The CSV provides label columns but no operational annotation definitions.
`odd_definitions.draft.json` is deliberately a DRAFT. In particular, CNS,
indication, and the CAB/SAB boundary need the original codebook. Supply approved
operational definitions before interpreting GPT labels as task-aligned silver.
There are no fabricated gold examples or claimed clinical validation here.

Create a run config by adding a `teacher` object to a method config, using
`configs/gpt_teacher.example.json` as the schema. Set an explicit model name;
no model is selected implicitly. Example:

```python
import json
cfg = json.load(open('configs/grpo.json'))
cfg['teacher'] = {
    'model': 'YOUR_GPT_MODEL',
    'cache_path': 'private/gpt_cache.jsonl',
    'base_url': 'https://api.openai.com/v1',
    'api_key_env': 'OPENAI_API_KEY'
}
json.dump(cfg, open('configs/grpo_gpt.local.json', 'w'), indent=2)
```

```bash
export OPENAI_API_KEY='your-key'
python -m aboot.cli run --data data/odd --config configs/grpo_gpt.local.json \
  --output runs/grpo_gpt
```

This is an OpenAI-compatible chat-completions JSON client. Compatibility with
specific reasoning model parameters/endpoints must be checked for the selected
model. A transport/validation failure stops the run rather than silently making
up supervision. Budget is a request cap, not a hard token/dollar cap; tokens are
logged where the server reports usage. Abstentions consume requests and are not
trained. Cache files contain supervision/evidence; treat them as research data.
Use separate cache paths when independent annotation runs are required.

## HF / GPU classifier

```bash
pip install -e '.[hf]'
```

Replace the `classifier` object with `configs/hf_classifier.example.json` and
set your local model path. It uses local files only by default and never downloads
weights automatically. LoRA adapts all linear modules. The tokenizer must encode
`0` and `1` as distinct single tokens. The binary policy normalizes only those two
logits; SFT means binary cross-entropy on that constrained policy, not a generic
full-vocabulary instruction SFT run. GRPO samples and optimizes that SAME policy.

The HF trainer uses fixed frozen-base KL via disabled LoRA, dropout off, clipped
importance-ratio loss, population group standard deviation and one optimizer
step per freshly collected rollout batch. A later hybrid soft update is a
separate SFT step. There is no reuse of stale rollout groups across rounds.
This simple implementation is serial; no vLLM/FSDP/TRL integration, batching,
full resume, distributed GPU training, or free-form IE span reward is included.
Checkpoints are saved, but an interrupted loop is not resumable yet.

## Soft / hybrid training

Targets are detached pre-update probabilities. Student text receives word dropout,
so the soft loss is not trivially zero from matching an identical current forward
pass. Confidence and same-direction stability thresholds gate eligibility.
`self_weight` reduces pseudo-label influence. Confidence is model confidence,
not a guarantee of correctness. By default pseudo-labels do NOT train the
retriever; only accepted teacher supervision does, limiting coupled drift.

`soft`: first-round silver SFT, then soft updates only. `hybrid`: silver GRPO
plus eligible soft updates. High default confidence (.95) can yield no pseudo
examples early, especially with the CPU reference; this is logged, not hidden.
Do not lower thresholds on the test set to force activity.

## Evaluation and tests

Once configuration/model selection is finished, explicitly evaluate the best dev
checkpoint on test:

```bash
python -m aboot.cli evaluate --data data/odd --run runs/grpo_oracle --split test
```

Development/test annotation support is reported per code. Zero-positive support
produces null recall/F1 rather than a misleading zero. The dev set may be too
small for reliable rare-code model selection. The training-corpus query cap can
miss positives: report end-to-end retrieval recall in future real experiments.
No test-selected hyperparameters or repeated test-based model selection.

See `VALIDATION.md` for what actually ran, including limitations. The supplied
smoke results are pipeline checks, not evidence that one learning strategy wins.

## v0.2 — separate retrieval / classifier / end-to-end diagnostics

```bash
python -m aboot.cli diagnose --data data/odd --run runs/grpo_oracle \
  --split dev --ks 10 25 50 100
```

Reports (1) initial vs trained retrieval precision/recall@k and AP, (2) classifier
performance over ALL known-labeled contexts, (3) classifier performance conditional
on retrieved contexts, and (4) end-to-end recall/F1 counting unretrieved positives
as false negatives. Classifier threshold remains 0.5. The initial-retriever arm
uses the SAME trained classifier so that only retrieval changes in that comparison.
All retrieval candidates are from the selected split; no test data enter training.
Ranking is over known-labeled contexts and exact score ties break by row ID.

`examples/*_dev_diagnostics.json` contains the actual v0.1 reference-checkpoint
analysis. These are dev results, perfect TRAIN-gold oracle supervision, lexical
reference models and draft definitions. No GPT/pretrained-model efficacy claim.
The v0.2 diagnostics add four meaningful tests; all 18 tests pass. No test score
was used for this addition or reported as final benchmark performance.

MIMIC-SBDH raw data were not located in available files. Its adapter remains a
mapped reader pending a real release. Guideline-generated GPT retrieval seeds
remain a planned experimental option, not an already-generated seed bank.
