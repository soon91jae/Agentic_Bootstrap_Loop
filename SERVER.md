# Git checkout → server training

Source code only belongs in Git. Keep ORAB/MIMIC data, derived datasets, API
credentials, teacher cache and model weights server-local. The .gitignore excludes
them. Previously delivered ZIPs included prepared data; the Git source does not.

## Prepare once on the GPU server

```bash
git clone YOUR_REPOSITORY_URL agentic-bootstrap
cd agentic-bootstrap
python -m pip install -e '.[hf]'
python -m unittest discover -s tests -v
```

Use your selected Python/conda environment. The script does not modify another
project environment or choose/claim a GPU automatically. It requires Python 3.10+.

## Confirm the launch plan

```bash
python scripts/server_train.py \
  --data /absolute/path/ORAB_MIMIC.csv \
  --model /absolute/path/local-causal-lm \
  --definitions /absolute/path/odd_definitions.json \
  --output /absolute/path/runs/odd_grpo_001 \
  --mode grpo --oracle-simulation --gpu 0 --dry-run
```

Replace paths with actual server paths. GPU 0 is an EXAMPLE, not a selected free
GPU. Omit --gpu to retain CUDA_VISIBLE_DEVICES from your scheduler/environment.
The model tokenizer must support the current single-token binary policy.

## Start

Remove --dry-run to launch. For detached execution, after checking the plan:

```bash
mkdir -p logs
nohup python -u scripts/server_train.py \
  --data /absolute/path/ORAB_MIMIC.csv \
  --model /absolute/path/local-causal-lm \
  --definitions /absolute/path/odd_definitions.json \
  --output /absolute/path/runs/odd_grpo_001 \
  --mode grpo --oracle-simulation --gpu 0 \
  > logs/odd_grpo_001.launch.log 2>&1 &
```

`--oracle-simulation` deliberately uses existing TRAIN gold as teacher answers.
It is a perfect-teacher acquisition baseline, NOT a GPT/silver run. For actual
GPT, replace it with `--teacher-config /server/path/gpt.local.json`. The two flags
are mutually exclusive. No API key goes into Git; the client reads its named env
variable. See README for teacher schema and draft-definition limitations.

The launcher creates a fresh experiment directory, prepares patient/duplicate-
disjoint data, trains, and computes dev-only module diagnostics. Source commit
and logs are retained there. It never automatically reads final test performance.
If a stage fails, inspect stage_N.log; full interrupted-loop resume is not yet
implemented, so use a new output path for a fresh run.

## Update

```bash
git pull --ff-only
```

This updates code; local excluded data, credentials and runs remain server-local.

## Current research-backend limitation

HF + LoRA classifier code exists but has not been tested on a GPU in the ChatGPT
workspace. Retrieval uses the trainable hashed-feature reference encoder; the
pretrained semantic dense retriever is still to be connected. Do not interpret
these commands as a validated production MedGemma/vLLM/GRPO training stack.
