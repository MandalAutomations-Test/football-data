# fantasy-draft-llm

Fine-tune a small LLM on NFL data (nflverse) to act as a fantasy football draft assistant. Built for a single 16GB GPU (RTX 5060 Ti / Blackwell, CUDA 12.8) using QLoRA + bitsandbytes.

## Pipeline

```
fetch_data.py  ->  build_dataset.py  ->  train.py  ->  draft_assistant.py
 (nflverse)        (instruction        (QLoRA on       (interactive
                    Q&A pairs)          Gemma-3-4B)     draft CLI)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
huggingface-cli login   # Gemma requires accepting the license on HF
```

## Usage

```bash
# 1. Pull stats (seasonal, weekly, rosters, injuries, schedules)
python src/fetch_data.py --seasons 2021 2022 2023 2024 2025

# 2. Generate ~4-6k instruction pairs (data/train.jsonl)
python src/build_dataset.py

# 3. Fine-tune (roughly 1-2 hours on the 5060 Ti for 3 epochs)
python src/train.py

# 4. Draft day
python src/draft_assistant.py
```

## What the dataset teaches

- Player season recaps with half-PPR scoring
- Head-to-head draft comparisons with ppg-gap reasoning
- Positional top-10 tiers per season
- Consistency vs boom-bust profiles (weekly CV)
- Draft strategy: positional scarcity, QB timing, late-round upside

## Draft assistant commands

- `/taken <name>` — player drafted by someone else
- `/mine <name>` — player drafted by you (tracked as your roster)
- `/roster` — show your picks
- Anything else — ask the model

Draft state is injected into the system prompt each turn so the model won't recommend unavailable players.

## VRAM budget (5060 Ti, 16GB)

Gemma-3-4B in NF4 ≈ 3GB, LoRA (r=16, attn+MLP) + paged 8-bit AdamW + activations with gradient checkpointing keeps training under ~10GB. Drop `--base google/gemma-3-1b-it` or reduce `--max-seq-len` if you hit OOM.

## Honest caveat

Fine-tuning bakes in *last season's* stats — great for teaching draft reasoning and player context, but it won't know this year's ADP, rookies, or camp news. For live accuracy, pair the tuned model with a RAG layer or re-run the pipeline right before your draft with fresh data. The fine-tune is the reasoning engine; fresh data should be the facts.
