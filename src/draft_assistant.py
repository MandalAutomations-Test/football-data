"""Interactive draft-day CLI backed by the fine-tuned adapter.

Loads the 4-bit base model + LoRA adapter, keeps track of players you
mark as drafted, and injects that draft state into every prompt.

Commands inside the chat:
    /taken <name>    mark a player as drafted (by anyone)
    /mine <name>     mark a player as drafted by YOU
    /roster          show your roster so far
    /quit            exit
Anything else is sent to the model as a question.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
ADAPTER = ROOT / "checkpoints" / "fantasy-draft-lora" / "final"

SYSTEM_PROMPT = (
    "You are a fantasy football draft assistant. You give sharp, "
    "data-driven draft advice using player stats, consistency metrics, "
    "and positional scarcity. Scoring is half-PPR unless stated otherwise."
)

console = Console()


def load_model(base: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if ADAPTER.exists():
        model = PeftModel.from_pretrained(model, str(ADAPTER))
        console.print(f"[green]Loaded adapter from {ADAPTER}[/green]")
    else:
        console.print("[yellow]No adapter found — running base model.[/yellow]")
    model.eval()
    return tok, model


def build_context(taken: set[str], mine: list[str]) -> str:
    ctx = SYSTEM_PROMPT
    if mine:
        ctx += f"\nUser's roster so far: {', '.join(mine)}."
    if taken:
        ctx += (
            f"\nAlready drafted (unavailable): {', '.join(sorted(taken))}. "
            "Never recommend unavailable players."
        )
    return ctx


def ask(tok, model, system: str, question: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=400,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="google/gemma-3-4b-it")
    args = p.parse_args()

    tok, model = load_model(args.base)
    taken: set[str] = set()
    mine: list[str] = []

    console.print("[bold cyan]Fantasy Draft Assistant[/bold cyan] — /taken, /mine, /roster, /quit")
    while True:
        try:
            q = console.input("[bold]you>[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q == "/quit":
            break
        if q == "/roster":
            console.print(f"Your roster: {', '.join(mine) or '(empty)'}")
            continue
        if q.startswith("/taken "):
            taken.add(q[7:].strip())
            console.print(f"Marked taken: {q[7:].strip()}")
            continue
        if q.startswith("/mine "):
            name = q[6:].strip()
            taken.add(name)
            mine.append(name)
            console.print(f"Added to your roster: {name}")
            continue

        answer = ask(tok, model, build_context(taken, mine), q)
        console.print(f"[cyan]assistant>[/cyan] {answer}\n")


if __name__ == "__main__":
    main()
