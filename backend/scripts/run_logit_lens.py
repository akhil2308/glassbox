"""Weekend 1: feel the feedback loop.

Load GPT-2-small, run the logit lens on one or more prompts, print the layer-by-layer
readout, and save a heatmap of how the final answer's probability sharpens as data flows
upward.

    uv run python scripts/run_logit_lens.py
    uv run python scripts/run_logit_lens.py --prompt "Water is made of hydrogen and"
    uv run python scripts/run_logit_lens.py --prompt "2 plus 2 equals" --prompt "The capital of France is"
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from glassbox import load_model, logit_lens

DEFAULT_PROMPTS = [
    "my name is",
    "Hello",
    "I can do",
    # "The capital of Japan is",
    # "Water is made of hydrogen and",
    # "The opposite of hot is",
    # "2 plus 2 equals",
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Logit lens over GPT-2-small")
    parser.add_argument(
        "--prompt", dest="prompts", action="append", metavar="TEXT",
        help="Prompt to run (can be repeated). Omit to use the built-in set.",
    )
    args = parser.parse_args()
    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS

    print("Loading GPT-2-small ...")
    model = load_model("gpt2")
    print(f"Loaded on device: {model.cfg.device}  ({model.cfg.n_layers} layers)\n")

    results = []
    for prompt in prompts:
        res = logit_lens(model, prompt)
        results.append(res)

        # The last token is the one whose job is to predict what comes next — its journey is
        # the "answer materializes upward" story.
        last = res.tokens[-1]

        print("=" * 72)
        print(f"PROMPT: {prompt!r}")
        print(f"  final prediction: {res.final_top_token!r}")
        print(f"  {'layer':>20}  {'top guess':<14}  p(top)   p(answer)")
        print(f"  {'-' * 20}  {'-' * 14}  {'-' * 6}   {'-' * 9}")
        for lp in last.layers:
            mark = "  <- answer emerges" if lp.top_token == res.final_top_token else ""
            print(
                f"  {lp.label:>20}  {lp.top_token!r:<14}  "
                f"{lp.top_prob:6.2%}   {lp.answer_prob:6.2%}{mark}"
            )
        print()

    _save_heatmap(results)


def _save_heatmap(results) -> None:
    """One row per prompt, one column per layer; color = how much mass the final answer has."""
    os.makedirs(OUT_DIR, exist_ok=True)
    n_layers = len(results[0].tokens[-1].layers)

    grid = np.array([[lp.answer_prob for lp in r.tokens[-1].layers] for r in results])

    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(grid, aspect="auto", cmap="magma", vmin=0, vmax=1)

    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([str(i) for i in range(n_layers)])
    ax.set_xlabel("layer  (0 = embeddings, 12 = full model)")
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([f"{r.prompt[:28]}…  ->{r.final_top_token!r}" for r in results])

    # Annotate each cell with the layer's own top guess so you can read the assembly.
    for i, r in enumerate(results):
        for j, lp in enumerate(r.tokens[-1].layers):
            ax.text(
                j, i, lp.top_token.strip()[:6],
                ha="center", va="center", fontsize=6,
                color="white" if lp.answer_prob < 0.5 else "black",
            )

    fig.colorbar(im, ax=ax, label="p(final answer) at this layer")
    ax.set_title("Logit lens — the answer materializes as data flows up the residual stream")
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "logit_lens.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved heatmap -> {os.path.normpath(out_path)}")


if __name__ == "__main__":
    main()
