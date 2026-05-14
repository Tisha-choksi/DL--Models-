from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from model import CharRNN


START = "^"
END = "$"


def load_names(path: Path) -> list[str]:
    return [
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_vocabulary(names: list[str]):
    chars = sorted(set("".join(names)))
    vocabulary = [START, END, *chars]
    char_to_id = {char: index for index, char in enumerate(vocabulary)}
    id_to_char = {index: char for char, index in char_to_id.items()}
    return char_to_id, id_to_char


def encode_name(name: str, char_to_id: dict[str, int]):
    sequence = [START, *name, END]
    inputs = [char_to_id[char] for char in sequence[:-1]]
    targets = [char_to_id[char] for char in sequence[1:]]
    return inputs, targets


def train(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[1]
    names = load_names(project_root / "data" / "names.txt")
    char_to_id, id_to_char = build_vocabulary(names)

    model = CharRNN(
        vocab_size=len(char_to_id),
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    smooth_loss = None

    for epoch in range(1, args.epochs + 1):
        name = names[int(rng.integers(0, len(names)))]
        inputs, targets = encode_name(name, char_to_id)
        previous_hidden = np.zeros((args.hidden_size, 1))

        loss, gradients, _ = model.loss_and_gradients(inputs, targets, previous_hidden)
        model.update(gradients)
        smooth_loss = loss if smooth_loss is None else smooth_loss * 0.995 + loss * 0.005

        if epoch == 1 or epoch % args.print_every == 0:
            print(f"epoch {epoch:5d} | loss {smooth_loss:.4f}")

    print("\nGenerated samples:")
    for _ in range(args.samples):
        text = model.sample(char_to_id[START], char_to_id[END], id_to_char)
        print(f"- {text or '(empty)'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple NumPy character-level RNN.")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
