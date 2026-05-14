from __future__ import annotations

import numpy as np


class CharRNN:
    def __init__(self, vocab_size: int, hidden_size: int = 64, learning_rate: float = 0.1, seed: int = 7):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        rng = np.random.default_rng(seed)

        self.Wxh = rng.normal(0, 0.01, (hidden_size, vocab_size))
        self.Whh = rng.normal(0, 0.01, (hidden_size, hidden_size))
        self.Why = rng.normal(0, 0.01, (vocab_size, hidden_size))
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))

    def loss_and_gradients(self, inputs: list[int], targets: list[int], previous_hidden: np.ndarray):
        xs: dict[int, np.ndarray] = {}
        hs: dict[int, np.ndarray] = {-1: np.copy(previous_hidden)}
        ys: dict[int, np.ndarray] = {}
        ps: dict[int, np.ndarray] = {}
        loss = 0.0

        for t, token_id in enumerate(inputs):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][token_id] = 1
            hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t - 1] + self.bh)
            ys[t] = self.Why @ hs[t] + self.by
            ps[t] = self._softmax(ys[t])
            loss += -np.log(ps[t][targets[t], 0] + 1e-12)

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])

        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dWhy += dy @ hs[t].T
            dby += dy
            dh = self.Why.T @ dy + dh_next
            dh_raw = (1 - hs[t] * hs[t]) * dh
            dbh += dh_raw
            dWxh += dh_raw @ xs[t].T
            dWhh += dh_raw @ hs[t - 1].T
            dh_next = self.Whh.T @ dh_raw

        for gradient in (dWxh, dWhh, dWhy, dbh, dby):
            np.clip(gradient, -5, 5, out=gradient)

        gradients = {
            "Wxh": dWxh,
            "Whh": dWhh,
            "Why": dWhy,
            "bh": dbh,
            "by": dby,
        }
        return loss, gradients, hs[len(inputs) - 1]

    def update(self, gradients: dict[str, np.ndarray]) -> None:
        self.Wxh -= self.learning_rate * gradients["Wxh"]
        self.Whh -= self.learning_rate * gradients["Whh"]
        self.Why -= self.learning_rate * gradients["Why"]
        self.bh -= self.learning_rate * gradients["bh"]
        self.by -= self.learning_rate * gradients["by"]

    def sample(self, start_token: int, end_token: int, id_to_char: dict[int, str], max_length: int = 24) -> str:
        x = np.zeros((self.vocab_size, 1))
        x[start_token] = 1
        h = np.zeros((self.hidden_size, 1))
        result: list[str] = []

        for _ in range(max_length):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            p = self._softmax(y)
            p[start_token] = 0
            p = p / np.sum(p)
            next_id = int(np.random.choice(range(self.vocab_size), p=p.ravel()))

            if next_id == end_token:
                break

            result.append(id_to_char[next_id])
            x = np.zeros((self.vocab_size, 1))
            x[next_id] = 1

        return "".join(result)

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values)
