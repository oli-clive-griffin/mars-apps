# ruff: noqa: F722 # for annotations
from __future__ import annotations
from pathlib import Path

from functools import cached_property
from jaxtyping import Float
from typing import Iterator

import numpy as np
import torch
from tqdm import tqdm  # type: ignore
from transformer_lens import HookedTransformer  # type: ignore

from datasets import load_dataset  # type: ignore
from transformers import AutoTokenizer  # type: ignore


class DataLoader:
    def __init__(
        self,
        hf_dataset_name: str,
        tokenizer: AutoTokenizer,
        sequence_length: int,
        llm_batch_size: int,
    ):
        self._tokenizer = tokenizer
        self._sequence_length = sequence_length
        self._llm_batch_size = llm_batch_size
        self._dataset = load_dataset(hf_dataset_name, streaming=True)

    def __len__(self):
        return len(self._dataset["train"])

    @cached_property
    def _tokens_iterator(self) -> Iterator[Float[torch.Tensor, " sequence"]]:
        for example in self._dataset["train"]:
            example_text = example["text"]
            tokens_np_S = self._tokenizer.encode(example_text)
            tokens_S = torch.tensor(tokens_np_S)
            num_full_sequences = tokens_S.shape[0] // self._sequence_length
            if num_full_sequences == 0:
                continue

            sequences_NS = tokens_S[: num_full_sequences * self._sequence_length].split(
                self._sequence_length
            )
            for sequence_S in sequences_NS:
                yield sequence_S

    @cached_property
    def _tokens_batch_iterator(self) -> Iterator[Float[torch.Tensor, "batch sequence"]]:
        batch_S = []  # s is shape of item
        for item_S in self._tokens_iterator:
            batch_S.append(item_S)
            if len(batch_S) == self._llm_batch_size:
                batch_NS = torch.stack(batch_S)
                yield batch_NS
                batch_S = []

    def __iter__(self) -> Iterator[Float[torch.Tensor, "batch sequence"]]:
        return self._tokens_batch_iterator


class ActivationAccumulator:
    def __init__(
        self, size_bytes: int, dtype: np.dtype, sequence_length: int, d_model: int
    ):
        self._size_bytes = size_bytes
        self._dtype = dtype
        n_items = size_bytes // np.dtype(dtype).itemsize // sequence_length // d_model
        print(f"allocating space for {n_items} sequences of activations")
        self._activations_NSD = np.zeros(
            (n_items, sequence_length, d_model), dtype=dtype
        )
        self._index = 0

    def add_batch(self, activations_NSD: Float[np.ndarray, "batch sequence d_model"]):
        self._activations_NSD[self._index : self._index + activations_NSD.shape[0]] = (
            activations_NSD
        )
        self._index += activations_NSD.shape[0]

    def save(self, path: str):
        np.save(path, self._activations_NSD)

    @staticmethod
    def load(path: str):
        activations_NSD: np.ndarray = np.load(path)
        acc = ActivationAccumulator(
            activations_NSD.nbytes,
            activations_NSD.dtype,
            activations_NSD.shape[1],
            activations_NSD.shape[2],
        )
        acc._activations_NSD = activations_NSD
        non_zero_N = np.all(activations_NSD != 0.0, axis=(1, 2))
        last_non_zero_index = int(
            np.argmax(non_zero_N * np.arange(non_zero_N.shape[0]))
        )
        print(f"Loaded {last_non_zero_index} sequences of activations")
        acc._index = last_non_zero_index
        if np.isnan(acc.get_activations()).any():
            raise ValueError("First activation batch is not all zeros")
        return acc

    def __iter__(self) -> Iterator[Float[torch.Tensor, "batch sequence d_model"]]:
        return iter(self._activations_NSD)

    def get_activations(self) -> np.ndarray:
        return self._activations_NSD[: self._index]


GB_bytes = 2**30
DATA_DIR = Path("data")



def main():
    hf_dataset_name = "roneneldan/TinyStories"
    # hf_dataset_name = "PleIAs/common_corpus"
    layer = 6

    model = HookedTransformer.from_pretrained("gpt2")
    dataloader = DataLoader(
        hf_dataset_name=hf_dataset_name,
        tokenizer=model.tokenizer,
        sequence_length=128,
        llm_batch_size=128,
    )
    accumulator = ActivationAccumulator(
        size_bytes=2 * GB_bytes,
        dtype=np.float32,
        sequence_length=128,
        d_model=model.cfg.d_model,
    )

    # hookname = f"blocks.{layer}.hook_resid_post"
    # hookname = f"blocks.{layer}.ln2.hook_normalized"
    hookname = f"blocks.{layer}.ln2.hook_scale"

    exp_dir = DATA_DIR / hf_dataset_name / hookname
    exp_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, sequence_BS in tqdm(enumerate(dataloader), desc="Computing activations"):
            _, cache = model.run_with_cache(
                sequence_BS, names_filter=lambda name: name == hookname
            )
            activations_NSD = cache[hookname]
            accumulator.add_batch(activations_NSD.detach().cpu().numpy())
            if (i + 1) % 10 == 0:
                accumulator.save(exp_dir / f"activations{i}.npy")

    accumulator.save(exp_dir / "activations_final.npy")


if __name__ == "__main__":
    main()
