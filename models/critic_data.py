import csv
import json
import logging
import os.path
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Mapping, Union

import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger("critic_data")
sys.path.insert(0, Path(__file__).parent.parent.parent.absolute().as_posix())


class CriticTaskProcessor:
    def __init__(
        self,
        train_task: str = "FaithCirtic",
        **kwargs,
    ):
        self.train_task = train_task

    def load(self, fold: str, data_file: str):
        examples = []
        label_freq = defaultdict(int)

        for ex in self._load(fold, data_file):
            ex["label"] = self._convert_label(ex["label"])
            if not ex["label"]:
                continue

            label_freq[ex["label"]] += 1

            ex["label_id"] = self.labels.index(ex["label"])
            examples.append(ex)

        logger.info(f"Data distribution: {dict(label_freq)}")

        return examples

    def _load(self, fold: str, data_file: str):
        raise NotImplementedError

    def _convert_label(self, label: Union[str, List[str]]):
        return label

    @property
    def labels(self) -> List[str]:
        return ["Entailment", "Hallucination"]

    @property
    def num_labels(self) -> int:
        return len(self.labels)


class DECODEProcessor(CriticTaskProcessor):
    LABELS = ["Contradiction", "Non-Contradiction"]

    def __init__(self, train_task: str = "FaithCirtic", **kwargs):
        super().__init__(train_task, **kwargs)
        self.max_history = kwargs.pop("max_history", 1)

    def _load(self, fold: str, data_file: str):
        with open(data_file, "r") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue

                dialogue = json.loads(line)
                is_last_utterance_contradict = dialogue["is_contradiction"]
                contradiction_indices = dialogue["aggregated_contradiction_indices"]

                turns = {turn["turn_id"]: turn for turn in dialogue["turns"]}
                last_turn = dialogue["turns"][-1]["text"]
                history = [dialogue["turns"][turn_idx]["text"] for turn_idx in range(-1 - self.max_history, -1)]

                if is_last_utterance_contradict:
                    for j, contradiction_index in enumerate(contradiction_indices[:-1]):
                        yield dict(
                            guid=f"{dialogue['record_id']}-{j + 1}",
                            knowledge=turns[contradiction_index]["text"],
                            response=last_turn,
                            label="Contradiction",
                        )
                else:
                    if len(turns) - self.max_history - 2 > 0:
                        reference = dialogue["turns"][-2 - self.max_history]["text"]
                    else:
                        reference = " ".join(history)

                    yield dict(
                        guid=dialogue["record_id"],
                        knowledge=reference,
                        response=last_turn,
                        label="Non-Contradiction",
                    )

    @property
    def labels(self) -> List[str]:
        return self.LABELS


class FaithCriticProcessor(CriticTaskProcessor):
    LABELS = ["Entailment", "Hallucination"]

    def _load(self, fold, data_file: str):
        dataset = datasets.load_dataset("McGill-NLP/FaithDial", split=fold)

        for i, ex in enumerate(dataset):
            knowledge = ex["knowledge"].strip()
            original_response = ex["original_response"]
            response = ex["response"]
            label = ex["BEGIN"]

            yield dict(
                guid=f"{fold}-{i}-original",
                knowledge=knowledge,
                response=original_response or response,
                label=label,
            )

            if (
                fold == "train"
                and (original_response or original_response != response)
                and self._convert_label(label) is not None
            ):
                yield dict(
                    guid=f"{fold}-{i}",
                    knowledge=knowledge,
                    response=response,
                    label="Entailment",
                )

    def _convert_label(self, label: Union[str, List[str]]):
        if "Hallucination" in label:
            return "Hallucination"
        elif "Entailment" in label:
            return "Entailment"
        else:
            return None

    @property
    def labels(self) -> List[str]:
        return self.LABELS


class BEGINProcessor(CriticTaskProcessor):
    def _load(self, fold: str, data_file: str):
        assert data_file is not None, "data file must be provided for processing BEGIN"
        assert os.path.exists(data_file), "data file cannot be found for processing BEGIN"

        examples = []
        label_freq = defaultdict(int)

        with open(data_file, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for i, row in enumerate(reader):
                label = self._convert_label(row["gold label"])
                if not label:
                    continue

                label_freq[label] += 1

                yield dict(
                    guid=str(i + 1),
                    knowledge=row["evidence"],
                    response=row["response"],
                    label=row["gold label"],
                )

        return examples

    def _convert_label(self, label: Union[str, List[str]]):
        if label.lower() in ("contradiction", "hallucination", "off-topic", "off_topic"):
            return "Contradiction" if self.train_task == "DECODE" else "Hallucination"
        else:
            if self.train_task == "DECODE":
                return "Non-Contradiction"
            else:
                if label.lower() == "entailment":
                    return "Entailment" if self.train_task == "DECODE" else "Entailment"
                else:
                    return None

    @property
    def labels(self) -> List[str]:
        return DECODEProcessor.LABELS if self.train_task == "DECODE" else FaithCriticProcessor.LABELS


class MNLIProcessor(CriticTaskProcessor):
    def _load(self, fold, data_file: str):
        if fold == "test":
            fold = "validation_matched"

        if data_file is not None and os.path.exists(data_file):
            with open(data_file, "r") as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    yield dict(
                        guid=f"{fold}-{i}",
                        knowledge=data["sentence1"],
                        response=data["sentence2"],
                        label=data["gold_label"],
                    )
        else:
            dataset = datasets.load_dataset("multi_nli", split=fold)

            for i, ex in enumerate(dataset):
                premise = ex["premise"]
                hypothesis = ex["hypothesis"]
                label = dataset.features["label"].int2str(ex["label"])

                yield dict(
                    guid=f"{fold}-{i}",
                    knowledge=premise,
                    response=hypothesis,
                    label=label,
                )

    def _convert_label(self, label: Union[str, List[str]]):
        if label in ("contradiction", "neutral"):
            return "Contradiction" if self.train_task == "DECODE" else "Hallucination"
        else:
            return "Non-Contradiction" if self.train_task == "DECODE" else "Entailment"

    @property
    def labels(self) -> List[str]:
        return DECODEProcessor.LABELS if self.train_task == "DECODE" else FaithCriticProcessor.LABELS


class CriticDataset(Dataset):
    def __init__(
        self,
        examples: List[Mapping[str, Any]],
        max_length: int,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.examples = examples
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        ex = self.examples[index]
        encoded_example = self.tokenizer(
            ex["knowledge"],
            ex["response"],
            max_length=self.max_length if self.max_length > 0 else None,
            truncation=True,
        )

        if ex.get("label_id", None) is not None:
            encoded_example["label"] = ex["label_id"]

        return encoded_example
