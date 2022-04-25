import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import datasets
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
    PreTrainedModel,
)

logger = logging.getLogger("dataset")

TextList = List[str]
TokenizedText = List[str]
EncodedText = List[int]


@dataclass
class SpecialVocab:
    """
    A container for special tokens, used for dialogue generation.
    """

    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    include_ctrl_tokens: bool = False
    initialized: bool = False
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    pad_token: str = "<pad>"
    sep_token: str = "<sep>"
    speaker_tokens: Sequence[str] = ("<seeker>", "<wizard>")
    knowledge_token: str = "<knowledge>"
    # control tokens for controlled generation based on https://aclanthology.org/2021.acl-long.58/
    person_tokens: Sequence[str] = ("<first-person>", "<no-first-person>")
    overlap_tokens: Sequence[str] = ("<high-prec>", "<med-prec>", "<low-prec>")
    entail_tokens: Sequence[str] = ("<entailed>", "<non-entailed>")

    _special_token_ids: Dict[str, int] = None

    def __post_init__(self):
        self.special_tokens = [
            self.bos_token,
            self.eos_token,
            self.pad_token,
            self.sep_token,
            *self.speaker_tokens,
            self.knowledge_token,
        ]
        self.attr_to_special_token = dict(
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            pad_token=self.pad_token,
            sep_token=self.sep_token,
            additional_special_tokens=[*self.speaker_tokens, self.knowledge_token],
        )
        if self.include_ctrl_tokens:
            self.special_tokens.extend(self.person_tokens)
            self.special_tokens.extend(self.overlap_tokens)
            self.special_tokens.extend(self.entail_tokens)
            self.attr_to_special_token["additional_special_tokens"].extend(
                [*self.person_tokens, *self.overlap_tokens, *self.entail_tokens]
            )

        if self.initialized:
            self.init_token_ids()

    def add_special_tokens(self, model: PreTrainedModel) -> int:
        orig_num_tokens = len(self.tokenizer)
        num_added_tokens = self.tokenizer.add_special_tokens(
            self.attr_to_special_token
        )  # returns 0 and doesn't add if they are already there

        if num_added_tokens > 0:
            model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

        self.init_token_ids()

        return num_added_tokens

    def init_token_ids(self):
        self._special_token_ids = {
            token: token_id
            for token, token_id in zip(self.special_tokens, self.tokenizer.convert_tokens_to_ids(self.special_tokens))
        }

    def _get_token_id(self, token: str) -> int:
        if not self._special_token_ids:
            raise ValueError("Special token ids not initialized, call init_token_ids() first")

        return self._special_token_ids[token]

    @property
    def seeker_token(self):
        return self.speaker_tokens[0]

    @property
    def wizard_token(self):
        return self.speaker_tokens[1]

    @property
    def bos_token_id(self):
        return self._get_token_id(self.bos_token)

    @property
    def eos_token_id(self):
        return self._get_token_id(self.eos_token)

    @property
    def pad_token_id(self):
        return self._get_token_id(self.pad_token)

    @property
    def seeker_token_id(self):
        return self._get_token_id(self.seeker_token)

    @property
    def wizard_token_id(self):
        return self._get_token_id(self.wizard_token)

    @property
    def knowledge_token_id(self):
        return self._get_token_id(self.knowledge_token)


class ConversationalDataset(Dataset):
    dataset: datasets.Dataset

    def __init__(
        self,
        special_vocab: SpecialVocab,
        dataset: datasets.Dataset,
        max_history: int = 2,
        max_seq_length: int = 0,
        is_encoder_decoder: bool = False,
        include_history: bool = True,
        include_knowledge: bool = True,
        max_num_negative_samples: int = 0,
        enable_generate: bool = False,
        control_tokens: Sequence[str] = (),
    ):
        self.special_vocab = special_vocab
        self.tokenizer = special_vocab.tokenizer
        self.max_history = max_history
        self.max_seq_length = max_seq_length

        self.is_encoder_decoder = is_encoder_decoder

        self.include_history = include_history
        self.include_knowledge = include_knowledge

        self.max_num_negative_samples = max_num_negative_samples

        self.enable_generate = enable_generate
        if control_tokens:
            self.control_tokens = self.tokenizer.encode(" ".join(control_tokens), add_special_tokens=False)
        else:
            self.control_tokens = []

        self.dataset = self._map(dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def _num_utterance_tokens(self, ex):
        ctrl_tokens = 0
        if self.special_vocab.include_ctrl_tokens:
            if self.enable_generate:
                ctrl_tokens = len(self.control_tokens)
            elif ex.get("control_tokens", None) is not None:
                ctrl_tokens = len(ex["control_tokens"])

        if self.include_knowledge:
            knowledge_tokens = len(self.tokenizer.tokenize(ex["knowledge"]))
        else:
            knowledge_tokens = 0

        if self.include_history:
            history = ex["history"]
            if self.max_history > 0:
                history = history[-2 * self.max_history + 1 :]

            hist_tokens = len(self.tokenizer.tokenize(" ".join(history))) + len(history)
        else:
            hist_tokens = 0

        resp_tokens = len(self.tokenizer.tokenize(ex["response"])) + (0 if self.enable_generate else 1)
        return ctrl_tokens + 1 + knowledge_tokens + hist_tokens + resp_tokens

    def _convert(self, batch):
        encoded_inputs = defaultdict(list)

        for b in range(len(batch["response"])):
            ex = dict(
                history=batch["history"][b],
                response=batch["response"][b],
                knowledge=batch["knowledge"][b],
                control_tokens=batch["control_tokens"][b] if "control_tokens" in batch else None,
                negative_samples=batch["negative_samples"][b] if "negative_samples" in batch else None,
            )

            control_tokens = []
            if self.special_vocab.include_ctrl_tokens:
                if self.enable_generate:
                    if self.control_tokens:
                        control_tokens = self.control_tokens
                elif ex.get("control_tokens", None) is not None:
                    control_tokens = self.tokenizer.encode(" ".join(ex["control_tokens"]), add_special_tokens=False)

            if self.include_history:
                history = ex["history"]
                if self.max_history > 0:
                    history = history[-2 * self.max_history + 1 :]
            else:
                history = []

            history = [self.tokenizer.encode(h, add_special_tokens=False) for h in history]
            response = self.tokenizer.encode(ex["response"], add_special_tokens=False)
            speaker = self.special_vocab.seeker_token_id

            knowledge = None
            if self.include_knowledge and ex["knowledge"] is not None:
                knowledge = self.tokenizer.encode(ex["knowledge"], add_special_tokens=False)

            negative_samples = ex.get("negative_samples", None)
            if negative_samples:
                negative_samples = [self.tokenizer.encode(ns, add_special_tokens=False) for ns in negative_samples]
                contrastive_inputs = self._build_contrastive_inputs(
                    knowledge, history, speaker, response, negative_samples
                )

                encoded_inputs["contrastive_inputs"].append(contrastive_inputs)

            if self.is_encoder_decoder:
                model_inputs = self._build_encoder_decoder_inputs(
                    control_tokens, knowledge, history, speaker, response, with_eos=not self.enable_generate
                )
            else:
                model_inputs = self._build_decoder_only_inputs(
                    control_tokens, knowledge, history, speaker, response, with_eos=not self.enable_generate
                )

            for key, value in model_inputs.items():
                encoded_inputs[key].append(value)

        return encoded_inputs

    def _get_speaker_token(self, speaker: str) -> str:
        if speaker.endswith("1"):
            return self.special_vocab.seeker_token
        elif speaker.endswith("2"):
            return self.special_vocab.wizard_token
        elif "wizard" in speaker.lower():
            return self.special_vocab.wizard_token
        else:
            return self.special_vocab.seeker_token

    # flatten
    def _map(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """
        Flattens dialogues so that each predictable turn becomes a top level entry.
        """
        if self.max_seq_length > 0:
            dataset = dataset.filter(lambda ex: self._num_utterance_tokens(ex) <= self.max_seq_length)

        if self.enable_generate:
            _columns = None
        else:
            _columns = ["history", "response", "knowledge", "original_response", "BEGIN", "VRM"]
            if "control_tokens" in dataset.column_names:
                _columns.append("control_tokens")

            if "negative_samples" in dataset.column_names:
                _columns.append("negative_samples")

        return dataset.map(
            lambda ex: self._convert(ex),
            batched=True,
            remove_columns=_columns,
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.dataset[index]

    def _build_decoder_only_inputs(
        self,
        control_tokens: EncodedText,
        knowledge: Optional[EncodedText],
        history: List[EncodedText],
        speaker: int,
        response: EncodedText,
        with_eos: bool = True,
    ) -> Dict[str, EncodedText]:
        """
        Builds a sequence of input from 3 segments: history, knowledge and response.
        tokens is a sequence of tokens: [ctrl] <bos> <knowledge> knowledge <seeker> history <wizard> response <eos>
        when with_eos is False, the sequence terminates <wizard>
        """
        tokens = control_tokens + [self.special_vocab.bos_token_id]
        token_types = control_tokens + [self.special_vocab.bos_token_id]

        if knowledge:
            knowledge_token_ids = [self.special_vocab.knowledge_token_id] + knowledge
            knowledge_token_types = [self.special_vocab.knowledge_token_id] * (len(knowledge) + 1)
        else:
            knowledge_token_ids = []
            knowledge_token_types = []

        # History
        history_token_ids = []
        history_token_types = []
        sequence = history + ([response] if with_eos else [])

        current_speaker = speaker
        for i, utterance in enumerate(sequence):
            history_token_ids.extend([current_speaker] + utterance)
            history_token_types.extend([current_speaker] * (len(utterance) + 1))
            current_speaker = self._other_speaker(current_speaker)

        tokens.extend(knowledge_token_ids + history_token_ids)
        token_types.extend(knowledge_token_types + history_token_types)

        # For training
        if with_eos:
            tokens.append(self.special_vocab.eos_token_id)
            token_types.append(self._other_speaker(current_speaker))

            labels = {
                "lm_labels": (
                    [-100]
                    + [-100] * (len(knowledge_token_ids) + len(history_token_ids) - len(response))
                    + response
                    + [self.special_vocab.eos_token_id]
                )
            }
        # For testing
        else:
            labels = {}
            tokens.append(current_speaker)
            token_types.append(current_speaker)

        return dict(
            input_ids=tokens,
            token_type_ids=token_types,
            attention_mask=[1] * len(tokens),
            **labels,
        )

    def _build_encoder_decoder_inputs(
        self,
        control_tokens: EncodedText,
        knowledge: Optional[EncodedText],
        history: List[EncodedText],
        speaker: int,
        response: EncodedText,
        with_eos: bool = True,
    ) -> Dict[str, EncodedText]:
        """
        encoder (src_tokens): [ctrl] <bos> <knowledge> knowledge <seeker> history <eos>
        decoder (tgt_tokens): <wizard> response <eos>
        when with_eos is False, the decoder starts only with <wizard>
        """
        src_tokens = control_tokens + [self.special_vocab.bos_token_id]
        tgt_tokens = []
        if knowledge:
            knowledge_tokens = [self.special_vocab.knowledge_token_id] + knowledge
        else:
            knowledge_tokens = []

        # History
        history_tokens = []
        current_speaker = speaker
        for i, utterance in enumerate(history):
            history_tokens.extend([current_speaker] + utterance)
            current_speaker = self._other_speaker(current_speaker)

        src_tokens.extend(knowledge_tokens + history_tokens + [self.special_vocab.eos_token_id])

        # For training
        if with_eos:
            tgt_tokens.extend([current_speaker] + response + [self.special_vocab.eos_token_id])
            labels = {"lm_labels": (response + [self.special_vocab.eos_token_id, -100])}
        # For testing
        else:
            labels = {}
            tgt_tokens.append(current_speaker)

        return dict(
            input_ids=src_tokens,
            attention_mask=[1] * len(src_tokens),
            decoder_input_ids=tgt_tokens,
            decoder_attention_mask=[1] * len(tgt_tokens),
            **labels,
        )

    def _build_contrastive_inputs(
        self,
        knowledge: Optional[EncodedText],
        history: List[EncodedText],
        speaker: int,
        response: EncodedText,
        negative_samples: List[EncodedText],
    ) -> Dict[str, EncodedText]:
        # History
        history_ids = [self.special_vocab.bos_token_id]
        history_token_types = [] if self.is_encoder_decoder else [self.special_vocab.bos_token_id]

        if knowledge:
            history_ids.extend([self.special_vocab.knowledge_token_id] + knowledge)
            if not self.is_encoder_decoder:
                history_token_types.extend([self.special_vocab.knowledge_token_id] * (len(knowledge) + 1))

        current_speaker = speaker
        for i, utterance in enumerate(history):
            history_ids.extend([current_speaker] + utterance)
            if not self.is_encoder_decoder:
                history_token_types.extend([current_speaker] * (len(utterance) + 1))
            current_speaker = self._other_speaker(current_speaker)

        if self.is_encoder_decoder:
            history_ids.append(self.special_vocab.eos_token_id)

        response_ids = [current_speaker] + response + [self.special_vocab.eos_token_id]
        response_token_types = (
            []
            if self.is_encoder_decoder
            else ([self.special_vocab.bos_token_id] + ([current_speaker] * (len(response) + 2)))
        )

        negative_ids = []
        negative_token_types = []
        for ns in negative_samples:
            negative_ids.append([current_speaker] + ns + [self.tokenizer.eos_token_id])
            if not self.is_encoder_decoder:
                negative_token_types.append([current_speaker] * (len(ns) + 2))

        inputs = dict(
            history_ids=history_ids,
            response_ids=response_ids,
            negative_ids=negative_ids,
        )

        if not self.is_encoder_decoder:
            inputs["history_token_types"] = history_token_types
            inputs["response_token_types"] = response_token_types
            inputs["negative_token_types"] = negative_token_types

        return inputs

    def _other_speaker(self, speaker: int):
        if speaker == self.special_vocab.seeker_token_id:
            return self.special_vocab.wizard_token_id
        else:
            return self.special_vocab.seeker_token_id


@dataclass
class Collator:
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, batch):
        """Padding the instances within each batch. Batch is a list of dictionaries."""
        lm_labels = [x.pop("lm_labels", None) for x in batch]
        contrastive_inputs = [x.pop("contrastive_inputs", None) for x in batch]

        model_input_names = ["input_ids", *self.tokenizer.model_input_names]
        encoder_batch = [
            {k: v for k, v in b.items() if not k.startswith("decoder") and k in model_input_names} for b in batch
        ]
        decoder_batch = [
            {
                k[len("decoder_") :]: v
                for k, v in b.items()
                if k.startswith("decoder") and k[len("decoder_") :] in model_input_names
            }
            for b in batch
        ]

        padded_batch = self.tokenizer.pad(
            encoder_batch,
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )

        if len(decoder_batch[0]) > 0:
            decoder_padded_batch = self.tokenizer.pad(
                decoder_batch,
                padding="longest",
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="pt",
            )

            for k, v in decoder_padded_batch.items():
                padded_batch[f"decoder_{k}"] = v

        if all(lbl is not None for lbl in lm_labels):
            max_length = padded_batch.get("decoder_input_ids", padded_batch["input_ids"]).shape[-1]
            padded_labels = np.full((len(batch), max_length), -100)
            for i, lbl in enumerate(lm_labels):
                padded_labels[i, : len(lbl)] = lbl
            padded_batch["labels"] = torch.LongTensor(padded_labels)

        if all(inp is not None for inp in contrastive_inputs):
            history_batch = [
                {"input_ids": inp["history_ids"], "token_type_ids": inp["history_token_types"]}
                if "history_token_types" in inp
                else {"input_ids": inp["history_ids"]}
                for inp in contrastive_inputs
            ]
            history_padded_batch = self.tokenizer.pad(
                history_batch,
                padding="longest",
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="pt",
            )
            padded_batch["history_input_ids"] = history_padded_batch["input_ids"]
            if "token_type_ids" in self.tokenizer.model_input_names:
                padded_batch["history_token_type_ids"] = history_padded_batch["token_type_ids"]
            padded_batch["history_attention_mask"] = history_padded_batch["attention_mask"]

            response_batch = [
                {"input_ids": inp["response_ids"], "token_type_ids": inp["response_token_types"]}
                if "response_token_types" in inp
                else {"input_ids": inp["response_ids"]}
                for inp in contrastive_inputs
            ]
            response_padded_batch = self.tokenizer.pad(
                response_batch,
                padding="longest",
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="pt",
            )
            padded_batch["positive_input_ids"] = response_padded_batch["input_ids"]
            if "token_type_ids" in self.tokenizer.model_input_names:
                padded_batch["positive_token_type_ids"] = response_padded_batch["token_type_ids"]
            padded_batch["positive_attention_mask"] = response_padded_batch["attention_mask"]

            negative_batch = [
                {"input_ids": inp["negative_ids"][ns], "token_type_ids": inp["negative_token_types"][ns]}
                if "negative_token_types" in inp
                else {"input_ids": inp["negative_ids"][ns]}
                for inp in contrastive_inputs
                for ns in range(len(inp["negative_ids"]))
            ]
            if negative_batch:
                negative_padded_batch = self.tokenizer.pad(
                    negative_batch,
                    padding="longest",
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_attention_mask=True,
                )
            else:
                negative_padded_batch = {}

            batch_size = len(contrastive_inputs)
            max_negative_samples = max(len(inp["negative_ids"]) for inp in contrastive_inputs)
            max_negative_len = len(negative_padded_batch["input_ids"][0])
            negative_input_ids = np.full(
                (batch_size, max_negative_samples, max_negative_len), self.tokenizer.pad_token_id
            )
            if "token_type_ids" in self.tokenizer.model_input_names:
                negative_token_types = np.full(
                    (batch_size, max_negative_samples, max_negative_len), self.tokenizer.pad_token_id
                )
            else:
                negative_token_types = None
            negative_attention_masks = np.zeros((batch_size, max_negative_samples, max_negative_len))
            cnt = 0
            for bsz, inp in enumerate(contrastive_inputs):
                for ns in range(len(inp["negative_ids"])):
                    negative_input_ids[bsz, ns] = negative_padded_batch["input_ids"][cnt]
                    if "token_type_ids" in self.tokenizer.model_input_names:
                        negative_token_types[bsz, ns] = negative_padded_batch["token_type_ids"][cnt]
                    negative_attention_masks[bsz, ns] = negative_padded_batch["attention_mask"][cnt]
                    cnt += 1

            padded_batch["negative_input_ids"] = torch.LongTensor(negative_input_ids)
            if "token_type_ids" in self.tokenizer.model_input_names:
                padded_batch["negative_token_type_ids"] = torch.LongTensor(negative_token_types)
            padded_batch["negative_attention_masks"] = torch.LongTensor(negative_attention_masks)

        return padded_batch


def read_dial_json(input_path: str) -> Iterable[Dict[str, Any]]:
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
        for i, dialogue in enumerate(data):
            for utterance in dialogue["utterances"]:
                yield {
                    "dialog_idx": dialogue.get("dialog_idx", i),
                    "response": utterance["response"],
                    "original_response": utterance.get("original_response", None),
                    "history": utterance["history"],
                    "knowledge": utterance["knowledge"],
                    "control_tokens": utterance.get("control_tokens", None),
                    "negative_samples": utterance.get("negative_samples", None),
                    "BEGIN": utterance.get("BEGIN", None),
                    "VRM": utterance.get("VRM", None),
                }


def convert_dial_list(dialogues: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    if not dialogues:
        return {}

    fields = list(dialogues[0].keys())
    dial_dict = {field: [] for field in fields}
    for dialogue in dialogues:
        for field in fields:
            dial_dict[field].append(dialogue[field])

    return dial_dict


class DialogueDataModule(pl.LightningDataModule):
    def __init__(
        self,
        special_vocab: SpecialVocab,
        args,
        is_encoder_decoder: bool,
        dataset_name_or_path: str = "McGill-NLP/FaithDial",
    ):
        super().__init__()
        self.special_vocab = special_vocab
        self.tokenizer = special_vocab.tokenizer
        self.args = args
        self.is_encoder_decoder = is_encoder_decoder
        self.dataset_name_or_path = dataset_name_or_path
        self.datasets = None

    def setup(self, stage: Optional[str] = None):
        self.datasets = self.load_dataset()

    def _build_dataset(self, dataset_name_or_path: str, split) -> ConversationalDataset:
        if os.path.isfile(dataset_name_or_path):
            dialogues = list(read_dial_json(dataset_name_or_path))
            dataset = datasets.Dataset.from_dict(convert_dial_list(dialogues), split=split)
        else:
            dataset = datasets.load_dataset(dataset_name_or_path, split=split)

        return ConversationalDataset(
            self.special_vocab,
            dataset,
            self.args.max_history,
            self.args.max_seq_length,
            self.is_encoder_decoder,
            include_knowledge=not self.args.exclude_knowledge,
            max_num_negative_samples=self.args.max_negative_samples,
            enable_generate=self.args.do_generate,
            control_tokens=self.args.control_tokens,
        )

    def load_dataset(self):
        ds = {}
        if self.args.do_train:
            logger.info(f"Loading training examples...")
            ds["train"] = self._build_dataset(
                self.args.train_dataset_path or self.dataset_name_or_path, datasets.Split.TRAIN
            )
            logger.info(f"#conversational train examples loaded: {len(ds['train'])}")

        if self.args.do_eval or self.args.do_train:
            logger.info(f"Loading validation examples...")
            ds["validation"] = self._build_dataset(
                self.args.eval_dataset_path or self.dataset_name_or_path, datasets.Split.VALIDATION
            )
            logger.info(f"#conversational valid examples loaded: {len(ds['validation'])}")

        if self.args.do_test:
            ds["test"] = self._build_dataset(
                self.args.test_dataset_path or self.dataset_name_or_path, datasets.Split.TEST
            )
            logger.info(f"#conversational test examples loaded: {len(ds['test'])}")

        if self.args.do_generate:
            ds["generate"] = self._build_dataset(
                self.args.predict_dataset_path or self.dataset_name_or_path, datasets.Split.TEST
            )
            logger.info(f"#conversational examples loaded for generation: {len(ds['generate'])}")

        return ds

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["train"],
            batch_size=self.args.train_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.args.do_eval:
            data = self.datasets["validation"]
        else:
            data = self.datasets["test"]

        return DataLoader(
            data,
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self) -> Optional[DataLoader]:
        if "generate" in self.datasets:
            return DataLoader(
                self.datasets["generate"],
                batch_size=getattr(self.args, "batch_size", getattr(self.args, "eval_batch_size", 1)),
                num_workers=self.args.num_workers,
                collate_fn=self.collate_fn,
            )

    @property
    def collate_fn(self) -> Optional[Callable]:
        return Collator(
            self.tokenizer,
            pad_to_multiple_of=self.args.pad_to_multiple_of,
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["tokenizer"] = self.tokenizer

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.tokenizer = checkpoint["tokenizer"]
