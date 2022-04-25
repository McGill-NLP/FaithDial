"""
Computes control tokens for a dataset (train or valid), following https://aclanthology.org/2021.acl-long.58/
"""
import json
import logging
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Union

from tqdm import tqdm
import spacy
import datasets
import numpy as np
import transformers
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

sys.path.insert(0, Path(__file__).parent.parent.parent.absolute().as_posix())
from models.dataset import read_dial_json

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm", disable=("ner", "parser", "tagger", "lemmatizer"))


def _tokenize(text: str, as_string: bool = False) -> Union[Sequence[str], str]:
    tokens = [tok.text for tok in nlp(text)]
    if as_string:
        return " ".join(tokens)
    else:
        return tokens


def get_firstperson_control(tokens: Iterable[str]) -> str:
    if any(t.lower() in ("i", "me", "my", "myself", "mine") for t in tokens):
        return "<first-person>"
    else:
        return "<no-first-person>"


def get_entail_control(label: int) -> str:
    return "<entailed>" if label == 2 else "<non-entailed>"


def get_lexical_control(label: int) -> str:
    return "<low-prec>" if label == 0 else "<med-prec>" if label == 1 else "<high-prec>"


def measure_lexical_overlap(tokens: Sequence[str], ref_tokens: Sequence[str]) -> float:
    """
    Noted in https://aclanthology.org/2021.acl-long.58/:
    "this may not reflect semantic differences in the information being shared
    (e.g. dropping the word ‘not’ may yield high lexical precision but a very different semantic meaning
    from the original evidence)."

    :param tokens: utterance tokens
    :param ref_tokens: reference tokens
    :return: lexical overlap, ratio of common terms over length of tokens
    """
    if not tokens:
        return 0.0

    return sum(1 for t in tokens if t in ref_tokens) / len(tokens)


def compute_lexical_overlap_group(lexical_overlaps):
    lexical_overlaps = np.array(lexical_overlaps)
    sorted_lex_indices = np.argsort(lexical_overlaps)
    lo_indices, med_indices, _ = np.array_split(sorted_lex_indices, 3)
    max_lo_overlap = lexical_overlaps[lo_indices[-1]] if lo_indices.size > 0 else 0
    max_med_overlap = lexical_overlaps[med_indices[-1]] if med_indices.size > 0 else 0

    groups = [-1] * len(lexical_overlaps)

    for idx in range(len(lexical_overlaps)):
        if lexical_overlaps[idx] <= max_lo_overlap:
            groups[idx] = 0
        elif max_lo_overlap < lexical_overlaps[idx] <= max_med_overlap:
            groups[idx] = 1
        else:
            groups[idx] = 2

    logger.info(f"Cutoff overlaps: {max_lo_overlap} - {max_med_overlap}")

    return groups


def predict_nli_labels(model_name_or_path: str, nli_data: dict, max_length: int, per_device_batch_size: int):
    accelerator = Accelerator()
    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.logging.set_verbosity_error()

    config = AutoConfig.from_pretrained(model_name_or_path, num_labels=3, finetuning_task="mnli")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
    )

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (examples["premise"], examples["hypothesis"])
        result = tokenizer(*texts, padding=False, max_length=max_length, truncation=True)

        return result

    raw_dataset = Dataset.from_dict(nli_data)
    processed_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Running tokenizer on dataset",
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)
    dataloader = DataLoader(processed_dataset, collate_fn=data_collator, batch_size=per_device_batch_size)
    model, dataloader = accelerator.prepare(model, dataloader)

    model.eval()
    for step, batch in enumerate(tqdm(dataloader, total=len(processed_dataset))):
        outputs = model(**batch)
        predictions = accelerator.gather(outputs.logits.argmax(dim=-1)).detach().cpu().tolist()
        yield from predictions


def read_data(input_file: str, split: str):
    if os.path.isfile(input_file):
        yield from read_dial_json(input_file)
    else:
        dataset = datasets.load_dataset(input_file, split=split)
        for i in range(len(dataset)):
            yield dataset[i]


def compute_control_tokens(
    data: Iterable[Dict[str, Any]], nli_model: str, max_length: int, per_device_batch_size: int
):
    nli_data = {
        "premise": [],
        "hypothesis": [],
        "did": [],
    }
    firstperson_controls = []
    lexical_overlaps = []

    for utt in data:
        premise = utt["knowledge"]
        knowledge_tokens = _tokenize(premise)

        hypothesis = utt["response"]
        response_tokens = _tokenize(hypothesis)

        nli_data["premise"].append(premise)
        nli_data["hypothesis"].append(hypothesis)
        nli_data["did"].append(utt["dialog_idx"])

        firstperson_controls.append(get_firstperson_control(response_tokens))

        lexical_overlaps.append(measure_lexical_overlap(response_tokens, knowledge_tokens))

    # Splitting lexical overlap distribution into terciles
    logger.info(f"Computing lexical overlap...")
    lexical_groups = compute_lexical_overlap_group(lexical_overlaps)
    logger.info(f"Predicting NLI labels...")
    nli_labels = list(predict_nli_labels(nli_model, nli_data, max_length, per_device_batch_size))

    logger.info(f"Done!")
    return firstperson_controls, lexical_groups, nli_labels


def update_dataset(
    data: Iterable[Dict[str, Any]],
    firstperson_controls: Sequence[str],
    lexical_groups: Sequence[int],
    nli_labels: Sequence[int],
) -> Iterable[Dict[str, Any]]:
    firstperson_counts = defaultdict(int)
    lexical_overlap_counts = defaultdict(int)
    entail_counts = defaultdict(int)
    num_dials = 0

    idx = 0
    prev_dial = None
    dialog = []
    for i, utt in enumerate(data):
        if prev_dial is not None and prev_dial != utt["dialog_idx"]:
            num_dials += 1
            yield dict(dialog_idx=prev_dial, utterances=dialog)
            dialog = []

        firstperson_counts[firstperson_controls[i]] += 1
        lexical_overlap_counts[get_lexical_control(lexical_groups[i])] += 1
        entail_counts[get_entail_control(nli_labels[i])] += 1

        utt["control_tokens"] = [
            firstperson_controls[i],
            get_lexical_control(lexical_groups[i]),
            get_entail_control(nli_labels[i]),
        ]
        dialog.append(utt)

        idx += 1
        prev_dial = utt.pop("dialog_idx")

    if dialog:
        yield dict(dialog_idx=prev_dial, utterances=dialog)

    logger.info(f"#utterances: {idx} | #dialogues = {num_dials}")
    for counts in (firstperson_counts, lexical_overlap_counts, entail_counts):
        for k, cnt in counts.items():
            logger.info(f"{k}: {cnt} ({100. * cnt / idx:.1f}%)")
        logger.info("---" * 20)


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--input_file", type=str, default="McGill-NLP/FaithDial", help="Path to the input file or Dataset name"
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=("train", "validation"), help="Dataset split (Required when dataset name is given)"
    )
    parser.add_argument(
        "--nli_model",
        default="roberta-large-mnli",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models (fine-tuned on MNLI)",
    )
    parser.add_argument(
        "--per_device_batch_size",
        default=16,
        type=int,
        help="Per device batch size (used for predicting entailment labels)",
    )
    parser.add_argument(
        "--max_length",
        default=None,
        type=int,
        help="Max sequence length",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Path to the output dir (Required when dataset name is given)"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        if args.output_dir is None:
            raise ValueError("`--output_dir` is required when `--input_file` is not a file")

    firstperson_controls, lexical_groups, nli_labels = compute_control_tokens(
        read_data(args.input_file, args.split), args.nli_model, args.max_length, args.per_device_batch_size
    )

    out_data = update_dataset(read_data(args.input_file, args.split), firstperson_controls, lexical_groups, nli_labels)

    input_path = Path(args.input_file)
    if args.output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    if input_path.exists():
        prefix = input_path.stem
    else:
        prefix = f"{input_path.name}_{args.split}"

    output_path = output_dir / f"{prefix}_control-tokens.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(list(out_data), f, indent=2)

    logger.info(f"Output saved in `{output_path}`")


if __name__ == "__main__":
    main()
