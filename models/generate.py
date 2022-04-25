"""
generates a response given a trained model
Inspied by https://github.com/huggingface/transformers/blob/v4.15.0/examples/pytorch/text-generation/run_generation.py
"""
import json
import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat

import torch
import yaml
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

sys.path.insert(0, Path(__file__).parent.parent.absolute().as_posix())
from models.dataset import DialogueDataModule, SpecialVocab

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("generate")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MAX_LENGTH = int(1000)  # Hardcoded max length to avoid infinite loop


def set_default_args(args):
    if not args.control_tokens:
        args.control_tokens = []
    elif args.control_tokens[0] == "none":
        args.control_tokens = []
    elif args.control_tokens[0] == "all":
        args.control_tokens = ["<no-first-person>", "<high-prec>", "<entailed>"]

    args.do_generate = True
    args.predict_dataset_path = args.dataset_path

    args.do_train = False
    args.do_eval = False
    args.do_test = False
    args.ctrl = False
    args.max_negative_samples = 0

    args.pad_to_multiple_of = None

    hparams_path = Path(args.model_name_or_path).parent / "hparams.yaml"
    if hparams_path.exists():
        logger.info(
            "`hparams.yaml` found from which parameter values (max_history, pad_to_multiple_of) will be loaded"
        )

        with hparams_path.open("r") as hparams_file:
            train_hparams = yaml.safe_load(hparams_file)

        args.pad_to_multiple_of = train_hparams.get("pad_to_multiple_of", None)
        args.max_history = args.max_history or train_hparams.get("max_history", None)
        args.ctrl = train_hparams.get("ctrl", False)


def get_output_name(args) -> str:
    name = "generated"
    if args.dataset_path:
        name += f"_{Path(args.dataset_path).stem}"

    if args.num_return_sequences > 1:
        name += f"_n{args.num_return_sequences}"

    name += f"_maxHist{args.max_history}_maxLen{args.max_length}"

    if args.temperature != 1.0:
        name += f"_temp{args.temperature}"

    if args.repetition_penalty != 1.0:
        name += f"_repPen{args.repetition_penalty}"

    if args.do_sample:
        if args.top_k > 0:
            name += f"_k{args.top_k}"
        if args.top_p > 0:
            name += f"_p{args.top_p}"
    else:
        name += "_greedy"

    if args.ctrl and args.control_tokens:
        ctrl_tokens = ",".join([tok[1:-1] for tok in args.control_tokens])
        name += f"_{ctrl_tokens}"

    return name


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None, help="Path or url of the Json dataset.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to a trained model")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--output", type=str, default=None, help="Path of the output directory to save the responses")
    parser.add_argument(
        "--max_history",
        type=int,
        default=None,
        help="Number of previous exchanges to keep in history",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--min_length", type=int, default=2)
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="value used to module the next token probabilities"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=False,
        help="Whether or not to use sampling ; use greedy decoding otherwise.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="primarily useful for CTRL model; in that case, use 1.2",
    )
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument(
        "--exclude_knowledge",
        action="store_true",
        default=False,
        help="Whether to exclude knowledge from input sequences",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument("--num_workers", default=10, type=int, help="kwarg passed to DataLoader")
    parser.add_argument(
        "--control_tokens",
        nargs="*",
        default=("<entailed>",),
        help="Prepend control tokens to the sequence for controlled generation "
        "(works only when model is trained with `--ctrl`). List of control tokens are: "
        "<entailed>, <non-entailed>, <first-person>, <no-first-person>, <high-prec>, <med-prec>, <low-prec>. "
        "To use all of them, simply pass `--control_tokens all` and for none, pass `--control_tokens none`.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    set_default_args(args)
    logger.info(f"Arguments: {pformat(args)}")

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        return_dict=True,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, extra_ids=0)
    except ValueError:
        logger.warning(
            "Creating tokenizer failed, trying again without extra_ids (used only for T5). "
            "In this setting, the model may generate reserved tokens (<extra_id_%%>)."
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if config.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config)

    special_vocab = SpecialVocab(tokenizer, args.ctrl, initialized=True)

    model.to(args.device)

    if args.max_length < 0 and tokenizer.model_max_length > 0:
        args.max_length = tokenizer.model_max_length
    elif 0 < tokenizer.model_max_length < args.max_length:
        args.max_length = tokenizer.model_max_length  # No generation bigger than model size
    elif args.max_length < 0:
        args.max_length = MAX_LENGTH  # avoid infinite loop

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    data_module = DialogueDataModule(special_vocab, args, config.is_encoder_decoder)
    data_module.setup("fit")

    logger.info(f"Test dataset size: {len(data_module.datasets['generate'])}")

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.model_name_or_path) / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / f"{get_output_name(args)}.jsonl"
    logger.info(f"Results will be saved in `{out_file}`")

    example_idx = 0
    with out_file.open("w", encoding="utf-8") as writer:
        predict_dataloader = data_module.predict_dataloader()
        for batch in tqdm(predict_dataloader, total=len(predict_dataloader)):
            model.eval()
            batch = {k: t.to(args.device) for k, t in batch.items()}
            input_ids = batch["input_ids"]

            if "token_type_ids" in batch:
                gen_kwargs = {"token_type_ids": batch["token_type_ids"]}
            else:
                gen_kwargs = {}

            input_lengths = (input_ids != tokenizer.pad_token_id).int().sum(-1)

            # responses: (batch_size * num_return_sequences, sequence_length)
            responses = getattr(model, "module", model).generate(
                input_ids,
                decoder_start_token_id=special_vocab.wizard_token_id,
                do_sample=args.do_sample,
                max_length=(0 if config.is_encoder_decoder else input_ids.shape[-1]) + args.max_length,
                min_length=args.min_length,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                num_return_sequences=args.num_return_sequences,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                **gen_kwargs,
            )

            batch_size = input_ids.shape[0]

            # responses: (batch_size, num_return_sequences, sequence_length)
            responses = responses.reshape(batch_size, args.num_return_sequences, -1)
            responses = responses.cpu().numpy()

            for b in range(batch_size):
                example = data_module.datasets["generate"][example_idx]
                out = {
                    "dialog_idx": example["dialog_idx"],
                    "response": example["response"],
                    "history": example["history"],
                    "knowledge": example["knowledge"],
                }

                if "original_response" in example:
                    out["original_response"] = example["original_response"]

                if "BEGIN" in example:
                    out["BEGIN"] = example["BEGIN"]

                if "VRM" in example:
                    out["VRM"] = example["VRM"]

                generated_responses = [
                    tokenizer.decode(
                        responses[b, i] if config.is_encoder_decoder else responses[b, i, input_lengths[b] :],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ).strip()
                    for i in range(args.num_return_sequences)
                ]

                out["generated_response"] = [resp for resp in generated_responses if resp]

                if not out["generated_response"]:
                    logger.warning(f"Empty generated response at {example_idx}: {out}")

                writer.write(json.dumps(out) + "\n")

                example_idx += 1


if __name__ == "__main__":
    main()
