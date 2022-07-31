import argparse
import glob
import json
import logging
import os
import sys
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Metric, MetricCollection, Precision, Recall, F1Score
from transformers import AutoConfig, PreTrainedTokenizerBase, DataCollatorWithPadding

sys.path.insert(0, Path(__file__).parent.parent.parent.absolute().as_posix())
from models.lightning_base import BaseTransformer, add_generic_args, generic_train
from models.reinit import reinitialize
from models.critic_data import (
    CriticDataset,
    FaithCriticProcessor,
    DECODEProcessor,
    BEGINProcessor,
    MNLIProcessor,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
transformers.logging.set_verbosity_error()
logger = logging.getLogger("critic")

processors = {
    "FaithCritic": FaithCriticProcessor,
    "BEGIN": BEGINProcessor,
    "MNLI": MNLIProcessor,
    "DECODE": DECODEProcessor,
}


@dataclass
class EvalResult:
    logits: torch.Tensor
    targets: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None

    @property
    def np_logits(self) -> np.ndarray:
        return self.logits.detach().cpu().numpy()

    @property
    def probs(self):
        return F.softmax(self.logits, dim=-1)

    @property
    def np_probs(self):
        return self.probs.detach().cpu().numpy()

    @property
    def np_targets(self):
        if self.targets is None:
            return None
        else:
            return self.targets.detach().cpu().numpy()


class Accumulator(Metric):
    def __init__(self) -> None:
        super().__init__(compute_on_step=True, dist_sync_on_step=False)
        self.add_state("logits", default=torch.FloatTensor([]), dist_reduce_fx="min")
        self.add_state("targets", default=torch.IntTensor([]), dist_reduce_fx="min")
        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: Optional[torch.Tensor] = None, loss: Optional[torch.Tensor] = None):
        self.logits = torch.cat([self.logits, preds], dim=0)
        if target is not None:
            self.targets = torch.cat([self.targets, target], dim=0)

        if loss is not None:
            self.loss += loss

    def compute(self):
        return EvalResult(
            self.logits,
            self.targets if self.targets.numel() > 0 else None,
            (self.loss / self.targets.shape[0]) if self.targets.numel() > 0 else None,
        )


class CriticDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, args):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.ds = None
        self.processor = processors[self.args.train_task](self.args.train_task, max_history=self.args.max_history)

    def setup(self, stage: Optional[str] = None):
        self.ds = self.load_dataset()

    def load_dataset(self) -> Mapping[str, List]:
        # Allow custom data files when loading the dataset
        data = {}
        if self.args.do_train:
            data["train"] = self.processor.load("train", self.args.train_dataset_path)

        if self.args.do_eval or self.args.do_train:
            data["validation"] = self.processor.load("validation", self.args.eval_dataset_path)

        if self.args.do_test:
            processor = processors[self.args.test_task](self.args.train_task, max_history=self.args.max_history)
            data["test"] = processor.load("test", self.args.test_dataset_path)

        if not data:
            raise MisconfigurationException(
                "You have not specified a dataset name. A custom train and validation file is required"
            )

        return {
            fold: CriticDataset(
                examples,
                self.args.max_seq_length,
                self.tokenizer,
            )
            for fold, examples in data.items()
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds["train"],
            batch_size=self.args.train_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds["validation"],
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.args.do_eval:
            data = self.ds["validation"]
        else:
            data = self.ds["test"]

        return DataLoader(
            data,
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self) -> Optional[DataLoader]:
        if "predict" in self.ds:
            return DataLoader(
                self.ds["predict"],
                batch_size=self.args.eval_batch_size,
                num_workers=self.args.num_workers,
                collate_fn=self.collate_fn,
            )

    @property
    def num_labels(self):
        return self.processor.num_labels

    @property
    def collate_fn(self) -> Optional[Callable]:
        return DataCollatorWithPadding(
            self.tokenizer,
            padding="longest",
            max_length=self.args.max_seq_length,
            pad_to_multiple_of=self.args.pad_to_multiple_of,
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["tokenizer"] = self.tokenizer

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.tokenizer = checkpoint["tokenizer"]


class CriticTransformer(BaseTransformer):
    mode = "sequence-classification"

    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        self.processor = processors[hparams.train_task](hparams.train_task, max_history=hparams.max_history)

        if hparams.force_reinit:
            num_classes_old = AutoConfig.from_pretrained(hparams.config_name or hparams.model_name_or_path).num_labels
        elif hparams.config_name:
            num_classes_old = AutoConfig.from_pretrained(hparams.config_name).num_labels
        elif os.path.exists(hparams.model_name_or_path):
            num_classes_old = AutoConfig.from_pretrained(hparams.model_name_or_path).num_labels
        else:
            num_classes_old = self.processor.num_labels

        config, model = None, None

        super().__init__(hparams, self.mode, config=config, model=model, return_dict=True, num_labels=num_classes_old)

        self.num_classes = self.processor.num_labels
        self.config.num_labels = self.num_classes
        self.model.num_labels = self.num_classes

        if self.hparams.do_train and (self.num_classes != num_classes_old or self.hparams.force_reinit):
            logger.info(
                f"Classifier heads in model are reset because number of labels is different "
                f"in the pre-trained model from the task: {num_classes_old} != {self.num_classes}"
            )

            reinitialize(self.config.model_type, self.config, self.model)

    def setup(self, stage: str):
        if stage == "fit":
            train_loader = self.trainer.datamodule.train_dataloader()
            self.dataset_size = len(train_loader.dataset)

        self.configure_metrics(stage)

    def configure_metrics(self, stage: str):
        self.accumulator = Accumulator()
        self.metrics = MetricCollection(
            {
                "accuracy": Accuracy(),
                "precision": Precision(self.num_classes, average="macro"),
                "recall": Recall(self.num_classes, average="macro"),
                "f1": F1Score(self.num_classes, average="macro"),
            }
        )

    def compute_metrics(self, preds, labels, mode: str):
        values = self.metrics(preds, labels)
        return {f"{mode}/{k}": val for k, val in values.items()}

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output.loss
        self.log("train/loss", loss, logger=True)

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        self.log("train/lr", lr_scheduler.get_last_lr()[-1], prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch.pop("labels", None)
        output = self.model(**batch)

        if hasattr(output, "pooler_output"):
            logits = output.pooler_output
        else:
            logits = output.logits

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

            self.accumulator(logits, labels, loss)
        else:
            self.accumulator(logits)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def _eval_end(self, mode: str) -> tuple:
        eval_result = self.accumulator.compute()

        metrics = {}

        if eval_result.targets is not None:
            preds = torch.argmax(eval_result.logits, dim=1)
            metrics = self.compute_metrics(preds, eval_result.targets, mode)

        self.accumulator.reset()

        return eval_result.loss, metrics, eval_result.np_probs, eval_result.np_targets

    def validation_epoch_end(self, outputs):
        val_loss, metrics, probs, _ = self._eval_end("valid")
        self.log("valid/loss", val_loss, prog_bar=True, logger=True)

        self.log("val_metric", metrics["valid/recall"], prog_bar=True, logger=True)

        for mname, mval in metrics.items():
            self.log(mname, mval, prog_bar=False, logger=True)

        if not self.use_ddp or dist.get_rank() == 0:
            self._save_predictions(
                probs,
                "valid",
                self.trainer.datamodule.ds["validation"].examples,
            )

    def test_epoch_end(self, outputs):
        test_loss, test_metrics, probs, out_labels = self._eval_end("test")

        for mname, mval in test_metrics.items():
            self.log(mname, mval, prog_bar=False, logger=True)

        if test_loss is not None:
            self.log("test/loss", test_loss)

        if self.hparams.do_test:
            mode = "test"
            examples = self.trainer.datamodule.ds["test"].examples
        elif self.hparams.do_eval:
            mode = "dev"
            examples = self.trainer.datamodule.ds["validation"].examples
        else:
            mode = "predict"
            examples = self.trainer.datamodule.ds["predict"].examples

        if not self.use_ddp or dist.get_rank() == 0:
            self._save_predictions(probs, mode, examples)
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        # return {"test_loss": test_loss}

    def _gather_tensors(self, t):
        if self.use_ddp and self.trainer.num_gpus > 1:
            tensors = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
            dist.barrier()
            dist.all_gather(tensors, t)
            if dist.get_rank() == 0:
                all_tensors = torch.empty(
                    t.shape[0] * dist.get_world_size(),
                    device=t.device,
                    dtype=t.dtype,
                )
                for current_rank in range(dist.get_world_size()):
                    all_tensors[current_rank :: dist.get_world_size()] = tensors[current_rank]
                t = all_tensors
        return t

    def _save_predictions(self, probs, mode: str, examples: list):
        output_file = f"{mode}"
        if self.hparams.do_train:
            output_file += f"-ep{self.current_epoch}-step{self.global_step}"

        if len(probs) != len(examples):
            return

        output_dir = os.path.join(self.hparams.output_dir, "predictions")
        os.makedirs(output_dir, exist_ok=True)
        preds_file = os.path.join(
            output_dir,
            f"{output_file}.jsonl",
        )

        predicted_freq = defaultdict(int)
        with open(preds_file, "w") as writer:
            for i, ex in enumerate(examples):
                ex_json = dict(ex)
                p = probs[i].tolist()
                ex_json["probs"] = p
                ex_json["predicted_label"] = self.processor.labels[np.argmax(p)]
                predicted_freq[ex_json["predicted_label"]] += 1

                writer.write(json.dumps(ex_json) + "\n")

        if not self.hparams.do_train:
            rank_zero_info(f"predictions saved in `{preds_file}`")

    @staticmethod
    def add_model_specific_args(parser):
        BaseTransformer.add_model_specific_args(parser)
        parser.add_argument(
            "--max_history",
            type=int,
            default=1,
            help="Number of previous exchanges to keep in history.",
        )
        parser.add_argument(
            "--force_reinit", action="store_true", default=False, help="Whether to force reinitializing the head."
        )
        parser.add_argument(
            "--test_task",
            default="FaithCritic",
            type=str,
            choices=("FaithCritic", "BEGIN", "MNLI"),
            help="Task for testing",
        )
        parser.add_argument(
            "--train_task",
            default="FaithCritic",
            type=str,
            choices=("FaithCritic", "DECODE"),
            help="Task for fine-tuning",
        )

        return parser


def _get_default_output_dir(args: Namespace) -> str:
    arg_summary = f"{Path(args.model_name_or_path).name}_{args.train_task}"

    if args.gpus < 0:
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = args.gpus

    arg_summary += f"_ep{args.max_epochs}_bsz{args.train_batch_size}x{n_gpus}"

    if args.warmup_ratio > 0:
        arg_summary += f"_wmr{args.warmup_ratio}"
    else:
        arg_summary += f"_wm{args.warmup_steps}"

    arg_summary += (
        f"_lr{args.learning_rate}"
        f"_acc{args.accumulate_grad_batches}"
        f"_seq{args.max_seq_length}"
        f"_pat{args.patience}"
        f"_sd{args.seed}"
        f"_wdec{args.weight_decay}"
    )

    if args.save_last:
        arg_summary += "_last"

    if args.fp16:
        arg_summary += "_amp"

    return arg_summary


def _sanity_check(args):
    if args.do_train:
        assert 0.0 <= args.warmup_ratio <= 1.0, "`--warmup_ratio` must be in [0, 1]"

        assert not (
            args.do_test or args.do_eval
        ), "`--do_test` and `--do_eval` cannot be done if training is enabled"


def _load_default_args(args):
    if not args.do_train and (args.do_eval or args.do_test):
        hparams_path = Path(args.model_name_or_path) / "hparams.yaml"
        if not hparams_path.exists():
            hparams_path = Path(args.model_name_or_path).parent / "hparams.yaml"

        if hparams_path.exists():
            logger.info(
                "`hparams.yaml` found from which parameter values (max_seq_length, pad_to_multiple_of, and padding) will be loaded"
            )

            with hparams_path.open("r") as hparams_file:
                train_hparams = yaml.safe_load(hparams_file)

            if args.max_seq_length == 0:
                args.max_seq_length = train_hparams.get("max_seq_length", 0)

            if args.pad_to_multiple_of is None:
                args.pad_to_multiple_of = train_hparams.get("pad_to_multiple_of", None)

            args.train_task = train_hparams.get("train_task", args.train_task)


def main():
    parser = argparse.ArgumentParser()
    add_generic_args(parser)
    CriticTransformer.add_model_specific_args(parser)

    args = parser.parse_args()

    _sanity_check(args)
    _load_default_args(args)

    # If output_dir not provided, a folder will be generated in pwd
    if args.output_dir is None:
        if os.path.exists(args.model_name_or_path):
            args.output_dir = args.model_name_or_path
        else:
            args.output_dir = "./checkpoints"
            os.makedirs(args.output_dir, exist_ok=True)

    if args.do_train:
        args.output_dir = os.path.join(args.output_dir, _get_default_output_dir(args))
        os.makedirs(args.output_dir, exist_ok=True)

    model = CriticTransformer(args)
    data_module = CriticDataModule(model.tokenizer, args)
    data_module.setup("fit")

    extra_callbacks = []

    if args.do_train and args.patience > 0:
        extra_callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="val_metric",
                min_delta=args.min_delta,
                patience=args.patience,
                verbose=False,
                mode="max",
            )
        )

    ckpt_kwargs = {"save_last": True} if args.save_last else {"save_top_k": 1}
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename="pl-{epoch:02d}-{val_metric:.3f}",
        monitor="val_metric",
        mode="max",
        verbose=True,
        **ckpt_kwargs,
    )

    trainer_kwargs = dict(
        reload_dataloaders_every_n_epochs=1,
        weights_summary="top",
    )

    if args.do_train:
        trainer_kwargs["profiler"] = "simple"

    logger = (
        pl_loggers.TensorBoardLogger(
            save_dir=args.output_dir,
            name="train_logs",
            default_hp_metric=False,
        )
        if args.do_train
        else False
    )

    trainer = generic_train(
        model,
        args,
        logger,
        extra_callbacks,
        checkpoint_callback,
        data_module=data_module,
        **trainer_kwargs,
    )

    # Optionally, predict on dev set and write to output_dir
    if args.do_test or args.do_eval:
        checkpoints = list(
            sorted(
                glob.glob(
                    os.path.join(args.output_dir, "checkpointepoch=*.ckpt"),
                    recursive=True,
                )
            )
        )
        if checkpoints:
            model = model.load_from_checkpoint(checkpoints[-1])

        trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
