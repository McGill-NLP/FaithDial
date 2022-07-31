"""
Edited from https://github.com/huggingface/transformers/blob/v3.4.0/examples/lightning_base.py
"""
import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Union, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.utilities.enums import DistributedType
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelForCausalLM,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
}


# update this and the import above to support new schedulers from transformers.optimization
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule_with_warmup,  # not supported for now
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


class LiteProgressBar(pl.callbacks.progress.TQDMProgressBar):
    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__(refresh_rate, process_position)

    def get_metrics(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> Dict[str, Union[int, str]]:
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items


class BaseTransformer(pl.LightningModule):
    def __init__(
        self,
        hparams: argparse.Namespace,
        mode="base",
        config=None,
        tokenizer=None,
        model=None,
        **config_kwargs,
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        self.save_hyperparameters(hparams)

        self.output_dir = Path(self.hparams.output_dir)
        if self.hparams.do_train:
            save_hparams_to_yaml(str(self.output_dir / "hparams.yaml"), self.hparams)
        cache_dir = self.hparams.cache_dir
        if config is None:
            self.config = AutoConfig.from_pretrained(
                self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
                cache_dir=cache_dir,
                **config_kwargs,
            )
        else:
            self.config: PretrainedConfig = config

        extra_model_params = (
            "encoder_layerdrop",
            "decoder_layerdrop",
            "dropout",
            "dropout_rate",
            "attention_dropout",
        )
        for p in extra_model_params:
            if getattr(self.hparams, p, None) and hasattr(self.config, p):
                setattr(self.config, p, getattr(self.hparams, p))

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
                cache_dir=cache_dir,
            )
        else:
            self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = tokenizer
        self.model_type = MODEL_MODES[mode]
        if model is None:
            self.model = self.model_type.from_pretrained(
                self.hparams.model_name_or_path,
                from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
                config=self.config,
                cache_dir=cache_dir,
            )
        else:
            self.model = model

    def load_hf_checkpoint(self, *args, **kwargs):
        self.model = self.model_type.from_pretrained(*args, **kwargs)

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        total_steps = self.total_steps()
        if self.hparams.warmup_ratio > 0:
            warmup_steps = self.hparams.warmup_ratio * total_steps
        else:
            warmup_steps = self.hparams.warmup_steps

        if self.hparams.lr_scheduler != "constant":
            scheduler = get_schedule_func(self.opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        else:
            scheduler = get_schedule_func(self.opt, num_warmup_steps=warmup_steps)

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                scale_parameter=False,
                relative_step=False,
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                eps=self.hparams.adam_epsilon,
            )
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def get_number_of_gpus(self):
        if self.hparams.gpus == -1:
            return torch.cuda.device_count()
        else:
            return self.hparams.gpus

    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.get_number_of_gpus())
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        return (self.dataset_size / effective_batch_size) * self.hparams.max_epochs

    def setup(self, stage: str):
        if stage in ("generate", "predict"):
            self.dataset_size = len(self.trainer.datamodule.datasets["generate"])
        elif stage == "test":
            self.dataset_size = len(self.trainer.datamodule.test_dataloader().dataset)
        elif stage in ("train", "fit"):
            self.dataset_size = len(self.trainer.datamodule.datasets["train"])

        self.configure_metrics(stage)

    def configure_metrics(self, stage: str) -> Optional[Any]:
        pass

    @property
    def use_ddp(self):
        return self.trainer._distrib_type in (DistributedType.DDP, DistributedType.DDP_SPAWN)

    def get_dataloader(
        self,
        mode: str,
        batch_size: int,
        shuffle: bool = False,
        data_path: Optional[str] = None,
        dataset=None,
        **kwargs,
    ):
        if dataset is None:
            assert data_path is not None
            dataset = self.get_dataset(mode, data_path, **kwargs)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.get_collator(mode),
            # num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            "train",
            self.hparams.train_batch_size,
            shuffle=True,
            dataset=self.train_dataset,
        )

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            "valid",
            self.hparams.eval_batch_size,
            shuffle=False,
            data_path=self.hparams.eval_dataset_path,
        )

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            "test", self.hparams.eval_batch_size, shuffle=False, data_path=self.hparams.test_dataset_path
        )

    def predict_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            "predict", self.hparams.eval_batch_size, shuffle=False, data_path=self.hparams.predict_dataset_path
        )

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        rank_zero_info(f"Saving a model at step={self.global_step} in epoch={self.current_epoch}...")
        save_path = self.output_dir.joinpath("best_model")
        self.model.config.save_step = self.global_step
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            required=True,
            type=str,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument(
            "--config_name",
            default="",
            type=str,
            help="Pretrained config name or path if not the same as model_name",
        )
        parser.add_argument(
            "--tokenizer_name",
            default=None,
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default=None,
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument(
            "--dropout_rate",
            type=float,
            help="Dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--attention_dropout",
            type=float,
            help="Attention dropout probability (Optional). Goes into model.config",
        )
        ### Optimization
        parser.add_argument(
            "--learning_rate",
            default=6.25e-5,
            type=float,
            help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--lr_scheduler",
            default="linear",
            choices=arg_to_scheduler_choices,
            metavar=arg_to_scheduler_metavar,
            type=str,
            help="Learning rate scheduler",
        )
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument(
            "--warmup_ratio",
            default=0.0,
            type=float,
            help="Linear warmup proportional to train steps. Overrides `--warmup_steps`.",
        )
        parser.add_argument(
            "--max_grad_norm",
            dest="gradient_clip_val",
            default=1.0,
            type=float,
            help="Max gradient norm",
        )
        parser.add_argument("--adafactor", action="store_true")
        parser.add_argument(
            "--gradient_accumulation_steps",
            dest="accumulate_grad_batches",
            type=int,
            default=4,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )

        parser.add_argument("--num_workers", default=8, type=int, help="Number of workers, passed to DataLoader")
        parser.add_argument("--num_train_epochs", dest="max_epochs", default=10, type=int)
        parser.add_argument("--train_batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)
        parser.add_argument(
            "--gpus",
            default=-1,
            type=int,
            help="The number of GPUs allocated for this, it is by default -1 meaning all available",
        )
        parser.add_argument(
            "--fp16",
            action="store_true",
            help="Whether to use 16-bit (mixed) precision instead of 32-bit",
        )
        parser.add_argument(
            "--eval_interval",
            dest="val_check_interval",
            type=float,
            default=0.5,
            help="Run an evaluation X times (float) within an epoch",
        )
        parser.add_argument(
            "--pad_to_multiple_of",
            type=int,
            default=None,
            help="Pad sequence to multiple of the given value. ",
        )

        # Early Stopping
        parser.add_argument(
            "--patience",
            type=int,
            default=5,
            help="Number of validation steps to wait if no improvement and then stop the training (for early stopping).",
        )
        parser.add_argument(
            "--min_delta",
            type=float,
            default=0.0,
            help="An absolute change of less than `min_delta`, will count as no improvement (for early stopping).",
        )


class LoggingCallback(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        lr_scheduler = trainer.lr_schedulers[0]["scheduler"]
        lrs = {f"lr/group_{i}": lr for i, lr in enumerate(lr_scheduler.get_last_lr())}
        pl_module.logger.log_metrics(lrs, trainer.global_step)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Validation results *****")

        for cl in trainer.callbacks:
            if isinstance(cl, pl.callbacks.EarlyStopping):
                rank_zero_info(f"early stop {cl.wait_count}/{cl.patience} - best = {cl.best_score.item()}")
                break

        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                rank_zero_info("{} = {}".format(key, str(metrics[key])))
        rank_zero_info("")

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Test results *****")
        metrics = trainer.callback_metrics
        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def add_generic_args(parser):
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run evaluation on the test set.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="enables cudnn.deterministic for reproducibility.",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        default=False,
        help="Whether to overwrite the model's directory.",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=0,
        help="Max sequence length (larger samples will be excluded from data).",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path of the checkpoint directories.",
    )

    parser.add_argument(
        "--save_last",
        action="store_true",
        default=False,
        help="Whether to save last model",
    )

    parser.add_argument("--train_dataset_path", type=str, default=None, help="Path or url of the train dataset.")
    parser.add_argument("--eval_dataset_path", type=str, default=None, help="Path or url of the validation dataset.")
    parser.add_argument("--test_dataset_path", type=str, default=None, help="Path or url of the test dataset.")
    parser.add_argument("--predict_dataset_path", type=str, default=None, help="Path or url of the predict dataset.")


def generic_train(
    model: BaseTransformer,
    args: argparse.Namespace,
    logger: Union[pl_loggers.LightningLoggerBase, bool] = True,  # can pass WandbLogger() here
    extra_callbacks=None,
    checkpoint_callback=None,
    logging_callback=None,
    data_module=None,
    **extra_train_kwargs,
):
    extra_callbacks = extra_callbacks or []

    pl.seed_everything(args.seed)

    # add custom checkpoints
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
    if logging_callback is None:
        logging_callback = LoggingCallback()

    train_params = extra_train_kwargs or {}

    if args.fp16:
        train_params["precision"] = 16

    if args.gpus > 1 or (args.gpus == -1 and torch.cuda.device_count() > 1):
        train_params["strategy"] = "ddp"

    train_params["accumulate_grad_batches"] = args.accumulate_grad_batches

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[logging_callback, checkpoint_callback, LiteProgressBar()] + extra_callbacks,
        logger=logger,
        **train_params,
    )

    if args.do_train:
        trainer.fit(model, datamodule=data_module)

    return trainer
