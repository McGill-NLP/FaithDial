import argparse
import glob
import logging
import sys
import os
import shutil
from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from loss_dropper import LossDropper
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.distributed import rank_zero_info
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, Path(__file__).parent.parent.absolute().as_posix())
from models.dataset import DialogueDataModule, SpecialVocab
from models.lightning_base import generic_train, add_generic_args, BaseTransformer
from models.metrics import LossDropAccumulator
from models.modeling_nce import InfoNCE

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("dialog")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATASET_NAME = "McGill-NLP/FaithDial"


class DialogueTransformer(BaseTransformer):
    def __init__(self, hparams: argparse.Namespace):
        """Initialize a model, tokenizer and config."""
        config = AutoConfig.from_pretrained(
            hparams.config_name if hparams.config_name else hparams.model_name_or_path,
            cache_dir=hparams.cache_dir,
            return_dict=True,
        )

        mode = "summarization" if config.is_encoder_decoder else "language-modeling"

        tokenizer = AutoTokenizer.from_pretrained(
            hparams.tokenizer_name if hparams.tokenizer_name else hparams.model_name_or_path,
            cache_dir=hparams.cache_dir,
            extra_ids=0,
        )

        super().__init__(hparams, mode, tokenizer=tokenizer, return_dict=True)
        self.special_vocab = SpecialVocab(self.tokenizer, self.hparams.ctrl)
        self.special_vocab.add_special_tokens(self.model)

        self.config.pad_token = self.tokenizer.pad_token
        self.config.pad_token_id = self.tokenizer.pad_token_id

        if self.hparams.enable_infonce:
            self.nce_head = InfoNCE(
                self.model,
                self.tokenizer.pad_token_id,
                self.hparams.inbatch_negatives,
                encoder_emb_method=self.hparams.nce_encoder_emb,
                project=self.hparams.nce_project,
            )
        else:
            self.nce_head = None

        if self.hparams.loss_truncation:
            self.dropper = LossDropper(
                dropc=self.hparams.loss_dropc,
                min_count=self.hparams.drop_recompute,
                recompute=self.hparams.drop_recompute,
            )
        else:
            self.dropper = None

    def configure_metrics(self, stage: str):
        if self.hparams.loss_truncation:
            self.dropper_accum = LossDropAccumulator()
        else:
            self.dropper_accum = None

    def training_step(self, batch, batch_nb):
        history_batch = {k: batch.pop(k) for k in list(batch.keys()) if k.startswith("history_")}
        positive_batch = {k: batch.pop(k) for k in list(batch.keys()) if k.startswith("positive_")}
        negative_batch = {k: batch.pop(k) for k in list(batch.keys()) if k.startswith("negative_")}

        output = self.model(**batch)

        if self.hparams.loss_truncation:
            loss_fn = torch.nn.NLLLoss(reduction="none")

            labels = batch["labels"]
            if self.config.is_encoder_decoder:
                lm_loss = loss_fn(
                    F.log_softmax(output.logits.view(-1, output.logits.size(-1)), dim=-1), labels.view(-1)
                )
            else:
                shift_logits = output.logits[..., :-1, :].contiguous().view(-1, output.logits.size(-1))
                shift_labels = labels[..., 1:].contiguous().view(-1)
                lm_loss = loss_fn(F.log_softmax(shift_logits, dim=-1), shift_labels)

            lm_loss = lm_loss.view(batch["input_ids"].shape[0], -1)
            lm_loss = lm_loss.mean(dim=1)
            drop_mask = self.dropper(lm_loss)
            self.dropper_accum(drop_mask)
            self.log("train/drop_mask", (drop_mask == 0).int().sum(), prog_bar=False, reduce_fx=torch.sum)
            self.log("train/acc_drop_mask", self.dropper_accum.compute(), prog_bar=False, reduce_fx=torch.sum)
            lm_loss *= drop_mask
            lm_loss = lm_loss.mean()
        else:
            lm_loss = output.loss

        loss = lm_loss

        if self.hparams.enable_infonce and negative_batch:
            self.log("train/lm_loss", lm_loss, prog_bar=True, logger=True)

            lm_ppl = torch.clamp(torch.exp(lm_loss), max=100, min=0)
            self.log("train/lm_ppl", lm_ppl, prog_bar=False)

            nce_output = self.nce_head(**history_batch, **positive_batch, **negative_batch)
            self.log("train/nce_loss", nce_output.loss, prog_bar=True, logger=True)
            self.log("train/nce_hits-at-1", nce_output.hits_at_1, prog_bar=True, logger=True)
            loss += self.hparams.nce_coef * nce_output.loss

        self.log("train/loss", loss, prog_bar=False)
        ppl = torch.clamp(torch.exp(loss), max=100, min=0)
        self.log("train/ppl", ppl, prog_bar=True, logger=True)

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        self.log("train/lr", lr_scheduler.get_last_lr()[-1], prog_bar=False)
        self.log("lr", lr_scheduler.get_last_lr()[-1], prog_bar=True, logger=False)

        return loss

    def validation_step(self, batch, batch_nb):
        # if we dont send labels to model, it doesnt return losses
        labels = batch.pop("labels", None)
        assert labels is not None

        lm_output = self.model(**batch)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        if self.config.is_encoder_decoder:
            val_loss = loss_fn(lm_output.logits.view(-1, lm_output.logits.size(-1)), labels.view(-1))
        else:
            shift_logits = lm_output.logits[..., :-1, :].contiguous().view(-1, lm_output.logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)
            val_loss = loss_fn(shift_logits, shift_labels)

        num_tokens = (labels > 0).int().sum()
        mean_val_loss = val_loss.detach().cpu() / num_tokens.detach().cpu()

        return mean_val_loss

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def _eval_end(self, outputs, mode: str):
        mean_loss = torch.stack(outputs).mean().detach().cpu()
        self.log(f"{mode}/loss", mean_loss, prog_bar=True)

        ppl = torch.exp(mean_loss)
        self.log(f"{mode}/ppl", ppl, prog_bar=True)

    def validation_epoch_end(self, outputs: List[torch.Tensor]):
        self._eval_end(outputs, "valid")

    def test_epoch_end(self, outputs: List[torch.Tensor]):
        self._eval_end(outputs, "test")

    def on_train_epoch_end(self):
        if self.hparams.loss_truncation:
            rank_zero_info(f"  Total number of dropped examples: {self.dropper_accum.compute()}")
            self.dropper_accum.reset()

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
            "--exclude_knowledge",
            action="store_true",
            default=False,
            help="Whether to exclude knowledge from input sequences.",
        )
        parser.add_argument(
            "--max_negative_samples",
            type=int,
            default=0,
            help="Max number of negative samples",
        )
        parser.add_argument(
            "--enable_infonce",
            action="store_true",
            default=False,
            help="Whether to use InfoNCE",
        )
        parser.add_argument(
            "--nce_coef",
            type=float,
            default=0.1,
            help="NCE loss coefficient",
        )
        parser.add_argument(
            "--inbatch_negatives",
            action="store_true",
            default=False,
            help="Whether to use inbatch negative sampling (only when InfoNCE is enabled)",
        )
        parser.add_argument(
            "--nce_encoder_emb",
            type=str,
            choices=("first_token", "mean_pool", "dec_first"),
            default="first_token",
            help="For encoder-decoder models, how to build history representations from encoder. "
            "See https://arxiv.org/pdf/2108.08877.pdf",
        )

        parser.add_argument(
            "--loss_truncation",
            action="store_true",
            default=False,
            help="Whether to use loss truncation (https://aclanthology.org/2020.acl-main.66/)",
        )
        parser.add_argument(
            "--loss_dropc",
            type=float,
            default=0.4,
            help="Fraction of data for loss truncation",
        )
        parser.add_argument(
            "--drop_recompute",
            type=int,
            default=10000,
            help="Recompute cutoff points every X steps",
        )

        parser.add_argument(
            "--ctrl",
            action="store_true",
            default=False,
            help="Whether to use controlled generation (https://aclanthology.org/2021.acl-long.58/)",
        )


def _sanity_check(args):
    if args.do_train:
        assert 0.0 <= args.warmup_ratio <= 1.0, "`--warmup_ratio` must be in [0, 1]"

        assert not (
            args.do_test or args.do_eval
        ), "`--do_test` or `--do_eval` cannot be done if training is enabled"


def _set_default_args(args):
    args.control_tokens = []
    args.do_generate = False


def main():
    parser = argparse.ArgumentParser()
    add_generic_args(parser)
    DialogueTransformer.add_model_specific_args(parser)
    args = parser.parse_args()

    _sanity_check(args)
    _set_default_args(args)

    if args.output_dir is None:
        if os.path.exists(args.model_name_or_path):
            args.output_dir = args.model_name_or_path
        else:
            args.output_dir = "./checkpoints"
            os.makedirs(args.output_dir, exist_ok=True)

    odir = Path(args.output_dir)
    if args.overwrite_output_dir and odir.exists():
        shutil.rmtree(odir)
    odir.mkdir(parents=True, exist_ok=True)

    model = DialogueTransformer(args)
    data_module = DialogueDataModule(model.special_vocab, args, model.config.is_encoder_decoder)
    data_module.setup("fit")

    extra_callbacks = []
    if args.do_train and args.patience > 0:
        extra_callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="valid/ppl",
                min_delta=args.min_delta,
                patience=args.patience,
                verbose=False,
                mode="min",
            )
        )

    logger = pl_loggers.TensorBoardLogger(
        save_dir=args.output_dir,
        name="train_logs" if args.do_train else "valid_logs",
        default_hp_metric=False,
    )

    checkpoint_kwargs = {}
    if args.save_last:
        checkpoint_kwargs["save_last"] = True
    else:
        checkpoint_kwargs["save_top_k"] = 1

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        monitor="valid/ppl",
        mode="min",
        **checkpoint_kwargs,
    )

    trainer_kwargs = dict(
        reload_dataloaders_every_n_epochs=1,
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

    if args.do_test or args.do_generate:
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-epoch=*.ckpt"), recursive=True)))
        if checkpoints:
            model = model.load_from_checkpoint(checkpoints[-1])

        trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
