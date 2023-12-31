# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import logging
import math
import os
import random
import re
import time
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from math import floor, ceil

import numpy as np
import mindspore
from packaging import version
from mindspore import nn
from mindspore.utils.data.dataloader import DataLoader
from mindspore.utils.data.dataset import Dataset
from mindspore.utils.data.distributed import DistributedSampler
from mindspore.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from tqdm.auto import tqdm, trange
import transformers
from transformers import AutoModelForSequenceClassification
from transformers.data.data_collator import DataCollator,default_data_collator
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

import utils
from training_args import TrainingArguments, is_tpu_available
from trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, PredictionOutput, TrainOutput

from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer
import mindspore.nn.functional as F

import textattack
import csv
from textattack.models.wrappers.huggingface_model_wrapper import HuggingFaceModelWrapper
from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
from textattack.datasets import HuggingFaceDataset
from textattack.attack_results import SuccessfulAttackResult, MaximizedAttackResult, FailedAttackResult
import copy

try:
    from apex import amp
    _has_apex = True
except ImportError:
    _has_apex = False


def is_apex_available():
    return _has_apex


if is_tpu_available():
    import mindspore_xla.core.xla_model as xm
    import mindspore_xla.debug.metrics as met
    import mindspore_xla.distributed.parallel_loader as pl

try:
    from mindspore.utils.tensorboard import SummaryWriter
    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter
        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


def is_tensorboard_available():
    return _has_tensorboard


try:
    import wandb
    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn("W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except ImportError:
    _has_wandb = False


def is_wandb_available():
    return _has_wandb


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    mindspore.manual_seed(seed)
    mindspore.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


@contextmanager
def mindspore_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        mindspore.distributed.barrier()
    yield
    if local_rank == 0:
        mindspore.distributed.barrier()


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not mindspore.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = mindspore.distributed.get_world_size()
        if rank is None:
            if not mindspore.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = mindspore.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def get_tpu_sampler(dataset: Dataset):
    if xm.xrt_world_size() <= 1:
        return RandomSampler(dataset)
    return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())


class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for mindspore,
    optimized for Transformers.
    """

    model: PreTrainedModel
    args: TrainingArguments
    data_collator: DataCollator
    train_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool
    tb_writer: Optional["SummaryWriter"] = None
    optimizers: Tuple[mindspore.optim.Optimizer, mindspore.optim.lr_scheduler.LambdaLR] = None
    global_step: Optional[int] = None
    epoch: Optional[float] = None

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        prediction_loss_only=False,
        tb_writer: Optional["SummaryWriter"] = None,
        optimizers: Tuple[mindspore.optim.Optimizer, mindspore.optim.lr_scheduler.LambdaLR] = None,
    ):
        """
        Trainer is a simple but feature-complete training and eval loop for mindspore,
        optimized for Transformers.

        Args:
            prediction_loss_only:
                (Optional) in evaluation and prediction, only return the loss
        """
        self.model = model.to(args.device)
        self.args = args
        if data_collator is not None:
            self.data_collator = data_collator
        else:
            self.data_collator = default_data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.prediction_loss_only = prediction_loss_only
        self.optimizers = optimizers
        if tb_writer is not None:
            self.tb_writer = tb_writer
        elif is_tensorboard_available() and self.is_world_master():
            self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
        if not is_tensorboard_available():
            logger.warning(
                "You are instantiating a Trainer but Tensorboard is not installed. You should consider installing it."
            )
        if is_wandb_available():
            self._setup_wandb()
        else:
            logger.info(
                "You are instantiating a Trainer but W&B is not installed. To use wandb logging, "
                "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
            )
        set_seed(self.args.seed)
        # Create output directory if needed
        if self.is_world_master():
            os.makedirs(self.args.output_dir, exist_ok=True)
        if is_tpu_available():
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            self.model.config.xla_device = True

    def isTwoModelsEq(self,model1,model2):
        d1 = model1.state_dict()
        d2 = model2.state_dict()
        for k,v in d1.items():
            if not d2[k].equal(v):
                return False
        return True
    def attack(self,training_args,data_args,my_args,model,tokenizer):
        model_backup = copy.deepcopy(model)
        eq = self.isTwoModelsEq(model,model_backup)
        model_backup.load_state_dict(model.state_dict())
        eq = self.isTwoModelsEq(model,model_backup)
        training_args_backup = copy.deepcopy(training_args)
        logger.info("Attacking after training!")
        training_args_backup.do_lower_case = True
        training_args_backup.results_file = 'attack_log.csv'
        # training_args_backup.dataset_name = 'glue'
        training_args_backup.task_name = data_args.task_name
        # training_args_backup.valid = 'validation'
        training_args_backup.num_examples = my_args.num_examples  # 1000
        # training_args_backup.num_examples = 100  # 1000
        training_args_backup.seed = 42
        training_args_backup.save_perturbed = 0
        training_args_backup.perturbed_file = "bert_textfooler.csv"
        model_wrapper = HuggingFaceModelWrapper(model_backup, tokenizer)
        attack = TextFoolerJin2019.build(model_wrapper)
        dataset = HuggingFaceDataset(my_args.dataset_name,
                                     subset="sst2" if data_args.task_name == "sst-2" else data_args.task_name,
                                     split=data_args.valid)
        # for attack
        attack_args = textattack.AttackArgs(num_examples=training_args_backup.num_examples,
                                            disable_stdout=True, random_seed=training_args_backup.seed)
        attacker = textattack.Attacker(attack, dataset, attack_args)
        num_results = 0
        num_successes = 0
        num_failures = 0

        if training_args_backup.save_perturbed:
            with open(training_args_backup.perturbed_file, 'w', encoding='utf-8', newline="") as f:
                csv_writer = csv.writer(f, delimiter='\t')
                csv_writer.writerow(['sentence', 'label'])
                f.close()

        # attacking
        for result in attacker.attack_dataset():
            logger.info(result)
            num_results += 1
            if (
                    type(result) == SuccessfulAttackResult
                    or type(result) == MaximizedAttackResult
            ):
                num_successes += 1
            if type(result) == FailedAttackResult:
                num_failures += 1

            if training_args_backup.save_perturbed:
                with open(training_args_backup.perturbed_file, 'a', encoding='utf-8', newline="") as f:
                    csv_writer = csv.writer(f, delimiter='\t')
                    csv_writer.writerow(
                        [result.perturbed_result.attacked_text.text, result.perturbed_result.ground_truth_output])

        logger.info("[Succeeded / Failed / Total] {} / {} / {}".format(num_successes, num_failures, num_results))

        # compute metric
        original_accuracy = (num_successes + num_failures) * 100.0 / num_results
        accuracy_under_attack = num_failures * 100.0 / num_results
        attack_succ = (original_accuracy - accuracy_under_attack) * 100.0 / original_accuracy
        # out_csv = open(training_args.results_file, 'a', encoding='utf-8', newline="")
        # csv_writer = csv.writer(out_csv)
        # csv_writer.writerow([training_args.model_name_or_path, original_accuracy, accuracy_under_attack, attack_succ])
        # out_csv.close()
        logger.info("[Accuracy / Aua / Attack_success] {} / {} / {}".format(original_accuracy, accuracy_under_attack,
                                                                            attack_succ))

        eq = self.isTwoModelsEq(model,model_backup)
        return





    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if is_tpu_available():
            train_sampler = get_tpu_sampler(self.train_dataset)
        else:
            train_sampler = (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator.collate_batch,
        )

        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_tpu_available():
            sampler = SequentialDistributedSampler(
                eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(eval_dataset)
        else:
            sampler = SequentialSampler(eval_dataset)

        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator.collate_batch,
        )

        return data_loader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        # We use the same batch_size as for eval.
        if is_tpu_available():
            sampler = SequentialDistributedSampler(
                test_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(test_dataset)
        else:
            sampler = SequentialSampler(test_dataset)

        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator.collate_batch,
        )

        return data_loader

    def get_optimizers(
        self, num_training_steps: int
    ) -> Tuple[mindspore.optim.Optimizer, mindspore.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.
        """
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def _setup_wandb(self):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can override this method to customize the setup if needed.  Find more information at https://docs.wandb.com/huggingface
        You can also override the following environment variables:

        Environment:
            WANDB_WATCH:
                (Optional, ["gradients", "all", "false"]) "gradients" by default, set to "false" to disable gradient logging
                or "all" to log gradients and parameters
            WANDB_PROJECT:
                (Optional): str - "huggingface" by default, set this to a custom string to store results in a different project
            WANDB_DISABLED:
                (Optional): boolean - defaults to false, set to "true" to disable wandb entirely
        """
        logger.info('Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"')
        wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"), config=vars(self.args))
        # keep track of model topology and gradients
        if os.getenv("WANDB_WATCH") != "false":
            wandb.watch(
                self.model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, self.args.logging_steps)
            )

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get num of examples from a DataLoader, by accessing its Dataset.
        """
        return len(dataloader.dataset)

    def train(self,train_dataloader, eval_dataloader, train_dataset=None, model_path: Optional[str] = None, **kwargs):
        """
        Main training entry point.
        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
        # attack every epoch
        # attack_every_epoch = kwargs.get('attack_every_epoch', False)
        tokenizer = kwargs.get('tokenizer', None)
        training_args = kwargs.get('training_args', None)
        data_args = kwargs.get('data_args', None)
        my_args = kwargs.get('my_args', None)

        # For recording the learnable coefficients for self-attention heads and
        # intermediate neurons during the searching stage of EarlyBERT
        save_self_slimming = kwargs.get('save_self_slimming', False)
        save_inter_slimming = kwargs.get('save_inter_slimming', False)
        config = kwargs.get('config')
        # Initialize the list for recording the learnable coefficients in EarlyBERT.
        # Separately record the coefficients in different layers.
        if save_self_slimming:
            self_slimming_coef_records = [[] for _ in range(config.num_hidden_layers)]
        else:
            self_slimming_coef_records = []
        if save_inter_slimming:
            inter_slimming_coef_records = [[] for _ in range(config.num_hidden_layers)]
        else:
            inter_slimming_coef_records = []

        lottery_ticket_training = kwargs.get('lottery_ticket_training', False)

        train_dataloader = train_dataloader
        # train_dataloader = self.get_train_dataloader()
        int_num_train_epochs = round(self.args.num_train_epochs)
        # If `args.num_train_epochs` is a non-integer float, calculate the max number
        # of training steps and use it to early-stop the training.
        if abs(int_num_train_epochs - self.args.num_train_epochs) > 1e-3:
            # self.args.max_steps = round(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            # align with first stage
            self.args.max_steps = len(train_dataset) * self.args.num_train_epochs // self.args.train_batch_size
        # We can also directly set `args.max_steps` from the command line argument.
        # This will overwrites the `args.num_train_epochs` argument.
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
            and not lottery_ticket_training
            # We skip loading the optimizer and scheduler states for the
            # efficient training phase of EarlyBERT .
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                mindspore.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(mindspore.load(os.path.join(model_path, "scheduler.pt")))
        model = self.model

        # Train!
        total_train_batch_size = (
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (mindspore.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

            if lottery_ticket_training:
                self.global_step = 0
                logger.info("  Starting lottery-ticket training")

        if not lottery_ticket_training:
            output_dir = os.path.join(self.args.output_dir, "checkpoint-0")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            tokenizer = kwargs.get('tokenizer')
            tokenizer.save_pretrained(output_dir)

            mindspore.save(self.args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)

            mindspore.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            mindspore.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)
        # Save the model initialization for LT
        # In all cases (even distributed/parallel), self.model is always a reference
        # to the model we want to save.
        if hasattr(model, "module"):
            assert model.module is self.model
        else:
            assert model is self.model

        output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")
        self.save_model(output_dir)

        if self.is_world_master():
            self._rotate_checkpoints()

        elif self.is_world_master():
            mindspore.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            mindspore.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, ceil(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )
        for epoch in train_iterator:
            avg_loss = utils.ExponentialMovingAverage()

            if (self.args.max_epochs > 0 and epoch >= self.args.max_epochs) or \
               (self.args.max_steps  > 0 and self.global_step >= self.args.max_steps):
                train_iterator.close()
                break

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master())

            for step, inputs in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                tr_tmp_loss = self._training_step(model, inputs, optimizer, **kwargs)
                batch_loss = tr_tmp_loss
                avg_loss.update(batch_loss)

                tr_loss+=tr_tmp_loss

                # used in search stage
                if save_self_slimming:
                    idx_layer = 0
                    for m in model.modules():
                        if isinstance(m, BertSelfAttention) and m.self_slimming:
                            self_slimming_coef_records[idx_layer].append(m.slimming_coef.detach().cpu().numpy().reshape(-1))
                            idx_layer += 1
                if save_inter_slimming:
                    idx_layer = 0
                    for m in model.modules():
                        if isinstance(m, BertLayer) and m.inter_slimming:
                            inter_slimming_coef_records[idx_layer].append(m.slimming_coef.detach().cpu().numpy().reshape(-1))
                            idx_layer += 1

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):

                    mindspore.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    # update
                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + float(step + 1) / len(epoch_iterator)
                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                        # backward compatibility for mindspore schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(mindspore.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss = tr_loss

                        self._log(logs)
                        # todo modified
                        if self.args.evaluate_during_training:
                            eval_result = self.evaluate(eval_dataloader=eval_dataloader)
                            logger.info("***** Eval results at global step {} *****".format(self.global_step))
                            for key, value in eval_result.items():
                                logger.info("  %s = %s", key, value)


                epoch_iterator.set_description(f'epoch: {epoch: d}, '
                                     f'loss: {avg_loss.get_metric(): 0.4f}, '
                                     f'lr: {optimizer.param_groups[0]["lr"]: .3e}')
                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break


            # if attack_every_epoch:
            #     logger.info("Saving after training epoch{}".format(epoch))
            #     print("Saving after training epoch{}".format(epoch))
            #     mindspore.save(model.state_dict(), self.args.output_dir + "model_epoch{}.pt".format(epoch))

            # Save checkpoints at the end of each epoch
            # In all cases (even distributed/parallel), self.model is always a reference
            # to the model we want to save.
            if hasattr(model, "module"):
                assert model.module is self.model
            else:
                assert model is self.model
            # Save model checkpoint

            output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{epoch+1}epoch")
            self.save_model(output_dir)
            # tokenizer.save_pretrained(output_dir)

            if self.is_world_master():
                self._rotate_checkpoints()

            elif self.is_world_master():
                mindspore.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                mindspore.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        if self.tb_writer:
            self.tb_writer.close()
        # epoch_s = [6,7,8,9,10]
        # if attack_every_epoch:
        #     for i in range(int(self.args.num_train_epochs)):
        #         # if i not in epoch_s:
        #         #     continue
        #         logger.info("Attacking after training epoch{}".format(i))
        #         print("Attacking after training epoch{}".format(i))
        #         model.load_state_dict(mindspore.load(self.args.output_dir + "model_epoch{}.pt".format(i)))
        #         self.attack(training_args, data_args, my_args, model, tokenizer)


        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step,
                           tr_loss / self.global_step,
                           self_slimming_coef_records,
                           inter_slimming_coef_records)


    def _log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.tb_writer:
            for k, v in logs.items():
                self.tb_writer.add_scalar(k, v, self.global_step)
        if is_wandb_available():
            wandb.log(logs, step=self.global_step)
        output = json.dumps({**logs, **{"step": self.global_step}})
        if iterator is not None:
            iterator.write(output)
        else:
            print(output)


    def _training_step(
        self, model: nn.Module, inputs: Dict[str, mindspore.Tensor], optimizer: mindspore.optim.Optimizer, **kwargs
    ) -> float:

        model.train()
        model_inputs = inputs[0]
        labels = inputs[1]
        model_inputs = {k: v.to(self.args.device) for k, v in model_inputs.items()}
        labels = labels.to(self.args.device)
        model_inputs['labels'] = labels
        inputs = model_inputs
        # for k, v in inputs.items():
        #     inputs[k] = v.to(self.args.device)
        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        lottery_ticket_training = kwargs.get('lottery_ticket_training', 0.0)
        # l1 loss
        if not lottery_ticket_training:
            l1_loss_self_coef = kwargs.get('l1_loss_self_coef', 0.0)
            if l1_loss_self_coef > 0.0:
                l1_self_loss = 0.0
                for m in model.modules():
                    if isinstance(m, BertSelfAttention) and m.self_slimming:
                        l1_self_loss += m.slimming_coef.abs().sum()
                loss += l1_self_loss * l1_loss_self_coef

            l1_loss_inter_coef = kwargs.get('l1_loss_inter_coef', 0.0)
            if l1_loss_inter_coef > 0.0:
                l1_inter_loss = 0.0
                for m in model.modules():
                    if isinstance(m, BertLayer) and m.inter_slimming:
                        l1_inter_loss += m.slimming_coef.abs().sum()
                loss += l1_inter_loss * l1_loss_inter_coef
        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()

    def is_local_master(self) -> bool:
        if is_tpu_available():
            return xm.is_master_ordinal(local=True)
        else:
            return self.args.local_rank in [-1, 0]

    def is_world_master(self) -> bool:
        """
        This will be True only in one process, even in distributed mode,
        even when training on multiple machines.
        """
        if is_tpu_available():
            return xm.is_master_ordinal(local=False)
        else:
            return self.args.local_rank == -1 or mindspore.distributed.get_rank() == 0
 
    def save_model(self, output_dir: Optional[str] = None):
        """
        Saving best-practices: if you use default names for the model,
        you can reload it using from_pretrained().

        Will only save from the world_master process (unless in TPUs).
        """

        if is_tpu_available():
            self._save_tpu(output_dir)
        elif self.is_world_master():
            self._save(output_dir)

    def _save_tpu(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        logger.info("Saving model checkpoint to %s", output_dir)

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            mindspore.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")

        xm.rendezvous("saving_checkpoint")
        self.model.save_pretrained(output_dir)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # if not isinstance(self.model, PreTrainedModel):
        #     raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        mindspore.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def evaluate(
        self, eval_dataloader, eval_dataset: Optional[Dataset] = None, prediction_loss_only: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent.

        Args:
            eval_dataset: (Optional) Pass a dataset if you wish to override
            the one on the instance.
        Returns:
            A dict containing:
                - the eval loss
                - the potential metrics computed from the predictions
        """
        # eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_dataloader = eval_dataloader

        output = self._prediction_loop(eval_dataloader, description="Evaluation")

        self._log(output.metrics)

        if self.args.tpu_metrics_debug:
            # tpu-comment: Logging debug metrics for mindspore/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output.metrics

    def predict(self, test_dataset: Dataset) -> PredictionOutput:
        """
        Run prediction and return predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in evaluate().
        """
        test_dataloader = self.get_test_dataloader(test_dataset)

        return self._prediction_loop(test_dataloader, description="Prediction")

    def _prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = mindspore.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in mindspore.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: mindspore.Tensor = None
        label_ids: mindspore.Tensor = None
        model.eval()

        if is_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        for inputs in tqdm(dataloader, desc=description):

            # modified
            # for k, v in inputs.items():
            #     inputs[k] = v.to(self.args.device)
            model_inputs = inputs[0]
            labels = inputs[1]
            model_inputs = {k: v.to(self.args.device) for k, v in model_inputs.items()}
            labels = labels.to(self.args.device)
            model_inputs['labels'] = labels
            inputs = model_inputs
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])


            with mindspore.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach()
                else:
                    preds = mindspore.cat((preds, logits.detach()), dim=0)
                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["labels"].detach()
                    else:
                        label_ids = mindspore.cat((label_ids, inputs["labels"].detach()), dim=0)

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = self.distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = self.distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
        elif is_tpu_available():
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            if preds is not None:
                preds = xm.mesh_reduce("eval_preds", preds, mindspore.cat)
            if label_ids is not None:
                label_ids = xm.mesh_reduce("eval_label_ids", label_ids, mindspore.cat)

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def distributed_concat(self, tensor: mindspore.Tensor, num_total_examples: int) -> mindspore.Tensor:
        assert self.args.local_rank != -1

        output_tensors = [tensor.clone() for _ in range(mindspore.distributed.get_world_size())]
        mindspore.distributed.all_gather(output_tensors, tensor)

        concat = mindspore.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        output = concat[:num_total_examples]
        return output


