# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import logging
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from open_r1.configs import GRPOConfig
from open_r1.prompts import get_system_prompt
from open_r1.rewards import create_reward_functions
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.logging import init_wandb_training

logger = logging.getLogger(__name__)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_configs (`dict[str, dict]`):
            Dict of reward functions and arguments. Valid keys: "short_answer_accuracy", "strict_format", "soft_format".
    """
    system_prompt_name: str = 'simple'
    system_prompt: Optional[str] = None
    data_files: Optional[dict[str, str]] = None
    test_size: float = 0.05
    question_key: str = 'question'
    answer_key: str = 'answer_aliases'
    reward_configs: dict[str, dict] = field(
        default_factory=lambda: {
            'short_answer_accuracy': {
                'scale': 2.0,
                'score': 'exact_match',
                'normalize_answer': True,
            },
            'strict_format': {
                'scale': 0.5,
            },
            'soft_format': {
                'scale': 0.5,
            },
        },
        metadata={
            'help': 'Dict of reward functions and arguments. Valid keys: "short_answer_accuracy", "strict_format", "soft_format".'
        },
    )

    def __post_init__(self):
        if self.system_prompt_name:
            self.system_prompt = get_system_prompt(self.system_prompt_name)


def main(
        script_args: GRPOScriptArguments,
        training_args: GRPOConfig,
        model_args: ModelConfig,
):
    if Path(training_args.output_dir).exists():
        raise ValueError(f'output_dir already exists: {training_args.output_dir}')

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = load_dataset(
        script_args.dataset_name,
        data_files=script_args.data_files,
    )

    # Get reward functions
    reward_funcs = create_reward_functions(script_args.reward_configs)

    # Setup data
    logger.info("*** Load data ***")

    # Format into conversation
    def make_conversation(example):
        return {
            'prompt': [
                {'role': 'system', 'content': script_args.system_prompt},
                {'role': 'user', 'content': example[script_args.question_key]},
            ],
            'answer': example[script_args.answer_key],
        }

    dataset = dataset.map(make_conversation)
    for split in list(dataset.keys()):
        column_names = dataset[split].column_names
        column_names = [name for name in column_names if name not in ('prompt', 'answer')]
        dataset[split] = dataset[split].remove_columns(column_names)

    if 'test' not in dataset:
        dataset = dataset['train'].train_test_split(
            test_size=script_args.test_size,
            shuffle=False,
        )

    for k, d in dataset.items():
        logger.info(f"{k} size: {len(d)}")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
