#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The Numina Team. All rights reserved.
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
"""
Script to instruction fine-tune causal language models on a Hub dataset

Adapted from huggingface/transformers: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""
import logging
import math
import random
import sys

import datasets
import torch
import transformers
import wandb
from accelerate import Accelerator
from aimo.configs import DataConfig, ModelConfig, SFTConfig
from aimo.utils import (
    H4ArgumentParser,
    apply_chat_template,
    check_hub_revision_exists,
    get_checkpoint,
    get_datasets,
    get_tokenizer,
    hf_login,
    init_wandb_training,
)
from training.aimo.utils.callbacks import LoraMergeCallback # Added import
from transformers import set_seed, BitsAndBytesConfig, AutoModelForCausalLM
# Import PeftModel and LoraConfig related classes
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

logger = logging.getLogger(__name__)


def main():
    # Accelerator is not strictly needed for single-GPU if not using its launch features
    # but SFTConfig might rely on its presence for some defaults.
    # If running directly, accelerator.is_main_process will be True.
    accelerator = Accelerator()

    parser = H4ArgumentParser((ModelConfig, DataConfig, SFTConfig))
    model_config, data_config, sft_config = parser.parse()
    
    # Conditionally check Hub revision
    if sft_config.push_to_hub and sft_config.hub_model_id:
        logger.info("push_to_hub is True and hub_model_id is set, checking Hub revision.")
        check_hub_revision_exists(sft_config)
    else:
        logger.info("Skipping Hub revision check (push_to_hub is False or hub_model_id not set).")

    set_seed(sft_config.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = sft_config.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {sft_config.local_rank}, device: {sft_config.device}, n_gpu: {sft_config.n_gpu}"
        + f" distributed training: {bool(sft_config.local_rank != -1)}, 16-bits training: {sft_config.fp16}"
    )
    logger.info(f"Model parameters {model_config}")
    logger.info(f"Data parameters {data_config}")
    logger.info(f"Training/evaluation parameters {sft_config}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(sft_config)
    if last_checkpoint is not None and sft_config.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Setup WandB
    if sft_config.wandb_enabled:
        init_wandb_training(sft_config)

    # Login to HuggingFace Hub if needed
    if sft_config.push_to_hub:
        logger.info("push_to_hub is True, attempting Hugging Face login (if HF_TOKEN is set).")
        hf_login() # This function itself only logs in if HF_TOKEN is set
    else:
        logger.info("Skipping Hugging Face login (push_to_hub is False).")

    ###############
    # Load datasets
    ###############
    logger.info("*** Load datasets ***")

    raw_datasets = get_datasets(
    data_config,
    splits=data_config.dataset_splits,
    )
    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_config, data_config, set_pad_token=sft_config.packing)

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    
    # Determine compute dtype for BitsAndBytesConfig
    compute_dtype_torch = getattr(torch, model_config.bnb_4bit_compute_dtype, torch.float16)
    logger.info(f"Using bnb_4bit_compute_dtype: {model_config.bnb_4bit_compute_dtype} ({compute_dtype_torch})")


    bits_and_bytes_config = None
    if model_config.load_in_4bit:
        bits_and_bytes_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype_torch,
        )
        logger.info(f"BitsAndBytesConfig created: {bits_and_bytes_config}")
    else:
        logger.info("4-bit quantization (load_in_4bit) is NOT enabled in ModelConfig.")

    # Load the base model
    logger.info(f"Loading base model: {model_config.model_name_or_path}")
    base_model_torch_dtype = compute_dtype_torch if model_config.load_in_4bit else getattr(torch, model_config.torch_dtype, None)

    loaded_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation if model_config.attn_implementation else None,
        torch_dtype=base_model_torch_dtype,
        quantization_config=bits_and_bytes_config,
        use_cache=False if sft_config.gradient_checkpointing else True,
        device_map={"": accelerator.device} if not bits_and_bytes_config else None # device_map for single GPU if not quantized, None for quantized and let Trainer handle.
                                                                                  # For BNB quantized models, device_map="auto" or specific device is often used,
                                                                                  # but SFTTrainer should handle this too. Explicitly setting for main process.
    )
    logger.info(f"Base model loaded. Device: {loaded_model.device}")
    if hasattr(loaded_model, 'is_loaded_in_4bit') and loaded_model.is_loaded_in_4bit:
        logger.info("Model successfully loaded in 4-bit.")
    elif model_config.load_in_4bit:
        logger.warning("Model was configured for 4-bit loading, but is_loaded_in_4bit is not true or not present.")


    # Prepare model for k-bit training if quantized (e.g. gradient checkpointing, layer norm casting)
    if model_config.load_in_4bit or model_config.load_in_8bit: #Though 8bit is not used here
        logger.info("Preparing k-bit model for training (e.g., gradient checkpointing compatibility).")
        loaded_model = prepare_model_for_kbit_training(
            loaded_model, use_gradient_checkpointing=sft_config.gradient_checkpointing
        )
        logger.info("Model prepared for k-bit training.")


    final_model_to_train = loaded_model # Start with the (potentially quantized) base model

    if model_config.use_peft:
        logger.info("*** Initializing and Applying PEFT (LoRA/QDora) Config ***")
        if not model_config.lora_target_modules:
            logger.error("FATAL: lora_target_modules is not set in ModelConfig. PEFT requires target modules.")
            raise ValueError("lora_target_modules must be specified for PEFT fine-tuning.")
        
        peft_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            target_modules=model_config.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            use_dora=model_config.lora_use_dora,
            bias="none", # Common choice: "none", "all", "lora_only"
        )
        logger.info(f"PEFT Config: {peft_config}")
        
        final_model_to_train = get_peft_model(loaded_model, peft_config)
        logger.info("PEFT adapters applied to the model.")
        final_model_to_train.print_trainable_parameters() # Crucial for debugging
    elif model_config.load_in_4bit:
        # This case should now be caught by the ValueError if PEFT isn't used with a quantized model.
        logger.error("Model is loaded in 4-bit but PEFT is not enabled. This will lead to an error during training.")
        # The Trainer will raise the error you saw.

    # Apply chat template to datasets
    raw_datasets = raw_datasets.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer, "task": "sft"})

    # Initialize the custom callback list
    custom_callbacks = []
    if sft_config.enable_periodic_lora_merge:
        if sft_config.lora_merge_steps is not None and sft_config.lora_merge_steps > 0:
            logger.info(f"Initializing LoraMergeCallback: enabled with merge interval {sft_config.lora_merge_steps} steps.")
            lora_merge_callback = LoraMergeCallback(
                model_config=model_config, 
                sft_config=sft_config, # sft_config is the TrainingArguments instance
                merge_interval=sft_config.lora_merge_steps
            )
            custom_callbacks.append(lora_merge_callback)
        else:
            logger.warning("LoraMergeCallback is enabled but 'lora_merge_steps' is not a positive integer. Callback will not be added.")
    else:
        logger.info("LoraMergeCallback is not enabled ('enable_periodic_lora_merge' is False).")

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    with sft_config.main_process_first(desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the processed training set:\n\n{train_dataset[index]['text']}")
    
    logger.info(f"Initializing SFTTrainer with the prepared model.")
    trainer = SFTTrainer(
        model=final_model_to_train,     # Pass the (PeftModel) instance
        args=sft_config,
        train_dataset=train_dataset if sft_config.do_train else None,
        eval_dataset=eval_dataset if sft_config.do_eval else None,
        dataset_text_field="text",
        max_seq_length=data_config.block_size,
        tokenizer=tokenizer,
        packing=sft_config.packing,
        callbacks=custom_callbacks,  # Pass the custom_callbacks list here
        # peft_config is not needed here as the model is already a PeftModel
    )

    ###############
    # Training loop
    ###############
    if sft_config.do_train:
        logger.info("*** Train ***")
        checkpoint = None
        if sft_config.resume_from_checkpoint is not None:
            checkpoint = sft_config.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_config.max_train_samples if data_config.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        if wandb.run is not None: # Check if wandb is initialized
            wandb.config.update(model_config, allow_val_change=True)
            wandb.config.update(data_config, allow_val_change=True)

    ##########
    # Evaluate
    ##########
    if sft_config.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = (
            data_config.max_eval_samples if data_config.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # SFTTrainer.save_model will save the adapters by default if it's a PeftModel
    trainer.save_model(sft_config.output_dir)
    logger.info(f"Model (adapters) saved to {sft_config.output_dir}")

    # To save the full model (merged) if needed, or if not using PEFT (not recommended for quantized)
    # if not model_config.use_peft:
    #    loaded_model.save_pretrained(sft_config.output_dir)

    kwargs_hub = {
        "finetuned_from": model_config.model_name_or_path,
        "dataset": list(data_config.dataset_mixer.keys()),
        "dataset_tags": list(data_config.dataset_mixer.keys()),
        "tags": ["aimo", "qdora", "4bit", "local-training"],
    }

    if accelerator.is_main_process:
        if sft_config.push_to_hub and sft_config.hub_model_id:
            logger.info("Attempting to create model card (push_to_hub is True).")
            try:
                trainer.create_model_card(**kwargs_hub)
            except Exception as e:
                logger.warning(f"Could not create model card, likely due to no Hub login or repo issue: {e}")
        else:
            logger.info("Skipping model card creation (push_to_hub is False or hub_model_id not set).")
        
        # Ensure local config saving happens regardless of Hub status
        if hasattr(trainer.model.config, "use_cache"): # Ensure attribute exists
             trainer.model.config.use_cache = True
        # The main config (for adapters if PEFT, or full model if not) is saved by trainer.save_model()
        # trainer.model.config.save_pretrained(sft_config.output_dir) # This might be redundant for PEFT


    if sft_config.push_to_hub:
        logger.info("Pushing to hub...")
        # For PEFT models, SFTTrainer.push_to_hub will push adapters.
        # If you want to push the merged model, you'd need to merge and then push.
        trainer.push_to_hub(**kwargs_hub)

if __name__ == "__main__":
    main()