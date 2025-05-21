import logging
import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, PeftModel
from aimo.configs import ModelConfig # Make sure this import path is correct based on your structure

logger = logging.getLogger(__name__)

class LoraMergeCallback(TrainerCallback):
    def __init__(self, model_config: ModelConfig, sft_config: TrainingArguments, merge_interval: int):
        super().__init__()
        self.model_config = model_config
        self.sft_config = sft_config # This is actually an instance of SFTConfig (TrainingArguments subclass)
        self.merge_interval = merge_interval
        self.last_merge_step = 0
        logger.info(f"LoraMergeCallback initialized with merge_interval: {self.merge_interval}")

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Using on_step_begin to ensure model/optimizer are updated *before* the step computation
        
        if self.merge_interval is None or self.merge_interval <= 0:
            return

        # Check if it's time to merge (current step is a multiple of merge_interval, but not step 0)
        # We subtract 1 from global_step because global_step is 1-indexed for the first step,
        # and we want to merge *after* completing `merge_interval` steps.
        # So, if merge_interval is 100, we merge at the beginning of step 101 (after 100 steps are done).
        # However, on_step_begin is called *before* the step. So if global_step is 100, it means 99 steps completed.
        # Let's adjust: merge when global_step is exactly merge_interval, 2*merge_interval etc.
        # Or, more simply, if (state.global_step - self.last_merge_step) >= self.merge_interval
        # and state.global_step > 0 to avoid merging at step 0.
        
        # We want to merge *after* step `self.merge_interval`, `2*self.merge_interval`, etc.
        # `on_step_begin` for step `N` means `N-1` steps have been completed.
        # So, if `state.global_step -1` is a multiple of `merge_interval` and not 0.
        effective_completed_steps = state.global_step -1 # state.global_step is 1-indexed

        if effective_completed_steps > 0 and            effective_completed_steps >= self.merge_interval and            (effective_completed_steps % self.merge_interval == 0):
            
            logger.info(f"Step {state.global_step}: Triggering LoRA merge and reinitialization.")
            
            trainer = kwargs.get("trainer") # SFTTrainer instance should be passed in kwargs
            if not trainer:
                logger.error("Trainer instance not found in callback kwargs. Cannot proceed with LoRA merge.")
                control.should_training_stop = True
                return

            model = trainer.model # Get the model directly from the trainer
            if not isinstance(model, PeftModel):
                logger.warning(f"Model is not a PeftModel. Skipping LoRA merge. Model type: {type(model)}")
                return
            
            logger.info(f"Current model device before merge: {model.device if hasattr(model, 'device') else 'N/A'}")
            # Store if the model was k-bit trained to re-apply preparation
            was_kbit_trained = (hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit) or                                (hasattr(model, 'is_loaded_in_8bit') and model.is_loaded_in_8bit)
            if was_kbit_trained:
                logger.info("Model was k-bit trained. Will re-prepare after LoRA reinitialization.")

            # 1. Merge LoRA weights
            logger.info(f"Merging LoRA adapters for model: {type(model)}")
            try:
                # PeftModel.merge_and_unload() merges the LoRA weights into the base model
                # and returns the base model.
                base_model = model.merge_and_unload()
                logger.info(f"Successfully merged and unloaded LoRA adapters. Base model type: {type(base_model)}")
                logger.info(f"Base model device after merge: {base_model.device if hasattr(base_model, 'device') else 'N/A'}")

            except Exception as e:
                logger.error(f"Error during LoRA merge_and_unload: {e}", exc_info=True)
                control.should_training_stop = True 
                return

            # 2. Reinitialize LoRA Adapters
            logger.info("Reinitializing LoRA adapters...")
            if not self.model_config.use_peft or not self.model_config.lora_target_modules:
                logger.error("PEFT is not configured or target modules are missing. Cannot reinitialize LoRA.")
                control.should_training_stop = True
                return

            # Create new LoraConfig. Ensure it's consistent with initial setup.
            peft_config = LoraConfig(
                r=self.model_config.lora_r,
                lora_alpha=self.model_config.lora_alpha,
                lora_dropout=self.model_config.lora_dropout,
                target_modules=self.model_config.lora_target_modules,
                task_type="CAUSAL_LM", # Ensure this matches the task type used in sft.py
                use_dora=self.model_config.lora_use_dora,
                bias="none", 
            )
            
            # Apply PEFT to the base_model
            # The base_model should be on the correct device already if device_map="auto" was used
            # or if the trainer handles device placement.
            new_peft_model = get_peft_model(base_model, peft_config)
            logger.info(f"Successfully reinitialized PEFT model. New PEFT model type: {type(new_peft_model)}")
            logger.info(f"New PEFT model device: {new_peft_model.device if hasattr(new_peft_model, 'device') else 'N/A'}")
            
            # 3. Re-prepare for k-bit training if it was originally
            if was_kbit_trained:
                logger.info("Re-preparing model for k-bit training...")
                # Ensure gradient_checkpointing setting is correctly obtained from sft_config
                use_gc = getattr(self.sft_config, 'gradient_checkpointing', False)
                new_peft_model = prepare_model_for_kbit_training(
                    new_peft_model, use_gradient_checkpointing=use_gc
                )
                logger.info("Model re-prepared for k-bit training.")
                logger.info(f"Model device after k-bit re-preparation: {new_peft_model.device if hasattr(new_peft_model, 'device') else 'N/A'}")


            # Ensure the model is on the correct device, as expected by the Trainer
            # The trainer `args.device` (which is SFTConfig.device) should be the target device.
            target_device = args.device
            new_peft_model.to(target_device)
            logger.info(f"Moved new PEFT model to device: {target_device}")

            trainer.model = new_peft_model 
            logger.info("Updated trainer.model with the new PEFT model.")
            
            if hasattr(trainer.model, 'print_trainable_parameters'):
                trainer.model.print_trainable_parameters()
            else:
                # Manual check for trainable parameters if print_trainable_parameters is not available
                trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in trainer.model.parameters())
                logger.info(f"Trainable parameters: {trainable_params} || All parameters: {total_params} || Trainable %: {100 * trainable_params / total_params:.2f}")


            # 4. Reinitialize Optimizer and Scheduler
            logger.info("Reinitializing optimizer and scheduler...")
            # SFTTrainer uses `create_optimizer_and_scheduler`.
            # We need the `num_training_steps` for the scheduler.
            # `state.max_steps` should provide the total number of training steps.
            if hasattr(trainer, "create_optimizer_and_scheduler"):
                trainer.create_optimizer_and_scheduler(num_training_steps=state.max_steps)
                logger.info("Optimizer and scheduler re-created using trainer.create_optimizer_and_scheduler().")
            elif hasattr(trainer, "create_optimizer"): # Fallback if only create_optimizer is present
                 trainer.create_optimizer()
                 logger.info("Optimizer re-created using trainer.create_optimizer().")
                 if hasattr(trainer, "create_scheduler"):
                     trainer.create_scheduler(num_training_steps=state.max_steps, optimizer=trainer.optimizer)
                     logger.info("Scheduler re-created using trainer.create_scheduler().")
                 else:
                     logger.warning("Trainer has create_optimizer but not create_scheduler. LR scheduling might be incorrect.")
            else:
                logger.error("Trainer does not have a recognized method to recreate the optimizer (create_optimizer_and_scheduler or create_optimizer). Training quality will be affected.")
                # Not stopping training, but logging a severe error.
            
            self.last_merge_step = state.global_step # Record the step at which merge was initiated
            logger.info(f"LoRA merge and reinitialization complete, initiated at the beginning of step {state.global_step}. Next merge trigger will be around step {state.global_step + self.merge_interval}.")
