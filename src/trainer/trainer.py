"""
Clinical Fine-tuning Trainer Module

This module provides the main training function for fine-tuning language models
on clinical decision-making tasks using PEFT (Parameter-Efficient Fine-Tuning) with LoRA.
"""

import os
from typing import Callable, Dict, Optional, Tuple
import gc

import torch
from datasets import load_dataset, Split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModelForCausalLM
from trl import SFTTrainer

from src.trainer.evaluation import (
    compute_metrics,
    preprocess_logits_for_metrics,
    EvaluationArgKey
)

from src.file_utils import ensure_directory_exists, read_yaml
from src.huggingface import connect_to_hf


def clinical_finetuning_trainer(
        data_files: Dict[Split, str],
        model_output_dir: str,
        prompt_formatting_func: Callable,
        evaluation_args_yml_path: str,
        pretrained_model_id: str,
        training_args_yml_path: Optional[str] = None,
        peft_model_id_or_path: Optional[str] = None,
        should_quant_to_4bit: bool = True,
        dataset_type: str = "parquet"
    ) -> None:
    """
    Fine-tune a language model for clinical decision-making tasks.

    This function handles the complete training pipeline including:
    - Model and tokenizer loading with optional 4-bit quantization
    - Dataset preparation and prompt formatting
    - PEFT/LoRA configuration
    - Optional weight injection strategies
    - Training with evaluation and checkpointing
    - Model saving

    Args:
        data_files: Dictionary mapping dataset splits (TRAIN, VALIDATION) to file paths.
        model_output_dir: Directory where trained model and checkpoints will be saved.
        prompt_formatting_func: Function to format dataset examples into training prompts.
        evaluation_args_yml_path: Path to YAML file containing evaluation configuration:
            - last_prompt_token_id: Token ID marking end of prompt sequence
            - positive_token_id: Token ID for positive (Yes) responses
            - negative_token_id: Token ID for negative (No) responses
            - prediction_threshold: Threshold for binary classification (default: 0.5)
            - k: Number of top tokens to consider in evaluation (default: 1000)
        pretrained_model_id: HuggingFace model identifier or path to pretrained model.
        training_args_yml_path: Path to YAML file with training hyperparameters (optional).
        peft_model_id_or_path: Path to existing PEFT model for continued training (optional).
        injection_strategy: Strategy for custom weight initialization (default: NONE).
        injection_params: Parameters for weight injection strategy:
            - layers_names: List of layer names for SELECTED_LAYERS strategy
            - percent_injection: Percentage for RANDOM_PERCENTAGE strategy
            - should_shuffle: Whether to shuffle injected weights
        should_quant_to_4bit: Whether to use 4-bit quantization (default: True).
        dataset_type: Format of dataset files, e.g., "parquet" or "csv" (default: "parquet").

    Raises:
        ValueError: If required configuration parameters are missing or invalid.
        AssertionError: If CUDA is not available or Flash Attention not supported.

    Returns:
        None. Trained model is saved to model_output_dir.

    Notes:
        - Requires CUDA GPU with compute capability >= 8.0 for Flash Attention
        - Uses LoRA for parameter-efficient fine-tuning
        - Automatically handles memory cleanup after training
    """

    # Initialize environment
    connect_to_hf()
    
    # Load configuration
    training_args = _load_training_args(training_args_yml_path)
    evaluation_args = _validate_and_read_evaluation_args(
        evaluation_args_yml_path,
        should_contain_last_prompt_key=True
    )

    # Load and prepare datasets
    train_dataset, valid_dataset = _load_and_prepare_datasets(
        data_files=data_files,
        dataset_type=dataset_type,
        prompt_formatting_func=prompt_formatting_func
    )

    print(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(valid_dataset)}")

    # Load model and tokenizer
    model, tokenizer = _load_model_and_tokenizer(
        pretrained_model_id=pretrained_model_id,
        should_quant_to_4bit=should_quant_to_4bit
    )

    # Configure LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
    peft_config = _create_lora_config(training_args)

    # Load existing PEFT model if provided (for continued training)
    if peft_model_id_or_path is not None:
        model = PeftModelForCausalLM.from_pretrained(
            model=model,
            model_id=peft_model_id_or_path,
            is_trainable=True,
        )
        model.print_trainable_parameters()

    # Configure training arguments
    training_arguments = _create_training_arguments(
        model_output_dir=model_output_dir,
        training_args=training_args,
        should_quant_to_4bit=should_quant_to_4bit
    )

    # Create evaluation metric functions
    compute_metrics_func = _create_compute_metrics_func(evaluation_args)
    preprocess_logits_func = _create_preprocess_logits_func(evaluation_args)

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        dataset_text_field='prompt',
        peft_config=peft_config,
        max_seq_length=training_args.get('max_seq_length', 1024),
        tokenizer=tokenizer,
        packing=False,
        compute_metrics=compute_metrics_func,
        preprocess_logits_for_metrics=preprocess_logits_func
    )

    # Train the model
    print("\nStarting training...")
    trainer.train()
    print("Training completed!")

    # Save the trained model
    _save_model(trainer, model_output_dir)

    # Clean up memory
    _cleanup_memory(model, trainer, train_dataset, valid_dataset)


# ============================================================================
# Helper Functions
# ============================================================================

def _load_training_args(training_args_yml_path: Optional[str]) -> Dict:
    """Load training arguments from YAML file."""
    if training_args_yml_path is None:
        return {}
    return read_yaml(training_args_yml_path).get('args', {})


def _load_and_prepare_datasets(
    data_files: Dict[Split, str],
    dataset_type: str,
    prompt_formatting_func: Callable
) -> Tuple:
    """Load datasets and apply prompt formatting."""
    train_dataset = load_dataset(dataset_type, data_files=data_files, split=Split.TRAIN)
    valid_dataset = load_dataset(dataset_type, data_files=data_files, split=Split.VALIDATION)

    def add_prompt(example):
        example['prompt'] = prompt_formatting_func(example)
        return example

    train_dataset = train_dataset.map(add_prompt)
    valid_dataset = valid_dataset.map(add_prompt)

    return train_dataset, valid_dataset


def _create_quantization_config() -> BitsAndBytesConfig:
    """Create 4-bit quantization configuration for memory efficiency."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


def _load_model_and_tokenizer(
    pretrained_model_id: str,
    should_quant_to_4bit: bool
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the pretrained model and tokenizer."""
    # Load model with optional quantization
    quantization_config = _create_quantization_config() if should_quant_to_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=False,  # Incompatible with gradient checkpointing
    )

    quant_status = "with 4-bit quantization" if should_quant_to_4bit else "without quantization"
    print(f"Model loaded {quant_status}")

    # Load and configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'  # Prevents warnings

    model.config.pretraining_tp = 1

    return model, tokenizer


def _create_lora_config(training_args: Dict) -> LoraConfig:
    """
    Create LoRA configuration for parameter-efficient fine-tuning.

    Based on QLoRA paper and Sebastian Raschka's experiments.
    """
    return LoraConfig(
        lora_alpha=training_args.get('lora_alpha', 128),
        r=training_args.get('lora_rank', 256),
        lora_dropout=training_args.get('lora_dropout', 0.05),
        bias="none",
        target_modules=training_args.get('target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        task_type="CAUSAL_LM",
    )


def _create_training_arguments(
    model_output_dir: str,
    training_args: Dict,
    should_quant_to_4bit: bool
) -> TrainingArguments:
    """Create training arguments configuration."""
    save_total_limit = training_args.get('save_total_limit')
    if save_total_limit is not None:
        print(f"Checkpoint limit: {save_total_limit}")

    return TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=training_args.get('num_train_epochs', 5),
        per_device_train_batch_size=training_args.get('per_device_train_batch_size', 3),
        per_device_eval_batch_size=training_args.get('per_device_eval_batch_size', 4),
        gradient_accumulation_steps=training_args.get('gradient_accumulation_steps', 2),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs=training_args.get(
            'gradient_checkpointing_kwargs',
            {'use_reentrant': False}
        ),
        optim=training_args.get('optim', 'adamw_torch_fused'),
        logging_steps=training_args.get('logging_steps', 100),
        save_strategy=training_args.get('eval_strategy', 'epoch'),
        save_steps=training_args.get('eval_steps', None),
        eval_strategy=training_args.get('eval_strategy', 'epoch'),
        eval_steps=training_args.get('eval_steps', None),
        learning_rate=float(training_args.get('learning_rate', 2e-4)),
        bf16=True,
        tf32=should_quant_to_4bit,
        max_grad_norm=training_args.get('max_grad_norm', 0.3),
        warmup_ratio=training_args.get('warmup_ratio', 0.03),
        weight_decay=training_args.get('weight_decay', 0),
        lr_scheduler_type=training_args.get('lr_scheduler_type', 'constant'),
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        report_to="tensorboard",
        seed=training_args.get('seed', 42),
        data_seed=training_args.get('data_seed', 42),
        ddp_find_unused_parameters=False,
    )


def _create_compute_metrics_func(evaluation_args: Dict) -> Callable:
    """Create the compute_metrics function with evaluation parameters."""
    def compute_metrics_wrapper(eval_pred: Tuple[torch.Tensor, torch.Tensor]):
        return compute_metrics(
            eval_pred,
            last_prompt_token_id=evaluation_args.get(EvaluationArgKey.LAST_PROMPT_TOKEN_ID.value),
            positive_token_id=evaluation_args.get(EvaluationArgKey.POSITIVE_TOKEN_ID.value),
            negative_token_id=evaluation_args.get(EvaluationArgKey.NEGATIVE_TOKEN_ID.value),
            pred_threshold=evaluation_args.get(EvaluationArgKey.PREDICTION_THRESHOLD.value, 0.5)
        )
    return compute_metrics_wrapper


def _create_preprocess_logits_func(evaluation_args: Dict) -> Callable:
    """Create the preprocess_logits_for_metrics function with evaluation parameters."""
    def preprocess_logits_wrapper(logits: torch.Tensor, labels: torch.Tensor):
        return preprocess_logits_for_metrics(
            logits,
            labels,
            last_prompt_token_id=evaluation_args.get(EvaluationArgKey.LAST_PROMPT_TOKEN_ID.value),
            k=evaluation_args.get(EvaluationArgKey.K.value, 1000)
        )
    return preprocess_logits_wrapper


def _save_model(trainer: SFTTrainer, model_output_dir: str) -> None:
    """
    Save the trained PEFT model.

    This step facilitates continuation of training by saving the PEFT configuration,
    ensuring the model can be reloaded with the same settings.

    Note: Checkpoints are useful for inference but may have issues when reloading for training.
    Reference: https://discuss.huggingface.co/t/transformers-trainer-accelerate-fsdp-how-do-i-load-my-model-from-a-checkpoint/61585/3
    """
    saved_model_path = f"{model_output_dir}/saved_model"
    ensure_directory_exists(saved_model_path)
    trainer.save_model(saved_model_path)
    print(f"\nModel saved to: {saved_model_path}")


def _cleanup_memory(model, trainer, train_dataset, valid_dataset) -> None:
    """Clean up memory by deleting large objects and clearing CUDA cache."""
    print("\nCleaning up memory...")

    del model
    del trainer
    del train_dataset
    del valid_dataset

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()
    print("Memory cleanup completed")


def _validate_and_read_evaluation_args(
    evaluation_args_yml_path: str,
    should_contain_last_prompt_key: bool = True
) -> Dict:
    """
    Load and validate evaluation arguments from YAML configuration.

    Args:
        evaluation_args_yml_path: Path to evaluation arguments YAML file.
        should_contain_last_prompt_key: Whether last_prompt_token_id is required.

    Returns:
        Dictionary containing validated evaluation arguments.

    Raises:
        ValueError: If required configuration is missing or invalid.
    """
    # Validate path
    if not evaluation_args_yml_path:
        raise ValueError("evaluation_args_yml_path is required but not provided")

    if not os.path.exists(evaluation_args_yml_path):
        raise FileNotFoundError(
            f"Evaluation args file not found: {evaluation_args_yml_path}"
        )

    print(f"Loading evaluation args from: {evaluation_args_yml_path}")

    # Load configuration
    evaluation_args = read_yaml(evaluation_args_yml_path).get('args', {})

    # Validate required fields
    if should_contain_last_prompt_key:
        if evaluation_args.get(EvaluationArgKey.LAST_PROMPT_TOKEN_ID.value) is None:
            raise ValueError(
                f"Required field '{EvaluationArgKey.LAST_PROMPT_TOKEN_ID.value}' "
                "is missing in evaluation_args"
            )

    positive_token_id = evaluation_args.get(EvaluationArgKey.POSITIVE_TOKEN_ID.value)
    negative_token_id = evaluation_args.get(EvaluationArgKey.NEGATIVE_TOKEN_ID.value)

    if positive_token_id is None or negative_token_id is None:
        raise ValueError(
            f"Required fields '{EvaluationArgKey.POSITIVE_TOKEN_ID.value}' and "
            f"'{EvaluationArgKey.NEGATIVE_TOKEN_ID.value}' are required in evaluation_args"
        )

    print(f"✓ Evaluation args validated: {evaluation_args}")
    return evaluation_args

