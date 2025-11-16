"""
Clinical Fine-tuning Script

This script fine-tunes a language model for clinical decision-making tasks.
It loads configuration from YAML files and trains the model using clinical expert-designed prompts.

The prompts are designed to elicit clear, deterministic responses (Yes/No) which yield better
results than ambiguous or hesitant phrases based on experimental findings.

Usage:
    python clinic_finetuning_trainer.py <config_yaml_path> <model_output_dir>
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from datasets import Split
from src.trainer.trainer import clinical_finetuning_trainer
from src.file_utils import call_function_by_path, read_yaml


def format_training_prompt(example: dict, prompt_template_func_path: str) -> str:
    """
    Format a training example into a prompt using the specified template function.

    Args:
        example: Training example containing patient data
        prompt_template_func_path: Python path to the prompt formatting function

    Returns:
        Formatted prompt string
    """
    return call_function_by_path(
        prompt_template_func_path,
        example=example,
        is_train=True
    )


def load_configuration(yaml_path: str) -> dict:
    """
    Load and validate configuration from YAML file.

    Args:
        yaml_path: Path to configuration YAML file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If YAML file doesn't exist
        KeyError: If required configuration keys are missing
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    config = read_yaml(yaml_path)

    # Validate required fields
    if 'dataset_dir' not in config:
        raise KeyError("Required field 'dataset_dir' missing in configuration")

    return config


def print_configuration(config: dict, model_output_dir: str, yaml_path: str):
    """Print configuration summary for logging."""
    print("\n" + "="*70)
    print("CLINICAL FINE-TUNING CONFIGURATION")
    print("="*70)
    print(f"Process ID: {os.getpid()}")
    print(f"Config file: {yaml_path}")
    print(f"\nDataset:")
    print(f"  Directory: {config['dataset_dir']}")
    print(f"  Train file: {config.get('train_data_file_name', 'train.parquet')}")
    print(f"\nModel:")
    print(f"  Pretrained ID: {config.get('pretrained_model_id')}")
    print(f"  Output directory: {model_output_dir}")
    print(f"  4-bit quantization: {config.get('should_quant_to_4bit', False)}")
    print(f"\nTraining:")
    print(f"  Training args config: {config.get('training_args_yml_path')}")
    print(f"  Evaluation args config: {config.get('evaluation_args_yml_path')}")
    print(f"  PEFT model path: {config.get('saved_peft_model_id_or_path')}")
    print(f"\nInjection:")
    print(f"  Strategy: {config.get('injection_strategy', 'NONE')}")
    print(f"  Parameters: {config.get('injection_params', {})}")
    print(f"\nPrompt:")
    print(f"  Template function: {config.get('prompt_template_func_path', 'templates.clinic_instructions.clear_clinical_prompt_formatting_func')}")
    print("="*70 + "\n")


def main():
    """Main execution function."""
    # Parse command-line arguments
    if len(sys.argv) < 3:
        print("Usage: python clinic_finetuning_trainer.py <config_yaml_path> <model_output_dir>")
        sys.exit(1)

    yaml_path = sys.argv[1]
    model_output_dir = sys.argv[2]

    # Load configuration
    config = load_configuration(yaml_path)

    # Extract configuration values with defaults
    dataset_dir = config['dataset_dir']
    train_data_file = config.get('train_data_file_name', 'train.parquet')
    validation_data_file = 'valid.parquet'

    training_args_path = config.get('training_args_yml_path')
    evaluation_args_path = config.get('evaluation_args_yml_path')
    peft_model_path = config.get('saved_peft_model_id_or_path')
    pretrained_model_id = config.get('pretrained_model_id')

    should_quant_to_4bit = config.get('should_quant_to_4bit', False)
    prompt_template_func_path = config.get(
        'prompt_template_func_path',
        'templates.clinic_instructions.clear_clinical_prompt_formatting_func'
    )

    # Print configuration for logging
    print_configuration(config, model_output_dir, yaml_path)

    # Create prompt formatting function
    def prompt_formatter(example: dict) -> str:
        return format_training_prompt(example, prompt_template_func_path)

    # Prepare data files
    data_files = {
        Split.TRAIN: os.path.join(dataset_dir, train_data_file),
        Split.VALIDATION: os.path.join(dataset_dir, validation_data_file)
    }

    # Run training
    clinical_finetuning_trainer(
        data_files=data_files,
        model_output_dir=model_output_dir,
        prompt_formatting_func=prompt_formatter,
        peft_model_id_or_path=peft_model_path,
        training_args_yml_path=training_args_path,
        evaluation_args_yml_path=evaluation_args_path,
        pretrained_model_id=pretrained_model_id,
        should_quant_to_4bit=should_quant_to_4bit
    )

    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
