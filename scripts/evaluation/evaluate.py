#!/usr/bin/env python3
"""
Unified evaluation script for model checkpoints.

Usage:
    # Evaluate all epochs/checkpoints in a directory
    python evaluate.py --config config.yaml --models_dir path/to/models --eval_set all

    # Evaluate a specific model checkpoint
    python evaluate.py --config config.yaml --model_path path/to/checkpoint --eval_set test --output_dir eval_output
"""

import sys
import os
import os.path as o
import argparse

# Update path to root for script
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "../..")))

from datasets import Split, load_dataset
from src.evaluation.evaluate_by_first_word import eval_by_first_word, eval_model_epochs_by_first_word
from src.file_utils import call_function_by_path, ensure_directory_exists, read_yaml


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified evaluation script for model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--eval_set",
        type=str,
        choices=["train", "valid", "test", "all"],
        default="all",
        help="Which dataset split(s) to evaluate on (default: all)"
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--model_path",
        type=str,
        help="Path to specific model checkpoint to evaluate"
    )
    mode_group.add_argument(
        "--models_dir",
        type=str,
        help="Directory containing multiple model checkpoints to evaluate all epochs"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for evaluation results (required when using --model_path)"
    )

    return parser.parse_args()


def load_config(config_path):
    """Load and extract configuration from YAML file."""
    args = read_yaml(config_path)

    # Get first word evaluation config
    first_word_eval = args.get('first_word_eval', {})
    if not first_word_eval:
        raise ValueError("Config must contain 'first_word_eval' section with 'positive' and 'negative' tokens")

    positive_token = first_word_eval.get('positive')
    negative_token = first_word_eval.get('negative')

    if not positive_token or not negative_token:
        raise ValueError("first_word_eval must contain both 'positive' and 'negative' tokens")

    config = {
        'dataset_dir': args['dataset_dir'],
        'prompt_template_func_path': args['prompt_template_func_path'],
        'template_name': args.get('template_name', 'clinic_instruction'),
        'dataset_type': args.get('dataset_type', 'parquet'),
        'report_pred_threshold': args.get('report_pred_threshold', 0.5),
        'model_prefix': args.get('model_prefix', 'checkpoint-'),
        'positive_token': positive_token,
        'negative_token': negative_token
    }

    return config


def print_configuration(args, config, mode):
    """Print configuration for debugging."""
    print("=" * 60)
    print(f"{'EVALUATION CONFIGURATION':^60}")
    print("=" * 60)
    print(f"Mode: {mode}")
    if mode == "specific":
        print(f"Model Path: {args.model_path}")
        print(f"Output Directory: {args.output_dir}")
    else:
        print(f"Models Directory: {args.models_dir}")
        print(f"Model Prefix: {config['model_prefix']}")
    print(f"Config File: {args.config}")
    print(f"Dataset Directory: {config['dataset_dir']}")
    print(f"Eval Set: {args.eval_set}")
    print(f"Prompt Template: {config['prompt_template_func_path']}")
    print(f"Template Name: {config['template_name']}")
    print(f"Dataset Type: {config['dataset_type']}")
    print(f"Prediction Threshold: {config['report_pred_threshold']}")
    print(f"Positive Token: {config['positive_token']}")
    print(f"Negative Token: {config['negative_token']}")
    print("=" * 60)
    print()


def create_prompt_wrapper(prompt_template_func_path):
    """Create a wrapper function for prompt creation."""
    def eval_create_prompt(example):
        return call_function_by_path(
            prompt_template_func_path,
            example=example,
            is_train=False
        )
    return eval_create_prompt


def get_data_files(dataset_dir, dataset_type, eval_set):
    """Get data file paths based on available files and eval_set."""
    datasets_files = os.listdir(dataset_dir)
    data_files = {}

    if eval_set in ["train", "all"] and f'cont20_train.{dataset_type}' in datasets_files:
        data_files[Split.TRAIN] = f"{dataset_dir}/cont20_train.{dataset_type}"

    if eval_set in ["valid", "all"] and f'valid.{dataset_type}' in datasets_files:
        data_files[Split.VALIDATION] = f"{dataset_dir}/valid.{dataset_type}"

    if eval_set in ["test", "all"] and f'test.{dataset_type}' in datasets_files:
        data_files[Split.TEST] = f"{dataset_dir}/test.{dataset_type}"

    return data_files


def evaluate_specific_model(args, config):
    """Evaluate a specific model checkpoint."""
    print(f"\n{'=' * 60}")
    print(f"{'EVALUATING SPECIFIC MODEL':^60}")
    print(f"{'=' * 60}\n")

    # Ensure output directory exists
    ensure_directory_exists(args.output_dir)

    # Get data files
    data_files = get_data_files(
        config['dataset_dir'],
        config['dataset_type'],
        args.eval_set
    )

    if not data_files:
        raise ValueError(f"No data files found for eval_set '{args.eval_set}' in {config['dataset_dir']}")

    print(f"Data files found: {list(data_files.keys())}\n")

    # Load datasets
    datasets = {}
    for split, file_path in data_files.items():
        print(f"Loading {split} dataset from {file_path}...")
        dataset = load_dataset(config['dataset_type'], data_files={split: file_path}, split=split)

        # Map split to string name
        split_name = {
            Split.TRAIN: 'train',
            Split.VALIDATION: 'valid',
            Split.TEST: 'test'
        }.get(split, str(split))

        datasets[split_name] = dataset

    # Create prompt function
    create_prompt_func = create_prompt_wrapper(config['prompt_template_func_path'])

    # Get first response words from config
    positive_token = config['positive_token']
    negative_token = config['negative_token']

    print(f"\nStarting evaluation...")
    print(f"Positive token: {positive_token}")
    print(f"Negative token: {negative_token}\n")

    # Run evaluation
    eval_by_first_word(
        model_path=args.model_path,
        datasets=datasets,
        positive_token_str=positive_token,
        negative_token_str=negative_token,
        create_prompt_func=create_prompt_func,
        output_dir_path=args.output_dir,
        pred_threshold=config['report_pred_threshold']
    )

    print(f"\n{'=' * 60}")
    print(f"{'EVALUATION COMPLETE':^60}")
    print(f"{'=' * 60}")
    print(f"Results saved to: {args.output_dir}\n")


def evaluate_all_epochs(args, config):
    """Evaluate all model checkpoints in a directory."""
    print(f"\n{'=' * 60}")
    print(f"{'EVALUATING ALL EPOCHS':^60}")
    print(f"{'=' * 60}\n")

    # Get data files (exclude train for multiple epochs)
    data_files = {}
    dataset_dir = config['dataset_dir']
    dataset_type = config['dataset_type']

    if args.eval_set in ["valid", "all"]:
        valid_path = f"{dataset_dir}/valid.{dataset_type}"
        if os.path.exists(valid_path):
            data_files[Split.VALIDATION] = valid_path

    if args.eval_set in ["test", "all"]:
        test_path = f"{dataset_dir}/test.{dataset_type}"
        if os.path.exists(test_path):
            data_files[Split.TEST] = test_path

    if not data_files:
        raise ValueError(f"No data files found for eval_set '{args.eval_set}' in {dataset_dir}")

    print(f"Data files found: {list(data_files.keys())}\n")

    # Create prompt function
    create_prompt_func = create_prompt_wrapper(config['prompt_template_func_path'])

    # Get first response words from config
    positive_token = config['positive_token']
    negative_token = config['negative_token']

    print(f"Positive token: {positive_token}")
    print(f"Negative token: {negative_token}\n")

    # Run evaluation on all epochs
    eval_model_epochs_by_first_word(
        models_dir=args.models_dir,
        model_prefix=config['model_prefix'],
        data_files=data_files,
        positive_token_str=positive_token,
        negative_token_str=negative_token,
        create_prompt_func=create_prompt_func
    )

    print(f"\n{'=' * 60}")
    print(f"{'EVALUATION COMPLETE':^60}")
    print(f"{'=' * 60}")
    print(f"Results saved in: {args.models_dir}\n")


def main():
    """Main entry point for the unified evaluation script."""
    # Parse arguments
    args = parse_args()

    # Validate arguments
    if args.model_path and not args.output_dir:
        raise ValueError("--output_dir is required when using --model_path")

    # Load configuration
    config = load_config(args.config)

    # Determine mode
    mode = "specific" if args.model_path else "all_epochs"

    # Print configuration
    print_configuration(args, config, mode)

    # Run appropriate evaluation
    if mode == "specific":
        evaluate_specific_model(args, config)
    else:
        evaluate_all_epochs(args, config)


if __name__ == "__main__":
    main()
