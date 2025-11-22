#!/usr/bin/env python3
"""
Zero-shot evaluation script for language models on clinical outcome prediction.

This script evaluates language models without fine-tuning by examining the probability
distribution over tokens for binary classification (0/1 or False/True).

Usage:
    python zero_shot_evaluate.py --config config.yaml
    python zero_shot_evaluate.py --model_path path/to/model --data_path path/to/data.parquet \
                                  --output_dir results/ --data_name mimic --true_token_id 29896 \
                                  --false_token_id 29900
"""

import sys
import os
import os.path as o
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM

# Update path to root for script
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "../..")))

from src.templates.clinic_instruction import clinical_prompt_formatting_func_with_label_definition
from src.file_utils import read_yaml, ensure_directory_exists


def is_valid_first_word(first_word_id, valid_ids):
    """Check if the predicted first word ID is in the valid set."""
    return first_word_id in valid_ids


def predict(model, tokenizer, data, true_id, false_id, data_name, device):
    """
    Perform zero-shot prediction on the dataset.

    Args:
        model: The language model
        tokenizer: The tokenizer
        data: DataFrame containing the dataset
        true_id: Token ID for "true" or "1" class
        false_id: Token ID for "false" or "0" class
        data_name: Name of the dataset (mimic, tasmc, shebamc)
        device: Device to run inference on

    Returns:
        results_df: DataFrame with prediction results
        bad_responses_df: DataFrame with bad/invalid responses (or None)
    """
    bad_responses = []
    results = []

    print(f"Using device: {device}")
    print(f"Dataset: {data_name}")
    print(f"Number of samples: {len(data)}")
    print(f"True token ID: {true_id}, False token ID: {false_id}")
    print()

    for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing samples"):
        # Create prompt using the clinical prompt template
        prompt = clinical_prompt_formatting_func_with_label_definition(row, data_name)

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt")

        # Get model predictions
        with torch.no_grad():
            outputs = model(
                inputs['input_ids'].to(device),
                inputs['attention_mask'].to(device)
            )

        # Get logits for the last token in the prompt
        last_logits = outputs.logits[:, -1, :]

        # Convert logits to probabilities
        probabilities = softmax(last_logits, dim=-1)

        # Get probabilities for true and false tokens
        true_prob = probabilities[0, true_id].item()
        false_prob = probabilities[0, false_id].item()

        # Normalize probabilities
        normalized_true_prob = true_prob / (true_prob + false_prob) if (true_prob + false_prob) > 0 else 0.5

        # Get the most likely next token
        next_token_probabilities, next_token_ids = probabilities.topk(1, dim=-1)
        first_word_probability = next_token_probabilities.item()
        first_word_id = next_token_ids.item()
        first_word = tokenizer.decode(first_word_id)

        # Store results
        res = {
            "index": index,
            "true_label": int(row['label']),
            "first_word_response": first_word,
            "first_word_id": first_word_id,
            "first_word_prob": first_word_probability,
            "true_prob": true_prob,
            "false_prob": false_prob,
            "normalized_true_prob": normalized_true_prob,
            "predicted_label": 1 if normalized_true_prob >= 0.5 else 0
        }

        # Check if the response is valid (first token is either true or false)
        if not is_valid_first_word(first_word_id, [true_id, false_id]):
            res["is_valid_response"] = False

            bad_responses.append({
                "index": index,
                "prompt": prompt,
                "first_word": first_word,
                "first_word_id": first_word_id,
                "true_label": int(row['label'])
            })
        else:
            res["is_valid_response"] = True

        results.append(res)

    # Create DataFrames
    results_df = pd.DataFrame(results)
    bad_responses_df = pd.DataFrame(bad_responses) if len(bad_responses) > 0 else None

    return results_df, bad_responses_df


def calculate_metrics(results_df):
    """
    Calculate evaluation metrics from results.

    Args:
        results_df: DataFrame with prediction results

    Returns:
        metrics: Dictionary with evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

    y_true = results_df['true_label'].values
    y_pred = results_df['predicted_label'].values
    y_prob = results_df['normalized_true_prob'].values

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_prob)

    # Calculate valid response rate
    valid_rate = results_df['is_valid_response'].mean() if 'is_valid_response' in results_df else 1.0

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc,
        'valid_response_rate': valid_rate,
        'num_samples': len(results_df)
    }

    return metrics


def print_metrics(metrics):
    """Print evaluation metrics in a formatted way."""
    print("\n" + "=" * 60)
    print(f"{'EVALUATION METRICS':^60}")
    print("=" * 60)
    print(f"Number of samples: {metrics['num_samples']}")
    print(f"Valid response rate: {metrics['valid_response_rate']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print("=" * 60 + "\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation for language models on clinical outcome prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file (alternative to individual args)"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the pre-trained model"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the data file (parquet format)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save evaluation results"
    )

    parser.add_argument(
        "--data_name",
        type=str,
        choices=['mimic', 'tasmc', 'shebamc'],
        help="Name of the dataset (mimic, tasmc, or shebamc)"
    )

    parser.add_argument(
        "--true_token_id",
        type=int,
        help="Token ID for the 'true'/'1' class"
    )

    parser.add_argument(
        "--false_token_id",
        type=int,
        help="Token ID for the 'false'/'0' class"
    )

    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="Trust remote code when loading the model"
    )

    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=['float16', 'float32', 'bfloat16'],
        help="Torch dtype for model inference"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (default: auto-detect cuda/cpu)"
    )

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    config = read_yaml(config_path)

    required_fields = ['model_path', 'data_path', 'output_dir', 'data_name',
                       'true_token_id', 'false_token_id']

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Config file must contain '{field}' field")

    return config


def print_configuration(config):
    """Print configuration for debugging."""
    print("=" * 60)
    print(f"{'ZERO-SHOT EVALUATION CONFIGURATION':^60}")
    print("=" * 60)
    print(f"Model Path: {config['model_path']}")
    print(f"Data Path: {config['data_path']}")
    print(f"Output Directory: {config['output_dir']}")
    print(f"Dataset Name: {config['data_name']}")
    print(f"True Token ID: {config['true_token_id']}")
    print(f"False Token ID: {config['false_token_id']}")
    print(f"Trust Remote Code: {config.get('trust_remote_code', False)}")
    print(f"Torch Dtype: {config.get('torch_dtype', 'float16')}")
    print("=" * 60)
    print()


def main():
    """Main entry point for zero-shot evaluation."""
    args = parse_args()

    # Load configuration
    if args.config:
        config = load_config(args.config)
        # Override with command line args if provided
        if args.model_path:
            config['model_path'] = args.model_path
        if args.data_path:
            config['data_path'] = args.data_path
        if args.output_dir:
            config['output_dir'] = args.output_dir
        if args.data_name:
            config['data_name'] = args.data_name
        if args.true_token_id is not None:
            config['true_token_id'] = args.true_token_id
        if args.false_token_id is not None:
            config['false_token_id'] = args.false_token_id
    else:
        # Use command line arguments
        required = ['model_path', 'data_path', 'output_dir', 'data_name',
                    'true_token_id', 'false_token_id']
        for field in required:
            if getattr(args, field) is None:
                raise ValueError(f"--{field} is required when not using --config")

        config = {
            'model_path': args.model_path,
            'data_path': args.data_path,
            'output_dir': args.output_dir,
            'data_name': args.data_name,
            'true_token_id': args.true_token_id,
            'false_token_id': args.false_token_id,
            'trust_remote_code': args.trust_remote_code,
            'torch_dtype': args.torch_dtype
        }

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print configuration
    print_configuration(config)

    # Ensure output directory exists
    ensure_directory_exists(config['output_dir'])

    # Load data
    print("Loading data...")
    df = pd.read_parquet(config['data_path'])
    print(f"Loaded {len(df)} samples")
    print()

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    torch_dtype = getattr(torch, config.get('torch_dtype', 'float16'))
    trust_remote = config.get('trust_remote_code', False)

    model = AutoModelForCausalLM.from_pretrained(
        config['model_path'],
        device_map="auto",
        trust_remote_code=trust_remote,
        torch_dtype=torch_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_path'],
        trust_remote_code=trust_remote
    )
    print("Model and tokenizer loaded successfully")
    print()

    # Run prediction
    print("Starting zero-shot evaluation...")
    results_df, bad_responses_df = predict(
        model=model,
        tokenizer=tokenizer,
        data=df,
        true_id=config['true_token_id'],
        false_id=config['false_token_id'],
        data_name=config['data_name'],
        device=device
    )

    # Calculate metrics
    metrics = calculate_metrics(results_df)
    print_metrics(metrics)

    # Save results
    results_path = os.path.join(config['output_dir'], 'results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

    if bad_responses_df is not None:
        bad_responses_path = os.path.join(config['output_dir'], 'bad_responses.csv')
        bad_responses_df.to_csv(bad_responses_path, index=False)
        print(f"Bad responses saved to: {bad_responses_path}")
        print(f"Number of bad responses: {len(bad_responses_df)} ({len(bad_responses_df)/len(results_df)*100:.2f}%)")
    else:
        print("No bad responses detected")

    # Save metrics
    metrics_path = os.path.join(config['output_dir'], 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Zero-Shot Evaluation Metrics\n")
        f.write("=" * 40 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"Metrics saved to: {metrics_path}")

    print("\n" + "=" * 60)
    print(f"{'EVALUATION COMPLETE':^60}")
    print("=" * 60)


if __name__ == "__main__":
    main()

