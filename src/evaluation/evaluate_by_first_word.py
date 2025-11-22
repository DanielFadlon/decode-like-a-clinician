"""
Evaluation module for language models using first-word prediction strategy.

This module provides functionality to evaluate language models by analyzing the
probability distribution of the first predicted token, specifically for binary
classification tasks.
"""

import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


__all__ = [
    "TokenLabel",
    "evaluate_by_first_word",
    "evaluate_model_checkpoints",
]


@dataclass
class TokenLabel:
    """Represents a label with its token ID and text representation."""
    token_id: int
    text: str


def _is_valid_token(token_id: int, valid_token_ids: List[int]) -> bool:
    """
    Check if a token ID is in the list of valid token IDs.

    Args:
        token_id: The token ID to validate.
        valid_token_ids: List of acceptable token IDs.

    Returns:
        True if token_id is valid, False otherwise.
    """
    return token_id in valid_token_ids


def _predict_dataset(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        positive_label: TokenLabel,
        negative_label: TokenLabel,
        create_prompt_func: Callable,
        device: torch.device = torch.device('cuda')
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Generate predictions for a dataset using first-word prediction strategy.

    For each instance in the dataset, this function:
    1. Creates a prompt using the provided function
    2. Gets the model's next token probability distribution
    3. Extracts probabilities for positive and negative labels
    4. Records any invalid predictions (tokens outside expected labels)

    Args:
        model: Pre-trained language model for inference.
        tokenizer: Tokenizer corresponding to the model.
        dataset: Dataset containing instances to predict on.
        positive_label: Token label for the positive class.
        negative_label: Token label for the negative class.
        create_prompt_func: Function that takes a dataset instance and returns a prompt string.
        device: Device to run inference on (default: CUDA).

    Returns:
        Tuple containing:
        - DataFrame with prediction results for all instances
        - DataFrame with invalid predictions (or None if all valid)
    """
    predictions = []
    invalid_predictions = []

    model.eval()

    for idx in tqdm(range(len(dataset)), desc="Predicting"):
        instance = dataset[idx]
        prompt = create_prompt_func(instance)
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model(
                inputs['input_ids'].to(device),
                inputs['attention_mask'].to(device)
            )

        # Extract logits for the next token prediction (last position in sequence)
        next_token_logits = outputs.logits[:, -1, :]

        # Convert logits to probabilities
        token_probabilities = softmax(next_token_logits, dim=-1)

        # Extract probabilities for positive and negative labels
        positive_prob = token_probabilities[0, positive_label.token_id].item()
        negative_prob = token_probabilities[0, negative_label.token_id].item()

        # Normalize probabilities to sum to 1.0 for binary classification
        normalized_positive_prob = positive_prob / (positive_prob + negative_prob)

        # Get the most likely next token
        top_token_prob, top_token_id = token_probabilities.topk(1, dim=-1)
        predicted_token_id = top_token_id.item()
        predicted_token_text = tokenizer.decode(predicted_token_id)
        predicted_token_prob = top_token_prob.item()

        prediction_result = {
            "index": idx,
            "predicted_token": predicted_token_text,
            "predicted_token_prob": predicted_token_prob,
            "positive_prob": positive_prob,
            "negative_prob": negative_prob,
            "normalized_positive_prob": normalized_positive_prob
        }

        # Track predictions that don't match expected positive/negative tokens
        valid_token_ids = [positive_label.token_id, negative_label.token_id]
        if not _is_valid_token(predicted_token_id, valid_token_ids):
            prediction_result["is_invalid"] = True
            invalid_predictions.append({
                "index": idx,
                "prompt": prompt,
                "predicted_token": predicted_token_text
            })

        predictions.append(prediction_result)

    predictions_df = pd.DataFrame(predictions)
    invalid_predictions_df = (
        pd.DataFrame(invalid_predictions) if invalid_predictions else None
    )

    return predictions_df, invalid_predictions_df


def evaluate_by_first_word(
        model_path: str,
        datasets: Dict[str, Dataset],
        positive_token: str,
        negative_token: str,
        create_prompt_func: Callable,
        output_dir: Optional[str] = None,
        prediction_threshold: float = 0.5,
        save_results: bool = True
    ) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a language model using first-word prediction strategy.

    This function evaluates a language model's performance on binary classification
    tasks by analyzing the probability of the first predicted token. It computes
    metrics (AUC, accuracy) for each provided dataset split.

    Args:
        model_path: Path to the pre-trained model (can include PEFT adapter).
        datasets: Dictionary mapping dataset split names to Dataset objects.
        positive_token: String token representing the positive class (e.g., "Yes").
        negative_token: String token representing the negative class (e.g., "No").
        create_prompt_func: Function that takes a dataset instance and returns a prompt.
        output_dir: Directory to save evaluation results. If None, results aren't saved.
        prediction_threshold: Probability threshold for binary classification (default: 0.5).
        save_results: Whether to save detailed results to CSV files (default: True).

    Returns:
        Dictionary mapping dataset names to their evaluation metrics (AUC, accuracy).
        Structure: {dataset_name: {"auc": float, "accuracy": float}}

    Example:
        >>> results = evaluate_by_first_word(
        ...     model_path="path/to/model",
        ...     datasets={"train": train_ds, "test": test_ds},
        ...     positive_token="Yes",
        ...     negative_token="No",
        ...     create_prompt_func=lambda x: f"Question: {x['text']}\nAnswer:"
        ... )
    """
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Encode label tokens to IDs
    positive_token_id = tokenizer.encode(positive_token, add_special_tokens=False)[0]
    negative_token_id = tokenizer.encode(negative_token, add_special_tokens=False)[0]
    print(f"Positive token '{positive_token}' -> ID: {positive_token_id}")
    print(f"Negative token '{negative_token}' -> ID: {negative_token_id}")

    positive_label = TokenLabel(token_id=positive_token_id, text=positive_token)
    negative_label = TokenLabel(token_id=negative_token_id, text=negative_token)

    evaluation_results = {}
    scores_file_path = (
        f"{output_dir}/scores.txt" if output_dir and save_results else None
    )

    for dataset_name, dataset in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating on: {dataset_name}")
        print(f"{'=' * 60}\n")

        # Generate predictions
        predictions_df, invalid_predictions_df = _predict_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            positive_label=positive_label,
            negative_label=negative_label,
            create_prompt_func=create_prompt_func,
            device=device
        )

        # Extract predictions and ground truth
        y_pred_probs = np.array(predictions_df['normalized_positive_prob'])
        y_true = np.array(dataset['label'])

        # Save detailed results if requested
        if output_dir and save_results:
            predictions_df.to_csv(
                f'{output_dir}/{dataset_name}_predictions.csv',
                index=False
            )
            if invalid_predictions_df is not None:
                invalid_predictions_df.to_csv(
                    f'{output_dir}/{dataset_name}_invalid_predictions.csv',
                    index=False
                )
                print(f"Warning: {len(invalid_predictions_df)} invalid predictions detected")

        # Calculate metrics
        evaluation_results[dataset_name] = {}
        # Accuracy calculation
        y_pred = (y_pred_probs > prediction_threshold).astype(int)

        auc = roc_auc_score(y_true, y_pred_probs)
        accuracy = accuracy_score(y_true, y_pred)
        evaluation_results[dataset_name]["auc"] = auc
        evaluation_results[dataset_name]["accuracy"] = accuracy

        print(f"{dataset_name} - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return evaluation_results


def _extract_checkpoint_number(checkpoint_name: str) -> int:
    """
    Extract numeric identifier from checkpoint name for sorting.

    Args:
        checkpoint_name: Name of the checkpoint (e.g., "checkpoint-100", "epoch-5").

    Returns:
        Integer extracted from the checkpoint name.

    Raises:
        ValueError: If no number is found in the checkpoint name.

    Examples:
        >>> _extract_checkpoint_number("checkpoint-100")
        100
        >>> _extract_checkpoint_number("epoch-5")
        5
    """
    match = re.search(r'\d+', checkpoint_name)
    if not match:
        raise ValueError(f"No number found in checkpoint name: {checkpoint_name}")
    return int(match.group())


def _sort_checkpoints(checkpoint_results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Sort checkpoint results by numeric identifier in checkpoint names.

    Assumes checkpoint names contain a numeric identifier used for ordering
    (e.g., checkpoint-100, checkpoint-200, epoch-1, epoch-2).

    Args:
        checkpoint_results: Dictionary mapping checkpoint names to their metrics.

    Returns:
        OrderedDict with checkpoints sorted by their numeric identifiers.
    """
    sorted_names = sorted(
        checkpoint_results.keys(),
        key=_extract_checkpoint_number
    )
    return {name: checkpoint_results[name] for name in sorted_names}


def evaluate_model_checkpoints(
        checkpoints_dir: str,
        checkpoint_prefix: str,
        data_files: Dict[str, str],
        positive_token: str,
        negative_token: str,
        create_prompt_func: Callable,
        prediction_threshold: float = 0.5,
        dataset_format: str = "parquet"
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate multiple model checkpoints using first-word prediction strategy.

    This function iterates through all model checkpoints in a directory that match
    the specified prefix and evaluates each one on the provided datasets. Results
    are saved to individual directories for each checkpoint, and visualization plots
    are generated showing metric trends across checkpoints.

    Args:
        checkpoints_dir: Directory containing model checkpoints.
        checkpoint_prefix: Prefix to filter checkpoint directories (e.g., "checkpoint-").
        data_files: Dictionary mapping split names to dataset file paths.
        positive_token: String token representing the positive class.
        negative_token: String token representing the negative class.
        create_prompt_func: Function that takes a dataset instance and returns a prompt.
        prediction_threshold: Probability threshold for binary classification (default: 0.5).
        dataset_format: Format of the dataset files (default: "parquet").

    Returns:
        Dictionary mapping checkpoint names to their evaluation results.
        Structure: {checkpoint_name: {dataset_name: {"auc": float, "accuracy": float}}}

    Example:
        >>> results = evaluate_model_checkpoints(
        ...     checkpoints_dir="outputs/training_run",
        ...     checkpoint_prefix="checkpoint-",
        ...     data_files={"train": "data/train.parquet", "test": "data/test.parquet"},
        ...     positive_token="Yes",
        ...     negative_token="No",
        ...     create_prompt_func=lambda x: f"Question: {x['text']}\nAnswer:"
        ... )
    """
    # Find all checkpoints matching the prefix
    all_items = os.listdir(checkpoints_dir)
    checkpoint_names = [
        item for item in all_items
        if item.startswith(checkpoint_prefix) and os.path.isdir(os.path.join(checkpoints_dir, item))
    ]

    if not checkpoint_names:
        raise ValueError(f"No checkpoints found with prefix '{checkpoint_prefix}' in {checkpoints_dir}")

    print(f"Found {len(checkpoint_names)} checkpoints to evaluate")

    all_results = {}

    for checkpoint_name in checkpoint_names:
        print(f"\n{'#' * 70}")
        print(f"# Evaluating checkpoint: {checkpoint_name}")
        print(f"{'#' * 70}\n")

        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
        eval_output_dir = os.path.join(checkpoints_dir, "eval", checkpoint_name)

        os.makedirs(eval_output_dir, exist_ok=True)

        # Load datasets
        datasets = {}
        for split_name, file_path in data_files.items():
            datasets[split_name] = load_dataset(
                dataset_format,
                data_files={split_name: file_path},
                split=split_name
            )

        # Evaluate checkpoint
        all_results[checkpoint_name] = evaluate_by_first_word(
            model_path=checkpoint_path,
            datasets=datasets,
            positive_token=positive_token,
            negative_token=negative_token,
            create_prompt_func=create_prompt_func,
            output_dir=eval_output_dir,
            prediction_threshold=prediction_threshold
        )

    # Sort results by checkpoint number
    all_results = _sort_checkpoints(all_results)

    # Generate visualizations for each dataset split
    for split_name in data_files.keys():
        auc_scores = []
        accuracy_scores = []

        for checkpoint_name in all_results.keys():
            auc_scores.append(all_results[checkpoint_name][split_name]["auc"])
            accuracy_scores.append(all_results[checkpoint_name][split_name]["accuracy"])

    return all_results

