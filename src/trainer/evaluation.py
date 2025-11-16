from enum import Enum

class EvaluationArgKey(Enum):
    POSITIVE_TOKEN_ID = 'positive_token_id'
    NEGATIVE_TOKEN_ID = 'negative_token_id'
    LAST_PROMPT_TOKEN_ID = 'last_prompt_token_id'
    K = 'k'
    PREDICTION_THRESHOLD = 'prediction_threshold'

from typing import Dict, Optional, Tuple
from sklearn.metrics import accuracy_score, auc, roc_curve, f1_score
import numpy as np
import torch
from scipy.special import softmax

"""
    Evaluation a binary classification task based on the first response token in the sequence.
"""

def compute_metrics(
        eval_pred: Tuple[np.ndarray, np.ndarray],
        positive_token_id: int,
        negative_token_id: int,
        last_prompt_token_id: Optional[int] = -1,
        pred_threshold: float = 0.5
    ) -> Dict[str, float]:
    """
    Compute the metrics for a binary classification task based on the first response token in the sequence.

    Parameters:
    -----------

    eval_pred: Tuple[np.ndarray, np.ndarray]
        The logits_inforamtion (from preproceesing logits function) and the labels for the evaluation dataset.

    positive_token_id: int
        The token id for the positive class.

    negative_token_id: int
        The token id for the negative class.

    last_prompt_token_id: Optional[int]
        The token id for the last prompt token. If not provided, the first response token will be used.

    pred_threshold: float
        The threshold to convert the probabilities to binary classes. Default is 0.5.

    Returns:
    --------
    Dict[str, float]
        The dictionary containing the computed metrics.

    Notes:
    ------
    To achieve the probabilities for the positive and negative classes, we should consider the previos token logits.
    If the last_prompt_token_id is provided, we will use it directly to get the probabilities.
    Otherwise, we will use the first response token in the sequence and then decrease one.
    """
    should_find_by_first_response_token = last_prompt_token_id == -1
    logits_info, labels = eval_pred
    logits, ids = logits_info

    num_packs, num_instances_in_pack, k = logits.shape
    flatten_logits = logits.reshape(num_packs * num_instances_in_pack, k)
    flatten_ids = ids.reshape(num_packs * num_instances_in_pack, k)

    probs = softmax(flatten_logits, axis=1)

    def get_token_probs(epsilon=1e-10):
        # Initialize arrays for positive and negative probabilities - epsilon for those who not return in topK
        positive_probs = np.full((num_packs * num_instances_in_pack,), epsilon)
        negative_probs = np.full((num_packs * num_instances_in_pack,), epsilon)

        # Iterate over each instance and assign the correct probabilities
        for i in range(num_packs * num_instances_in_pack):
            ids_i = flatten_ids[i]
            probs_i = probs[i]

            # Check if the positive token is in the top-k ids
            if positive_token_id in ids_i:
                positive_probs[i] = probs_i[ids_i == positive_token_id].item()

            # Check if the negative token is in the top-k ids
            if negative_token_id in ids_i:
                negative_probs[i] = probs_i[ids_i == negative_token_id].item()

        return positive_probs, negative_probs

    positive_probs, negative_probs = get_token_probs()

    normalized_positive_prob = positive_probs / (positive_probs + negative_probs)
    y_pred_probs = normalized_positive_prob.tolist()

    count_guesses = 0
    y_true = []
    mask = []
    for label_i in labels:
        if should_find_by_first_response_token:
            first_response_token_indexes = np.where((label_i == negative_token_id) | (label_i == positive_token_id))[0]
        else:
            first_response_token_indexes = np.where((label_i == last_prompt_token_id))[0] + 1

        # for packing support
        for first_response_token_idx in first_response_token_indexes:
            if first_response_token_idx >= len(label_i):
                mask.append(False)
                count_guesses += 1
                print(f"Warning: first_response_token_idx is out of bounds. Guess number: {count_guesses}")
            else:
                y_true.append(1 if label_i[first_response_token_idx] == positive_token_id else 0)
                mask.append(True)

    y_pred_probs = np.array(y_pred_probs)[mask].astype(float)
    y_true = np.array(y_true).astype(int)

    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    y_pred = np.where(y_pred_probs > pred_threshold, 1, 0)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": auc(fpr, tpr),
        "threshold": pred_threshold
    }


def preprocess_logits_for_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    last_prompt_token_id: Optional[int] = -1,
    positive_token_id: Optional[int] = -1,
    negative_token_id: Optional[int] = -1,
    k: int = 1000
):
    """
    Original Trainer may have a memory leak OR just vocab size is too large.
    This is a workaround to avoid storing too many tensors that are not needed.

    Assuming the sum of the top-k logit's probabilities is (close to) ~1.
    Then, pass just the top-k logits and their indices to the compute_metrics method.

    Additionally, we related just to the first response token in the sequence.
    Since, our evaluation method is based on the first response token.

    Given logits shape (batch_size, aeq_length, vocab_size),
    for each instance, we extract the top-k logits and their indices.

    Imprtant Note:
    --------------
        You must pass the last prompt token id or the first response token id(positive and negative).
        We use one of them to extract the right probabilities for the positive and negative classes (from the last prompt token).


    Parameters:
    -----------

    logits: shape (batch_size, seq_length, vocab_size)

    labels: shape (batch_size, seq_length)

    last_prompt_token_id: Optional[int]
        The token id for the last prompt token. If not provided, the first response token will be used.

    positive_token_id: Optional[int]
        The token id for the positive class.

    negative_token_id: Optional[int]
        The token id for the negative class.

    k: int
        The number of top-k logits to extract. Default is 1000.

    Return:
    -------

    top-k logits: shape (batch_size, num_instances_in_pack, k)

    top-k ids: shape (batch_size, num_instances_in_pack, k)
    """

    if last_prompt_token_id == -1 and (positive_token_id == -1 or negative_token_id == -1):
        raise ValueError("You should provide on of the following: \n 1. positive and negative tokens ids.\n 2. last prompt token id.")

    should_find_by_first_response_token = last_prompt_token_id == -1
    last_prompt_token_indexes = []
    if should_find_by_first_response_token:
        last_prompt_token_indexes = [torch.where((label == negative_token_id) | (label == positive_token_id))[0] - 1 for label in labels]
    else:
        last_prompt_token_indexes = [torch.where((label == last_prompt_token_id))[0] for label in labels]

    max_length = max([len(idxs) for idxs in last_prompt_token_indexes])
    # Pad the tensors to the maximum pack length (some packs may have less instances than others)
    padded_logits = []
    for logit, idxs in zip(logits, last_prompt_token_indexes):
        padded_logit = torch.zeros(max_length, * logit.shape[1:], device=logit.device)
        padded_logit[:len(idxs)] = logit[idxs]
        padded_logits.append(padded_logit)

    stacked_logits = torch.stack(padded_logits)

    # Extract the top-k logits and their indices
    topk_logits, topk_ids = torch.topk(stacked_logits, k, dim=-1)

    return topk_logits, topk_ids
