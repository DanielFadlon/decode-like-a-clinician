"""
Clinical Fine-tuning Trainer Package

This package provides a complete training pipeline for fine-tuning language models
on clinical decision-making tasks using PEFT (Parameter-Efficient Fine-Tuning) with LoRA.
"""

from src.trainer.trainer import clinical_finetuning_trainer
from src.trainer.utils import validate_resources, validate_and_read_evaluation_args

__all__ = [
    'clinical_finetuning_trainer',
    'validate_resources',
    'validate_and_read_evaluation_args',
]

