"""
    Prompt instructions designed by clinic experts.
    (Use clear, deterministic - our experiments show that determined Yes/No
    yield better results than 'expected outcomes' hesitate phrases.

    The prompt might change based on the input dataset text field.
    In this type of experiment, we only modify the description of the patient's condition.
"""

CLEAR_CLINIC_POSITIVE_PHRASE = "Presence of the composite clinical outcome will occur"
CLEAR_CLINIC_NEGATIVE_PHRASE = "No, the composite clinical outcome will be absent"

CLEAR_CLINIC_POSITIVE_FIRST_WORD = " Presence"
CLEAR_CLINIC_NEGATIVE_FIRST_WORD = " No"

CLINIC_RESPONSE_TEMPLATE = " ### Answer:"

def clinical_prompt_formatting_func(example, is_train):
    """
    Formats the prompt for the clinical outcome prediction task.

    Args:
        example (dict): A dictionary containing the example data with keys 'text' and 'label'.
        is_train (bool): A flag indicating whether the example is for training.

    Returns:
        str: The formatted prompt.
    """

    text = example['text']
    label = int(example['label'])
    excepted_outcome = CLEAR_CLINIC_POSITIVE_PHRASE if label == 1 else CLEAR_CLINIC_NEGATIVE_PHRASE

    prompt = (
        f"Given the health condition of a patient: {text}\n\n"
        "The task is to predict the presence of composite clinical outcome of the hospitalization.\n"
        "Is the composite clinical outcome expected to be present?"
    )

    if is_train:
        prompt += f" {excepted_outcome}"

    return prompt


def zero_shot_clinical_prompt_formatting_func(example):
    """
    Formats the prompt for the clinical outcome prediction task. For zero-shot inference.
    """
    prompt = clinical_prompt_formatting_func(example, is_train=False)
    prompt += "\n" + "The answer should be a clear, deterministic Presence/No."
    return prompt
