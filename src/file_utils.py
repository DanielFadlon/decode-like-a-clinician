import importlib
import os
import yaml


def call_function_by_path(path: str, *args, **kwargs):
    """
    Dynamically call a function by its full path (e.g. 'my_module.my_submodule.my_function').

    Args:
        path (str): The full dotted path to the function.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the function call.
    """
    module_path, func_name = path.rsplit('.', 1)

    module = importlib.import_module(module_path)
    func = getattr(module, func_name)

    return func(*args, **kwargs)


def read_yaml(file_path: str) -> dict:
    """
    Read and parse a YAML file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Dictionary containing the parsed YAML content.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory_path: Path to the directory to create.
    """
    os.makedirs(directory_path, exist_ok=True)
