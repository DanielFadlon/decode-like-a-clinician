#!/usr/bin/env python3
"""
Helper script to determine token IDs for zero-shot evaluation.

This script helps you find the correct token IDs for '0', '1', 'True', 'False',
or any other tokens you want to use for binary classification.

Usage:
    python get_token_ids.py --model_path path/to/model
    python get_token_ids.py --model_path path/to/model --tokens "Yes" "No"
    python get_token_ids.py --model_path path/to/model --trust_remote_code
"""

import argparse
from transformers import AutoTokenizer


def get_token_ids(model_path, tokens, trust_remote_code=False):
    """
    Get token IDs for specified tokens.

    Args:
        model_path: Path to the model
        tokens: List of token strings to encode
        trust_remote_code: Whether to trust remote code

    Returns:
        Dictionary mapping tokens to their IDs
    """
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code
    )

    token_ids = {}

    print("\n" + "=" * 60)
    print(f"{'TOKEN IDs':^60}")
    print("=" * 60)

    for token in tokens:
        # Try encoding with and without special tokens
        ids_with_special = tokenizer.encode(token, add_special_tokens=True)
        ids_without_special = tokenizer.encode(token, add_special_tokens=False)

        print(f"\nToken: '{token}'")
        print(f"  With special tokens: {ids_with_special}")
        print(f"  Without special tokens: {ids_without_special}")

        # Usually we want the first token without special tokens
        if ids_without_special:
            token_id = ids_without_special[0]
            token_ids[token] = token_id
            print(f"  → Recommended ID: {token_id}")

            # Verify by decoding
            decoded = tokenizer.decode(token_id)
            print(f"  → Decodes to: '{decoded}'")

    print("=" * 60)

    # Show vocabulary info
    print(f"\nTokenizer vocabulary size: {len(tokenizer)}")
    print(f"Model max length: {tokenizer.model_max_length}")

    return token_ids


def explore_related_tokens(model_path, base_tokens, trust_remote_code=False):
    """
    Explore tokens related to the base tokens.

    Args:
        model_path: Path to the model
        base_tokens: List of base token strings
        trust_remote_code: Whether to trust remote code
    """
    print(f"\nLoading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code
    )

    print("\n" + "=" * 60)
    print(f"{'EXPLORING RELATED TOKENS':^60}")
    print("=" * 60)

    for token in base_tokens:
        print(f"\nBase token: '{token}'")

        # Try different variations
        variations = [
            token,
            token.lower(),
            token.upper(),
            token.capitalize(),
            f" {token}",  # With leading space
            f"{token} ",  # With trailing space
            f" {token.lower()}",
            f" {token.upper()}",
        ]

        seen_ids = set()
        for variant in variations:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if ids and ids[0] not in seen_ids:
                seen_ids.add(ids[0])
                decoded = tokenizer.decode(ids[0])
                print(f"  '{variant}' → ID: {ids[0]}, Decodes to: '{decoded}'")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Get token IDs for zero-shot evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model or model name on HuggingFace Hub"
    )

    parser.add_argument(
        "--tokens",
        type=str,
        nargs="+",
        default=["0", "1"],
        help="Tokens to get IDs for (default: '0' '1')"
    )

    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="Trust remote code when loading tokenizer"
    )

    parser.add_argument(
        "--explore",
        action="store_true",
        default=False,
        help="Explore related token variations"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Get token IDs
    token_ids = get_token_ids(
        args.model_path,
        args.tokens,
        args.trust_remote_code
    )

    # Optionally explore variations
    if args.explore:
        explore_related_tokens(
            args.model_path,
            args.tokens,
            args.trust_remote_code
        )

    # Print configuration snippet
    print("\n" + "=" * 60)
    print(f"{'CONFIGURATION SNIPPET':^60}")
    print("=" * 60)
    print("\n# Add these to your zero_shot_config.yaml:")

    if len(args.tokens) >= 2:
        print(f"true_token_id: {token_ids.get(args.tokens[1], 'UNKNOWN')}  # Token: '{args.tokens[1]}'")
        print(f"false_token_id: {token_ids.get(args.tokens[0], 'UNKNOWN')}  # Token: '{args.tokens[0]}'")
    else:
        for token, token_id in token_ids.items():
            print(f"{token}_token_id: {token_id}  # Token: '{token}'")

    print()


if __name__ == "__main__":
    main()

