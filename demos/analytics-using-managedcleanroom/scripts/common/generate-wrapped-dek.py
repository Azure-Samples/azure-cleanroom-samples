#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Script to wrap an existing DEK with a KEK public key.

The DEK is generated during 'datastore add' and stored locally as a 32-byte binary file.
This script reads the DEK from local storage and wraps it with the KEK public key.

Usage from PowerShell:
    $wrappedDek = python3 generate-wrapped-dek.py `
        --dek-file <path-to-dek-bin-file> `
        --kek-public-key-file <path-to-kek-pem-file>
"""

import argparse
import base64
import sys
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


def wrap_dek(dek_file: str, kek_public_key_file: str) -> str:
    """
    Wrap a DEK with a KEK public key.

    :param dek_file: Path to the DEK file (raw 32-byte key).
    :param kek_public_key_file: Path to the KEK PEM file (private or public key).
    :return: Base64-encoded wrapped DEK.
    """
    # Read the DEK from local file.
    dek_path = Path(dek_file)
    if not dek_path.exists():
        raise FileNotFoundError(f"DEK file not found: {dek_file}")

    with open(dek_path, "rb") as f:
        dek_bytes = f.read()

    if len(dek_bytes) != 32:
        raise ValueError(f"Expected 32-byte DEK, got {len(dek_bytes)} bytes")

    # Read the KEK key from PEM file.
    kek_path = Path(kek_public_key_file)
    if not kek_path.exists():
        raise FileNotFoundError(f"KEK key file not found: {kek_public_key_file}")

    with open(kek_path, "rb") as f:
        kek_pem_bytes = f.read()

    # Try to load as public key first, then as private key (to extract public key).
    try:
        public_key = serialization.load_pem_public_key(kek_pem_bytes)
    except Exception:
        # If it's a private key PEM, extract the public key from it.
        from cryptography.hazmat.primitives.asymmetric import rsa

        private_key = serialization.load_pem_private_key(kek_pem_bytes, password=None)
        public_key = private_key.public_key()

    # Wrap the DEK with the KEK using RSA-OAEP with SHA-256.
    wrapped_dek_bytes = public_key.encrypt(
        dek_bytes,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    # Base64 encode the wrapped DEK.
    return base64.b64encode(wrapped_dek_bytes).decode()


def main():
    parser = argparse.ArgumentParser(description="Wrap a DEK with a KEK public key")
    parser.add_argument(
        "--dek-file",
        required=True,
        help="Path to the DEK file (generated during datastore add)",
    )
    parser.add_argument(
        "--kek-public-key-file",
        required=True,
        help="Path to the KEK PEM file (private or public key)",
    )

    args = parser.parse_args()

    try:
        wrapped_dek = wrap_dek(args.dek_file, args.kek_public_key_file)
        print(wrapped_dek)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
