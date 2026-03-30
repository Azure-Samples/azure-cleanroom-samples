#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Script to create a KEK locally with a security policy.

This script generates an RSA private key and writes it locally along with
the SKR policy JSON, and optionally imports it to Azure Key Vault/Managed HSM.

Usage from PowerShell:
    python3 create-kek.py `
        --kek-name $kekName `
        --output-dir $outputDir `
        --skr-policy-json $skrPolicyJson `
        --key-vault-url $keyVaultUrl
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def create_kek(
    kek_name: str,
    output_dir: str,
    skr_policy_json: str,
    key_vault_url: Optional[str] = None,
) -> None:
    """
    Create a KEK locally and optionally import to Azure Key Vault.

    :param kek_name: Name of the KEK to create.
    :param output_dir: Directory to write the key and policy files.
    :param skr_policy_json: Full SKR policy as JSON string.
    :param key_vault_url: Optional Azure Key Vault/Managed HSM URL to import the key.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pem_file_path = output_path / f"{kek_name}.pem"
    skr_file_path = output_path / f"{kek_name}-skr-policy.json"

    # Generate RSA private key.
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # Write private key to PEM file.
    private_key_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )

    with open(pem_file_path, "wb") as private_key_file:
        private_key_file.write(private_key_bytes)

    # Write SKR policy to JSON file.
    skr_policy = json.loads(skr_policy_json)
    with open(skr_file_path, "w") as f:
        json.dump(skr_policy, f, indent=2)

    print(f"✅ Created KEK '{kek_name}' in {output_dir}", file=sys.stderr)
    print(f"   Private key: {pem_file_path}", file=sys.stderr)
    print(f"   SKR policy: {skr_file_path}", file=sys.stderr)

    # Import to Azure Key Vault if URL is provided.
    if key_vault_url:
        vault_param = (
            "--hsm-name"
            if ".managedhsm.azure.net" in key_vault_url.lower()
            else "--vault-name"
        )
        hostname = urlparse(key_vault_url).hostname
        assert hostname is not None, f"Invalid Key Vault URL: {key_vault_url}"
        kv_name = hostname.split(".")[0]

        cmd = (
            f"az keyvault key import --name {kek_name} --pem-file {pem_file_path} "
            + f"--policy {skr_file_path} {vault_param} {kv_name} --exportable true "
            + f"--protection hsm --ops encrypt wrapKey --immutable false"
        )

        print(f"   Importing KEK to {key_vault_url}...", file=sys.stderr)
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=False
        )

        if result.returncode != 0:
            print(f"❌ Failed to import KEK to Key Vault:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            sys.exit(1)

        print(f"✅ Successfully imported KEK to {key_vault_url}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Create a KEK locally with security policy"
    )
    parser.add_argument("--kek-name", required=True, help="Name of the KEK to create")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the key and policy files",
    )
    parser.add_argument(
        "--skr-policy-json", required=True, help="Full SKR policy as JSON string"
    )
    parser.add_argument(
        "--key-vault-url",
        required=False,
        help="Optional Azure Key Vault or Managed HSM URL to import the KEK",
    )

    args = parser.parse_args()

    create_kek(
        kek_name=args.kek_name,
        output_dir=args.output_dir,
        skr_policy_json=args.skr_policy_json,
        key_vault_url=args.key_vault_url,
    )


if __name__ == "__main__":
    main()
