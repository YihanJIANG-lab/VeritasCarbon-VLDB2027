"""
Upload the full VeritasCarbon-ESG-35K dataset to Hugging Face.

Prerequisites:
1. Hugging Face account: https://huggingface.co/join
2. Access token: https://huggingface.co/settings/tokens (create a WRITE token)
3. Install: pip install datasets huggingface_hub

Usage:
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxx
    python scripts/upload_to_huggingface.py

Or login interactively:
    hf auth login
    python scripts/upload_to_huggingface.py --no-token
"""

import os
import sys
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi, login

REPO_ID = "YihanJIANG-lab/VeritasCarbon-ESG-35K"
JSONL_PATH = Path("hf_dataset/veritascarbon_esg_35k.jsonl")
README_PATH = Path("hf_dataset/README.md")


def load_jsonl(path: Path):
    """Generator that yields records from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def main():
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
        print("Logged in via HF_TOKEN environment variable.")
    else:
        print("No HF_TOKEN found. Attempting to use cached credentials...")

    if not JSONL_PATH.exists():
        print(f"Error: {JSONL_PATH} not found. Run this script from the repo root.")
        sys.exit(1)

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
        print(f"Dataset repo '{REPO_ID}' is ready.")
    except Exception as e:
        print(f"Repo creation/check failed: {e}")
        print("If the repo already exists, this is fine. Continuing...")

    # Upload JSONL
    print(f"Uploading {JSONL_PATH} ({JSONL_PATH.stat().st_size / 1024 / 1024:.1f} MB)...")
    api.upload_file(
        path_or_fileobj=str(JSONL_PATH),
        path_in_repo="veritascarbon_esg_35k.jsonl",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print("JSONL uploaded.")

    # Upload README
    if README_PATH.exists():
        print(f"Uploading {README_PATH}...")
        api.upload_file(
            path_or_fileobj=str(README_PATH),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        print("README uploaded.")

    # Also push as a Hugging Face Dataset object for `load_dataset()` compatibility
    print("Building Hugging Face Dataset object for direct loading...")
    dataset = Dataset.from_json(str(JSONL_PATH))
    print(f"Dataset object ready: {len(dataset)} records.")

    dataset.push_to_hub(REPO_ID, private=False)
    print(f"\nSuccess! Dataset is live at: https://huggingface.co/datasets/{REPO_ID}")
    print("Users can now load it with:")
    print(f'  from datasets import load_dataset')
    print(f'  ds = load_dataset("{REPO_ID}", split="train")')


if __name__ == "__main__":
    main()
