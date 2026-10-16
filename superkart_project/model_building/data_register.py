"""Registers the raw dataset on the Hugging Face Hub as a dataset repository.

Run by the `register-dataset` job of the GitHub Actions workflow. The token comes from the
HF_TOKEN repository secret, never from a hard-coded string.
"""
import os

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

repo_id = "khemanta/superkart-sales-prediction"
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating it...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=os.getenv("HF_TOKEN"))
    print(f"Dataset repo '{repo_id}' created.")

api.upload_folder(
    folder_path="superkart_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"Uploaded superkart_project/data -> https://huggingface.co/datasets/{repo_id}")
