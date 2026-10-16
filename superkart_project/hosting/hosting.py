"""Pushes the deployment folder to the Hugging Face Space.

Run by the `deploy-hosting` job of the GitHub Actions workflow.
"""
import os

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

api = HfApi(token=os.getenv("HF_TOKEN"))

SPACE_REPO = "khemanta/superkart-sales-frontend"

try:
    api.repo_info(repo_id=SPACE_REPO, repo_type="space")
    print(f"Space '{SPACE_REPO}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{SPACE_REPO}' not found. Creating it...")
    create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="docker",
                private=False, token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="superkart_project/deployment",
    repo_id=SPACE_REPO,
    repo_type="space",
    path_in_repo="",
)
print(f"Deployment files pushed to https://huggingface.co/spaces/{SPACE_REPO}")
