"""Pushes the Gradio front end to a free Hugging Face Space.

Used instead of hosting.py while the account is on the free tier, where Docker Spaces return
402 Payment Required. The model, the inputs and the predictions are identical.
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
    create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="gradio",
                private=False, token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="superkart_project/deployment_gradio",
    repo_id=SPACE_REPO,
    repo_type="space",
    path_in_repo="",
)
print(f"Gradio app pushed to https://huggingface.co/spaces/{SPACE_REPO}")
