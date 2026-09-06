#!/usr/bin/env python3
"""
scripts/update_endpoint.py

Updates the active LLM server URL in GitHub repository variables or secrets
so that dependent projects like 'yasamarium/llm' (on Vercel) can automatically discover it.
"""

import os
import sys
import requests

GITHUB_API_URL = "https://api.github.com"

def set_repo_variable(token: str, repo: str, var_name: str, var_value: str):
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = f"{GITHUB_API_URL}/repos/{repo}/actions/variables/{var_name}"
    
    # Try updating existing variable
    patch_resp = requests.patch(url, headers=headers, json={"name": var_name, "value": var_value}, timeout=10)
    if patch_resp.status_code == 204:
        print(f"[SUCCESS] Updated variable {var_name} in {repo} to {var_value}")
        return True
    
    # If not found (404), create it
    if patch_resp.status_code == 404:
        create_url = f"{GITHUB_API_URL}/repos/{repo}/actions/variables"
        create_resp = requests.post(create_url, headers=headers, json={"name": var_name, "value": var_value}, timeout=10)
        if create_resp.status_code in (201, 204):
            print(f"[SUCCESS] Created variable {var_name} in {repo} with value {var_value}")
            return True
        else:
            print(f"[WARNING] Failed to create variable in {repo}: HTTP {create_resp.status_code} - {create_resp.text}")
    else:
        print(f"[WARNING] Failed to update variable in {repo}: HTTP {patch_resp.status_code} - {patch_resp.text}")
    return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python update_endpoint.py <PUBLIC_URL>")
        return

    public_url = sys.argv[1].strip()
    if not public_url:
        return

    token = os.getenv("GH_PAT") or os.getenv("GITHUB_TOKEN")
    target_repo = os.getenv("TARGET_REPO", "yasamarium/llm")
    current_repo = os.getenv("GITHUB_REPOSITORY", "yasamarium/llmserver")

    if not token:
        print("[INFO] No GH_PAT token configured. Skipping automated variable update.")
        return

    print(f"Syncing public endpoint URL ({public_url}) to GitHub repository variables...")
    
    # 1. Update in target repo (yasamarium/llm)
    if target_repo:
        set_repo_variable(token, target_repo, "LLMSERVER_URL", public_url)

    # 2. Also update in current repo for convenience
    if current_repo and current_repo != target_repo:
        set_repo_variable(token, current_repo, "LLMSERVER_URL", public_url)

if __name__ == "__main__":
    main()
