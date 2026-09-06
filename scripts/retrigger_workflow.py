#!/usr/bin/env python3
"""
scripts/retrigger_workflow.py

Safely triggers a new GitHub Actions workflow run before the 6-hour runner limit.
Safety safeguards:
1. Checks for already queued or in-progress runs to prevent parallel jobs.
2. Checks run frequency (circuit breaker) to prevent infinite loops.
3. Exits cleanly without error if GH_PAT is not provided.
"""

import os
import sys
import time
from datetime import datetime, timezone, timedelta
import requests

GITHUB_API_URL = "https://api.github.com"

def retrigger():
    pat = os.getenv("GH_PAT") or os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY", "yasamarium/llmserver")
    workflow_id = os.getenv("WORKFLOW_ID", "server.yml")
    ref = os.getenv("GITHUB_REF_NAME", "main")
    
    if not pat:
        print("[INFO] No GH_PAT token provided. Workflow re-triggering skipped.")
        print("To enable continuous 24/7 runner chaining, add 'GH_PAT' as a GitHub Secret.")
        return

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {pat}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # 1. Check existing active runs for this workflow
    runs_url = f"{GITHUB_API_URL}/repos/{repo}/actions/workflows/{workflow_id}/runs"
    try:
        resp = requests.get(runs_url, headers=headers, params={"status": "in_progress", "per_page": 5}, timeout=15)
        if resp.status_code == 200:
            in_prog = resp.json().get("workflow_runs", [])
            # Filter out current run if GITHUB_RUN_ID is set
            curr_run_id = os.getenv("GITHUB_RUN_ID")
            other_active = [r for r in in_prog if str(r.get("id")) != str(curr_run_id)]
            if other_active:
                print(f"[GUARD] Active workflow run already exists (Run ID: {other_active[0]['id']}). Skipping trigger to avoid parallel executions.")
                return

        resp_queued = requests.get(runs_url, headers=headers, params={"status": "queued", "per_page": 5}, timeout=15)
        if resp_queued.status_code == 200:
            queued = resp_queued.json().get("workflow_runs", [])
            if queued:
                print(f"[GUARD] Queued workflow run already detected (Run ID: {queued[0]['id']}). Skipping trigger.")
                return

        # 2. Circuit breaker: Check if too many runs were started in the last 2 hours
        all_recent = requests.get(runs_url, headers=headers, params={"per_page": 10}, timeout=15)
        if all_recent.status_code == 200:
            recent_runs = all_recent.json().get("workflow_runs", [])
            now_utc = datetime.now(timezone.utc)
            two_hours_ago = now_utc - timedelta(hours=2)
            runs_last_2h = 0
            for r in recent_runs:
                created_at_str = r.get("created_at")
                if created_at_str:
                    created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                    if created_at > two_hours_ago:
                        runs_last_2h += 1

            if runs_last_2h >= 4:
                print(f"[CIRCUIT BREAKER] Too many recent workflow triggers ({runs_last_2h} runs in the last 2 hours). Aborting automatic dispatch to avoid runaway loop.")
                return

        # 3. Dispatch new workflow run
        dispatch_url = f"{GITHUB_API_URL}/repos/{repo}/actions/workflows/{workflow_id}/dispatches"
        payload = {"ref": ref}
        print(f"Dispatching new workflow run for repo={repo}, workflow={workflow_id}, ref={ref}...")
        dispatch_resp = requests.post(dispatch_url, headers=headers, json=payload, timeout=15)

        if dispatch_resp.status_code == 204:
            print("[SUCCESS] New workflow run triggered successfully.")
        else:
            print(f"[ERROR] Failed to dispatch workflow: HTTP {dispatch_resp.status_code} - {dispatch_resp.text}")

    except Exception as e:
        print(f"[ERROR] Error occurred while attempting to re-trigger workflow: {e}")

if __name__ == "__main__":
    retrigger()
