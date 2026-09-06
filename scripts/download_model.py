#!/usr/bin/env python3
"""
scripts/download_model.py

Downloads the Qwen3 4B GGUF model if not already present.
Features:
- Checks if model already exists and is non-empty.
- Resumable downloading (HTTP Range header) for interrupted connections.
- Atomic write (downloads to temporary file first, then renames).
- Optional SHA-256 integrity verification.
- Safe progress logging with download speed and percentage.
"""

import os
import sys
import time
import hashlib
import requests
from pathlib import Path

# Add project root to sys.path so we can import config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings


def format_size(bytes_num: float) -> str:
    """Format bytes into human-readable representation."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_num < 1024.0:
            return f"{bytes_num:.2f} {unit}"
        bytes_num /= 1024.0
    return f"{bytes_num:.2f} PB"


def verify_sha256(filepath: Path, expected_hash: str) -> bool:
    """Compute and verify SHA256 hash of the downloaded file."""
    print(f"Verifying SHA-256 checksum for {filepath}...")
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):  # 8MB chunks
            sha256.update(chunk)
    actual_hash = sha256.hexdigest().lower()
    matches = actual_hash == expected_hash.lower()
    if matches:
        print(f"Checksum verification PASSED ({actual_hash})")
    else:
        print(f"Checksum verification FAILED! Expected: {expected_hash}, Got: {actual_hash}")
    return matches


def download_model():
    model_path = Path(settings.MODEL_PATH)
    target_dir = model_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check if model already exists and is valid
    if model_path.exists():
        existing_size = model_path.stat().st_size
        # A 4B GGUF Q4_K_M model should typically be > 1.5 GB
        if existing_size > 50 * 1024 * 1024:
            print(f"Model already exists at: {model_path} ({format_size(existing_size)})")
            if settings.MODEL_SHA256:
                if not verify_sha256(model_path, settings.MODEL_SHA256):
                    print("Existing model failed checksum verification. Re-downloading...")
                else:
                    print("Existing model verified successfully. Download skipped.")
                    return
            else:
                print("Download skipped. (Model is present and valid size)")
                return
        else:
            print(f"Found partial or invalid file ({existing_size} bytes). Re-downloading...")
            model_path.unlink(missing_ok=True)

    # If mock mode is explicitly requested, skip downloading huge model
    if settings.MOCK_MODEL:
        print("MOCK_MODEL is set to true. Creating a placeholder model file for mock inference.")
        with open(model_path, "w") as f:
            f.write("# Mock model placeholder for test execution")
        print(f"Mock placeholder created at {model_path}")
        return

    url = settings.MODEL_URL
    temp_path = model_path.with_suffix(".tmp")
    print(f"Preparing to download model:")
    print(f"  URL:  {url}")
    print(f"  File: {model_path.name}")
    print(f"  Dest: {model_path}")

    # 2. Check existing downloaded bytes for resumption
    downloaded_bytes = 0
    if temp_path.exists():
        downloaded_bytes = temp_path.stat().st_size
        print(f"Resuming download from byte offset: {format_size(downloaded_bytes)}")

    headers = {}
    if downloaded_bytes > 0:
        headers["Range"] = f"bytes={downloaded_bytes}-"

    max_retries = 5
    backoff = 2

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Connecting (attempt {attempt}/{max_retries})...")
            response = requests.get(url, headers=headers, stream=True, timeout=30)

            # Handle Range response
            if response.status_code == 416:  # Range Not Satisfiable
                print("Range not satisfiable; server file may have changed or download is complete.")
                temp_path.unlink(missing_ok=True)
                downloaded_bytes = 0
                headers.pop("Range", None)
                continue

            if response.status_code not in (200, 206):
                raise RuntimeError(
                    f"HTTP error {response.status_code} while fetching model: {response.text[:200]}"
                )

            total_size = int(response.headers.get("content-length", 0)) + downloaded_bytes
            print(f"Total file size: {format_size(total_size)}")

            mode = "ab" if downloaded_bytes > 0 else "wb"
            start_time = time.time()
            last_report_time = start_time
            bytes_since_report = 0

            with open(temp_path, mode) as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB buffer
                    if chunk:
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
                        bytes_since_report += len(chunk)

                        current_time = time.time()
                        if current_time - last_report_time >= 5.0:  # Report every 5 seconds
                            elapsed = current_time - last_report_time
                            speed = bytes_since_report / elapsed if elapsed > 0 else 0
                            percent = (
                                (downloaded_bytes / total_size * 100) if total_size > 0 else 0
                            )
                            print(
                                f"Progress: {format_size(downloaded_bytes)} / {format_size(total_size)} "
                                f"({percent:.1f}%) @ {format_size(speed)}/s"
                            )
                            last_report_time = current_time
                            bytes_since_report = 0

            # Completed write
            print("Download stream completed successfully.")
            break

        except (requests.RequestException, RuntimeError) as e:
            print(f"Download interrupted: {e}")
            if attempt < max_retries:
                print(f"Retrying in {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2
                # Update range header for next attempt
                if temp_path.exists():
                    downloaded_bytes = temp_path.stat().st_size
                    headers["Range"] = f"bytes={downloaded_bytes}-"
            else:
                print("Max retries exceeded. Download failed.")
                sys.exit(1)

    # 3. Optional SHA-256 verification
    if settings.MODEL_SHA256:
        if not verify_sha256(temp_path, settings.MODEL_SHA256):
            print("Downloaded file checksum mismatch! Removing corrupt file.")
            temp_path.unlink(missing_ok=True)
            sys.exit(1)

    # 4. Atomic rename from .tmp to final filename
    temp_path.replace(model_path)
    print(f"Successfully saved model to: {model_path} ({format_size(model_path.stat().st_size)})")


if __name__ == "__main__":
    download_model()
