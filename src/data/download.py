"""
FraudFlow v0.1.0

download.py — Download and validate the Elliptic Bitcoin dataset.

DVC stage: download
Usage: python -m src.data.download --output data/raw

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
from pathlib import Path

logger = logging.getLogger("fraudflow.download")

# Expected files and their approximate sizes (bytes)
EXPECTED_FILES = {
    "elliptic_txs_features.csv": 170_000_000,  # ~170 MB
    "elliptic_txs_classes.csv": 4_000_000,  # ~4 MB
    "elliptic_txs_edgelist.csv": 6_000_000,  # ~6 MB
}

KAGGLE_DATASET = "ellipticco/elliptic-data-set"


def download_elliptic(output_dir: Path) -> None:
    """
    Download the Elliptic Bitcoin dataset from Kaggle.

    Requires kaggle CLI credentials (~/.kaggle/kaggle.json).
    If files already exist and pass validation, download is skipped.

    Args:
        output_dir: Directory to write the CSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already present
    if _validate_files(output_dir):
        logger.info("Dataset already present and valid — skipping download")
        return

    logger.info(f"Downloading {KAGGLE_DATASET} → {output_dir}")

    try:
        import kaggle

        kaggle.api.dataset_download_files(KAGGLE_DATASET, path=str(output_dir), unzip=True)
    except ImportError:
        logger.warning(
            "kaggle package not installed. Install with: pip install kaggle\n"
            "Then place credentials in ~/.kaggle/kaggle.json"
        )
        raise
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        raise

    # The dataset unzips with a subdirectory — flatten if needed
    nested = output_dir / "elliptic_bitcoin_dataset"
    if nested.is_dir():
        for f in nested.iterdir():
            dest = output_dir / f.name
            if not dest.exists():
                shutil.move(str(f), str(dest))
        nested.rmdir()

    if not _validate_files(output_dir):
        raise RuntimeError(f"Downloaded files failed validation. Check {output_dir}")

    logger.info("✓ Download complete and validated")


def _validate_files(directory: Path) -> bool:
    """Check that expected files exist and have reasonable sizes."""
    for fname, min_size in EXPECTED_FILES.items():
        fpath = directory / fname
        if not fpath.exists():
            logger.debug(f"Missing: {fname}")
            return False
        actual_size = fpath.stat().st_size
        # Allow 50% variance in size
        if actual_size < min_size * 0.5:
            logger.debug(f"File too small: {fname} ({actual_size:,} < {min_size:,})")
            return False
    return True


def compute_md5(filepath: Path) -> str:
    """Compute MD5 hash of a file for integrity checking."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Download Elliptic Bitcoin dataset")
    parser.add_argument("--output", "-o", default="data/raw", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    download_elliptic(Path(args.output))


if __name__ == "__main__":
    main()
