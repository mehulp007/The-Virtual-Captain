"""Download the raw datasets into data/raw.

Sources:
  kaggle     The Ultimate Ball-by-Ball Cricket Dataset (Cricsheet + Cricinfo tracking).
             Needs a Kaggle API token, see https://www.kaggle.com/docs/api
  cricsheet  Cricsheet's all-matches JSON archive, used to top up recent matches.

Usage:
  uv run python scripts/download_data.py              # both sources
  uv run python scripts/download_data.py --source kaggle
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from virtual_captain.paths import RAW_DIR

KAGGLE_SLUG = "ariadaikalam/the-ultimate-ball-by-ball-cricket-dataset"
CRICSHEET_URL = "https://cricsheet.org/downloads/all_json.zip"

SOURCES = ("kaggle", "cricsheet")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source", choices=(*SOURCES, "all"), default="all")
    p.add_argument("--out", type=Path, default=RAW_DIR, help="raw data directory")
    p.add_argument("--force", action="store_true", help="re-download even if files exist")
    return p.parse_args(argv)


def has_files(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def download_kaggle(dest: Path, force: bool = False) -> None:
    if has_files(dest) and not force:
        print(f"kaggle: {dest} already populated, skipping (use --force to refresh)")
        return
    # imported here because the kaggle package tries to authenticate on import
    from kaggle.api.kaggle_api_extended import KaggleApi

    dest.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"kaggle: downloading {KAGGLE_SLUG} -> {dest}")
    api.dataset_download_files(KAGGLE_SLUG, path=str(dest), unzip=True, quiet=False)


def download_cricsheet(dest: Path, force: bool = False) -> None:
    if has_files(dest) and not force:
        print(f"cricsheet: {dest} already populated, skipping (use --force to refresh)")
        return
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "all_json.zip"

    print(f"cricsheet: downloading {CRICSHEET_URL}")
    with requests.get(CRICSHEET_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            open(archive, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc="all_json.zip") as bar,
        ):
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest / "json")
    archive.unlink()
    print(f"cricsheet: extracted to {dest / 'json'}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sources = SOURCES if args.source == "all" else (args.source,)

    if "kaggle" in sources:
        download_kaggle(args.out / "ultimate", force=args.force)
    if "cricsheet" in sources:
        download_cricsheet(args.out / "cricsheet", force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
