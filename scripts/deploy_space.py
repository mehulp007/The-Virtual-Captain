"""Put the demo app on Hugging Face Spaces.

Stages the files a Docker Space needs (space/, the package, configs, the Streamlit theme and
data/app/bundle.npz from scripts/build_app_bundle.py) and uploads them. Needs a write token:
run `uv run hf auth login` once first.

Usage:
  uv run python scripts/deploy_space.py                  # <your user>/the-virtual-captain
  uv run python scripts/deploy_space.py --space someone/name
  uv run python scripts/deploy_space.py --stage-only build/space   # just lay the files out
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from virtual_captain.app.bundle import BUNDLE
from virtual_captain.paths import ROOT

NAME = "the-virtual-captain"


def stage(out: Path) -> Path:
    if not BUNDLE.exists():
        raise SystemExit(f"{BUNDLE} is missing: run scripts/build_app_bundle.py first")
    out.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "space").iterdir():
        shutil.copy(f, out / f.name)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "src" / "virtual_captain", out / "src" / "virtual_captain",
                    ignore=ignore, dirs_exist_ok=True)  # fmt: skip
    shutil.copytree(ROOT / "configs", out / "configs", dirs_exist_ok=True)
    shutil.copytree(ROOT / ".streamlit", out / ".streamlit", dirs_exist_ok=True)
    shutil.copy(ROOT / "LICENSE", out / "LICENSE")
    (out / "data" / "app").mkdir(parents=True, exist_ok=True)
    shutil.copy(BUNDLE, out / "data" / "app" / BUNDLE.name)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--space", help="owner/name of the Space")
    p.add_argument("--stage-only", type=Path, help="lay the files out here and stop")
    args = p.parse_args(argv)
    if args.stage_only:
        print(f"staged in {stage(args.stage_only)}")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    repo = args.space or f"{api.whoami()['name']}/{NAME}"
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        api.upload_folder(
            folder_path=stage(Path(tmp)),
            repo_id=repo,
            repo_type="space",
            commit_message="update the app",
            delete_patterns=["src/**", "configs/**"],
        )
    print(f"https://huggingface.co/spaces/{repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
