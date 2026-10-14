import importlib.util
from pathlib import Path

import virtual_captain
from virtual_captain import paths

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_version():
    assert virtual_captain.__version__


def test_paths_point_inside_repo():
    assert paths.ROOT == ROOT
    assert paths.RAW_DIR.is_relative_to(ROOT)


def test_download_args_default_to_all_sources():
    dl = _load_script("download_data")
    args = dl.parse_args([])
    assert args.source == "all"
    assert args.out == paths.RAW_DIR


def test_download_skips_populated_dir(tmp_path, capsys):
    dl = _load_script("download_data")
    (tmp_path / "already_here.csv").write_text("x")
    dl.download_cricsheet(tmp_path)
    assert "skipping" in capsys.readouterr().out
