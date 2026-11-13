"""Train the shot model on seasons before 2025 and evaluate it on 2025-26.

Usage:
  uv run python scripts/train_shot_model.py
  uv run python scripts/train_shot_model.py --epochs 3      # quick run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from virtual_captain.models import encoding as enc  # noqa: E402
from virtual_captain.models.baselines import BatterHistogram  # noqa: E402
from virtual_captain.models.shot_model import ShotModel, ShotNet, shot_log_prob  # noqa: E402
from virtual_captain.paths import DATA_DIR, PROCESSED_DIR, REPORTS_DIR, ROOT  # noqa: E402

CHECKPOINT = DATA_DIR / "models" / "shot_model.pt"


def load_frame() -> pl.DataFrame:
    lf = pl.scan_parquet(PROCESSED_DIR / "deliveries.parquet")
    return lf.pipe(enc.tracked_innings).pipe(enc.add_labels).pipe(enc.split_column).collect()


def to_device(df: pl.DataFrame, vocab: enc.Vocab, device: str) -> dict[str, torch.Tensor]:
    arrays = {**enc.encode(df, vocab), **enc.labels(df)}
    return {k: torch.from_numpy(np.array(v)).to(device) for k, v in arrays.items()}


def batch_loss(net, b, idx):
    logits, mix = net(b["x"][idx], b["batter"][idx], b["bowler"][idx])
    outcome = b["outcome"][idx]
    contact = F.cross_entropy(logits, outcome, reduction="none")
    in_play = outcome == enc.IN_PLAY
    shot = -shot_log_prob(
        {k: v[in_play] for k, v in mix.items()},
        b["theta"][idx][in_play],
        b["aerial"][idx][in_play],
        b["dist"][idx][in_play],
    )
    return contact, shot


@torch.no_grad()
def evaluate(net, b, batch_size=65536):
    net.eval()
    n = b["x"].shape[0]
    contact, shot = [], []
    for start in range(0, n, batch_size):
        idx = torch.arange(start, min(start + batch_size, n), device=b["x"].device)
        c, s = batch_loss(net, b, idx)
        contact.append(c)
        shot.append(s)
    contact, shot = torch.cat(contact), torch.cat(shot)
    return contact.mean().item(), shot.mean().item(), contact.cpu().numpy(), shot.cpu().numpy()


def train(args, frame, device):
    train_df = frame.filter(pl.col("split") == "train")
    vocab = enc.Vocab.fit(train_df)
    tr = to_device(train_df, vocab, device)
    va = to_device(frame.filter(pl.col("split") == "val"), vocab, device)

    config = {
        "net": {
            "n_components": args.components,
            "hidden": args.hidden,
            "batter_dim": args.batter_dim,
        },
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
    }
    net = ShotNet(enc.n_features(), vocab.n_batters, vocab.n_bowlers, **config["net"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    n = tr["x"].shape[0]
    best, best_state, bad = float("inf"), None, 0
    gen = torch.Generator(device=device).manual_seed(args.seed)
    for epoch in range(args.epochs):
        net.train()
        t0 = time.perf_counter()
        perm = torch.randperm(n, device=device, generator=gen)
        total = 0.0
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            contact, shot = batch_loss(net, tr, idx)
            loss = contact.mean() + shot.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(idx)
        sched.step()

        val_contact, val_shot, _, _ = evaluate(net, va)
        val = val_contact + val_shot
        mlflow.log_metrics(
            {"train_loss": total / n, "val_contact_nll": val_contact, "val_shot_nll": val_shot},
            step=epoch,
        )
        print(
            f"epoch {epoch + 1:2d}  train {total / n:.4f}  val contact {val_contact:.4f}  "
            f"shot {val_shot:.4f}  ({time.perf_counter() - t0:.0f}s)"
        )
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break

    net.load_state_dict(best_state)
    return ShotModel(net, vocab, config, device)


def fit_baseline(frame: pl.DataFrame) -> BatterHistogram:
    """Per-batter histograms with the smoothing strength picked on the validation split."""
    train_df = frame.filter(pl.col("split") == "train")
    val = frame.filter(pl.col("split") == "val")
    val_shots = val.filter(pl.col("outcome") == enc.IN_PLAY)
    best, best_score = None, float("inf")
    for outcome_prior, shot_prior in [(20, 50), (50, 200), (200, 1000), (1000, 5000)]:
        b = BatterHistogram(outcome_prior, shot_prior).fit(train_df)
        score = -b.outcome_log_prob(val).mean() - b.shot_log_prob(val_shots).mean()
        print(f"baseline priors {outcome_prior}/{shot_prior}: val nll {score:.4f}")
        if score < best_score:
            best, best_score = b, score
    return best


def evaluate_test(model: ShotModel, frame: pl.DataFrame, baseline: BatterHistogram) -> dict:
    test = frame.filter(pl.col("split") == "test")
    te = to_device(test, model.vocab, model.device)
    _, _, contact, shot = evaluate(model.net, te)
    shots = test.filter(pl.col("outcome") == enc.IN_PLAY)
    return {
        "test_balls": test.height,
        "test_shots": shots.height,
        "contact_nll": {
            "shot_model": float(contact.mean()),
            "batter_histogram": float(-baseline.outcome_log_prob(test).mean()),
            "population": float(-baseline.global_outcome_log_prob(test).mean()),
        },
        "shot_nll": {
            "shot_model": float(shot.mean()),
            "batter_histogram": float(-baseline.shot_log_prob(shots).mean()),
            "population": float(-baseline.global_shot_log_prob(shots).mean()),
        },
    }


def calibration(model: ShotModel, frame: pl.DataFrame) -> pl.DataFrame:
    """Observed vs simulated rates on the held-out seasons, by format and phase."""
    test = frame.filter(pl.col("split") == "test")
    sim = model.sample(test, n=1, seed=1)
    both = test.select("format", "phase", "outcome", "aerial", "theta", "dist").with_columns(
        sim["outcome"].alias("sim_outcome"),
        sim["aerial"].alias("sim_aerial"),
        sim["theta"].alias("sim_theta"),
        sim["dist"].alias("sim_dist"),
    )

    def rates(prefix):
        o, a, t, d = (f"{prefix}{c}" for c in ("outcome", "aerial", "theta", "dist"))
        play = pl.col(o) == enc.IN_PLAY
        return [
            (pl.col(o) == enc.IN_PLAY).mean().alias(f"{prefix}in_play"),
            (pl.col(o) == enc.FIELD_FREE_OUT).mean().alias(f"{prefix}bowled_lbw"),
            (pl.col(o) == enc.SIX).mean().alias(f"{prefix}six"),
            pl.col(a).filter(play).mean().alias(f"{prefix}aerial"),
            (pl.col(t).filter(play) >= np.pi).mean().alias(f"{prefix}off_side"),
            pl.col(d).filter(play).median().alias(f"{prefix}median_dist"),
        ]

    return both.group_by("format", "phase").agg(*rates(""), *rates("sim_")).sort("format", "phase")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--components", type=int, default=16)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--batter-dim", type=int, default=32)
    p.add_argument("--out", type=Path, default=CHECKPOINT)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame = load_frame()
    counts = frame.group_by("split").len().sort("split")
    print(dict(zip(counts["split"], counts["len"], strict=True)), "device:", device)

    (ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlruns' / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("shot_model")
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        model = train(args, frame, device)
        model.save(args.out)

        baseline = fit_baseline(frame)
        metrics = evaluate_test(model, frame, baseline)
        mlflow.log_metrics(
            {
                f"test_{kind}_{name}": v
                for kind in ("contact_nll", "shot_nll")
                for name, v in metrics[kind].items()
            }
        )
        cal = calibration(model, frame)

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "shot_model_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    cal.write_csv(REPORTS_DIR / "shot_model_calibration.csv", float_precision=3)
    print(json.dumps(metrics, indent=2))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3, tbl_hide_dataframe_shape=True):
        print(cal)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
