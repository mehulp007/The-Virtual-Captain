"""What happens when a ball is bowled, before any fielder gets involved.

Two heads share one network:

  contact   P(no contact | bowled, lbw or stumped | put into play | six)
  shot      for a ball put into play (other than a six), a mixture density over
              theta   direction (von Mises, wagon angle in radians)
              aerial  whether it went in the air (caught or dropped)
              dist    where it ended up, as a fraction of the boundary radius (log-normal)

Each mixture component has its own direction, chance of being in the air, and a separate
distance distribution for ground and aerial balls, so edges behind the wicket, lofted hits
and pushes into the ring come out as different components.
"""

import math
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F

from virtual_captain.models import encoding as enc

LOG_2PI = math.log(2 * math.pi)


class ShotNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_batters: int,
        n_bowlers: int,
        n_components: int = 10,
        hidden: int = 256,
        batter_dim: int = 16,
        bowler_dim: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.k = n_components
        self.batter_emb = nn.Embedding(n_batters, batter_dim)
        self.bowler_emb = nn.Embedding(n_bowlers, bowler_dim)
        nn.init.normal_(self.batter_emb.weight, std=0.01)
        nn.init.normal_(self.bowler_emb.weight, std=0.01)
        self.trunk = nn.Sequential(
            nn.Linear(n_features + batter_dim + bowler_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.contact = nn.Linear(hidden, len(enc.OUTCOMES))
        # per component: weight, cos/sin of mean angle, concentration, aerial logit,
        # and log-normal (mu, sigma) for ground and aerial balls
        self.shot = nn.Linear(hidden, n_components * 9)

    def forward(self, x, batter, bowler):
        h = self.trunk(torch.cat([x, self.batter_emb(batter), self.bowler_emb(bowler)], dim=1))
        return self.contact(h), self._mixture(self.shot(h))

    def _mixture(self, raw):
        k = self.k
        logit_w, c, s, kappa, aerial, mu_g, mu_a, sd_g, sd_a = raw.split(k, dim=1)
        return {
            "log_w": F.log_softmax(logit_w, dim=1),
            "loc": torch.atan2(s, c),
            "kappa": F.softplus(kappa) + 0.05,
            "aerial_logit": aerial,
            "mu": torch.stack([mu_g, mu_a], dim=-1),
            "sigma": torch.stack([F.softplus(sd_g), F.softplus(sd_a)], dim=-1) + 0.03,
        }


def shot_log_prob(mix, theta, aerial, dist):
    """log p(theta, aerial, dist) under the mixture, one value per row."""
    theta, aerial, dist = theta[:, None], aerial[:, None], dist[:, None]
    kappa = mix["kappa"]
    # log of the von Mises normaliser 2*pi*I0(kappa), computed stably via i0e
    log_i0 = torch.log(torch.special.i0e(kappa))
    log_vm = kappa * (torch.cos(theta - mix["loc"]) - 1) - LOG_2PI - log_i0
    logit = mix["aerial_logit"]
    log_air = -F.softplus(-logit) * aerial - F.softplus(logit) * (1 - aerial)
    a = aerial.long().unsqueeze(-1).expand(-1, mix["mu"].shape[1], 1)
    mu = mix["mu"].gather(-1, a).squeeze(-1)
    sigma = mix["sigma"].gather(-1, a).squeeze(-1)
    log_d = torch.log(dist)
    log_ln = -log_d - torch.log(sigma) - 0.5 * LOG_2PI - 0.5 * ((log_d - mu) / sigma) ** 2
    return torch.logsumexp(mix["log_w"] + log_vm + log_air + log_ln, dim=1)


@torch.no_grad()
def sample_shots(mix, generator: torch.Generator | None = None):
    """One (theta, aerial, dist) draw per row."""
    comp = torch.multinomial(mix["log_w"].exp(), 1, generator=generator).squeeze(1)
    rows = torch.arange(comp.shape[0], device=comp.device)
    loc, kappa = mix["loc"][rows, comp], mix["kappa"][rows, comp]
    theta = torch.distributions.VonMises(loc, kappa).sample() % (2 * math.pi)
    p_air = torch.sigmoid(mix["aerial_logit"][rows, comp])
    aerial = (torch.rand(p_air.shape, device=p_air.device, generator=generator) < p_air).long()
    mu = mix["mu"][rows, comp, aerial]
    sigma = mix["sigma"][rows, comp, aerial]
    eps = torch.randn(mu.shape, device=mu.device, generator=generator)
    dist = torch.exp(mu + sigma * eps).clamp(enc.D_MIN, enc.D_MAX)
    return theta, aerial.bool(), dist


class ShotModel:
    """A trained ShotNet plus everything needed to feed it rows of the deliveries table."""

    def __init__(self, net: ShotNet, vocab: enc.Vocab, config: dict, device: str = "cpu"):
        self.net = net.to(device).eval()
        self.vocab = vocab
        self.config = config
        self.device = device

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "vocab": self.vocab.to_dict(),
                "config": self.config,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "ShotModel":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        vocab = enc.Vocab(**ckpt["vocab"])
        net = ShotNet(enc.n_features(), vocab.n_batters, vocab.n_bowlers, **ckpt["config"]["net"])
        net.load_state_dict(ckpt["state_dict"])
        return cls(net, vocab, ckpt["config"], device)

    def _tensors(self, df: pl.DataFrame):
        e = enc.encode(df, self.vocab)
        keys = ("x", "batter", "bowler")
        return tuple(torch.from_numpy(np.array(e[k])).to(self.device) for k in keys)

    @torch.no_grad()
    def predict(self, df: pl.DataFrame, batch_size: int = 65536):
        """Contact probabilities (n, 3) and the mixture parameters for every row."""
        probs, mixes = [], []
        for start in range(0, df.height, batch_size):
            logits, mix = self.net(*self._tensors(df.slice(start, batch_size)))
            probs.append(F.softmax(logits, dim=1))
            mixes.append(mix)
        mix = {k: torch.cat([m[k] for m in mixes]) for k in mixes[0]}
        return torch.cat(probs), mix

    @torch.no_grad()
    def sample(self, df: pl.DataFrame, n: int = 1, seed: int = 0) -> pl.DataFrame:
        """Simulate n deliveries per row: contact outcome, then a shot if in play."""
        gen = torch.Generator(device=self.device).manual_seed(seed)
        probs, mix = self.predict(df)
        rep = {k: v.repeat_interleave(n, dim=0) for k, v in mix.items()}
        outcome = torch.multinomial(probs, n, replacement=True, generator=gen).reshape(-1)
        theta, aerial, dist = sample_shots(rep, gen)
        in_play = outcome == enc.IN_PLAY
        return pl.DataFrame(
            {
                "row": np.repeat(np.arange(df.height), n),
                "outcome": outcome.cpu().numpy(),
                "theta": torch.where(in_play, theta, torch.nan).cpu().numpy(),
                "aerial": (aerial & in_play).cpu().numpy(),
                "dist": torch.where(in_play, dist, torch.nan).cpu().numpy(),
            }
        )
