"""Prioritized, de-duplicated alerting plus a naive threshold monitor for comparison."""
from dataclasses import dataclass

WARNING, CRITICAL = 0.70, 0.90
CONSECUTIVE = 3   # sustained minutes required (filters one-tick motion artifacts)
COOLDOWN = 30     # minutes before re-alerting at the same tier

# classic fixed-threshold monitor limits: (low, high)
LIMITS = {"hr": (50, 100), "sbp": (90, 180), "spo2": (94, 101), "rr": (10, 20), "temp": (35.5, 38.0)}
LIMIT_IDX = {"hr": 0, "sbp": 1, "spo2": 2, "rr": 3, "temp": 4}


@dataclass
class Alert:
    t: int
    pid: str
    tier: str
    risk: float


class AlertEngine:
    def __init__(self):
        self.streak, self.last_t, self.last_tier = {}, {}, {}

    def update(self, t: int, pid: str, r: float):
        tier = "CRITICAL" if r >= CRITICAL else "WARNING" if r >= WARNING else None
        self.streak[pid] = self.streak.get(pid, 0) + 1 if tier else 0
        if not tier or self.streak[pid] < CONSECUTIVE:
            return None
        escalated = tier == "CRITICAL" and self.last_tier.get(pid) != "CRITICAL"
        cooled = t - self.last_t.get(pid, -10**9) >= COOLDOWN
        if escalated or cooled:
            self.last_t[pid], self.last_tier[pid] = t, tier
            return Alert(t, pid, tier, round(r, 3))
        return None


class NaiveMonitor:
    """Fires on every transition from in-range to out-of-range for any single vital."""

    def __init__(self):
        self.out = {}

    def update(self, pid: str, reading) -> bool:
        breach = any(not (lo <= reading[LIMIT_IDX[k]] <= hi) for k, (lo, hi) in LIMITS.items())
        fired = breach and not self.out.get(pid, False)
        self.out[pid] = breach
        return fired
