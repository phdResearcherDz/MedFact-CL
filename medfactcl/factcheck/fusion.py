"""Confidence-weighted fusion of claim verdicts into the trust score."""
from typing import Dict, List, Tuple

WEIGHTS = {"SUPPORTED": 1.0, "NOT ENOUGH INFO": 0.0, "REFUTED": -1.0}


def fuse(verdicts: List[Dict]) -> Tuple[float, float]:
    """verdicts: [{"label": y_i, "confidence": tau_i}, ...] -> (tau_d in [-1, 1], tau_x in [0, 1]).

    tau_d = sum_i tau_i w(y_i) / sum_i tau_i and tau_x = (tau_d + 1) / 2.
    No claim, or NEI-only claims, give tau_d = 0 and hence the neutral tau_x = 0.5."""
    num = sum(float(v["confidence"]) * WEIGHTS[v["label"]] for v in verdicts)
    den = sum(float(v["confidence"]) for v in verdicts)
    tau_d = num / den if den > 0 else 0.0
    tau_d = max(-1.0, min(1.0, tau_d))
    return tau_d, (tau_d + 1.0) / 2.0
