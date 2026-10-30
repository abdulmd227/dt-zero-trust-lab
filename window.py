"""Consequence under response delay and safe response windows (V1).

Sweeps the response delay from 0 to 900 s in 1 s steps for each scenario at full
magnitude and reports the first delay at which production loss reaches 1 min,
the SIF trips, and the tank overflows.

Run:  python window.py
"""
import numpy as np
import twin_sim as ts
from batch_twin import simulate_batch

DELAYS = np.arange(0, 901)


def sweep(params=None, mag=1.0):
    n = DELAYS.size
    scn = np.repeat(np.arange(4), n)
    dl = np.tile(DELAYS, 4)
    return scn, dl, simulate_batch(scn, np.full(scn.size, mag), dl, params)


def first(dl, mask):
    idx = np.nonzero(mask)[0]
    return int(dl[idx[0]]) if idx.size else None


def windows(params=None):
    scn, dl, r = sweep(params)
    out = {}
    for s in range(4):
        m = scn == s
        d = dl[m]
        out[ts.SCENARIOS[s]] = dict(
            loss_1min=first(d, r["loss_s"][m] >= 60),
            trip=first(d, r["trip"][m]),
            overflow=first(d, r["overflow"][m]),
            loss_curve_min=(r["loss_s"][m] / 60.0).tolist())
    return out


def delay_table(deltas=(0, 60, 120, 300, 600)):
    rows = {}
    for s in range(4):
        rows[ts.SCENARIOS[s]] = {d: ts.simulate(s, 1.0, d) for d in deltas}
    return rows


if __name__ == "__main__":
    w = windows()
    for k, v in w.items():
        print(f"{k:20s} loss>=1min: {v['loss_1min']}  trip: {v['trip']}  overflow: {v['overflow']}  "
              f"loss@600s: {v['loss_curve_min'][600]:.1f} min")
