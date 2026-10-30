"""Sensitivity of the safe response windows to plant physics (V6).

beta, gamma, c_v and q_max are each set to 0.8 or 1.2 times nominal (16 corners)
and every safe response window is recomputed.

Run:  python param_robust.py
"""
import itertools
import twin_sim as ts
from window import windows

KEYS = ("beta", "gamma", "c_v", "q_max")


def corners():
    res = []
    for f in itertools.product((0.8, 1.2), repeat=4):
        p = {k: ts.NOMINAL[k] * fk for k, fk in zip(KEYS, f)}
        w = windows(p)
        res.append(dict(factors=dict(zip(KEYS, f)),
                        windows={s: {k: v[k] for k in ("loss_1min", "trip", "overflow")} for s, v in w.items()}))
    return res


def summarize(res):
    S = ts.SCENARIOS
    def rng(s, k):
        v = [c["windows"][s][k] for c in res if c["windows"][s][k] is not None]
        return (min(v), max(v), sum(c["windows"][s][k] is None for c in res)) if v else (None, None, len(res))
    summ = {s: {k: rng(s, k) for k in ("loss_1min", "trip", "overflow")} for s in S}
    def earliest(c, k):
        vals = {s: c["windows"][s][k] for s in S if c["windows"][s][k] is not None}
        if not vals:
            return []
        m = min(vals.values())
        return [s for s, v in vals.items() if v == m]
    summ["earliest_trip_counts"] = {s: sum(s in earliest(c, "trip") for c in res) for s in S}
    summ["shortest_loss_counts"] = {s: sum(s in earliest(c, "loss_1min") for c in res) for s in S}
    summ["vendor_trip_corners"] = sum(c["windows"]["Vendor firmware"]["trip"] is not None for c in res)
    return summ


if __name__ == "__main__":
    import json
    print(json.dumps(summarize(corners()), indent=2))
