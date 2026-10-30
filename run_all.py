"""Reproduce every number in the paper and write results.json.

Run:  python run_all.py            (about 1 min on one core)
      python run_all.py --hash     (print only the SHA-256 of the deterministic results, used for V3)
"""
import hashlib
import json
import sys
import twin_sim as ts
import window
import param_robust
import batch_twin


def deterministic_results():
    reqs = ts.generate_requests()
    w = window.windows()
    delay = {s: {str(d): {k: v[k] for k in ("loss_s", "trip", "trip_t", "overflow", "peak_T", "peak_L")}
                 for d, v in rows.items()} for s, rows in window.delay_table().items()}
    corners = param_robust.corners()
    return dict(
        seed=ts.SEED,
        gains=dict(KP_L=ts.KP_L, KI_L=ts.KI_L, KP_T=ts.KP_T, KI_T=ts.KI_T),
        windows={s: {k: v[k] for k in ("loss_1min", "trip", "overflow")} | {"loss_600s_min": v["loss_curve_min"][600]}
                 for s, v in w.items()},
        delay_table=delay,
        policy=ts.policy_comparison(reqs),
        weight_sensitivity=ts.weight_sensitivity(reqs),
        policy_robustness=ts.policy_robustness(),
        plant_physics=param_robust.summarize(corners),
        scalar_batch_matches_of_256=batch_twin.equivalence_check(),
    )


def digest(res):
    return hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()


if __name__ == "__main__":
    res = deterministic_results()
    if "--hash" in sys.argv:
        print(digest(res))
        sys.exit(0)
    res["sha256"] = digest(res)
    res["timing"] = batch_twin.timing()   # machine dependent, excluded from the hash
    res["multicore"] = batch_twin.multicore_timing()
    res["memory"] = batch_twin.memory_footprint()
    with open("results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: res[k] for k in ("windows", "sha256", "timing")}, indent=2))
