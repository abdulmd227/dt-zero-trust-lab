"""Vectorized process twin: advances many independent episodes in lock-step (V5).

Each episode may have its own scenario, magnitude, response delay and plant
parameters. Arithmetic follows twin_sim.simulate operation by operation, so the
batch results match the scalar twin exactly.

Run:  python batch_twin.py   -> timing and scalar/batch equivalence check
"""
import time
import random
import numpy as np
import twin_sim as ts


def simulate_batch(scn, mag, delta, params=None):
    """scn, mag, delta: arrays of length N (scn = -1 for no attack).
    params: dict of arrays (or scalars) for A, q_max, c_v, beta, gamma, T_in, T_amb.
    Returns dict of arrays: loss_s, trip, trip_t, overflow, overflow_t, peak_T, peak_L, L_end, T_end."""
    scn = np.asarray(scn, dtype=np.int64)
    n = scn.size
    mag = np.asarray(mag, dtype=np.float64) * np.ones(n)
    delta = np.asarray(delta, dtype=np.int64) * np.ones(n, dtype=np.int64)
    p = dict(ts.NOMINAL)
    if params:
        p.update(params)
    P = {k: np.asarray(v, dtype=np.float64) * np.ones(n) for k, v in p.items()}
    A, q_max, c_v, beta, gamma, T_amb, T_in0 = (P[k] for k in ("A", "q_max", "c_v", "beta", "gamma", "T_amb", "T_in"))

    # steady state (same expression order as ts.steady_inputs)
    u_in0 = c_v * np.sqrt(np.full(n, ts.L_SP)) / q_max
    q0 = q_max * u_in0
    u_h0 = ts.L_SP * (gamma * (ts.T_SP - T_amb) - (q0 / (A * ts.L_SP)) * (T_in0 - ts.T_SP)) / beta

    L = np.full(n, ts.L_SP)
    T = np.full(n, ts.T_SP)
    iL, iT = u_in0.copy(), u_h0.copy()
    u_in, u_h = u_in0.copy(), u_h0.copy()
    tripped = np.zeros(n, bool)
    trip_t = np.full(n, -1)
    overflow = np.zeros(n, bool)
    ovf_t = np.full(n, -1)
    offspec = np.zeros(n, np.int64)
    peak_T, peak_L = T.copy(), L.copy()
    t_end = ts.ATTACK_S + delta
    is_cred, is_unsafe, is_ransom, is_vendor = (scn == ts.CRED), (scn == ts.UNSAFE), (scn == ts.RANSOM), (scn == ts.VENDOR)
    sp_T_att = ts.T_SP + 35.0 * mag
    sp_L_att = ts.L_SP + 1.6 * mag
    T_in_att = T_in0 - 12.0 * mag
    stuck_u = 0.4 + 0.6 * mag

    def pi(e, integ, kp, ki, mask):
        v = kp * e + integ
        cond = ((v > 0.0) & (v < 1.0)) | ((v >= 1.0) & (e < 0.0)) | ((v <= 0.0) & (e > 0.0))
        integ = np.where(mask & cond, integ + ki * e * ts.DT, integ)
        v = kp * e + integ
        u = np.where(v > 1.0, 1.0, np.where(v < 0.0, 0.0, v))
        return u, integ

    for t in range(ts.EPISODE_S):
        active = (scn >= 0) & (t >= ts.ATTACK_S) & (t < t_end)
        sp_T = np.where(active & is_cred, sp_T_att, ts.T_SP)
        sp_L = np.where(active & is_unsafe, sp_L_att, ts.L_SP)
        bypass = active & is_unsafe
        frozen = active & is_vendor
        T_in = np.where(is_vendor & (t >= ts.ATTACK_S), T_in_att, T_in0)
        stuck = is_ransom & (t >= ts.ATTACK_S + ts.RANSOM_FAULT_S) & (t < t_end)
        upd = (~tripped) & (~frozen)
        new_uin, iL = pi(sp_L - L, iL, ts.KP_L, ts.KI_L, upd)
        new_uh, iT = pi(sp_T - T, iT, ts.KP_T, ts.KI_T, upd)
        u_in = np.where(upd, new_uin, u_in)
        u_h = np.where(upd, new_uh, u_h)
        a_in = np.where(tripped, 0.0, np.where(stuck, stuck_u, u_in))
        a_h = np.where(tripped, 0.0, u_h)
        q_in = q_max * a_in
        dL = (q_in - c_v * np.sqrt(L)) / A
        dT = (q_in / (A * L)) * (T_in - T) + beta * a_h / L - gamma * (T - T_amb)
        L = L + dL * ts.DT
        T = T + dT * ts.DT
        L = np.where(L < ts.L_MIN, ts.L_MIN, L)
        hit = L >= ts.L_OVF
        ovf_t = np.where(hit & ~overflow, t + 1, ovf_t)
        overflow = overflow | hit
        L = np.where(hit, ts.L_OVF, L)
        new_trip = (~tripped) & ((T >= ts.T_TRIP) | ((L >= ts.L_TRIP) & ~bypass))
        trip_t = np.where(new_trip, t + 1, trip_t)
        tripped = tripped | new_trip
        offspec += ((~tripped) & (np.abs(T - ts.T_SP) > ts.SPEC))
        peak_T = np.maximum(peak_T, T)
        peak_L = np.maximum(peak_L, L)
    return dict(loss_s=offspec + np.where(tripped, ts.RESTART_S, 0), trip=tripped, trip_t=trip_t,
                overflow=overflow, overflow_t=ovf_t, peak_T=peak_T, peak_L=peak_L, L_end=L, T_end=T)


def equivalence_check(n=256, seed=7):
    """Compare batch and scalar twin on n randomly drawn episodes; returns number of exact matches."""
    rng = random.Random(seed)
    scn = [rng.randrange(4) for _ in range(n)]
    mag = [0.6 + 0.4 * rng.random() for _ in range(n)]
    dl = [rng.randrange(0, 901) for _ in range(n)]
    b = simulate_batch(scn, mag, dl)
    keys = ("loss_s", "trip", "trip_t", "overflow", "overflow_t", "peak_T", "peak_L", "L_end", "T_end")
    match = 0
    for i in range(n):
        s = ts.simulate(scn[i], mag[i], dl[i])
        match += all(s[k] == b[k][i].item() for k in keys)
    return match


def timing(n_batch=4096, n_scalar_episodes=20, n_score=200000):
    reqs = ts.generate_requests()
    t0 = time.perf_counter()
    for i in range(n_scalar_episodes):
        ts.simulate(i % 4, 1.0, 120)
    t_scalar_ep = (time.perf_counter() - t0) / n_scalar_episodes
    xs = [r["x"] for r in reqs]
    t0 = time.perf_counter()
    for i in range(n_score):
        ts.risk(xs[i % len(xs)])
    t_score = (time.perf_counter() - t0) / n_score
    rng = random.Random(11)
    scn = [rng.randrange(4) for _ in range(n_batch)]
    mag = [0.6 + 0.4 * rng.random() for _ in range(n_batch)]
    dl = [rng.randrange(0, 901) for _ in range(n_batch)]
    simulate_batch(scn[:64], mag[:64], dl[:64])  # warm-up
    t0 = time.perf_counter()
    simulate_batch(scn, mag, dl)
    t_batch = time.perf_counter() - t0
    return dict(scalar_step_us=t_scalar_ep / ts.EPISODE_S * 1e6, scalar_episode_ms=t_scalar_ep * 1e3,
                score_us=t_score * 1e6, batch_n=n_batch, batch_s=t_batch,
                batch_episodes_per_s=n_batch / t_batch,
                batch_step_us=t_batch / (n_batch * ts.EPISODE_S) * 1e6)


def _chunk(args):
    scn, mag, dl = args
    r = simulate_batch(scn, mag, dl)
    return r["loss_s"]


def multicore_timing(n_batch=4096, workers=(1, 2), seed=11, repeats=3):
    """V5: split one batch of independent episodes across worker processes.
    Reports the best of `repeats` runs; results are checked to equal the single-process batch."""
    import multiprocessing as mp
    rng = random.Random(seed)
    scn = [rng.randrange(4) for _ in range(n_batch)]
    mag = [0.6 + 0.4 * rng.random() for _ in range(n_batch)]
    dl = [rng.randrange(0, 901) for _ in range(n_batch)]
    ref = simulate_batch(scn, mag, dl)["loss_s"]
    out = {}
    for w in workers:
        k = n_batch // w
        parts = [(scn[i * k:(i + 1) * k], mag[i * k:(i + 1) * k], dl[i * k:(i + 1) * k]) for i in range(w)]
        with mp.get_context("fork").Pool(w) as pool:
            pool.map(_chunk, [(p[0][:8], p[1][:8], p[2][:8]) for p in parts])  # warm-up
            best = None
            for _ in range(repeats):
                t0 = time.perf_counter()
                res = pool.map(_chunk, parts)
                el = time.perf_counter() - t0
                best = el if best is None else min(best, el)
        same = bool((np.concatenate(res) == ref).all())
        out[w] = dict(seconds=best, episodes_per_s=n_batch / best, matches_single=same)
    base = out[workers[0]]["seconds"]
    for w in workers:
        out[w]["speedup"] = base / out[w]["seconds"]
    return out


def memory_footprint(n_batch=4096):
    """Peak resident memory of a process that runs one batch (MB) and the twin's per-episode state."""
    import resource
    simulate_batch([0] * n_batch, [1.0] * n_batch, [120] * n_batch)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return dict(peak_rss_mb=peak_mb, state_floats_per_episode=6)


if __name__ == "__main__":
    import json
    print(json.dumps(timing(), indent=2))
    print("scalar/batch exact matches:", equivalence_check(), "/ 256")
    print(json.dumps(multicore_timing(), indent=2))
    print(json.dumps(memory_footprint(), indent=2))
