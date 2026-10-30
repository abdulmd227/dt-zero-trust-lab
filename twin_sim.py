"""Process twin, safety function, scenarios, risk score and policy comparison.

Digital-Twin-Driven Experiential Learning for Cyber-Physical Security (MCSoC 2026).
Heated mixing tank (level L, temperature T) with two PI loops in the PLC and an
independent SIF. Explicit Euler integration with a 1 s step over a 40 min episode.

Run:  python twin_sim.py          -> prints the access-policy comparison (V2)
"""
import math
import random

# ----------------------------------------------------------------- parameters
NOMINAL = dict(
    A=2.0,          # tank area, m^2
    q_max=0.03,     # max inflow, m^3/s
    c_v=0.01,       # outflow coefficient, m^2.5/s
    beta=0.6,       # heater gain, K*m/s
    gamma=5e-4,     # heat-loss coefficient, 1/s
    T_in=20.0,      # nominal feed temperature, degC
    T_amb=20.0,     # ambient temperature, degC
)
L_SP, T_SP = 1.5, 60.0           # setpoints
SPEC = 5.0                       # product on spec while |T - 60| <= 5
T_TRIP, L_TRIP, L_OVF = 85.0, 2.5, 3.0
L_MIN = 0.05                     # numerical floor while a tripped tank drains
RESTART_S = 900                  # 15 min restart after a SIF trip
DT, EPISODE_S, ATTACK_S = 1.0, 2400, 300

# PI gains (velocity-limited by output clamping with conditional integration)
KP_L, KI_L = 2.0, 0.01           # level loop: 1/m, 1/(m*s)
KP_T, KI_T = 0.25, 0.005         # temperature loop: 1/K, 1/(K*s)

CRED, UNSAFE, RANSOM, VENDOR = 0, 1, 2, 3
SCENARIOS = ["Credential misuse", "Unsafe engineering", "Ransomware", "Vendor firmware"]
RANSOM_FAULT_S = 120             # valve sticks 2 min after the HMI is lost


def steady_inputs(p):
    """Valve opening and heater duty that hold L = 1.5 m and T = 60 degC."""
    u_in = p["c_v"] * math.sqrt(L_SP) / p["q_max"]
    q0 = p["q_max"] * u_in
    u_h = L_SP * (p["gamma"] * (T_SP - p["T_amb"]) - (q0 / (p["A"] * L_SP)) * (p["T_in"] - T_SP)) / p["beta"]
    return u_in, u_h


def _pi(e, integ, kp, ki):
    """PI with output clamp to [0, 1] and conditional integration (anti-windup)."""
    v = kp * e + integ
    if (0.0 < v < 1.0) or (v >= 1.0 and e < 0.0) or (v <= 0.0 and e > 0.0):
        integ = integ + ki * e * DT
    v = kp * e + integ
    u = 1.0 if v > 1.0 else (0.0 if v < 0.0 else v)
    return u, integ


def simulate(scn, mag=1.0, delta=0, p=None, record=False):
    """Simulate one 40 min episode.

    scn   scenario index (CRED, UNSAFE, RANSOM, VENDOR) or None for no attack
    mag   attack magnitude in [0, 1]
    delta defender response delay in s (effects run from ATTACK_S to ATTACK_S + delta)
    Returns loss_s (off-spec seconds + 900 s if tripped), trip, trip_t, overflow,
    overflow_t, peak_T, peak_L and, if record, the T and L traces.
    """
    p = p or NOMINAL
    A, q_max, c_v, beta, gamma, T_amb = p["A"], p["q_max"], p["c_v"], p["beta"], p["gamma"], p["T_amb"]
    u_in0, u_h0 = steady_inputs(p)
    L, T = L_SP, T_SP
    iL, iT = u_in0, u_h0
    u_in, u_h = u_in0, u_h0
    tripped, trip_t, overflow, ovf_t = False, -1, False, -1
    offspec, peak_T, peak_L = 0, T, L
    trT, trL = [], []
    t_end = ATTACK_S + delta
    for k in range(EPISODE_S):
        t = k
        active = scn is not None and ATTACK_S <= t < t_end
        sp_T = T_SP + 35.0 * mag if (active and scn == CRED) else T_SP
        sp_L = L_SP + 1.6 * mag if (active and scn == UNSAFE) else L_SP
        bypass = active and scn == UNSAFE
        frozen = active and scn == VENDOR
        T_in = p["T_in"] - 12.0 * mag if (scn == VENDOR and t >= ATTACK_S) else p["T_in"]
        stuck = scn == RANSOM and ATTACK_S + RANSOM_FAULT_S <= t < t_end
        if tripped:
            a_in, a_h = 0.0, 0.0
        else:
            if not frozen:
                u_in, iL = _pi(sp_L - L, iL, KP_L, KI_L)
                u_h, iT = _pi(sp_T - T, iT, KP_T, KI_T)
            a_in = (0.4 + 0.6 * mag) if stuck else u_in
            a_h = u_h
        q_in = q_max * a_in
        dL = (q_in - c_v * math.sqrt(L)) / A
        dT = (q_in / (A * L)) * (T_in - T) + beta * a_h / L - gamma * (T - T_amb)
        L = L + dL * DT
        T = T + dT * DT
        if L < L_MIN:
            L = L_MIN
        if L >= L_OVF:
            if not overflow:
                overflow, ovf_t = True, t + 1
            L = L_OVF
        if not tripped and (T >= T_TRIP or (L >= L_TRIP and not bypass)):
            tripped, trip_t = True, t + 1
        if not tripped and abs(T - T_SP) > SPEC:
            offspec += 1
        if T > peak_T:
            peak_T = T
        if L > peak_L:
            peak_L = L
        if record:
            trT.append(T)
            trL.append(L)
    out = dict(loss_s=offspec + (RESTART_S if tripped else 0), trip=tripped, trip_t=trip_t,
               overflow=overflow, overflow_t=ovf_t, peak_T=peak_T, peak_L=peak_L, L_end=L, T_end=T)
    if record:
        out["T"], out["L"] = trT, trL
    return out


# ------------------------------------------------------------------ risk score
WEIGHTS = (0.20, 0.15, 0.20, 0.15, 0.15, 0.15)   # identity, device, action, asset, process, context
ALLOW_BELOW, DENY_FROM = 0.35, 0.60


def risk(x, w=WEIGHTS):
    return sum(wi * xi for wi, xi in zip(w, x)) / sum(w)


def decide(x, w=WEIGHTS):
    r = risk(x, w)
    if r < ALLOW_BELOW:
        return "allow"
    if r < DENY_FROM:
        return "step-up"
    return "deny"


# ------------------------------------------------------------ request population
SEED = 20261004
_ACTION = {CRED: [(0.6, 1.0)], UNSAFE: [(1.0, 1.0)], RANSOM: [(1.0, 1.0)], VENDOR: [(1.0, 1.0)]}
_ASSET = {CRED: [(0.4, 0.7), (0.8, 0.3)], UNSAFE: [(1.0, 1.0)], RANSOM: [(0.4, 1.0)], VENDOR: [(0.8, 1.0)]}
_MAL = {  # identity, device, context distributions for malicious requests
    CRED:   ([0.2, 0.4, 0.4], [0.3, 0.3, 0.4], [0.4, 0.6]),
    UNSAFE: ([0.8, 0.2, 0.0], [0.8, 0.2, 0.0], [0.7, 0.3]),
    RANSOM: ([0.3, 0.4, 0.3], [0.1, 0.4, 0.5], [0.3, 0.7]),
    VENDOR: ([0.5, 0.4, 0.1], [0.2, 0.4, 0.4], [0.5, 0.5]),
}
_BEN = ([0.75, 0.20, 0.05], [0.6, 0.3, 0.1], [0.85, 0.15])
_PROC_MAL = [0.10, 0.80, 0.10]                   # stopped, running, abnormal
_PROC_BEN = {UNSAFE: [0.85, 0.15, 0.0]}          # legitimate bypasses are done on a stopped plant
_PROC_BEN_DEFAULT = [0.15, 0.75, 0.10]
_STRONG = {CRED: 0.25, UNSAFE: 1.0, RANSOM: 0.10, VENDOR: 0.60}   # attacker holds strong session
P_CONDUIT_MAL, P_CONDUIT_BEN = 0.70, 0.95
LVL3, LVL2 = [0.0, 0.5, 1.0], [0.0, 1.0]


def _pick(rng, values, probs):
    r, acc = rng.random(), 0.0
    for v, pr in zip(values, probs):
        acc += pr
        if r < acc:
            return v
    return values[-1]


def _pick_pairs(rng, pairs):
    return _pick(rng, [v for v, _ in pairs], [q for _, q in pairs])


def _mix(p, q, a):
    return [(1.0 - a) * pi + a * qi for pi, qi in zip(p, q)]


def generate_requests(seed=SEED, n_per=50, stealth=0.0):
    """stealth in [0, 1] moves the attackers' identity, device and context
    distributions toward the benign ones (0 = nominal, 1 = indistinguishable)."""
    rng = random.Random(seed)
    reqs = []
    for fam in (CRED, UNSAFE, RANSOM, VENDOR):
        for malicious in (True, False):
            for _ in range(n_per):
                if malicious:
                    idp, dvp, cxp = (_mix(m, b, stealth) for m, b in zip(_MAL[fam], _BEN))
                else:
                    idp, dvp, cxp = _BEN
                proc = _PROC_MAL if malicious else _PROC_BEN.get(fam, _PROC_BEN_DEFAULT)
                x = (_pick(rng, LVL3, idp), _pick(rng, LVL3, dvp), _pick_pairs(rng, _ACTION[fam]),
                     _pick_pairs(rng, _ASSET[fam]), _pick(rng, LVL3, proc), _pick(rng, LVL2, cxp))
                reqs.append(dict(
                    family=fam, malicious=malicious, x=x,
                    conduit=rng.random() < (P_CONDUIT_MAL if malicious else P_CONDUIT_BEN),
                    strong=(rng.random() < _STRONG[fam]) if malicious else True,
                    mag=0.6 + 0.4 * rng.random() if malicious else 0.0))
    return reqs


def stepup_passes(req):
    """Second factor: attacker passes only with an already-strong session.
    Supervisor approval is refused for a SIF bypass when the request declares a running process."""
    if not req["strong"]:
        return False
    if req["family"] == UNSAFE and req["x"][4] >= 0.5:
        return False
    return True


def executes(policy, req, w=WEIGHTS):
    if policy == "implicit":
        return True
    if policy == "perimeter":
        return req["conduit"]
    d = decide(req["x"], w)
    return d == "allow" or (d == "step-up" and stepup_passes(req))


def policy_comparison(reqs=None, deltas=(120, 300)):
    reqs = reqs or generate_requests()
    mal = [r for r in reqs if r["malicious"]]
    ben = [r for r in reqs if not r["malicious"]]
    cache = {}
    res = {}
    for pol in ("implicit", "perimeter", "zero_trust"):
        ex = [r for r in mal if executes(pol, r)]
        row = dict(
            malicious_executed=len(ex) / len(mal),
            benign_blocked=sum(not executes(pol, r) for r in ben) / len(ben),
            benign_stepup=(sum(decide(r["x"]) == "step-up" for r in ben) / len(ben)) if pol == "zero_trust" else 0.0,
            n_mal_executed=len(ex))
        for d in deltas:
            loss = trips = ovf = 0
            for r in ex:
                key = (id(r), d)
                if key not in cache:
                    cache[key] = simulate(r["family"], r["mag"], d)
                o = cache[key]
                loss += o["loss_s"]
                trips += o["trip"]
                ovf += o["overflow"]
            row[f"loss_min_{d}"] = loss / 60.0
            row[f"trips_{d}"] = trips
            row[f"overflows_{d}"] = ovf
        if pol == "zero_trust":
            through_stepup = [r for r in ex if decide(r["x"]) == "step-up"]
            row["mal_via_stepup"] = len(through_stepup)
            row["mal_via_stepup_vendor"] = sum(r["family"] == VENDOR for r in through_stepup)
            row["mal_executed_by_family"] = {SCENARIOS[f]: sum(r["family"] == f for r in ex) for f in range(4)}
            ovf_reqs = [r for r in ex if simulate(r["family"], r["mag"], 300)["overflow"]]
            row["overflow_requests_300"] = [dict(family=SCENARIOS[r["family"]], declared_process=r["x"][4],
                                                 decision=decide(r["x"])) for r in ovf_reqs]
        res[pol] = row
    return res


def weight_sensitivity(reqs=None, factors=(0.8, 1.0, 1.2)):
    """V4: scale each weight by 0.8, 1.0 or 1.2 (3^6 = 729 vectors)."""
    import itertools
    reqs = reqs or generate_requests()
    base = [decide(r["x"]) for r in reqs]
    mal = [r for r in reqs if r["malicious"]]
    ben = [r for r in reqs if not r["malicious"]]
    unchanged, mal_ex, ben_bl = [], [], []
    for f in itertools.product(factors, repeat=6):
        w = tuple(wi * fi for wi, fi in zip(WEIGHTS, f))
        unchanged.append(sum(decide(r["x"], w) == b for r, b in zip(reqs, base)) / len(reqs))
        mal_ex.append(sum(executes("zero_trust", r, w) for r in mal) / len(mal))
        ben_bl.append(sum(not executes("zero_trust", r, w) for r in ben) / len(ben))
    return dict(n_vectors=len(unchanged), unchanged_mean=sum(unchanged) / len(unchanged),
                unchanged_min=min(unchanged), mal_exec_min=min(mal_ex), mal_exec_max=max(mal_ex),
                ben_block_min=min(ben_bl), ben_block_max=max(ben_bl))


def policy_robustness(seeds=50, stealth_levels=(0.0, 0.25, 0.5), delta=300):
    """Re-run the policy comparison for many seeds and for stealthier attackers.
    Seeds are SEED + 1 ... SEED + seeds (the paper's seed itself is excluded)."""
    import numpy as np
    from batch_twin import simulate_batch
    out = {}
    for a in stealth_levels:
        rows = []
        for k in range(1, seeds + 1):
            reqs = generate_requests(SEED + k, stealth=a)
            mal = [r for r in reqs if r["malicious"]]
            ben = [r for r in reqs if not r["malicious"]]
            res = simulate_batch([r["family"] for r in mal], [r["mag"] for r in mal], np.full(len(mal), delta))
            loss = res["loss_s"]
            row = {}
            for pol in ("implicit", "perimeter", "zero_trust"):
                ex = np.array([executes(pol, r) for r in mal])
                row[pol] = dict(mal=float(ex.mean()), ben_block=sum(not executes(pol, r) for r in ben) / len(ben),
                                loss=float(loss[ex].sum()) / 60.0)
            rows.append(row)
        def stat(pol, key):
            v = np.array([r[pol][key] for r in rows])
            return dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()))
        red = np.array([1 - r["zero_trust"]["loss"] / r["implicit"]["loss"] for r in rows])
        out[str(a)] = dict(
            zt_mal=stat("zero_trust", "mal"), perim_mal=stat("perimeter", "mal"),
            zt_ben_block=stat("zero_trust", "ben_block"), perim_ben_block=stat("perimeter", "ben_block"),
            zt_loss_reduction=dict(mean=float(red.mean()), min=float(red.min()), max=float(red.max())),
            zt_below_perimeter_all=bool(all(r["zero_trust"]["mal"] < r["perimeter"]["mal"] for r in rows)),
            zt_loss_below_perimeter_all=bool(all(r["zero_trust"]["loss"] < r["perimeter"]["loss"] for r in rows)))
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(policy_comparison(), indent=2))
