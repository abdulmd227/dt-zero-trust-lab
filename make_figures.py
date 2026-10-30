"""Regenerate Fig. 2 (temperature traces) and Fig. 3 (production loss vs delay).

Run:  python make_figures.py   -> fig2_temperature.png, fig3_loss.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import twin_sim as ts
from window import windows

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.linewidth": 0.8})
W, H = 3.4, 1.9


def fig2(path="fig2_temperature.png"):
    fig, ax = plt.subplots(figsize=(W, H), dpi=600)
    t = np.arange(1, ts.EPISODE_S + 1) / 60.0
    ax.axhspan(55, 65, color="0.9", lw=0)
    ax.axhline(85, color="0.5", lw=0.7)
    ax.text(39.5, 86, "SIF trip 85 °C", ha="right", va="bottom", fontsize=6.5, color="0.35")
    ax.text(0.5, 56, "spec band", fontsize=6.5, color="0.35")
    for d, ls, lab in ((0, "-", "0 s"), (60, "--", "60 s"), (120, "-.", "120 s"), (300, ":", "300 s (SIF trip)")):
        r = ts.simulate(ts.CRED, 1.0, d, record=True)
        ax.plot(t, r["T"], ls, color="black", lw=0.9, label=lab)
    ax.set_xlim(-1, 41); ax.set_ylim(40, 92)
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Tank temperature (°C)")
    ax.legend(title="Response delay", fontsize=6.5, title_fontsize=6.5, frameon=False, loc="center right",
              bbox_to_anchor=(1.0, 0.62))
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def fig3(path="fig3_loss.png", w=None):
    w = w or windows()
    fig, ax = plt.subplots(figsize=(W, H), dpi=600)
    d = np.arange(0, 901)
    marks = [0, 60, 120, 300, 600, 900]
    styles = {"Credential misuse": ("o", "-"), "Unsafe engineering": ("s", "--"),
              "Ransomware": ("^", "-."), "Vendor firmware": ("D", ":")}
    for name, (mk, ls) in styles.items():
        c = np.array(w[name]["loss_curve_min"])
        ax.plot(d, c, ls, color="black", lw=0.9, marker=mk, markevery=marks, ms=3.2, label=name)
    ax.set_xlim(-20, 920); ax.set_ylim(-0.8, 19)
    ax.set_xlabel("Response delay (s)"); ax.set_ylabel("Production loss (min)")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    fig2()
    fig3()
    print("figures written")
