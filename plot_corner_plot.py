#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from getconfig import getEventInfo

# ---- corner (astrophysics standard) ----
import corner


# =========================
# Plot style (single-column friendly)
# =========================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['lines.linewidth'] = 0.9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['ytick.major.size'] = 3.0
plt.rcParams['ytick.minor.size'] = 1.5
plt.rcParams['xtick.major.size'] = 3.0
plt.rcParams['xtick.minor.size'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True


# =========================
# weighted_quantile (EXACTLY your old function)
# =========================
def weighted_quantile(values, quantiles, weights=None,
                      values_sorted=False, old_style=False):
    if len(values) == 0:
        return np.zeros_like(quantiles)
    values = np.array(values)
    quantiles = np.array(quantiles)
    if weights is None:
        weights = np.ones(len(values))
    weights = np.array(weights)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        weights = weights[sorter]

    weighted_quantiles = np.cumsum(weights) - 0.5 * weights
    if old_style:
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(weights)
    return np.interp(quantiles, weighted_quantiles, values)


# =========================
# Pretty-printing helpers
# =========================
def choose_fmt(name):
    fmts = {
        "Ml": "%.4f",
        "Ml_1": "%.4f",
        "Ml_2": "%.4f",
        "Dl": "%.3f",
        "MH_lens": "%.3f",
        "AV_lens": "%.3f",
    }
    return fmts.get(name, "%.4g")

def latex_line(label, med, low, up, fmt="%.4g", unit_suffix=""):
    s = (
        f"{label}: "
        + r"$"
        + (fmt % med)
        + r"^{+"
        + (fmt % (up - med))
        + r"}_{-"
        + (fmt % (med - low))
        + r"}$"
    )
    if unit_suffix:
        s += f" {unit_suffix}"
    return s


# =========================
# Load arrays + build normalized weights (same as your 1D plot)
# =========================
def _load_combined_h5_arrays(resultfile, config, mass_kind="primary"):
    """
    mass_kind (only matters if config.binary_lens=True):
        "primary" | "secondary" | "total"
    """
    with h5py.File(resultfile, "r") as f:
        g = f["weights"]

        Dl = g["Dl"][:]
        MH = g["MH_lens"][:]
        AV = g["AV_lens"][:]
        tw = g["total_weight"][:]  # float64

        if config.binary_lens:
            Ml1 = g["Ml_1"][:]
            Ml2 = g["Ml_2"][:]
            if mass_kind == "total":
                Ml = Ml1 + Ml2
                Ml_label = r"$M_{L,\rm tot}$"
            elif mass_kind == "secondary":
                Ml = Ml2
                Ml_label = r"$M_{L,2}$"
            else:
                Ml = Ml1
                Ml_label = r"$M_{L,1}$"
        else:
            Ml = g["Ml"][:]
            Ml_label = r"$M_{\rm L}$"

    mt = np.isfinite(tw) & (tw > 0)
    if not np.any(mt):
        raise ValueError("No finite positive values in total_weight.")

    # stable normalization in log space (same logic as your 1D plotting)
    log10w_max = np.nanmax(np.log10(tw[mt]))
    w = np.zeros_like(tw, dtype=np.float64)
    w[mt] = 10.0 ** (np.log10(tw[mt]) - log10w_max)

    wsum = np.sum(w)
    if not np.isfinite(wsum) or wsum <= 0:
        raise ValueError("Weights are non-finite or sum to zero after rescaling.")
    w /= wsum

    return (Ml.astype(np.float64, copy=False),
            Dl.astype(np.float64, copy=False),
            MH.astype(np.float64, copy=False),
            AV.astype(np.float64, copy=False),
            w, log10w_max, Ml_label)


# =========================
# Ranges + weighted quantiles (MH special strategy)
# =========================
def _make_ranges_and_quantiles(Ml, Dl, MH, AV, w,
                               mh_prior=(-1.0, 0.6),
                               mh_range_mode="2sigma"):
    """
    mh_range_mode:
        - "2sigma": MH axis range = median ± 2σ (σ from 16/84), clipped to mh_prior
        - "p995":   MH axis range = weighted [0.5%, 99.5%], clipped to mh_prior
        - "prior":  MH axis range = full prior bounds
    For other params: weighted [0.5%, 99.5%] to avoid tiny tails dominating axes.
    """
    q16, q50, q84 = 0.16, 0.50, 0.84
    q005, q995 = 0.005, 0.995

    qs = {}
    ranges = []

    def qstats(x, name):
        mm = np.isfinite(x) & np.isfinite(w) & (w > 0)
        xx = x[mm]
        ww = w[mm]
        if xx.size == 0:
            raise ValueError(f"No finite weighted samples for {name}.")
        a005, a995 = weighted_quantile(xx, [q005, q995], weights=ww)
        a16, a50, a84 = weighted_quantile(xx, [q16, q50, q84], weights=ww)
        return (a005, a16, a50, a84, a995)

    # Ml
    s = qstats(Ml, "Ml")
    qs["Ml"] = s
    ranges.append((s[0], s[4]))

    # Dl
    s = qstats(Dl, "Dl")
    qs["Dl"] = s
    ranges.append((s[0], s[4]))

    # MH (special)
    s = qstats(MH, "MH_lens")
    qs["MH_lens"] = s
    mh16, mh50, mh84 = s[1], s[2], s[3]
    sig = 0.5 * (mh84 - mh16)

    if mh_range_mode == "prior":
        lo, hi = mh_prior
    elif mh_range_mode == "p995":
        lo, hi = max(s[0], mh_prior[0]), min(s[4], mh_prior[1])
    else:  # "2sigma"
        lo = mh50 - 2.0 * sig
        hi = mh50 + 2.0 * sig
        lo, hi = max(lo, mh_prior[0]), min(hi, mh_prior[1])
    ranges.append((lo, hi))

    # AV
    s = qstats(AV, "AV_lens")
    qs["AV_lens"] = s
    ranges.append((s[0], s[4]))

    return ranges, qs


# =========================
# Corner plot driver
# =========================
def make_corner_plot(
    DIR,
    eventname,
    modelname, 
    mass_kind="primary",         # binary lens only: primary|secondary|total
    mh_prior=(-1.0, 0.6),
    mh_range_mode="2sigma",
    levels=(0.68, 0.95),         # "Contours enclose 68% and 95% posterior probability"
    bins=60,                      # fewer bins -> fewer tiny islands (no smoothing)
    figsize=(6.0, 6.0),
    dpi=800,
    # grayscale choices:
    density_cmap="Greys",         # <- change here if you want "gray_r" etc.
    contour_color="k",            # contour line color
    contour_lw=1.1,
    # truth lines:
    truths=None,                  # dict keys: Ml/Dl/MH_lens/AV_lens OR list [Ml,Dl,MH,AV]
    truth_color="orange",
    truth_lw=1.8,
    # toggles:
    show_density=True,            # TRUE: grayscale heatmap; FALSE: no background
    show_points=False,            # keep clean
    savepath=None,
):
    cfgfile = os.path.join(DIR, "config", f"{eventname.lower()}.cfg")
    config = getEventInfo(eventname, cfgfile=cfgfile)
#     modelname = config.model

    resultfile = os.path.join(
        DIR, "output", eventname, modelname,
        f"combine_weight_{eventname}_{modelname}.h5"
    )
    if not os.path.exists(resultfile):
        raise FileNotFoundError(f"Cannot find combined HDF5:\n  {resultfile}")

    Ml, Dl, MH, AV, w, log10w_max, Ml_label = _load_combined_h5_arrays(
        resultfile, config, mass_kind=mass_kind
    )

    # finite mask
    mm = (
        np.isfinite(Ml) & np.isfinite(Dl) & np.isfinite(MH) & np.isfinite(AV) &
        np.isfinite(w) & (w > 0)
    )
    Ml, Dl, MH, AV, w = Ml[mm], Dl[mm], MH[mm], AV[mm], w[mm]
    w /= np.sum(w)

    # diagnostics
    neff = 1.0 / np.sum(w**2)
    ws = np.sort(w)[::-1]
    top100 = ws[:100].sum() if ws.size >= 100 else np.sum(ws)

    print("==============================================")
    print(f"File: {resultfile}")
    print(f"Rows used: {len(w):,}")
    print(f"Weight used: total_weight (rescaled by subtracting max log10={log10w_max:.6f})")
    print(f"Neff = {neff:.2f}")
    print(f"Top100 mass = {top100:.4f}")
    print("MH axis strategy:", mh_range_mode, "with prior", mh_prior)
    print("Corner levels:", levels, "(credible fractions)")
    print("==============================================\n")

    # ranges + quantiles
    ranges, qs = _make_ranges_and_quantiles(
        Ml, Dl, MH, AV, w, mh_prior=mh_prior, mh_range_mode=mh_range_mode
    )

    unit_suffix = {
        "Ml": r"M_\odot",
        "Dl": r"\rm kpc",
        "MH_lens": r"\rm dex",
        "AV_lens": r"\rm mag",
    }

    # print summaries (16/50/84)
    def _print_stats(name, x):
        xmin, xmax = np.min(x), np.max(x)
        a005, a16, a50, a84, a995 = qs[name]
        fmt = choose_fmt(name)
        print(f"max {name} = {xmax}")
        print(f"min {name} = {xmin}")
        print(latex_line(name, a50, a16, a84, fmt=fmt, unit_suffix=unit_suffix.get(name, "")))
        print("")

    _print_stats("Ml", Ml)
    _print_stats("Dl", Dl)
    _print_stats("MH_lens", MH)
    _print_stats("AV_lens", AV)

    # samples matrix (ALL samples, weighted)
    samples = np.vstack([Ml, Dl, MH, AV]).T
    labels = [
        Ml_label + r" $(M_\odot)$",
        r"$D_{\rm L}$ (kpc)",
        r"$[{\rm M/H}]$ (dex)",
        r"$A_V$ (mag)",
    ]

    # truths
    truths_arr = None
    if truths is not None:
        if isinstance(truths, dict):
            truths_arr = [
                float(truths["Ml"]),
                float(truths["Dl"]),
                float(truths["MH_lens"]),
                float(truths["AV_lens"]),
            ]
        else:
            truths_arr = [float(x) for x in truths]

    # build figure
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # IMPORTANT choices for what you asked:
    #  - plot_density=True  -> show grayscale density
    #  - fill_contours=False -> NO filled contours (lines only)
    #  - weights=w -> TRUE weighted posterior (no resampling)
    fig = corner.corner(
        samples,
        weights=w,
        labels=labels,
        range=ranges,
        bins=bins,
        levels=levels,
        plot_density=show_density,
        plot_contours=True,
        fill_contours=False,
        plot_datapoints=show_points,
        color=contour_color,
        truths=truths_arr,
        truth_color=truth_color,
        show_titles=True,     # we will set titles ourselves
        smooth=None,           # NO smoothing
        smooth1d=None,         # NO smoothing
        fig=fig,
        # forwarded to hist2d in corner (works in standard corner versions)
        cmap=density_cmap,
    )

    # reshape axes
    axes = np.array(fig.axes).reshape((4, 4))

    # --- make contour lines a bit cleaner/thicker ---
    # (matplotlib collections include the contour LineCollections)
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            # lower triangle only has 2D content
            if i <= j:
                continue
            for coll in ax.collections:
                # contour lines are LineCollections; set linewidth
                try:
                    coll.set_linewidth(contour_lw)
                except Exception:
                    pass

    # --- diagonal: add quantile lines + titles (your preferred style) ---
    diag_names = ["Ml", "Dl", "MH_lens", "AV_lens"]
    for i, name in enumerate(diag_names):
        ax = axes[i, i]
        a005, a16, a50, a84, a995 = qs[name]
        fmt = choose_fmt(name)

        # quantile lines (marginal)
        ax.axvline(a50, color="k", lw=1.0)
        ax.axvline(a16, color="k", lw=0.9, ls="--")
        ax.axvline(a84, color="k", lw=0.9, ls="--")

        # MH: show prior bounds so truncation is explicit
        if name == "MH_lens":
            ax.axvline(mh_prior[0], color="k", lw=0.8, ls=":")
            ax.axvline(mh_prior[1], color="k", lw=0.8, ls=":")

        # title in the exact style you referenced: med^{+up}_{-low}
#         med_s = fmt % a50
#         up_s  = fmt % (a84 - a50)
#         lo_s  = fmt % (a50 - a16)
#         ax.set_title(rf"{name}: {med_s}$^{{+{up_s}}}_{{-{lo_s}}}$", fontsize=11)

    # --- truth linewidth bump (corner draws truth lines; we just thicken them) ---
    if truths_arr is not None:
        for ax in fig.axes:
            for ln in ax.lines:
                # corner sets truth line color; match by string when possible
                if ln.get_color() == truth_color:
                    ln.set_linewidth(truth_lw)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=dpi)
        print(f"Saved: {savepath}")

    return fig


# =========================
# Run (YOU fill in the truth values here)
# =========================
if __name__ == "__main__":
    eventname = "mock1"  # CHANGE
    modelname = "single_lens"

    truth = {
        "Ml": 0.5,
        "Dl": 7.50,
        "MH_lens": 0.1,
        "AV_lens": 4.00,
    }                    # CHANGE

    fig = make_corner_plot(
        DIR=DIR,
        eventname=eventname,
        modelname = modelname,
        mass_kind="primary",
        mh_prior=(-1.0, 0.6),
        mh_range_mode="p995",#"2sigma",
        levels=(0.68, 0.95),
        bins=35,
        figsize=(6, 6),
        dpi=300,
        density_cmap="Greys",      # <- grayscale density background
        contour_color="k",         # <- black contours
        contour_lw=1.1,
        truths=truth,
        truth_color="orange",
        truth_lw=1.8,
        show_density=True,         # <- density ON
        show_points=False,         # <- no scattered points
        savepath=None,             # e.g. f"{DIR}/output/{eventname}/corner_{eventname}.png"
    )
    
    for ax in fig.axes:
        ax.tick_params(zorder=10000)  # puts ticks above most artists
        # also ensure ticklines are above:
        for tl in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
            tl.set_zorder(10000)
            
    for ax in fig.axes:
        ax.set_axisbelow(False)
        for sp in ax.spines.values():
            sp.set_zorder(1000)
        ax.xaxis.set_zorder(1000)
        ax.yaxis.set_zorder(1000)
        ax.tick_params(zorder=1000)
    
    plt.savefig(
    'output/corner_plot.pdf',
    dpi=300, bbox_inches='tight'
    )

    # plt.show()
