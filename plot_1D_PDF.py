import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec

import sys
# sys.path.insert(1, '/work/zhangjiyuan/MetalLens')
from getconfig import getEventInfo

# =========================
# Plot style
# =========================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['ytick.major.size'] = 2.5
plt.rcParams['ytick.minor.size'] = 1.25
plt.rcParams['xtick.major.size'] = 2.5
plt.rcParams['xtick.minor.size'] = 1.25
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

# =========================
# weighted_quantile (UNCHANGED from your old code)
# =========================
def weighted_quantile(values, quantiles, weights=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    """
    if len(values)==0:
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
# Helpers: formatting + LaTeX printing
# =========================
def choose_fmt(name):
    fmts = {
        "Ds": "%.2f", "Dl": "%.2f",
        "Ml": "%.3f", "Ml_1": "%.3f", "Ml_2": "%.3f",
        "a_prep": "%.3f",
        "Age_lens": "%.2f", "Age_lens_snapped": "%.2f",
        "MH_lens": "%.3f", "MH_lens_snapped": "%.3f",
        "AV_lens": "%.3f", "RV_lens": "%.3f",
        "pirel": "%.4f", "thetaE": "%.3f",
        "murel_geo_E": "%.3f", "murel_geo_N": "%.3f", "murel_geo": "%.3f",
        "murel_hel_E": "%.3f", "murel_hel_N": "%.3f",
        "piE_E": "%.4f", "piE_N": "%.4f",
        "tE": "%.3f",
        "Intrinsic_Color_F062_F087_lens": "%.4f",
        "Intrinsic_Color_F087_F213_lens": "%.4f",
        "M_F213_lens": "%.4f",
        "E_F062_F087": "%.4f",
        "E_F087_F213": "%.4f",
        "A_F213": "%.4f",
        "Color_F062_F087_lens": "%.4f",
        "Color_F087_F213_lens": "%.4f",
        "F213_lens": "%.4f",
        # weights printed in log10 space usually
        "log10_total_weight": "%.3f",
        "log10_tEweight": "%.3f",
        "log10_piEweight": "%.3f",
        "log10_Color_F062_F087_lens_weight": "%.3f",
        "log10_Color_F087_F213_lens_weight": "%.3f",
        "log10_F213_lens_weight": "%.3f",
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
# Main plotting function (HDF5)
# =========================
def plot_from_combined_h5(
    DIR,
    eventname,
    modelname,
    bins_main=90,
    bins_w=120,
    ncols=5,
    figsize=(16, 12),
    dpi=300,
    color_fill="dodgerblue",
    alpha_fill=0.5,
    show_minmax_text=True,
    savepath=None,
):
    cfgfile = f"config/{eventname.lower()}.cfg"
    config = getEventInfo(eventname, cfgfile=cfgfile)
#     config = getEventInfo(eventname)
    # modelname = config.model

    resultfile = f"{DIR}/output/{eventname}/{modelname}/combine_weight_{eventname}_{modelname}.h5"
    if not os.path.exists(resultfile):
        raise FileNotFoundError(f"Cannot find combined HDF5: {resultfile}")

    # -------- read HDF5 --------
    with h5py.File(resultfile, "r") as f:
        g = f["weights"]
        # load everything we might plot (float32/64 -> float64 in memory)
        # we will filter by existence based on config and keys list below
        def read(name):
            if name not in g:
                return None
            return np.asarray(g[name], dtype=np.float64)

        # total weight is required
        total_weight = read("total_weight")
        if total_weight is None:
            raise KeyError(f"'total_weight' not found in {resultfile} under /weights")

        # Build main variable list in the same logical order as your saving code
        main_keys = [
            "Ds", "Dl",
            "Age_lens", "MH_lens", "Age_lens_snapped", "MH_lens_snapped",
            # mass fields (binary vs single) appended below
            "AV_lens", "RV_lens",
            "pirel", "thetaE",
            "murel_geo_E", "murel_geo_N", "murel_geo",
            "Intrinsic_Color_F062_F087_lens", "Intrinsic_Color_F087_F213_lens", "M_F213_lens",
            "E_F062_F087", "E_F087_F213", "A_F213",
            "murel_hel_E", "murel_hel_N",
            "piE_E", "piE_N",
            "tE",
            "Color_F062_F087_lens", "Color_F087_F213_lens", "F213_lens",
        ]
        if config.binary_lens:
            main_keys = main_keys[:6] + ["Ml_1", "Ml_2", "a_prep"] + main_keys[6:]
        else:
            main_keys = main_keys[:6] + ["Ml"] + main_keys[6:]

        # keep only keys that actually exist (robust if you tweak outputs later)
        main_keys = [k for k in main_keys if (k in g)]

        # -------- posterior weights for main panels (use total_weight) --------
        mt = np.isfinite(total_weight) & (total_weight > 0)
        if not np.any(mt):
            raise ValueError("No finite positive values in total_weight.")

        log10w_tot = np.full_like(total_weight, np.nan, dtype=np.float64)
        log10w_tot[mt] = np.log10(total_weight[mt])
        log10w_tot_max = np.nanmax(log10w_tot[mt])

        # rescale in log space -> linear weights, then normalize
        w = np.zeros_like(total_weight, dtype=np.float64)
        w[mt] = np.power(10.0, log10w_tot[mt] - log10w_tot_max)
        mw = np.isfinite(w) & (w > 0)

        wsum = np.sum(w[mw])
        if not np.isfinite(wsum) or wsum <= 0:
            raise ValueError("Weights are non-finite or sum to zero after rescaling.")
        w = w / wsum

        neff = 1.0 / np.sum(w[mw] ** 2)
        ws = np.sort(w[mw])[::-1]

        # -------- weight components to plot UNWEIGHTED in log10 space --------
        weight_keys = ["total_weight"]
        if config.use_tE and ("tEweight" in g):
            weight_keys.append("tEweight")
        if config.use_piE and ("piEweight" in g):
            weight_keys.append("piEweight")
        if config.use_Color_F062_F087_lens and ("Color_F062_F087_lens_weight" in g):
            weight_keys.append("Color_F062_F087_lens_weight")
        if config.use_Color_F087_F213_lens and ("Color_F087_F213_lens_weight" in g):
            weight_keys.append("Color_F087_F213_lens_weight")
        if config.use_F213_lens and ("F213_lens_weight" in g):
            weight_keys.append("F213_lens_weight")

        # -------- units for printing/labels --------
        unit_suffix = {
            "Ds": r"\rm kpc", "Dl": r"\rm kpc",
            "Ml": r"M_\odot", "Ml_1": r"M_\odot", "Ml_2": r"M_\odot",
            "a_prep": r"\rm au",
            "Age_lens": r"\rm Gyr", "Age_lens_snapped": r"\rm Gyr",
            "MH_lens": r"\rm dex", "MH_lens_snapped": r"\rm dex",
            "AV_lens": r"\rm mag", "RV_lens": r"\rm -",
            "pirel": r"\rm mas", "thetaE": r"\rm mas",
            "murel_geo": r"\rm mas/yr", "murel_geo_E": r"\rm mas/yr", "murel_geo_N": r"\rm mas/yr",
            "murel_hel_E": r"\rm mas/yr", "murel_hel_N": r"\rm mas/yr",
            "tE": r"\rm day",
            "Intrinsic_Color_F062_F087_lens": r"\rm mag",
            "Intrinsic_Color_F087_F213_lens": r"\rm mag",
            "M_F213_lens": r"\rm mag",
            "E_F062_F087": r"\rm mag",
            "E_F087_F213": r"\rm mag",
            "A_F213": r"\rm mag",
            "Color_F062_F087_lens": r"\rm mag",
            "Color_F087_F213_lens": r"\rm mag",
            "F213_lens": r"\rm mag",
        }

    # =========================
    # Layout (main variables + weight panels)
    # =========================
    nmain = len(main_keys)
    nrows_main = int(np.ceil(nmain / ncols))

    nweights = len(weight_keys)  # includes total_weight
    nrows_w = int(np.ceil(nweights / ncols))

    nrows_total = nrows_main + nrows_w

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(nrows_total, ncols)
    axs = [plt.subplot(gs[i]) for i in range(nrows_total * ncols)]
    plt.subplots_adjust(wspace=0.2, hspace=0.5)

    # quantiles (68.27% equal-tail)
    qlo, qmed, qhi = (0.5 - 0.6827 / 2), 0.5, (0.5 + 0.6827 / 2)

    # =========================
    # Print header summary
    # =========================
    print("==============================================")
    print(f"File: {resultfile}")
    print(f"Rows: {len(total_weight):,}")
    print(f"Weight used for MAIN panels: total_weight (rescaled by subtracting max log10={log10w_tot_max:.6f})")
    print("NOTE: Weight panels are UNWEIGHTED histograms in log10 space.")
    print("")
    print(f"Neff = {neff:.2f} (aim >> 1000; if low, increase sample size)")
    if ws.size >= 100:
        print(f"Top100 mass = {ws[:100].sum():.4f} (aim < 0.2; if high, increase sample size)")
    print("==============================================\n")

    # =========================
    # MAIN panels (weighted by total_weight)
    # =========================

    # Reopen once to fetch arrays (avoid keeping all arrays in memory if you prefer)
    with h5py.File(resultfile, "r") as f:
        g = f["weights"]

        for i, name in enumerate(main_keys):
            ax = axs[i]
            x = np.asarray(g[name], dtype=np.float64)

            mm = np.isfinite(x) & np.isfinite(w) & (w > 0)
            x = x[mm]
            ww = w[mm]

            if x.size == 0:
                ax.text(0.5, 0.5, "no finite samples", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                continue

            xmin, xmax = np.min(x), np.max(x)
            low, med, up = weighted_quantile(x, [qlo, qmed, qhi], weights=ww)
            fmt = choose_fmt(name)

            # print min/max + latex-friendly line
            print(f"max {name} = {xmax}")
            print(f"min {name} = {xmin}")
            print(latex_line(name, med, low, up, fmt=fmt, unit_suffix=unit_suffix.get(name, "")))
            print("")

            # histogram (weighted)
            ax.hist(
                x, bins=bins_main, weights=ww, density=True,
                histtype="bar", color=color_fill, alpha=alpha_fill
            )
            ax.hist(
                x, bins=bins_main, weights=ww, density=True,
                histtype="step", color="k", linewidth=0.8
            )

            # quantile lines
            ax.axvline(med, color="k", linestyle="-", linewidth=0.8)
            ax.axvline(low, color="k", linestyle="--", linewidth=0.7)
            ax.axvline(up,  color="k", linestyle="--", linewidth=0.7)

            xlabel_unit = unit_suffix.get(name, "")
            xlabel = name if (xlabel_unit == "" or xlabel_unit is None) else name + r'($%s$)' % xlabel_unit
            ax.set_xlabel(xlabel, fontsize=9)

            if i % ncols == 0:
                ax.set_ylabel("PDF", fontsize=9)

            ax.minorticks_on()

            # annotate
            txt = (
                f"med={fmt % med}\n"
                f"     -={fmt % (med-low)}\n"
                f"    +={fmt % (up-med)}"
            )
            if show_minmax_text:
                txt += f"\nmin={fmt % xmin}\nmax={fmt % xmax}"
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=7.5)

    # turn off unused MAIN slots
    last_main_slot = nrows_main * ncols
    for j in range(nmain, last_main_slot):
        axs[j].axis("off")

    # =========================
    # Weight panels (UNWEIGHTED, log10 space)
    # =========================
    base = nrows_main * ncols
    with h5py.File(resultfile, "r") as f:
        g = f["weights"]

        for k, wname in enumerate(weight_keys):
            ax = axs[base + k]

            arr = np.asarray(g[wname], dtype=np.float64)
            good = np.isfinite(arr) & (arr > 0)
            xw = np.log10(arr[good])

            title = f"log10({wname}) (UNWEIGHTED)"
            ax.set_title(title, fontsize=9)

            if xw.size == 0:
                ax.text(0.5, 0.5, "no finite positive samples", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                continue

            ax.hist(
                xw, bins=bins_w, density=True,
                histtype="bar", color="lightgray", alpha=0.9
            )
            ax.hist(
                xw, bins=bins_w, density=True,
                histtype="step", color="k", linewidth=0.9
            )

            wmax = np.max(xw)
            wmin = np.min(xw)
            ax.axvline(wmax, color="r", linestyle="-", linewidth=0.8)
            ax.axvline(wmax - 8, color="r", linestyle="--", linewidth=0.7)

            ax.text(
                0.02, 0.98,
                f"min={wmin:.3g}\nmax={wmax:.3g}\n(max-8)={(wmax-8):.3g}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.5
            )

            if k % ncols == 0:
                ax.set_ylabel("PDF", fontsize=9)

            ax.minorticks_on()

    # turn off unused WEIGHT slots
    last_weight_slot = base + nrows_w * ncols
    for j in range(base + nweights, last_weight_slot):
        axs[j].axis("off")

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight",dpi=300)
        print(f"Saved: {savepath}")

    # plt.show()


# =========================
# Run
# =========================
DIR = "."

eventname = "mock1"  # change here
modelname = "single_lens"

savepath = "output/%s/%s/1D_PDF.pdf"%(eventname, modelname)

plot_from_combined_h5(
    DIR=DIR,
    eventname=eventname,
    modelname=modelname,
    bins_main=90,
    bins_w=120,
    ncols=5,
    figsize=(16, 12),
    dpi=300,
    color_fill="dodgerblue",
    alpha_fill=0.5,
    show_minmax_text=True,
    savepath=savepath,
)
