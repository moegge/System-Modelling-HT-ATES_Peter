# -*- coding: utf-8 -*-
"""
Result Sweep Delft Gas x CO2 x Dispatch.py
==================================================================
Three-dimensional sweep: gas price x CO2 price x heat-pump dispatch threshold.

For every (gas price, CO2 price) cell the dispatch threshold is swept and the
CHEAPEST threshold is kept. The headline figure is one bar per cell showing the
system LCOH at that best threshold, so the bars answer:

    "under these fuel and carbon prices, what does the system cost once the
     heat pump is dispatched as well as this control can dispatch it?"

WHY THE THRESHOLD HAS TO BE SWEPT PER CELL
------------------------------------------
build_hp_dispatch thresholds the ALL-IN MARGINAL ELECTRICITY COST c_marg and
knows nothing about gas or carbon. The economically correct level is

    c_marg* = COP * (gas_price/eff + CO2_price * EF_gas)

i.e. it moves with both swept prices. At the base case (0.055 EUR/kWh gas,
150 EUR/t, COP 4.95) that is ~441 EUR/MWh. Cheap gas and zero carbon price
pull it down to ~180; expensive gas and a high carbon price push it past 1500.
The sweep finds the optimum empirically and the analysis compares it with the
formula above -- if the two disagree by more than one grid step, something in
the cost chain is not doing what the arithmetic says.

RUNTIME NOTE -- THERE IS 16x REDUNDANCY HERE
--------------------------------------------
The gas and CO2 prices affect ONLY the economics; they do not touch the
simulation. build_hp_dispatch reads electricity prices alone, and nothing in
system() / calc_heat sees a fuel price. So all 16 cells at a given threshold run
physically identical simulations and differ only in economic_analysis and
LCOE_calc_Yang. This script still runs them all, because splitting run_case into
a simulate half and a cost half is a bigger change than it is worth right now --
but that is the refactor that would turn 112 runs into 7. The consistency check
printed after the sweep verifies the invariance (HP heat must be bit-identical
across a threshold's cells); if it ever fails, a fuel price is leaking into the
physics.

OUTPUTS (into RESULTS_DIR, created next to this file)
-----------------------------------------------------
  gas_co2_best_lcoh.png       headline: system LCOH at the best threshold
  gas_co2_dispatch_value.png  (a) saving vs always-on, (b) found vs predicted
  gas_co2_data.csv            incremental cache; also the raw data
  gas_co2_summary.xlsx        all runs + best-per-cell + LCOH/threshold grids

RESUMABLE: every completed run is appended to the cache immediately, so an
interrupted sweep continues where it stopped and a pure re-plot costs seconds.
Delete the cache after ANY model change -- nothing in the key records the model
version.

    python "Result_Sweep_Delft_Gas_CO2_Dispatch.py"
==================================================================
"""

import os
import io
import contextlib
import traceback
import sys

# model_driver.py / main2_Peter.py / ATES_obj_Peter.py live one level up, in
# System-Modelling-HT-ATES_Peter; this file sits in the "Delft Case" subfolder.
_MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODEL_ROOT not in sys.path:
    sys.path.insert(0, _MODEL_ROOT)

import matplotlib
matplotlib.use("Agg")          # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np
import pandas as pd

from model_driver import (run_case, TIMESTEP, DEMAND_EXAMPLE,
                          DEMAND_T_IN, DEMAND_T_OUT)
from main2_Peter import build_hp_dispatch, demand_class

# ================================================================== #
#  SWEEP GRID                                                        #
# ================================================================== #

# --- Gas price [EUR/kWh_gas] ------------------------------------------------
# 0.055 is the paper's 55 EUR/MWh (Table 1). The others bracket it: a soft
# market, a firm one, and a 2022-style spike.
GAS_PRICE_GRID = [0.030, 0.055, 0.080, 0.120]

# --- CO2 price [EUR/tonne] --------------------------------------------------
# 0 = carbon not priced at all (the honest lower bound for a CAC discussion),
# 150 = the 2030 Dutch industry tax used in the paper.
CO2_PRICE_GRID = [0.0, 50.0, 100.0, 150.0, 200, 250, 300]

# --- Dispatch threshold on c_marg [EUR/MWh] --------------------------------
# Must span the break-even of every cell in the grid above, or the "optimum"
# found in the expensive-gas cells is just the top of the range. The banner
# prints the predicted break-even per cell -- check it sits inside this list.
ALWAYS_ON_THRESHOLD = 1e6    # gate never binds -> HP on every discharge hour
THRESHOLDS_EUR_MWH = [100, 200, 300, 430, 600, 900, 1500, ALWAYS_ON_THRESHOLD]

CONFIG = "GGAH"              # must include the HP or the threshold does nothing

# --- Network -------------------------------------------------------------
# 0.0 matches the dynamic-pricing script (run_case's own default). The network
# cost is identical in every cell, so it shifts all bars by the same amount and
# never changes which threshold wins -- but it does change the LEVEL, so set it
# if these numbers are to be compared with the Delft sweeps.
NETWORK_LENGTH_M = 0.0

# --- Gas boiler constants, for the break-even formula ----------------------
# These MUST match gas_boiler's defaults in main2_Peter. They are restated
# rather than imported because gas_boiler stores them per instance.
GAS_EFF            = 0.93     # [-] boiler efficiency
GAS_CO2_KG_PER_MWH = 200.0    # [kgCO2/MWh_heat]

# --- Output / behaviour -----------------------------------------------------
RESULTS_DIR = "results gas co2"
CACHE_CSV   = "gas_co2_data.csv"
SAVE_FIGURES = True
QUIET_RUNS   = True          # True -> swallow run_case's per-run console output

# ================================================================== #


def _break_even_threshold(gas_price, co2_price, cop):
    """
    The c_marg [EUR/MWh] at which the HP stops beating the gas boiler.

        gas marginal heat cost = gas_price/eff + CO2_price * EF
        HP marginal heat cost  = c_marg / COP
        equal when c_marg = COP * gas marginal

    Returns NaN if the COP is unknown.
    """
    if not np.isfinite(cop) or cop <= 0:
        return np.nan
    gas_marginal = (gas_price * 1000.0 / GAS_EFF
                    + co2_price * GAS_CO2_KG_PER_MWH / 1000.0)
    return cop * gas_marginal


def _n_timesteps():
    """Timestep count as run_case sees it (demand_class resamples the profile)."""
    dem = demand_class(T_in=DEMAND_T_IN, T_out=DEMAND_T_OUT,
                       example_demand=DEMAND_EXAMPLE)
    dem.adjust_for_timesetting(len_timestep=TIMESTEP)
    return len(dem.data)


def _load_cache(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  cache unreadable ({type(e).__name__}: {e}) -- starting fresh")
        return pd.DataFrame()


def _append_row(path, row):
    """
    Append one result row, reconciling columns with whatever is already there.
    A plain header-less append writes values in the CURRENT dict order under the
    OLD header, so adding a field silently shifts every later column.
    """
    if not os.path.exists(path):
        pd.DataFrame([row]).to_csv(path, index=False)
        return
    try:
        header = list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        pd.DataFrame([row]).to_csv(path, index=False)
        return

    if all(k in header for k in row):
        (pd.DataFrame([row]).reindex(columns=header)
         .to_csv(path, mode="a", index=False, header=False))
    else:
        old = pd.read_csv(path)
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(
            path, index=False)


def _save(fig, results_dir, name):
    if SAVE_FIGURES:
        path = os.path.join(results_dir, name + ".png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved figure -> {path}")
    plt.close(fig)


def _simulate(gas_price, co2_price, thr, tag):
    """One run_case call. Returns the result dict, or None if the run failed."""
    kwargs = dict(
        CONFIG=CONFIG,
        GAS_PRICE=gas_price,
        CO2_PRICE=co2_price,
        HP_DYNAMIC_DISPATCH=True,          # every row is dynamic; always-on is
        HP_THRESHOLD_EUR_MWH=float(thr),   # expressed as a huge threshold, so
        NETWORK_LENGTH_M=NETWORK_LENGTH_M, # all rows share one pricing model
        tag=tag,
        make_plots=False,
        write_excel=False,
    )
    buf = io.StringIO()
    try:
        if QUIET_RUNS:
            with contextlib.redirect_stdout(buf):
                return run_case(**kwargs)
        return run_case(**kwargs)
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        return None


def _mode_counts(ts_mode):
    """A/B/D discharge-step counts from the per-timestep mode array."""
    m = np.asarray(ts_mode, dtype=object)
    return (int((m == 'A').sum()), int((m == 'B').sum()), int((m == 'D').sum()))


def _row_from(gas_price, co2_price, thr, signal_on, res):
    a, b, d = _mode_counts(res.get("ts_mode", []))
    return {
        "gas_price":        float(gas_price),
        "co2_price":        float(co2_price),
        "threshold":        float(thr),
        "signal_on_steps":  signal_on,
        "mode_A":           a,
        "mode_B":           b,
        "mode_D":           d,
        "hp_GWh":           res.get("hp_GWh"),
        "hp_elec_GWh":      res.get("hp_elec_GWh"),
        "hp_mean_COP":      res.get("hp_mean_COP"),
        "hp_price_paid_eur_kwh": res.get("hp_price_paid_eur_kwh"),
        "hp_elec_cost_eur": res.get("hp_elec_cost_eur"),
        "ates_direct_GWh":  res.get("ates_direct_GWh"),
        "geo_to_demand_GWh": res.get("geo_to_demand_GWh"),
        "gas_GWh":          res.get("gas_GWh"),
        "demand_GWh":       res.get("demand_GWh"),
        "system_lcoh_yang": res.get("system_lcoh_yang"),
        "ates_lcoh":        res.get("ates_lcoh"),
        "gas_lcoh":         res.get("gas_lcoh"),
        "geo_lcoh":         res.get("geo_lcoh"),
        "total_CO2_t":      res.get("total_CO2_t"),
        "total_CO2_cost_eur": res.get("total_CO2_cost_eur"),
        "Reff":             res.get("Reff"),
    }


# ================================================================== #
#  ANALYSIS                                                          #
# ================================================================== #

def _best_per_cell(df):
    """
    Per (gas price, CO2 price): the threshold with the lowest system LCOH, what
    always-on would have cost, and the break-even the formula predicts.
    """
    rows = []
    for (g, c), sub in df.groupby(["gas_price", "co2_price"], sort=True):
        sub = sub.dropna(subset=["system_lcoh_yang"])
        if sub.empty:
            continue
        best = sub.loc[sub["system_lcoh_yang"].idxmin()]
        always = sub[np.isclose(sub["threshold"], ALWAYS_ON_THRESHOLD)]

        a_lcoh = (float(always["system_lcoh_yang"].iloc[0])
                  if not always.empty else np.nan)
        cop = (float(always["hp_mean_COP"].iloc[0]) if not always.empty
               else float(best["hp_mean_COP"]))

        rows.append({
            "gas_price":  g,
            "co2_price":  c,
            "best_threshold": float(best["threshold"]),
            "best_lcoh_eur_MWh": float(best["system_lcoh_yang"]) * 1000.0,
            "always_on_lcoh_eur_MWh": a_lcoh * 1000.0,
            "saving_vs_always_on_eur_MWh": (a_lcoh - float(best["system_lcoh_yang"])) * 1000.0,
            "predicted_breakeven_eur_MWh": _break_even_threshold(g, c, cop),
            "hp_mean_COP": cop,
            "hp_GWh_at_best":  float(best["hp_GWh"]),
            "gas_GWh_at_best": float(best["gas_GWh"]),
            "CO2_t_at_best":   float(best["total_CO2_t"]),
            "n_thresholds": len(sub),
        })
    return pd.DataFrame(rows)


def _physics_invariance(df):
    """
    Gas and CO2 prices must not touch the simulation. For each threshold the HP
    heat has to be identical across every (gas, CO2) cell; a non-zero spread
    means a fuel price is leaking into the physics.
    """
    worst, worst_thr = 0.0, np.nan
    for thr, sub in df.groupby("threshold"):
        v = sub["hp_GWh"].dropna().values
        if len(v) < 2:
            continue
        spread = float(np.max(v) - np.min(v))
        if spread > worst:
            worst, worst_thr = spread, thr
    return worst, worst_thr


# ================================================================== #
#  FIGURES                                                           #
# ================================================================== #

def _co2_colors(co2_vals):
    cmap = plt.get_cmap("viridis")
    return {c: cmap(t) for c, t in zip(co2_vals,
                                       np.linspace(0.15, 0.85, len(co2_vals)))}


def _grouped_bars(ax, best, value_col, co2_colors, annotate=None,
                  annotate_fmt="{:.0f}"):
    """
    One group per gas price, one bar per CO2 price within it. Returns the gas
    values in plot order so the caller can set the tick labels.
    """
    gas_vals = sorted(best["gas_price"].unique())
    co2_vals = sorted(best["co2_price"].unique())
    x = np.arange(len(gas_vals))
    width = 0.8 / max(len(co2_vals), 1)

    for j, c in enumerate(co2_vals):
        offset = (j - (len(co2_vals) - 1) / 2.0) * width
        vals, ann = [], []
        for g in gas_vals:
            m = best[(best["gas_price"] == g) & (best["co2_price"] == c)]
            vals.append(float(m[value_col].iloc[0]) if not m.empty else np.nan)
            ann.append(float(m[annotate].iloc[0])
                       if (annotate and not m.empty) else np.nan)
        bars = ax.bar(x + offset, vals, width, color=co2_colors[c],
                      edgecolor="white", linewidth=0.5)
        for bar, v, a in zip(bars, vals, ann):
            if not np.isfinite(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2.0, v,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)
            if annotate and np.isfinite(a):
                lbl = ("on" if a >= ALWAYS_ON_THRESHOLD
                       else annotate_fmt.format(a))
                ax.text(bar.get_x() + bar.get_width() / 2.0, v * 0.03,
                        lbl, ha="center", va="bottom", fontsize=6,
                        rotation=90, color="white")
    ax.set_xticks(x)
    return gas_vals, co2_vals


def _plot_best_lcoh(best, results_dir):
    """
    Headline figure: one bar per (gas price, CO2 price), height = system LCOH at
    that cell's cheapest dispatch threshold. The white number inside each bar is
    the winning threshold ('on' = always-on won).
    """
    co2_vals = sorted(best["co2_price"].unique())
    colors = _co2_colors(co2_vals)

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    gas_vals, _ = _grouped_bars(ax, best, "best_lcoh_eur_MWh", colors,
                                annotate="best_threshold")

    ax.set_xticklabels([f"{g * 1000:.0f}" for g in gas_vals])
    ax.set_xlabel("Gas price (\u20ac/MWh_gas)")
    ax.set_ylabel("System LCOH at the best dispatch threshold (\u20ac/MWh)")
    ax.set_title("System cost under the cheapest heat-pump dispatch threshold\n"
                 "(white label inside each bar = winning threshold on c_marg, "
                 "\u20ac/MWh; 'on' = always-on)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=colors[c],
                             label=f"CO\u2082 {c:.0f} \u20ac/t")
                       for c in co2_vals],
              title="Carbon price", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    _save(fig, results_dir, "gas_co2_best_lcoh")


def _plot_dispatch_value(best, results_dir):
    """
    (a) What optimising the threshold is worth against simply leaving the HP on.
        Zero means always-on already was optimal.
    (b) The threshold the sweep picked against the one the break-even formula
        predicts. Agreement within one grid step is the model behaving; a large
        gap means the cost chain is not doing what the arithmetic says.
    """
    co2_vals = sorted(best["co2_price"].unique())
    colors = _co2_colors(co2_vals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    gas_vals, _ = _grouped_bars(ax1, best, "saving_vs_always_on_eur_MWh", colors)
    ax1.set_xticklabels([f"{g * 1000:.0f}" for g in gas_vals])
    ax1.set_xlabel("Gas price (\u20ac/MWh_gas)")
    ax1.set_ylabel("LCOH saving vs always-on (\u20ac/MWh)")
    ax1.set_title("(a) Value of optimising the dispatch threshold", fontsize=9)
    ax1.axhline(0.0, color="k", lw=0.8)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.legend(handles=[Patch(facecolor=colors[c], label=f"CO\u2082 {c:.0f} \u20ac/t")
                        for c in co2_vals], fontsize=8)

    # (b) found vs predicted. Always-on wins are drawn at the top of the swept
    #     range, not at 1e6, or the axis is unreadable.
    finite_thr = [t for t in THRESHOLDS_EUR_MWH if t < ALWAYS_ON_THRESHOLD]
    cap = max(finite_thr) * 1.6
    for c in co2_vals:
        sub = best[best["co2_price"] == c].sort_values("gas_price")
        found = sub["best_threshold"].values.astype(float)
        found_plot = np.where(found >= ALWAYS_ON_THRESHOLD, cap, found)
        ax2.scatter(sub["predicted_breakeven_eur_MWh"].values, found_plot,
                    color=colors[c], s=45, zorder=3,
                    label=f"CO\u2082 {c:.0f} \u20ac/t")
    lim = max(cap, float(np.nanmax(best["predicted_breakeven_eur_MWh"])) * 1.1)
    ax2.plot([0, lim], [0, lim], ls="--", lw=1.0, color="0.5", label="1:1")
    ax2.axhline(cap, ls=":", lw=1.0, color="0.7")
    ax2.text(lim * 0.02, cap, " always-on won (plotted here)", fontsize=7,
             va="bottom", color="0.4")
    for t in finite_thr:
        ax2.axhline(t, lw=0.4, color="0.9", zorder=0)
    ax2.set_xlim(0, lim)
    ax2.set_ylim(0, lim)
    ax2.set_xlabel("Predicted break-even  COP \u00d7 (gas/eff + CO\u2082\u00b7EF)  "
                   "(\u20ac/MWh)")
    ax2.set_ylabel("Threshold the sweep picked (\u20ac/MWh)")
    ax2.set_title("(b) Found optimum vs theory  "
                  "(grey lines = the swept grid)", fontsize=9)
    ax2.grid(alpha=0.2)
    ax2.legend(fontsize=7)

    fig.tight_layout()
    _save(fig, results_dir, "gas_co2_dispatch_value")


# ================================================================== #

def main():
    here = os.path.dirname(os.path.abspath(__file__)) \
        if "__file__" in globals() else os.getcwd()
    results_dir = os.path.join(here, RESULTS_DIR)
    os.makedirs(results_dir, exist_ok=True)
    cache_path = os.path.join(results_dir, CACHE_CSV)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    n_steps = _n_timesteps()
    n_runs = len(GAS_PRICE_GRID) * len(CO2_PRICE_GRID) * len(THRESHOLDS_EUR_MWH)

    print("=" * 74)
    print("GAS PRICE x CO2 PRICE x DISPATCH THRESHOLD SWEEP")
    print("=" * 74)
    print(f"  gas prices      : {[f'{g*1000:.0f}' for g in GAS_PRICE_GRID]} EUR/MWh_gas")
    print(f"  CO2 prices      : {[f'{c:.0f}' for c in CO2_PRICE_GRID]} EUR/t")
    print(f"  thresholds      : "
          f"{[('on' if t >= ALWAYS_ON_THRESHOLD else f'{t:g}') for t in THRESHOLDS_EUR_MWH]}"
          f" EUR/MWh on c_marg")
    print(f"  configuration   : {CONFIG}   network {NETWORK_LENGTH_M/1000:.0f} km")
    print(f"  timesteps       : {n_steps}  ({TIMESTEP} s)")
    print(f"  TOTAL RUNS      : {n_runs}")
    print(f"  cache           : {cache_path}")

    # Predicted break-even per cell, at the base-case COP. Printed BEFORE the
    # runs so a badly chosen threshold grid is obvious in the first 5 seconds.
    COP_GUESS = 4.95
    print(f"\n  Predicted break-even c_marg [EUR/MWh] at COP {COP_GUESS:.2f} "
          f"-- the threshold grid must bracket these:")
    print("      gas\\CO2  " + "".join(f"{c:>10.0f}" for c in CO2_PRICE_GRID))
    for g in GAS_PRICE_GRID:
        print(f"      {g*1000:7.0f}  " + "".join(
            f"{_break_even_threshold(g, c, COP_GUESS):>10.0f}"
            for c in CO2_PRICE_GRID))
    lo, hi = min(t for t in THRESHOLDS_EUR_MWH if t < ALWAYS_ON_THRESHOLD), \
             max(t for t in THRESHOLDS_EUR_MWH if t < ALWAYS_ON_THRESHOLD)
    print(f"      swept range: {lo:g} .. {hi:g} (plus always-on)")
    print("=" * 74)

    # Dispatch signal depends ONLY on the threshold, so count the ON steps once
    # per threshold rather than once per run.
    on_steps = {}
    for thr in THRESHOLDS_EUR_MWH:
        on_steps[thr] = int(build_hp_dispatch(n_timesteps=n_steps,
                                              len_timestep=TIMESTEP,
                                              threshold_eur_mwh=float(thr),
                                              verbose=False).sum())

    cache = _load_cache(cache_path)
    done = set()
    if not cache.empty:
        done = {(round(float(r["gas_price"]), 6),
                 round(float(r["co2_price"]), 4),
                 round(float(r["threshold"]), 4))
                for _, r in cache.iterrows()}
        print(f"  resuming: {len(done)} of {n_runs} runs already cached\n")

    counter = 0
    for gas_price in GAS_PRICE_GRID:
        for co2_price in CO2_PRICE_GRID:
            for thr in THRESHOLDS_EUR_MWH:
                counter += 1
                key = (round(float(gas_price), 6), round(float(co2_price), 4),
                       round(float(thr), 4))
                if key in done:
                    continue
                thr_lbl = "always-on" if thr >= ALWAYS_ON_THRESHOLD else f"{thr:g}"
                print(f"[{counter:>4}/{n_runs}] gas {gas_price*1000:5.0f} "
                      f"| CO2 {co2_price:5.0f} | thr {thr_lbl:>9} "
                      f"({on_steps[thr]}/{n_steps} steps ON)", flush=True)
                tag = f"G{gas_price*1000:.0f}_C{co2_price:.0f}_T{thr:g}"
                res = _simulate(gas_price, co2_price, thr, tag)
                if res is None:
                    continue
                row = _row_from(gas_price, co2_price, thr, on_steps[thr], res)
                _append_row(cache_path, row)
                lcoh = row["system_lcoh_yang"]
                print("            -> system LCOH = "
                      + (f"{lcoh*1000:7.2f} EUR/MWh" if pd.notna(lcoh) else "nan")
                      + f" | HP {row['hp_GWh']:.2f} GWh"
                      + f" | gas {row['gas_GWh']:.2f} GWh"
                      + f" | CO2 {row['total_CO2_t']:,.0f} t", flush=True)

    df = _load_cache(cache_path)
    if df.empty:
        print("\nNo results -- nothing to plot.")
        return df

    print(f"\nCollected {len(df)} runs")

    spread, spread_thr = _physics_invariance(df)
    status = "PASS" if spread < 1e-9 else "CHECK"
    print(f"  [{status}] physics invariance: max HP-heat spread across the "
          f"(gas, CO2) cells of one threshold = {spread:.3e} GWh"
          + (f" (at threshold {spread_thr:g})" if status == "CHECK" else ""))
    if status == "CHECK":
        print("         A fuel price is reaching the simulation. Expected: gas "
              "and CO2 prices\n         enter economic_analysis only.")

    best = _best_per_cell(df)

    print("\nBEST DISPATCH THRESHOLD PER (GAS, CO2) CELL")
    show = best.copy()
    show["best_threshold"] = show["best_threshold"].map(
        lambda t: "always-on" if t >= ALWAYS_ON_THRESHOLD else f"{t:g}")
    show["gas_price"] = (show["gas_price"] * 1000).map("{:.0f}".format)
    print(show[["gas_price", "co2_price", "best_threshold",
                "predicted_breakeven_eur_MWh", "best_lcoh_eur_MWh",
                "always_on_lcoh_eur_MWh", "saving_vs_always_on_eur_MWh",
                "hp_mean_COP"]].to_string(
        index=False, float_format=lambda v: f"{v:.2f}"))

    n_on = int((best["best_threshold"] >= ALWAYS_ON_THRESHOLD).sum())
    print(f"\n  always-on was optimal in {n_on} of {len(best)} cells; "
          f"largest saving from gating the HP = "
          f"{best['saving_vs_always_on_eur_MWh'].max():.2f} EUR/MWh")

    _plot_best_lcoh(best, results_dir)
    _plot_dispatch_value(best, results_dir)

    # --- Wide grids -----------------------------------------------------------
    grid_lcoh = best.pivot_table(index="gas_price", columns="co2_price",
                                 values="best_lcoh_eur_MWh")
    grid_thr = best.pivot_table(index="gas_price", columns="co2_price",
                                values="best_threshold")
    grid_save = best.pivot_table(index="gas_price", columns="co2_price",
                                 values="saving_vs_always_on_eur_MWh")

    out = os.path.join(results_dir, "gas_co2_summary.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All runs", index=False)
        best.to_excel(writer, sheet_name="Best per cell", index=False)
        grid_lcoh.reset_index().to_excel(
            writer, sheet_name="LCOH grid", index=False)
        grid_thr.reset_index().to_excel(
            writer, sheet_name="Threshold grid", index=False)
        grid_save.reset_index().to_excel(
            writer, sheet_name="Saving grid", index=False)
    print(f"\nSaved -> {out}")
    return df, best


if __name__ == "__main__":
    main()