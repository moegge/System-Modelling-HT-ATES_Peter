# -*- coding: utf-8 -*-
"""
Result Sweep Delft LCOH vs LCOH vs GD and LCOH vs maxv 08.09..py
Producing Fig 6 & Fig 7
==================================================================
Reproduces two of David Geerts' parameter-exploration figures and extends both
with the heat-pump (GGAH) configurations:

  STUDY 1 -- Fig. 6(a): system LCOH vs the G/D ratio.
  STUDY 2 -- Fig. 7   : LCOH and RES vs the HT-ATES maximum pumping rate.

The two studies sweep DIFFERENT variables and therefore keep separate caches and
separate figures. Both can be run from this one file; either can be switched off
with RUN_GD_STUDY / RUN_MAXV_STUDY.

STUDY 1 -- G/D SWEEP (paper Section 4.3, first bullet)
------------------------------------------------------
Each demand profile is held FIXED and the geothermal capacity is swept so that
G/D runs from 0.25 to 3:

    GEO_POWER [kW] = gd * annual_demand_kWh / 8760

Curves keep the name of their BASE scenario (G/D = 1.62 / 1.07 / 0.65), which is
the ratio that profile has against the 7.4 MW doublet -- but ALONG a curve the
actual G/D is the x-axis value, not the label. Colours identify the DEMAND
PROFILE, not a ratio.

STUDY 2 -- max_V SWEEP (paper Section 4.3, second bullet)
----------------------------------------------------------
The geothermal doublet is held at its BASE size (7.4 MW), so each demand case
sits at its own base G/D, and the HT-ATES maximum pumping rate max_V is swept
from 30 to 350 m3/h. Reported per the paper's Fig. 7: HT-ATES component LCOH,
system LCOH, and RES, on a twin y-axis.

The paper notes (Section 5.3.2) that these lines are NOT smooth: the recovery
efficiency comes from a machine-learning model that is least accurate at low
injected volumes, so low pumping rates give erratic results. Wobbles at the left
of those curves are expected, not a bug -- check the Reff column before reading
anything into them.

CONFIGURATIONS
--------------
  * GG          gas + geothermal                        solid   (paper, study 1)
  * GGA         + HT-ATES, no heat pump                 dashed  (paper, both)
  * GGAH flat   + HP, flat electricity price            dotted  (added, both)
  * GGAH dyn    + HP, hourly spot dispatch + Eq. 2      dashdot (added, both)

G (gas only) is not drawn in study 1: with no geothermal it is a single point,
not a curve. It has no ATES either, so it is absent from study 2 as well.

OUTPUTS (into RESULTS_DIR, created next to this file)
-----------------------------------------------------
  study 1:
    gd_curve_lcoh.png                  all cases on one axes
    gd_curve_lcoh_panels.png           one panel per demand case
    gd_curve_lcoh_paper.png            GG + GGA only, paper axis limits
    gd_curve_lcoh_paper_uncropped.png  the same curves, autoscaled
    gd_curve_data.csv                  incremental cache; also the raw data
    gd_curve_summary.xlsx              results + wide LCOH table + optimum
  study 2:
    maxv_curve_GGA.png                 replica of paper Fig. 7
    maxv_curve_GGAH_flat.png           same layout, HP at a flat power price
    maxv_curve_GGAH_dyn.png            same layout, HP on hourly spot dispatch
    maxv_curve_lcoh_compare.png        the three configs' system LCOH together
    maxv_curve_data.csv                incremental cache; also the raw data
    maxv_curve_summary.xlsx            results + wide tables + optimum

RESUMABLE: every completed run is appended to its cache CSV immediately.
Re-running skips whatever is already there, so an interrupted sweep continues
where it stopped -- and a pure re-plot costs seconds, not an hour. Delete a
cache CSV to force a clean re-run of that study (do this after ANY model change:
a stale cache is otherwise mixed silently into the figures).

    python "Test File GD Curve Peter.py"
==================================================================
"""

import os
import io
import inspect
import contextlib
import traceback
import sys

# model_driver.py / main2_Peter.py / ATES_obj_Peter.py live one level up, in
# System-Modelling-HT-ATES_Peter; this file sits in the "Delft Case" subfolder.
_MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODEL_ROOT not in sys.path:
    sys.path.insert(0, _MODEL_ROOT)
    
import matplotlib
matplotlib.use("Agg")          # headless: no windows even if something calls show()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd

from model_driver import (run_case, DEMAND_T_IN, DEMAND_T_OUT,
                          GEO_T_OUT)
from main2_Peter import demand_class, build_hp_dispatch

# ================================================================== #
#  WHICH STUDIES TO RUN                                              #
# ================================================================== #

RUN_GD_STUDY   = True      # Fig. 6(a): system LCOH vs G/D
RUN_MAXV_STUDY = True      # Fig. 7   : LCOH + RES vs HT-ATES max_V

# ================================================================== #
#  SHARED SETUP                                                      #
# ================================================================== #

# --- The three demand cases -------------------------------------------------
# label -> (demand_class profile name, network length [m], colour)
# The label is the BASE G/D (profile vs the 7.4 MW doublet), matching the paper's
# legend. Network length is a property of the demand case, so it does NOT change
# along either sweep. 8 km / 15 km are from paper 4.2; 23 km for "Delft Total" is
# an assumption (= TU + City), the paper does not state it.
DEMAND_CASES = {
    "G/D = 1.62": dict(profile="TU Delft",    network_m=8000.0,  color="tab:blue"),
    "G/D = 1.07": dict(profile="Delft City",  network_m=15000.0, color="tab:orange"),
    "G/D = 0.65": dict(profile="Delft Total", network_m=23000.0, color="tab:green"),
}

# --- Configurations ---------------------------------------------------------
# label -> (CONFIG string, dynamic dispatch, matplotlib linestyle)
CONFIG_VARIANTS = [
    ("GG",         dict(cfg="GG",   dynamic=False, linestyle="-")),
    ("GGA",        dict(cfg="GGA",  dynamic=False, linestyle="--")),
    ("GGAH flat",  dict(cfg="GGAH", dynamic=False, linestyle=":")),
    ("GGAH dyn",   dict(cfg="GGAH", dynamic=True,  linestyle="-.")),
]

# --- Heat pump --------------------------------------------------------------
HP_POWER_EL = 3000.0             # [kW_el] compressor rating (study 1, and study 2
                                 # when HP_SIZING_MODE == "fixed_kW")

# In study 2 the ATES size is the swept variable, so a FIXED compressor rating
# means the ATES/HP balance changes along the curve (Ath/Hel runs from ~0.2 at
# max_V = 30 to ~2.7 at 350). That confounds "effect of well size" with "effect
# of the ATES/HP ratio".
#   "fixed_kW"    -> HP stays at HP_POWER_EL          (simple, ratio varies)
#   "fixed_ratio" -> HP = ates_nominal_kW / HP_RATIO  (ratio held, HP scales)
# ates_nominal_kW = (max_V/3600) * rho*cp * (GEO_T_OUT - DEMAND_T_OUT), the same
# expression run_case uses, so the reported ratio_ATES_HP matches the target.
HP_SIZING_MODE = "fixed_ratio"
HP_RATIO       = 2.48            # used only when HP_SIZING_MODE == "fixed_ratio"
                                 # (2.48 = 7431 kW_th / 3000 kW_el at max_V = 320)
RHO_CP = 4180.0                  # [kJ/m3.K] must match RHO_CP inside run_case

# --- Dynamic dispatch threshold --------------------------------------------
# build_hp_dispatch thresholds the ALL-IN MARGINAL COST c_marg = spot + adder,
# where adder = supplier markup + marginal energy tax + transport. Passing 60
# there therefore means "run when spot < 60 - adder", i.e. deeply NEGATIVE spot.
# Set the wholesale threshold you actually mean below; the adder is read from
# build_hp_dispatch's own defaults so the two can never drift apart.
HP_SPOT_THRESHOLD_EUR_MWH = 60.0     # run the HP when the day-ahead spot is below this

_p = inspect.signature(build_hp_dispatch).parameters
_ADDER_EUR_MWH = (_p["M"].default + _p["eb_marg_decision"].default
                  + _p["tau"].default) * 1000.0
HP_THRESHOLD_EUR_MWH = HP_SPOT_THRESHOLD_EUR_MWH + _ADDER_EUR_MWH

# --- Output / behaviour -----------------------------------------------------
RESULTS_DIR   = "results gd curve"
SAVE_FIGURES  = True
QUIET_RUNS    = True     # True -> swallow run_case's per-run console output
OPT_TOL       = 0.01     # optimum "range" = all x within this fraction of the min

HOURS_PER_YEAR = 8760

# ================================================================== #
#  STUDY 1 SETTINGS -- G/D sweep                                     #
# ================================================================== #

GD_SWEEP = [0.25, 0.5, 0.75, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
            1.6, 1.8, 2.0, 2.25, 2.5, 2.75, 3.0]

GD_CACHE_CSV = "gd_curve_data.csv"
GD_VARIANTS  = ["GG", "GGA", "GGAH flat", "GGAH dyn"]   # drawn in this order
GD_YLIM      = None      # e.g. (75, 130) to crop; None = autoscale

# Replica of Fig. 6(a): only what the paper draws, on the paper's axis ranges.
PAPER_VARIANTS = ["GG", "GGA"]
PAPER_XLIM     = (0.2, 3.0)
PAPER_YLIM     = (80.0, 127.0)     # paper y-axis spans roughly 80 to 127

# ================================================================== #
#  STUDY 2 SETTINGS -- max_V sweep                                   #
# ================================================================== #

# Paper 4.3: "The size is varied between 30 and 350 m3/h."
MAXV_SWEEP = [30, 50, 75, 100, 125, 150, 200, 250, 300, 350]   # [m3/h]

# The doublet is at its BASE size here, so each demand case sits at its own base
# G/D. 7.4 MW is the paper's doublet (320 m3/h of 75 C against a 55 C return).
FIXED_GEO_POWER_MW = 7.4

MAXV_CACHE_CSV = "maxv_curve_data.csv"
MAXV_VARIANTS  = ["GGA", "GGAH flat", "GGAH dyn"]   # GG has no ATES -> excluded

# Axis ranges for the Fig. 7 replica. The paper's left axis runs ~75-160
# euro/MWh and its right axis ~0.45-1.0.
MAXV_LCOH_YLIM = (70.0, 165.0)
MAXV_RES_YLIM  = (0.40, 1.02)
# Set either to None to autoscale that axis instead.

# ================================================================== #


def _annual_demand_kWh(profile):
    """Annual demand [kWh] of one profile, at the DHN temperatures used here."""
    dem = demand_class(T_in=DEMAND_T_IN, T_out=DEMAND_T_OUT, example_demand=profile)
    return float(np.sum(dem.data))


def _geo_power_for_gd(gd, annual_demand_kWh):
    """Nameplate G/D: GEO_POWER [kW] = gd * annual_demand / 8760."""
    return gd * annual_demand_kWh / HOURS_PER_YEAR


def _ates_nominal_kW(max_V):
    """Peak direct-HX power of a freshly charged well [kW]. Mirrors run_case."""
    return (max_V / 3600.0) * RHO_CP * (GEO_T_OUT - DEMAND_T_OUT)


def _hp_power_for(max_V):
    """Compressor rating for a given well size, per HP_SIZING_MODE."""
    if HP_SIZING_MODE == "fixed_ratio":
        return _ates_nominal_kW(max_V) / HP_RATIO
    if HP_SIZING_MODE == "fixed_kW":
        return HP_POWER_EL
    raise ValueError(f"HP_SIZING_MODE must be 'fixed_kW' or 'fixed_ratio', "
                     f"got {HP_SIZING_MODE!r}")


def _variant(label):
    """Look up a configuration variant by its label."""
    return dict(CONFIG_VARIANTS)[label]


def _res_fraction(row):
    """
    Renewable share of delivered heat, consistent across configurations:
        (geo to demand + ATES direct + HP source heat) / demand
    HP grid electricity is EXCLUDED, matching the RES convention used in the
    25.08 sweep. hp_GWh is the condenser output (Q_evap + P_el), so the source
    heat is hp_GWh - hp_elec_GWh.
    """
    dem = row.get("demand_GWh")
    if not dem or not np.isfinite(dem) or dem <= 0:
        return np.nan
    geo = np.nan_to_num(row.get("geo_to_demand_GWh", 0.0))
    ates = np.nan_to_num(row.get("ates_direct_GWh", 0.0))
    hp = np.nan_to_num(row.get("hp_GWh", 0.0))
    hp_el = np.nan_to_num(row.get("hp_elec_GWh", 0.0))
    return (geo + ates + max(hp - hp_el, 0.0)) / dem


def _load_cache(path):
    """Previously completed rows, or an empty frame."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  cache unreadable ({type(e).__name__}: {e}) -- starting fresh")
        return pd.DataFrame()


def _append_row(path, row):
    """Append one result row, writing the header only for a new file."""
    pd.DataFrame([row]).to_csv(path, mode="a", index=False,
                               header=not os.path.exists(path))


def _simulate(variant, case, geo_power, tag, ates_max_v=None, hp_power=None):
    """
    One run_case call. Returns the result dict, or None if the run failed.
    ates_max_v / hp_power are only passed on when supplied, so run_case's own
    defaults apply otherwise.
    """
    kwargs = dict(
        CONFIG=variant["cfg"],
        GEO_POWER=geo_power,
        DEMAND_EXAMPLE=case["profile"],
        NETWORK_LENGTH_M=case["network_m"],
        tag=tag,
        make_plots=False,
        write_excel=False,        # hundreds of workbooks would be useless here
    )
    if ates_max_v is not None:
        kwargs["ATES_MAX_V"] = float(ates_max_v)
    if variant["cfg"] == "GGAH":
        kwargs["HP_POWER_EL"] = float(hp_power if hp_power is not None else HP_POWER_EL)
        kwargs["HP_DYNAMIC_DISPATCH"] = variant["dynamic"]
        kwargs["HP_THRESHOLD_EUR_MWH"] = HP_THRESHOLD_EUR_MWH

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


def _base_row(case_label, case, variant_label, variant, res):
    """The output fields both studies record."""
    row = {
        "demand_case":    case_label,
        "profile":        case["profile"],
        "variant":        variant_label,
        "config":         variant["cfg"],
        "dynamic":        bool(variant["dynamic"]),
        "network_m":      case["network_m"],
        "demand_GWh":     res.get("demand_GWh"),
        "GD_achieved":    res.get("GD_ratio"),
        "ATES_MAX_V":     res.get("ATES_MAX_V"),
        "HP_POWER_EL":    res.get("HP_POWER_EL"),
        "ratio_ATES_HP":  res.get("ratio_ATES_HP"),
        "system_lcoh_yang": res.get("system_lcoh_yang"),
        "geo_lcoh":       res.get("geo_lcoh"),
        "ates_lcoh":      res.get("ates_lcoh"),
        "gas_lcoh":       res.get("gas_lcoh"),
        "Reff":           res.get("Reff"),
        "injected_volume_m3":  res.get("injected_volume_m3"),
        "extracted_volume_m3": res.get("extracted_volume_m3"),
        "geo_to_demand_GWh":   res.get("geo_to_demand_GWh"),
        "ates_direct_GWh":     res.get("ates_direct_GWh"),
        "hp_GWh":         res.get("hp_GWh"),
        "gas_GWh":        res.get("gas_GWh"),
        "unmet_GWh":      res.get("unmet_GWh"),
        "hp_elec_GWh":    res.get("hp_elec_GWh"),
        "hp_mean_COP":    res.get("hp_mean_COP"),
        "hp_price_paid_eur_kwh": res.get("hp_price_paid_eur_kwh"),
        "total_CO2_t":    res.get("total_CO2_t"),
    }
    row["RES"] = _res_fraction(row)
    return row


def _optimum_table(df, x_col, value_col, value_label, tol=OPT_TOL):
    """
    Per (demand case, variant): the x with the lowest `value_col`, plus the span
    of x staying within `tol` of it. The paper reports optima as RANGES because
    the curves are flat near the bottom.
    """
    rows = []
    for (case, variant), sub in df.groupby(["demand_case", "variant"], sort=False):
        sub = sub.dropna(subset=[value_col]).sort_values(x_col)
        if sub.empty:
            continue
        y = sub[value_col].values * 1000.0
        x = sub[x_col].values
        i = int(np.argmin(y))
        within = x[y <= y[i] * (1.0 + tol)]
        rows.append({
            "demand case": case,
            "variant": variant,
            f"optimal {x_col}": x[i],
            f"{x_col} within {tol:.0%} of min": f"{within.min():g} - {within.max():g}",
            f"min {value_label} [euro/MWh]": y[i],
        })
    return pd.DataFrame(rows)


# ================================================================== #
#  STUDY 1 -- G/D SWEEP                                              #
# ================================================================== #

def run_gd_study(results_dir):
    cache_path = os.path.join(results_dir, GD_CACHE_CSV)
    variants = [(lbl, _variant(lbl)) for lbl in GD_VARIANTS]
    n_runs = len(GD_SWEEP) * len(DEMAND_CASES) * len(variants)

    print("=" * 70)
    print("STUDY 1 -- SYSTEM LCOH vs G/D   (paper Fig. 6a)")
    print("=" * 70)
    print(f"  G/D points          : {len(GD_SWEEP)}  ({min(GD_SWEEP)} .. {max(GD_SWEEP)})")
    print(f"  demand cases        : {len(DEMAND_CASES)}")
    print(f"  configurations      : {len(variants)}  ({', '.join(GD_VARIANTS)})")
    print(f"  TOTAL RUNS          : {n_runs}")
    print(f"  HP rating           : {HP_POWER_EL:.0f} kW_el")
    print(f"  cache               : {cache_path}")

    annual = {}
    for case_label, case in DEMAND_CASES.items():
        annual[case_label] = _annual_demand_kWh(case["profile"])
        print(f"  {case_label:<12} {case['profile']:<12} "
              f"{annual[case_label] / 1e6:7.2f} GWh/yr, "
              f"{annual[case_label] / 1e6 / 8.760:6.2f} MW avg, "
              f"network {case['network_m'] / 1000:.0f} km")
    print("=" * 70)

    cache = _load_cache(cache_path)
    done = set()
    if not cache.empty:
        done = {(r["demand_case"], r["variant"], round(float(r["GD_target"]), 4))
                for _, r in cache.iterrows()}
        print(f"  resuming: {len(done)} of {n_runs} runs already cached\n")

    counter = 0
    for case_label, case in DEMAND_CASES.items():
        for variant_label, variant in variants:
            for gd in GD_SWEEP:
                counter += 1
                key = (case_label, variant_label, round(float(gd), 4))
                if key in done:
                    continue
                print(f"[{counter:>4}/{n_runs}] {case_label} | {variant_label:<10} "
                      f"| G/D = {gd:.2f}", flush=True)
                geo_power = _geo_power_for_gd(gd, annual[case_label])
                tag = f"GD_{case['profile']}_{variant_label}_{gd:.2f}".replace(" ", "")
                res = _simulate(variant, case, geo_power, tag)
                if res is None:
                    continue
                row = _base_row(case_label, case, variant_label, variant, res)
                row["GD_target"] = gd
                row["GEO_POWER_kW"] = geo_power
                _append_row(cache_path, row)
                lcoh = row["system_lcoh_yang"]
                print("           -> system LCOH = "
                      + (f"{lcoh * 1000:.1f} EUR/MWh" if pd.notna(lcoh) else "nan"),
                      flush=True)

    df = _load_cache(cache_path)
    if df.empty:
        print("\nStudy 1: no results -- nothing to plot.")
        return df

    df["_case_order"] = df["demand_case"].map({c: i for i, c in enumerate(DEMAND_CASES)})
    df["_var_order"] = df["variant"].map({l: i for i, l in enumerate(GD_VARIANTS)})
    df = (df.dropna(subset=["_case_order", "_var_order"])
            .sort_values(["_case_order", "_var_order", "GD_target"])
            .reset_index(drop=True))

    drift = (df["GD_achieved"] - df["GD_target"]).abs()
    if drift.max() > 1e-6:
        print(f"\n  WARNING: achieved G/D drifts from target by up to "
              f"{drift.max():.4f} -- check the demand profile / GEO_POWER path.")

    print(f"\nStudy 1: collected {len(df)} runs")
    _plot_gd_combined(df, results_dir)
    _plot_gd_panels(df, results_dir)
    _plot_gd_paper_replica(df, results_dir)

    wide = df.pivot_table(index="GD_target", columns=["demand_case", "variant"],
                          values="system_lcoh_yang") * 1000.0
    wide.columns = [f"{c} | {v} [euro/MWh]" for c, v in wide.columns]
    wide = wide.reset_index()

    opt = _optimum_table(df, "GD_target", "system_lcoh_yang", "system LCOH")
    if not opt.empty:
        print("\nOPTIMAL G/D PER CURVE (minimum of the system LCOH curve)")
        print(opt.to_string(index=False))
        print("  Paper reference: GG optimum between G/D 1.4 and 1.6; "
              "GGA optimum between 1.1 and 1.3.")

    out = os.path.join(results_dir, "gd_curve_summary.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.drop(columns=["_case_order", "_var_order"]).to_excel(
            writer, sheet_name="All runs", index=False)
        wide.to_excel(writer, sheet_name="LCOH vs GD", index=False)
        if not opt.empty:
            opt.to_excel(writer, sheet_name="Optimum", index=False)
    print(f"Saved -> {out}")
    return df


def _plot_gd_combined(df, results_dir):
    """Every curve on one axes. Colour = demand case, style = config."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for case_label, case in DEMAND_CASES.items():
        for variant_label in GD_VARIANTS:
            variant = _variant(variant_label)
            sub = (df[(df["demand_case"] == case_label)
                      & (df["variant"] == variant_label)]
                   .dropna(subset=["system_lcoh_yang"]).sort_values("GD_target"))
            if sub.empty:
                continue
            ax.plot(sub["GD_target"].values, sub["system_lcoh_yang"].values * 1000.0,
                    color=case["color"], linestyle=variant["linestyle"], linewidth=1.6)

    colour_handles = [Line2D([], [], color=c["color"], lw=1.6, label=lbl)
                      for lbl, c in DEMAND_CASES.items()]
    style_handles = [Line2D([], [], color="0.3", lw=1.6,
                            linestyle=_variant(lbl)["linestyle"], label=lbl)
                     for lbl in GD_VARIANTS]
    first = ax.legend(handles=colour_handles, loc="upper left", fontsize=8,
                      title="Demand case (base G/D)", title_fontsize=8)
    ax.add_artist(first)
    ax.legend(handles=style_handles, loc="upper center", fontsize=8,
              title="Configuration", title_fontsize=8)

    ax.set_xlabel(r"$G/D$")
    ax.set_ylabel("System LCOH (\u20ac/MWh)")
    ax.set_title("System LCOH vs G/D  (system = LCOE_calc_Yang, 60-yr horizon)")
    ax.grid(alpha=0.3)
    if GD_YLIM is not None:
        ax.set_ylim(*GD_YLIM)
    fig.tight_layout()
    _save(fig, results_dir, "gd_curve_lcoh")


def _plot_gd_panels(df, results_dir):
    """One panel per demand case, with the minimum of each curve marked."""
    cases = [c for c in DEMAND_CASES if (df["demand_case"] == c).any()]
    if not cases:
        return
    fig, axes = plt.subplots(1, len(cases), figsize=(4.6 * len(cases), 4.4),
                             sharey=True, squeeze=False)

    for ax_i, case_label in enumerate(cases):
        ax = axes[0][ax_i]
        for variant_label in GD_VARIANTS:
            variant = _variant(variant_label)
            sub = (df[(df["demand_case"] == case_label)
                      & (df["variant"] == variant_label)]
                   .dropna(subset=["system_lcoh_yang"]).sort_values("GD_target"))
            if sub.empty:
                continue
            gd = sub["GD_target"].values
            y = sub["system_lcoh_yang"].values * 1000.0
            ax.plot(gd, y, linestyle=variant["linestyle"], marker="o", ms=3, lw=1.5,
                    label=variant_label if ax_i == 0 else None)
            i = int(np.argmin(y))
            ax.plot(gd[i], y[i], marker="v", ms=7, mfc="none",
                    color=ax.lines[-1].get_color())

        ax.set_title(f"{case_label}  ({DEMAND_CASES[case_label]['profile']})", fontsize=9)
        ax.set_xlabel(r"$G/D$")
        ax.grid(alpha=0.3)
        if GD_YLIM is not None:
            ax.set_ylim(*GD_YLIM)
    axes[0][0].set_ylabel("System LCOH (\u20ac/MWh)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("System LCOH vs G/D per demand case  "
                 "(open triangle = minimum of each curve)", fontsize=10)
    fig.tight_layout()
    _save(fig, results_dir, "gd_curve_lcoh_panels")


def _plot_gd_paper_replica(df, results_dir):
    """
    Stripped-down replica of paper Fig. 6(a): GG and GGA only, paper axis ranges.
    Saved twice -- cropped (for shape) and autoscaled (for level), because a
    cropped axis hides curves leaving the top of the frame, which is exactly
    where the GGA lines head at G/D = 3.
    """
    fig, ax = plt.subplots(figsize=(5.2, 4.6))

    for case_label, case in DEMAND_CASES.items():
        for variant_label in PAPER_VARIANTS:
            variant = _variant(variant_label)
            sub = (df[(df["demand_case"] == case_label)
                      & (df["variant"] == variant_label)]
                   .dropna(subset=["system_lcoh_yang"]).sort_values("GD_target"))
            if sub.empty:
                continue
            ax.plot(sub["GD_target"].values, sub["system_lcoh_yang"].values * 1000.0,
                    color=case["color"], linestyle=variant["linestyle"], linewidth=1.3)

    handles = [Line2D([], [], color=c["color"], lw=1.3, label=lbl)
               for lbl, c in DEMAND_CASES.items()]
    handles += [Line2D([], [], color="k", lw=1.3,
                        linestyle=_variant(lbl)["linestyle"], label=lbl)
                for lbl in PAPER_VARIANTS]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=1.0)

    ax.set_xlabel(r"$G/D$")
    ax.set_ylabel("System LCOH (\u20ac/MWh)")
    ax.set_xlim(*PAPER_XLIM)
    ax.set_ylim(*PAPER_YLIM)
    fig.tight_layout()

    if SAVE_FIGURES:
        path = os.path.join(results_dir, "gd_curve_lcoh_paper.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved figure -> {path}")
        ax.set_xlim(auto=True)
        ax.set_ylim(auto=True)
        ax.relim()
        ax.autoscale_view()
        path = os.path.join(results_dir, "gd_curve_lcoh_paper_uncropped.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved figure -> {path}")
    plt.close(fig)


# ================================================================== #
#  STUDY 2 -- max_V SWEEP  (paper Fig. 7)                            #
# ================================================================== #

def run_maxv_study(results_dir):
    cache_path = os.path.join(results_dir, MAXV_CACHE_CSV)
    variants = [(lbl, _variant(lbl)) for lbl in MAXV_VARIANTS]
    geo_power = FIXED_GEO_POWER_MW * 1000.0
    n_runs = len(MAXV_SWEEP) * len(DEMAND_CASES) * len(variants)

    print("\n" + "=" * 70)
    print("STUDY 2 -- LCOH + RES vs HT-ATES max_V   (paper Fig. 7)")
    print("=" * 70)
    print(f"  max_V points        : {len(MAXV_SWEEP)}  "
          f"({min(MAXV_SWEEP)} .. {max(MAXV_SWEEP)} m3/h)")
    print(f"  demand cases        : {len(DEMAND_CASES)}")
    print(f"  configurations      : {len(variants)}  ({', '.join(MAXV_VARIANTS)})")
    print(f"  TOTAL RUNS          : {n_runs}")
    print(f"  geothermal (fixed)  : {FIXED_GEO_POWER_MW} MW -> base G/D per case")
    print(f"  HP sizing           : {HP_SIZING_MODE}"
          + (f" at {HP_POWER_EL:.0f} kW_el" if HP_SIZING_MODE == "fixed_kW"
             else f", Ath/Hel = {HP_RATIO:g}"))
    if HP_SIZING_MODE == "fixed_kW":
        r_lo = _ates_nominal_kW(min(MAXV_SWEEP)) / HP_POWER_EL
        r_hi = _ates_nominal_kW(max(MAXV_SWEEP)) / HP_POWER_EL
        print(f"                        -> Ath/Hel varies {r_lo:.2f} .. {r_hi:.2f} "
              f"across the sweep")
    print(f"  cache               : {cache_path}")
    print("=" * 70)

    cache = _load_cache(cache_path)
    done = set()
    if not cache.empty:
        done = {(r["demand_case"], r["variant"], round(float(r["MAXV_target"]), 4))
                for _, r in cache.iterrows()}
        print(f"  resuming: {len(done)} of {n_runs} runs already cached\n")

    counter = 0
    for case_label, case in DEMAND_CASES.items():
        for variant_label, variant in variants:
            for max_v in MAXV_SWEEP:
                counter += 1
                key = (case_label, variant_label, round(float(max_v), 4))
                if key in done:
                    continue
                hp_power = _hp_power_for(max_v) if variant["cfg"] == "GGAH" else None
                msg = (f"[{counter:>4}/{n_runs}] {case_label} | {variant_label:<10} "
                       f"| max_V = {max_v:g} m3/h")
                if hp_power is not None:
                    msg += f", HP = {hp_power:.0f} kW_el"
                print(msg, flush=True)
                tag = f"V_{case['profile']}_{variant_label}_{max_v:g}".replace(" ", "")
                res = _simulate(variant, case, geo_power, tag,
                                ates_max_v=max_v, hp_power=hp_power)
                if res is None:
                    continue
                row = _base_row(case_label, case, variant_label, variant, res)
                row["MAXV_target"] = float(max_v)
                row["GEO_POWER_kW"] = geo_power
                row["ates_nominal_kW"] = _ates_nominal_kW(max_v)
                _append_row(cache_path, row)
                lcoh, ates = row["system_lcoh_yang"], row["ates_lcoh"]
                print("           -> system LCOH = "
                      + (f"{lcoh * 1000:.1f}" if pd.notna(lcoh) else "nan")
                      + " | ATES LCOH = "
                      + (f"{ates * 1000:.1f}" if pd.notna(ates) else "nan")
                      + f" EUR/MWh | RES = {row['RES']:.3f}", flush=True)

    df = _load_cache(cache_path)
    if df.empty:
        print("\nStudy 2: no results -- nothing to plot.")
        return df

    df["_case_order"] = df["demand_case"].map({c: i for i, c in enumerate(DEMAND_CASES)})
    df["_var_order"] = df["variant"].map({l: i for i, l in enumerate(MAXV_VARIANTS)})
    df = (df.dropna(subset=["_case_order", "_var_order"])
            .sort_values(["_case_order", "_var_order", "MAXV_target"])
            .reset_index(drop=True))

    print(f"\nStudy 2: collected {len(df)} runs")

    # One Fig. 7 replica per configuration.
    fname_of = {"GGA": "maxv_curve_GGA",
                "GGAH flat": "maxv_curve_GGAH_flat",
                "GGAH dyn": "maxv_curve_GGAH_dyn"}
    subtitle_of = {
        "GGA": "gas + geothermal + HT-ATES  (no heat pump)  --  replica of paper Fig. 7",
        "GGAH flat": "gas + geothermal + HT-ATES + HP  (flat electricity price)",
        "GGAH dyn": f"gas + geothermal + HT-ATES + HP  (dynamic dispatch, "
                    f"spot < {HP_SPOT_THRESHOLD_EUR_MWH:.0f} \u20ac/MWh)",
    }
    for variant_label in MAXV_VARIANTS:
        _plot_maxv_fig7(df, variant_label, subtitle_of.get(variant_label, ""),
                        results_dir, fname_of.get(variant_label,
                                                  f"maxv_curve_{variant_label}"))
    _plot_maxv_compare(df, results_dir)

    # --- Wide tables --------------------------------------------------------
    wide_sys = df.pivot_table(index="MAXV_target", columns=["demand_case", "variant"],
                              values="system_lcoh_yang") * 1000.0
    wide_sys.columns = [f"{c} | {v} [euro/MWh]" for c, v in wide_sys.columns]
    wide_ates = df.pivot_table(index="MAXV_target", columns=["demand_case", "variant"],
                               values="ates_lcoh") * 1000.0
    wide_ates.columns = [f"{c} | {v} [euro/MWh]" for c, v in wide_ates.columns]
    wide_res = df.pivot_table(index="MAXV_target", columns=["demand_case", "variant"],
                              values="RES")
    wide_res.columns = [f"{c} | {v} [-]" for c, v in wide_res.columns]

    opt_sys = _optimum_table(df, "MAXV_target", "system_lcoh_yang", "system LCOH")
    opt_ates = _optimum_table(df, "MAXV_target", "ates_lcoh", "ATES LCOH")
    if not opt_ates.empty:
        print("\nOPTIMAL max_V PER CURVE (minimum of the HT-ATES component LCOH)")
        print(opt_ates.to_string(index=False))
        print("  Paper reference: minimum HT-ATES LCOH between 100 and 200 m3/h "
              "for G/D = 1.07 and 0.65;\n  at a relatively low pumping rate for "
              "G/D = 1.62. Lines are not smooth at low max_V -- the recovery-\n"
              "  efficiency model is least accurate there (paper 5.3.2).")

    out = os.path.join(results_dir, "maxv_curve_summary.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.drop(columns=["_case_order", "_var_order"]).to_excel(
            writer, sheet_name="All runs", index=False)
        wide_sys.reset_index().to_excel(writer, sheet_name="System LCOH", index=False)
        wide_ates.reset_index().to_excel(writer, sheet_name="ATES LCOH", index=False)
        wide_res.reset_index().to_excel(writer, sheet_name="RES", index=False)
        if not opt_sys.empty:
            opt_sys.to_excel(writer, sheet_name="Optimum system", index=False)
        if not opt_ates.empty:
            opt_ates.to_excel(writer, sheet_name="Optimum ATES", index=False)
    print(f"Saved -> {out}")
    return df


def _plot_maxv_fig7(df, variant_label, subtitle, results_dir, fname):
    """
    Paper Fig. 7 layout for ONE configuration: HT-ATES LCOH and system LCOH on
    the left axis, RES on the right, colour = demand case, three line styles.
    """
    sub_all = df[df["variant"] == variant_label]
    if sub_all.empty:
        return

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax_res = ax.twinx()

    for case_label, case in DEMAND_CASES.items():
        sub = (sub_all[sub_all["demand_case"] == case_label]
               .sort_values("MAXV_target"))
        if sub.empty:
            continue
        x = sub["MAXV_target"].values
        ax.plot(x, sub["system_lcoh_yang"].values * 1000.0,
                color=case["color"], linestyle="--", lw=1.5)
        ax.plot(x, sub["ates_lcoh"].values * 1000.0,
                color=case["color"], linestyle="-.", lw=1.5)
        ax_res.plot(x, sub["RES"].values,
                    color=case["color"], linestyle="-", lw=1.5)

    handles = [Line2D([], [], color=c["color"], lw=1.5, label=lbl)
               for lbl, c in DEMAND_CASES.items()]
    handles += [
        Line2D([], [], color="k", lw=1.5, linestyle="--", label="System LCOH"),
        Line2D([], [], color="k", lw=1.5, linestyle="-.", label="HT-ATES LCOH"),
        Line2D([], [], color="k", lw=1.5, linestyle="-", label="RES"),
    ]
    ax.legend(handles=handles, loc="center right", fontsize=8, framealpha=1.0)

    ax.set_xlabel("HT-ATES maximum pumping rate (m\u00b3/hour)")
    ax.set_ylabel("LCOH (\u20ac/MWh)")
    ax_res.set_ylabel("Renewable energy share")
    if MAXV_LCOH_YLIM is not None:
        ax.set_ylim(*MAXV_LCOH_YLIM)
    if MAXV_RES_YLIM is not None:
        ax_res.set_ylim(*MAXV_RES_YLIM)
    ax.grid(alpha=0.3)
    ax.set_title(f"{variant_label}\n{subtitle}", fontsize=9)
    fig.tight_layout()
    _save(fig, results_dir, fname)


def _plot_maxv_compare(df, results_dir):
    """
    System LCOH only, all three configurations together, one panel per demand
    case. This is the figure that answers "does the heat pump change how the
    well should be sized".
    """
    cases = [c for c in DEMAND_CASES if (df["demand_case"] == c).any()]
    if not cases:
        return
    fig, axes = plt.subplots(1, len(cases), figsize=(4.6 * len(cases), 4.4),
                             sharey=True, squeeze=False)

    for ax_i, case_label in enumerate(cases):
        ax = axes[0][ax_i]
        for variant_label in MAXV_VARIANTS:
            variant = _variant(variant_label)
            sub = (df[(df["demand_case"] == case_label)
                      & (df["variant"] == variant_label)]
                   .dropna(subset=["system_lcoh_yang"]).sort_values("MAXV_target"))
            if sub.empty:
                continue
            x = sub["MAXV_target"].values
            y = sub["system_lcoh_yang"].values * 1000.0
            ax.plot(x, y, linestyle=variant["linestyle"], marker="o", ms=3, lw=1.5,
                    label=variant_label if ax_i == 0 else None)
            i = int(np.argmin(y))
            ax.plot(x[i], y[i], marker="v", ms=7, mfc="none",
                    color=ax.lines[-1].get_color())

        ax.set_title(f"{case_label}  ({DEMAND_CASES[case_label]['profile']})", fontsize=9)
        ax.set_xlabel("max_V (m\u00b3/h)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("System LCOH (\u20ac/MWh)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("System LCOH vs HT-ATES pumping rate  "
                 "(open triangle = minimum of each curve)", fontsize=10)
    fig.tight_layout()
    _save(fig, results_dir, "maxv_curve_lcoh_compare")


# ================================================================== #

def _save(fig, results_dir, name):
    if SAVE_FIGURES:
        path = os.path.join(results_dir, name + ".png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved figure -> {path}")
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, RESULTS_DIR)
    os.makedirs(results_dir, exist_ok=True)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)

    print(f"dynamic dispatch: HP on when spot < {HP_SPOT_THRESHOLD_EUR_MWH:.1f} EUR/MWh "
          f"(= c_marg < {HP_THRESHOLD_EUR_MWH:.2f}, adder {_ADDER_EUR_MWH:.2f})\n")

    df_gd = run_gd_study(results_dir) if RUN_GD_STUDY else pd.DataFrame()
    df_v = run_maxv_study(results_dir) if RUN_MAXV_STUDY else pd.DataFrame()
    return df_gd, df_v


if __name__ == "__main__":
    main()