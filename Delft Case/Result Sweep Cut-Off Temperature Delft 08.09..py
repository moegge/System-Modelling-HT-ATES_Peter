# -*- coding: utf-8 -*-
"""
Result Sweep Cut-Off Temperature Delft 08.09..py
Producing Fig 8
==================================================================
Reproduces Fig. 8 of David Geerts' paper -- the impact of the DH cut-off (return)
temperature on system LCOH and renewable energy share -- and extends it with the
two heat-pump configurations.

FIGURE SETS (each one panel per demand case)
--------------------------------------------
  1. GG   + GGA          the paper replica, unchanged
  2. GGA  + GGAH flat    heat pump at a flat electricity price
  3. GGA  + GGAH dyn     heat pump on hourly spot dispatch (Eq. 2)
  4. GGA  + both GGAH    all three storage configurations at a glance

GGA is the common reference in sets 2-4 and its input data is identical to set
1 -- same runs, same cache rows -- so the GGA curve is literally the same line
in every figure.

WHAT THE PAPER DOES (Section 4.3, third bullet; results in 5.3.3)
------------------------------------------------------------------
The cut-off temperature is varied between 45 and 65 C, with "no additional costs
incurred to do so". Section 5.3.3 states the effect is TWOFOLD:

  1. "it changes the capacity of the geothermal doublet due to variations in dT
     (as shown in Eq. (1))"
  2. "it influences the energy that can be delivered by the HT-ATES, as this is
     also dependent on dT"

Effect 2 is automatic: the cut-off is passed to calc_heat as T_cutoff, so it
sets the HX floor.

Effect 1 is NOT automatic. run_case builds the doublet from a fixed POWER, which
would hold the capacity constant as the cut-off moves and silently drop half the
physics. The paper sizes the doublet by FLOW RATE (320 m3/h at 75 C, paper
4.1.1), so the capacity follows from the temperature difference. This script
derives

    GEO_POWER(T_cut) = flow * (T_geo - T_cut) * 1e-7 * 4186 * 1000 * 2.77777

which is geothermal.calc_output's own expression, evaluated here and passed in
as GEO_POWER. At 320 m3/h and a 55 C cut-off it returns 7441 kW, i.e. the
paper's 7.4 MW doublet, so the base point is unchanged and only the sweep
behaves differently. GEO_SIZING_MODE = "power" gives the fixed-capacity contrast.

CONSEQUENCE WORTH KNOWING: because the capacity moves, G/D moves along the
x-axis too -- about 1.5x the base value at 45 C down to 0.5x at 65 C. The curves
keep their base-scenario names, exactly as in the paper. GD_achieved is recorded
per run and plotted in the diagnostics figure, so the confounding is visible.

A THIRD CUT-OFF EFFECT, FOR THE GGAH RUNS ONLY
-----------------------------------------------
The heat pump's cold-side floor is T_floor = max(T_cut - delta_T_coldside, T_g).
Over 45-65 C with delta_T_coldside = 20 K that runs 25 -> 45 C, always above the
15 C ground temperature, so the max() never clips and there is no discontinuity
inside this sweep. It would clip below a 35 C cut-off.

The HP rating is held FIXED across the sweep (HP_SIZING_MODE = "fixed_kW",
3000 kW_el -- the value used in the max_V study, where it pairs with max_V = 320
at Ath/Hel = 2.48). Because the ATES nominal thermal power depends on the same
dT, the ratio then varies from 3.72 at 45 C to 1.24 at 65 C. Set HP_SIZING_MODE
= "fixed_ratio" to hold Ath/Hel instead, which makes the compressor rating vary
4495 -> 1498 kW. Neither is "right": fixed kW asks "I bought this heat pump, how
does the cut-off change its value", fixed ratio asks "how does the cut-off change
the value of a balanced pair". The banner prints whichever range applies.

FILES THIS SCRIPT NEEDS
-----------------------
  model_driver.py   run_case, DEMAND_T_IN, GEO_T_OUT   (imported)
  main2_Peter.py             demand_class, build_hp_dispatch    (imported)
  ATES_obj_Peter.py                              (imported by main2_Peter)
  results_AXI_V2                                 (read by ATES_obj.__init__)
  Predict_REFF_boostedregression.pkl             (read by predict_reff)
  Warmtevraag_Delft_parquet                      (read by demand_class)
  Netherlands.csv                                (GGAH dyn only, read by
                                                  build_hp_dispatch)
Run from the folder holding all of them.

OUTPUTS (into RESULTS_DIR, created next to this file)
-----------------------------------------------------
  cutoff_fig8.png                 GG + GGA, paper layout
  cutoff_fig8_<case>.png          each panel of the paper set on its own
  cutoff_fig8_hp_flat.png         GGA + GGAH flat
  cutoff_fig8_hp_dyn.png          GGA + GGAH dynamic
  cutoff_fig8_hp_both.png         GGA + both GGAH variants
  cutoff_diagnostics.png          geo capacity, G/D, Reff
  cutoff_diagnostics_hp.png       HP heat, electricity, COP, price paid
  cutoff_data.csv                 incremental cache; also the raw data
  cutoff_summary.xlsx             all runs + wide LCOH/RES tables + optimum

RESUMABLE: every completed run is appended to the cache immediately. _append_row
now reconciles columns, so adding the heat-pump fields does NOT invalidate rows
written by an earlier version of this script -- the existing GG/GGA runs are
reused and only the new GGAH runs are simulated. Still delete the cache after
any MODEL change, since nothing in the key records the model version.

    python "Test File Cut-Off Temperature Peter.py"
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

from model_driver import run_case, DEMAND_T_IN, GEO_T_OUT
from main2_Peter import demand_class, build_hp_dispatch

# ================================================================== #
#  SWEEP DEFINITION                                                  #
# ================================================================== #

# Paper 4.3: "It varies between 45 C and 65 C."
CUTOFF_SWEEP = [45.0, 47.5, 50.0, 52.5, 55.0, 57.5, 60.0, 62.5, 65.0]   # [C]

# --- The three demand cases -------------------------------------------------
# The label is the BASE G/D (against the 7.4 MW doublet at a 55 C cut-off).
# Network length is a property of the demand case and does not change along the
# sweep. 8 km / 15 km are from paper 4.2; 23 km for "Delft Total" is an
# assumption (= TU + City), not stated in the paper.
DEMAND_CASES = {
    "G/D = 1.62": dict(profile="TU Delft",    network_m=8000.0),
    "G/D = 1.07": dict(profile="Delft City",  network_m=15000.0),
    "G/D = 0.65": dict(profile="Delft Total", network_m=23000.0),
}

# --- Configurations ---------------------------------------------------------
# Colour identifies the CONFIGURATION here (one panel per demand case), which is
# the opposite of the Fig. 6 / Fig. 7 convention. It matches the paper's Fig. 8.
CONFIG_VARIANTS = {
    "GG":        dict(cfg="GG",   dynamic=False, color="tab:orange"),
    "GGA":       dict(cfg="GGA",  dynamic=False, color="tab:blue"),
    "GGAH flat": dict(cfg="GGAH", dynamic=False, color="tab:green"),
    "GGAH dyn":  dict(cfg="GGAH", dynamic=True,  color="tab:red"),
}

# Which configurations to SIMULATE. Everything the figure sets reference must
# appear here.
CUTOFF_VARIANTS = ["GGA", "GG", "GGAH flat", "GGAH dyn"]

# Which figures to DRAW: (variant list, filename, subtitle).
# GGA leads each heat-pump set so it is the visual reference.
PLOT_SETS = [
    (["GGA", "GG"],                     "cutoff_fig8",
     "paper replica"),
    (["GGA", "GGAH flat"],              "cutoff_fig8_hp_flat",
     "heat pump at a flat electricity price"),
    (["GGA", "GGAH dyn"],               "cutoff_fig8_hp_dyn",
     "heat pump on hourly spot dispatch"),
    (["GGA", "GGAH flat", "GGAH dyn"],  "cutoff_fig8_hp_both",
     "both heat-pump variants against GGA"),
]
# Single-panel versions are written for these sets only (by filename).
SINGLE_PANEL_SETS = ["cutoff_fig8"]

# --- Geothermal sizing ------------------------------------------------------
# "flow"  : doublet fixed at GEO_FLOW_M3H; capacity follows from (T_geo - T_cut).
#           The paper's definition; reproduces effect 1 of 5.3.3.
# "power" : doublet fixed at GEO_POWER_KW regardless of cut-off. Useful as a
#           contrast: running both isolates how much of Fig. 8 is the capacity
#           change and how much is the HT-ATES delta-T change.
GEO_SIZING_MODE = "flow"
GEO_FLOW_M3H    = 320.0          # [m3/h]  paper 4.1.1
GEO_POWER_KW    = 7441.0         # [kW]    only when GEO_SIZING_MODE == "power"

# --- DHN supply temperature -------------------------------------------------
# Held at 75 C throughout; only the RETURN (= cut-off) temperature is swept.
DHN_T_IN = DEMAND_T_IN           # 75 C from the test file

# --- ATES -------------------------------------------------------------------
ATES_MAX_V_BASE = 320.0          # [m3/h] base size, held fixed across the sweep

# --- Heat pump --------------------------------------------------------------
# "fixed_kW"    -> compressor stays at HP_POWER_EL; Ath/Hel varies with the
#                  cut-off (3.72 at 45 C to 1.24 at 65 C for 3000 kW).
# "fixed_ratio" -> compressor = ates_nominal_kW / HP_RATIO; the balance is held
#                  and the rating varies (4495 to 1498 kW for Ath/Hel = 2.48).
HP_SIZING_MODE      = "fixed_kW"
HP_POWER_EL         = 3000.0     # [kW_el] pairs with max_V 320 at Ath/Hel 2.48
HP_RATIO            = 2.48       # [-] used only in "fixed_ratio" mode
HP_DELTA_T_COLDSIDE = 20.0       # [K] run_case default, restated for the banner
ATES_T_GROUND       = 15.0       # [C] run_case default, restated for the banner
RHO_CP = 4180.0                  # [kJ/m3.K] must match RHO_CP inside run_case

# --- Dynamic dispatch threshold --------------------------------------------
# build_hp_dispatch thresholds the ALL-IN MARGINAL COST c_marg = spot + adder,
# where adder = supplier markup + marginal energy tax + transport. Passing 60
# there would mean "run when spot < 60 - adder", i.e. deeply NEGATIVE spot. Set
# the wholesale threshold you actually mean; the adder is read from
# build_hp_dispatch's own defaults so the two cannot drift apart.
HP_SPOT_THRESHOLD_EUR_MWH = 60.0

_p = inspect.signature(build_hp_dispatch).parameters
_ADDER_EUR_MWH = (_p["M"].default + _p["eb_marg_decision"].default
                  + _p["tau"].default) * 1000.0
HP_THRESHOLD_EUR_MWH = HP_SPOT_THRESHOLD_EUR_MWH + _ADDER_EUR_MWH

# --- Output / behaviour -----------------------------------------------------
RESULTS_DIR = "results cutoff"
CACHE_CSV   = "cutoff_data.csv"
SAVE_FIGURES = True
QUIET_RUNS   = True              # True -> swallow run_case's per-run output
OPT_TOL      = 0.01              # optimum "range" = all x within this of the min

# Axis ranges for the Fig. 8 replica. The paper's left axis runs ~78-130
# euro/MWh and its right axis 0.2-1.0. Set to None to autoscale.
FIG8_LCOH_YLIM = (78.0, 132.0)
FIG8_RES_YLIM  = (0.20, 1.02)
# The heat-pump sets can leave the paper's range; give them their own limits.
FIG8_HP_LCOH_YLIM = None         # None = autoscale
FIG8_HP_RES_YLIM  = (0.20, 1.02)


# ================================================================== #


def _geo_power_for_cutoff(t_cut):
    """
    Doublet capacity [kW] at a given cut-off temperature.

    The "flow" branch is geothermal.calc_output's own expression:
        power = flow_rate * (T_out - demand.T_out) * 1e-7 * 4186 * 1000 * 2.77777
    evaluated here so it can be passed to run_case as a fixed power. At 320 m3/h
    and a 55 C cut-off this returns 7441 kW = the paper's 7.4 MW doublet.
    """
    if GEO_SIZING_MODE == "flow":
        return GEO_FLOW_M3H * (GEO_T_OUT - t_cut) * 1e-7 * 4186 * 1000 * 2.77777
    if GEO_SIZING_MODE == "power":
        return GEO_POWER_KW
    raise ValueError(f"GEO_SIZING_MODE must be 'flow' or 'power', "
                     f"got {GEO_SIZING_MODE!r}")


def _ates_nominal_kW(t_cut, max_V=ATES_MAX_V_BASE):
    """
    Peak direct-HX power of a freshly charged well [kW]. Mirrors run_case, which
    computes (max_V/3600) * rho*cp * (GEO_T_OUT - DEMAND_T_OUT) -- and here
    DEMAND_T_OUT is the swept cut-off, so this moves along the sweep.
    """
    return (max_V / 3600.0) * RHO_CP * (GEO_T_OUT - t_cut)


def _hp_power_for_cutoff(t_cut):
    """Compressor rating [kW_el] at a given cut-off, per HP_SIZING_MODE."""
    if HP_SIZING_MODE == "fixed_kW":
        return HP_POWER_EL
    if HP_SIZING_MODE == "fixed_ratio":
        return _ates_nominal_kW(t_cut) / HP_RATIO
    raise ValueError(f"HP_SIZING_MODE must be 'fixed_kW' or 'fixed_ratio', "
                     f"got {HP_SIZING_MODE!r}")

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

    A plain `mode="a", header=False` append writes values in the CURRENT dict
    order under the OLD header, so adding a field silently shifts every later
    column. This version reorders to match the existing header, and rewrites the
    file when genuinely new columns appear -- which is what lets the heat-pump
    fields be added without discarding the GG/GGA rows from an earlier run.
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


def _res_fraction(row):
    """
    Renewable share of delivered heat:
        (geo to demand + ATES direct + HP source heat) / demand
    HP grid electricity is EXCLUDED, matching the convention in the other sweep
    scripts. For GG and GGA the heat-pump terms are zero, so this reduces to the
    paper's definition and the GGA curve is directly comparable to Fig. 8.
    """
    dem = row.get("demand_GWh")
    if not dem or not np.isfinite(dem) or dem <= 0:
        return np.nan
    geo = np.nan_to_num(row.get("geo_to_demand_GWh", 0.0))
    ates = np.nan_to_num(row.get("ates_direct_GWh", 0.0))
    hp = np.nan_to_num(row.get("hp_GWh", 0.0))
    hp_el = np.nan_to_num(row.get("hp_elec_GWh", 0.0))
    return (geo + ates + max(hp - hp_el, 0.0)) / dem


def _simulate(variant, case, t_cut, geo_power, hp_power, tag):
    """One run_case call at a given cut-off. Returns the result dict, or None."""
    kwargs = dict(
        CONFIG=variant["cfg"],
        GEO_POWER=geo_power,
        GEO_T_OUT=GEO_T_OUT,
        DEMAND_EXAMPLE=case["profile"],
        DEMAND_T_IN=DHN_T_IN,
        DEMAND_T_OUT=t_cut,               # <- the swept variable
        ATES_MAX_V=ATES_MAX_V_BASE,
        NETWORK_LENGTH_M=case["network_m"],
        tag=tag,
        make_plots=False,
        write_excel=False,
    )
    if variant["cfg"] == "GGAH":
        kwargs["HP_POWER_EL"] = float(hp_power)
        kwargs["HP_DELTA_T_COLDSIDE"] = HP_DELTA_T_COLDSIDE
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


def _row_from(case_label, case, variant_label, variant, t_cut, geo_power, res):
    row = {
        "demand_case":  case_label,
        "profile":      case["profile"],
        "variant":      variant_label,
        "config":       variant["cfg"],
        "dynamic":      bool(variant["dynamic"]),
        "T_cutoff":     float(t_cut),
        "GEO_POWER_kW": geo_power,
        "geo_flow_m3h": GEO_FLOW_M3H if GEO_SIZING_MODE == "flow" else np.nan,
        "network_m":    case["network_m"],
        "demand_GWh":   res.get("demand_GWh"),
        "GD_achieved":  res.get("GD_ratio"),
        "ATES_MAX_V":   res.get("ATES_MAX_V"),
        "ates_nominal_kW": res.get("ates_nominal_kW"),
        "HP_POWER_EL":  res.get("HP_POWER_EL"),
        "ratio_ATES_HP": res.get("ratio_ATES_HP"),
        "system_lcoh_yang": res.get("system_lcoh_yang"),
        "geo_lcoh":     res.get("geo_lcoh"),
        "ates_lcoh":    res.get("ates_lcoh"),
        "gas_lcoh":     res.get("gas_lcoh"),
        "Reff":         res.get("Reff"),
        "injected_volume_m3":  res.get("injected_volume_m3"),
        "extracted_volume_m3": res.get("extracted_volume_m3"),
        "geo_prod_GWh":      res.get("geo_prod_GWh"),
        "geo_to_demand_GWh": res.get("geo_to_demand_GWh"),
        "ates_direct_GWh":   res.get("ates_direct_GWh"),
        "hp_GWh":       res.get("hp_GWh"),
        "hp_elec_GWh":  res.get("hp_elec_GWh"),
        "hp_mean_COP":  res.get("hp_mean_COP"),
        "hp_elec_cost_eur": res.get("hp_elec_cost_eur"),
        "hp_price_paid_eur_kwh": res.get("hp_price_paid_eur_kwh"),
        "gas_GWh":      res.get("gas_GWh"),
        "unmet_GWh":    res.get("unmet_GWh"),
        "total_CO2_t":  res.get("total_CO2_t"),
    }
    row["RES"] = _res_fraction(row)
    return row


def _optimum_table(df, tol=OPT_TOL):
    """Per (demand case, variant): the cut-off minimising the system LCOH."""
    rows = []
    for (case, variant), sub in df.groupby(["demand_case", "variant"], sort=False):
        sub = sub.dropna(subset=["system_lcoh_yang"]).sort_values("T_cutoff")
        if sub.empty:
            continue
        y = sub["system_lcoh_yang"].values * 1000.0
        x = sub["T_cutoff"].values
        i = int(np.argmin(y))
        within = x[y <= y[i] * (1.0 + tol)]
        rows.append({
            "demand case": case,
            "variant": variant,
            "optimal T_cutoff [C]": x[i],
            f"T_cutoff within {tol:.0%} of min": f"{within.min():g} - {within.max():g}",
            "min system LCOH [euro/MWh]": y[i],
            "RES at that point": float(sub["RES"].values[i]),
            "LCOH at 45 C": float(np.interp(45.0, x, y)),
            "LCOH at 65 C": float(np.interp(65.0, x, y)),
        })
    return pd.DataFrame(rows)


# ================================================================== #
#  FIGURES                                                           #
# ================================================================== #

def _draw_panel(ax, df, case_label, variants, put_legend, lcoh_ylim, res_ylim):
    """One Fig. 8 panel: LCOH dashed on the left axis, RES solid on the right."""
    ax_res = ax.twinx()
    for variant_label in variants:
        variant = CONFIG_VARIANTS[variant_label]
        sub = (df[(df["demand_case"] == case_label)
                  & (df["variant"] == variant_label)]
               .dropna(subset=["system_lcoh_yang"]).sort_values("T_cutoff"))
        if sub.empty:
            continue
        x = sub["T_cutoff"].values
        ax.plot(x, sub["system_lcoh_yang"].values * 1000.0,
                color=variant["color"], linestyle="--", lw=1.5)
        ax_res.plot(x, sub["RES"].values,
                    color=variant["color"], linestyle="-", lw=1.5)

    if lcoh_ylim is not None:
        ax.set_ylim(*lcoh_ylim)
    if res_ylim is not None:
        ax_res.set_ylim(*res_ylim)
    ax.set_xlabel("Cut-off temperature (\u00b0C)")
    ax.grid(alpha=0.3)

    if put_legend:
        handles = [Line2D([], [], color=CONFIG_VARIANTS[l]["color"], lw=1.5, label=l)
                   for l in variants]
        handles += [
            Line2D([], [], color="k", lw=1.5, linestyle="--", label="LCOH"),
            Line2D([], [], color="k", lw=1.5, linestyle="-", label="RES"),
        ]
        ax.legend(handles=handles, loc="center left", fontsize=8, framealpha=1.0)
    return ax_res


def _plot_set(df, results_dir, variants, fname, subtitle):
    """Panels (a) (b) (c) side by side, one per demand case, for one variant set."""
    cases = [c for c in DEMAND_CASES if (df["demand_case"] == c).any()]
    if not cases:
        return
    paper_set = fname == "cutoff_fig8"
    lcoh_ylim = FIG8_LCOH_YLIM if paper_set else FIG8_HP_LCOH_YLIM
    res_ylim = FIG8_RES_YLIM if paper_set else FIG8_HP_RES_YLIM

    fig, axes = plt.subplots(1, len(cases), figsize=(4.8 * len(cases), 4.4),
                             squeeze=False)
    for i, case_label in enumerate(cases):
        ax_res = _draw_panel(axes[0][i], df, case_label, variants,
                             put_legend=(i == 0),
                             lcoh_ylim=lcoh_ylim, res_ylim=res_ylim)
        axes[0][i].set_title(f"({chr(97 + i)}) {case_label}   "
                             f"({DEMAND_CASES[case_label]['profile']})", fontsize=9)
        if i == 0:
            axes[0][i].set_ylabel("System LCOH (\u20ac/MWh)")
        if i == len(cases) - 1:
            ax_res.set_ylabel("Renewable Energy Share (RES)")
    fig.suptitle("Impact of the cut-off temperature on RES and system LCOH  --  "
                 + subtitle, fontsize=10)
    fig.tight_layout()
    _save(fig, results_dir, fname)

    if fname in SINGLE_PANEL_SETS:
        for case_label in cases:
            f, ax = plt.subplots(figsize=(5.0, 4.4))
            axr = _draw_panel(ax, df, case_label, variants, put_legend=True,
                              lcoh_ylim=lcoh_ylim, res_ylim=res_ylim)
            ax.set_ylabel("System LCOH (\u20ac/MWh)")
            axr.set_ylabel("Renewable Energy Share (RES)")
            ax.set_title(f"{case_label}  "
                         f"({DEMAND_CASES[case_label]['profile']})", fontsize=9)
            f.tight_layout()
            slug = case_label.replace("G/D = ", "GD").replace(".", "_")
            _save(f, results_dir, f"{fname}_{slug}")


def _plot_diagnostics(df, results_dir):
    """
    What moves underneath the figures. Geo capacity and G/D are configuration-
    independent (GGA rows shown); Reff is drawn per configuration, because the
    heat pump halves the injected volume and therefore changes it.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    colors = {c: col for c, col in zip(DEMAND_CASES,
                                       ["tab:blue", "tab:orange", "tab:green"])}
    styles = {"GGA": "-", "GGAH flat": "--", "GGAH dyn": ":"}

    for case_label in DEMAND_CASES:
        sub = (df[(df["demand_case"] == case_label) & (df["variant"] == "GGA")]
               .sort_values("T_cutoff"))
        if sub.empty:
            continue
        x = sub["T_cutoff"].values
        axes[0].plot(x, sub["GEO_POWER_kW"].values / 1000.0, "o-", ms=3,
                     color=colors[case_label], label=case_label)
        axes[1].plot(x, sub["GD_achieved"].values, "o-", ms=3,
                     color=colors[case_label], label=case_label)
        for v, ls in styles.items():
            s = (df[(df["demand_case"] == case_label) & (df["variant"] == v)]
                 .sort_values("T_cutoff"))
            if s.empty:
                continue
            axes[2].plot(s["T_cutoff"], s["Reff"], ls, marker="o", ms=3,
                         color=colors[case_label])

    axes[0].set_ylabel("Geothermal capacity (MW)")
    axes[1].set_ylabel("Achieved G/D (-)")
    axes[2].set_ylabel("Recovery efficiency Reff (-)")
    for ax in axes:
        ax.set_xlabel("Cut-off temperature (\u00b0C)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    reff_handles = [Line2D([], [], color=colors[c], lw=1.5, label=c)
                    for c in DEMAND_CASES]
    reff_handles += [Line2D([], [], color="0.3", lw=1.5, linestyle=ls, label=v)
                     for v, ls in styles.items()]
    axes[2].legend(handles=reff_handles, fontsize=7, ncol=2)
    fig.suptitle(f"What moves underneath the cut-off figures   "
                 f"(geothermal sized by {GEO_SIZING_MODE}; Reff per configuration)",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, results_dir, "cutoff_diagnostics")


def _plot_hp_diagnostics(df, results_dir):
    """Heat-pump behaviour across the cut-off sweep, both pricing variants."""
    hp = df[df["config"] == "GGAH"]
    if hp.empty:
        return
    colors = {c: col for c, col in zip(DEMAND_CASES,
                                       ["tab:blue", "tab:orange", "tab:green"])}
    styles = {"GGAH flat": "-", "GGAH dyn": "--"}
    panels = [("hp_GWh", "HP heat delivered (GWh/yr)"),
              ("hp_elec_GWh", "HP electricity (GWh/yr)"),
              ("hp_mean_COP", "Mean COP while running (-)"),
              ("hp_price_paid_eur_kwh", "Mean electricity price paid (\u20ac/kWh)")]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.0))
    for ax, (col, label) in zip(axes, panels):
        for case_label in DEMAND_CASES:
            for v, ls in styles.items():
                s = (hp[(hp["demand_case"] == case_label) & (hp["variant"] == v)]
                     .sort_values("T_cutoff"))
                if s.empty or col not in s:
                    continue
                ax.plot(s["T_cutoff"], s[col], ls, marker="o", ms=3,
                        color=colors[case_label],
                        label=f"{case_label} | {v}" if col == panels[0][0] else None)
        ax.set_xlabel("Cut-off temperature (\u00b0C)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Heat-pump diagnostics across the cut-off sweep   "
                 f"(HP sized by {HP_SIZING_MODE}; dynamic threshold "
                 f"spot < {HP_SPOT_THRESHOLD_EUR_MWH:.0f} \u20ac/MWh)", fontsize=10)
    fig.tight_layout()
    _save(fig, results_dir, "cutoff_diagnostics_hp")


# ================================================================== #

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, RESULTS_DIR)
    os.makedirs(results_dir, exist_ok=True)
    cache_path = os.path.join(results_dir, CACHE_CSV)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)

    variants = [(l, CONFIG_VARIANTS[l]) for l in CUTOFF_VARIANTS]
    n_runs = len(CUTOFF_SWEEP) * len(DEMAND_CASES) * len(variants)
    has_hp = any(CONFIG_VARIANTS[l]["cfg"] == "GGAH" for l in CUTOFF_VARIANTS)

    print("=" * 70)
    print("CUT-OFF TEMPERATURE SWEEP   (paper Fig. 8, plus heat-pump variants)")
    print("=" * 70)
    print(f"  cut-off points      : {len(CUTOFF_SWEEP)}  "
          f"({min(CUTOFF_SWEEP):g} .. {max(CUTOFF_SWEEP):g} C)")
    print(f"  demand cases        : {len(DEMAND_CASES)}")
    print(f"  configurations      : {', '.join(CUTOFF_VARIANTS)}")
    print(f"  DHN supply T_in     : {DHN_T_IN:g} C (fixed)")
    print(f"  ATES max_V          : {ATES_MAX_V_BASE:g} m3/h (fixed)")
    print(f"  TOTAL RUNS          : {n_runs}")
    print(f"  geothermal sizing   : {GEO_SIZING_MODE}"
          + (f" at {GEO_FLOW_M3H:g} m3/h" if GEO_SIZING_MODE == "flow"
             else f" at {GEO_POWER_KW:g} kW (capacity held constant)"))
    if GEO_SIZING_MODE == "flow":
        print("     T_cut [C] :  " + "  ".join(f"{t:6.1f}" for t in CUTOFF_SWEEP))
        print("     geo [MW]  :  " + "  ".join(
            f"{_geo_power_for_cutoff(t) / 1000:6.2f}" for t in CUTOFF_SWEEP))
    if has_hp:
        print(f"  HP sizing           : {HP_SIZING_MODE}"
              + (f" at {HP_POWER_EL:.0f} kW_el" if HP_SIZING_MODE == "fixed_kW"
                 else f", Ath/Hel = {HP_RATIO:g}"))
        print("     HP [kW_el]:  " + "  ".join(
            f"{_hp_power_for_cutoff(t):6.0f}" for t in CUTOFF_SWEEP))
        print("     Ath/Hel   :  " + "  ".join(
            f"{_ates_nominal_kW(t) / _hp_power_for_cutoff(t):6.2f}"
            for t in CUTOFF_SWEEP))
        print("     T_floor[C]:  " + "  ".join(
            f"{max(t - HP_DELTA_T_COLDSIDE, ATES_T_GROUND):6.1f}"
            for t in CUTOFF_SWEEP)
            + "   (clips at T_g only below a 35 C cut-off)")
        print(f"  dynamic dispatch    : HP on when spot < "
              f"{HP_SPOT_THRESHOLD_EUR_MWH:.1f} EUR/MWh "
              f"(= c_marg < {HP_THRESHOLD_EUR_MWH:.2f}, adder {_ADDER_EUR_MWH:.2f})")
    print(f"  cache               : {cache_path}")
    print("=" * 70)

    cache = _load_cache(cache_path)
    done = set()
    if not cache.empty:
        done = {(r["demand_case"], r["variant"], round(float(r["T_cutoff"]), 4))
                for _, r in cache.iterrows()}
        print(f"  resuming: {len(done)} of {n_runs} runs already cached\n")

    counter = 0
    for case_label, case in DEMAND_CASES.items():
        for variant_label, variant in variants:
            for t_cut in CUTOFF_SWEEP:
                counter += 1
                if (case_label, variant_label, round(float(t_cut), 4)) in done:
                    continue
                geo_power = _geo_power_for_cutoff(t_cut)
                hp_power = (_hp_power_for_cutoff(t_cut)
                            if variant["cfg"] == "GGAH" else np.nan)
                msg = (f"[{counter:>3}/{n_runs}] {case_label} | {variant_label:<10} "
                       f"| T_cut = {t_cut:4.1f} C  (geo {geo_power / 1000:.2f} MW")
                msg += (f", HP {hp_power:.0f} kW_el)" if np.isfinite(hp_power)
                        else ")")
                print(msg, flush=True)
                tag = (f"TC_{case['profile']}_{variant_label}_{t_cut:g}"
                       .replace(" ", ""))
                res = _simulate(variant, case, t_cut, geo_power, hp_power, tag)
                if res is None:
                    continue
                row = _row_from(case_label, case, variant_label, variant,
                                t_cut, geo_power, res)
                _append_row(cache_path, row)
                lcoh = row["system_lcoh_yang"]
                line = ("          -> system LCOH = "
                        + (f"{lcoh * 1000:.1f} EUR/MWh" if pd.notna(lcoh) else "nan")
                        + f" | RES = {row['RES']:.3f}"
                        + f" | G/D = {row['GD_achieved']:.2f}")
                if variant["cfg"] == "GGAH" and pd.notna(row["hp_mean_COP"]):
                    line += (f" | HP {row['hp_GWh']:.2f} GWh"
                             f" | COP {row['hp_mean_COP']:.2f}")
                print(line, flush=True)

    df = _load_cache(cache_path)
    if df.empty:
        print("\nNo results -- nothing to plot.")
        return df

    df["_case_order"] = df["demand_case"].map({c: i for i, c in enumerate(DEMAND_CASES)})
    df["_var_order"] = df["variant"].map({l: i for i, l in enumerate(CUTOFF_VARIANTS)})
    df = (df.dropna(subset=["_case_order", "_var_order"])
            .sort_values(["_case_order", "_var_order", "T_cutoff"])
            .reset_index(drop=True))

    print(f"\nCollected {len(df)} runs")
    print("\nAchieved G/D at each cut-off (GGA rows) -- it moves because the "
          "doublet capacity does:")
    piv = df[df["variant"] == "GGA"].pivot_table(
        index="T_cutoff", columns="demand_case", values="GD_achieved")
    print(piv.to_string(float_format=lambda v: f"{v:.2f}"))

    for variants_set, fname, subtitle in PLOT_SETS:
        if not df["variant"].isin(variants_set).any():
            continue
        _plot_set(df, results_dir, variants_set, fname, subtitle)
    _plot_diagnostics(df, results_dir)
    _plot_hp_diagnostics(df, results_dir)

    wide_lcoh = df.pivot_table(index="T_cutoff", columns=["demand_case", "variant"],
                               values="system_lcoh_yang") * 1000.0
    wide_lcoh.columns = [f"{c} | {v} [euro/MWh]" for c, v in wide_lcoh.columns]
    wide_res = df.pivot_table(index="T_cutoff", columns=["demand_case", "variant"],
                              values="RES")
    wide_res.columns = [f"{c} | {v} [-]" for c, v in wide_res.columns]

    opt = _optimum_table(df)
    if not opt.empty:
        print("\nOPTIMAL CUT-OFF PER CURVE (minimum of the system LCOH curve)")
        print(opt.to_string(index=False))
        print("  Paper 5.3.3: for G/D = 1.62 and 1.07 the GGA optimum is NOT at the")
        print("  lowest cut-off -- RES reaches 99% and lowering it further only adds")
        print("  storage operating cost without delivering more heat. The effect is")
        print("  most visible in the G/D = 0.65 case, where both the doublet and the")
        print("  HT-ATES produce much more heat at a low cut-off.")

    out = os.path.join(results_dir, "cutoff_summary.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.drop(columns=["_case_order", "_var_order"]).to_excel(
            writer, sheet_name="All runs", index=False)
        wide_lcoh.reset_index().to_excel(writer, sheet_name="System LCOH", index=False)
        wide_res.reset_index().to_excel(writer, sheet_name="RES", index=False)
        if not opt.empty:
            opt.to_excel(writer, sheet_name="Optimum", index=False)
    print(f"\nSaved -> {out}")
    return df


if __name__ == "__main__":
    main()