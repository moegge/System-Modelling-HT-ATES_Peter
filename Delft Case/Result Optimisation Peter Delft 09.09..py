# -*- coding: utf-8 -*-
"""
Result Optimisation Peter Delft 09.09..py
==================================================================
Optimisation sweep over the ATES / HP sizing space, for the three DELFT demand
cases of David Geerts' paper. Converted from the 26.08 optimisation file, which
swept the Amsterdam profile with a resized doublet.

WHAT CHANGED VS THE 26.08 VERSION
---------------------------------
The old file used GD_MODE = 'vary_geo': ONE demand profile (Amsterdam, ~50 GWh)
with the doublet back-solved per target G/D, giving roughly 9.2 / 6.1 / 3.7 MW.
This file uses the 'vary_demand' construction of the other scripts instead:

  * the doublet is FIXED at 7.4 MW (paper 4.1.1: 320 m3/h of 75 C water into a
    55 C return), the same value the cut-off and reference scripts use;
  * the three DELFT profiles are the demand cases, and G/D falls out of them at
    1.63 / 1.07 / 0.65 rather than being targeted;
  * network length is set per demand case, so the system LCOH is comparable to
    the Fig. 4 / Fig. 6 / Fig. 7 reproductions.

The consequence that matters: the optima this file finds are now transferable to
the cut-off script and the sweep scripts, because "G/D = 1.07" means the same
thing everywhere -- the Delft City profile against a 7.4 MW doublet. Under the
old construction it meant an Amsterdam profile against a 6.1 MW doublet, and an
optimal max_V found there had no claim on this system.

WHAT IS OPTIMISED
-----------------
The geothermal plant is FIXED. Two knobs are swept:
    ATES_MAXV_GRID  -- the ATES size, max_V [m3/h]
    RATIO_GRID      -- the sizing ratio Ath/Hel [-]
The HP rating is DERIVED, never given:
    ates_nominal_kW = (max_V/3600) * rho*cp * (GEO_T_OUT - DEMAND_T_OUT)
    HP_POWER_EL     = ates_nominal_kW / ratio
A fixed HP rating in kW would mean a different ATES/HP balance at every max_V,
which is exactly the confound this file exists to avoid.

The run grid is dimension-aware, so nothing runs more often than it varies:
    G, GG  -> no ATES, no HP  -> one run per demand case
    GGA    -> ATES, no HP     -> one run per demand case x max_V
    GGAH   -> ATES + HP       -> one run per demand case x max_V x ratio

System LCOH is David's LCOE_calc_Yang figure (pooled discounted cost / pooled
discounted heat over a COMMON 60-year horizon with reinvestment).

NOTE ON THE CAC CONVENTION
--------------------------
Every configuration is referenced to the gas-only baseline G within its demand
case, so the CAC here is the CUMULATIVE abatement cost versus fossil, not the
marginal cost of adding a component. That is forced: GGA is multi-row (one per
max_V) and cannot serve as a unique reference. The Fig. 5(b) reproduction uses
the marginal convention instead, so do not put the two side by side without
saying which is which.

NOT INCLUDED
------------
Dynamic electricity pricing. Every GGAH run here uses the flat elec_price. Adding
it would double the GGAH grid; better to add it selectively at the winning max_V
once the optimum is known.

Put this file in the SAME folder as model_driver.py, main2_Peter.py,
ATES_obj_Peter.py and the data files, then:

    python "Test File Optimisation Peter Delft.py"
==================================================================
"""

import os
import sys

# model_driver.py, main2_Peter.py and ATES_obj_Peter.py live one level up, in
# System-Modelling-HT-ATES_Peter; this file sits in the "Delft Case" subfolder.
_MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODEL_ROOT not in sys.path:
    sys.path.insert(0, _MODEL_ROOT)

import matplotlib
matplotlib.use("Agg")          # headless: no windows even if something calls show()
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from model_driver import (run_case, DEMAND_T_IN, DEMAND_T_OUT,
                          GEO_T_OUT, HP_DELTA_T_COLDSIDE,
                          ATES_T_GROUND)
from main2_Peter import demand_class

# ================================================================== #
#  DEFINE THE SWEEP GRID HERE                                        #
# ================================================================== #

# --- The three demand cases -------------------------------------------------
# Keyed by the BASE G/D (the ratio each profile has against the 7.4 MW doublet),
# which keeps the labels of the other scripts. Network length is a property of
# the demand case. 8 km / 15 km are from paper 4.2; 23 km for "Delft Total" is
# an assumption (= TU + City), not stated in the paper.
DEMAND_CASES = {
    1.62: dict(profile="TU Delft",    network_m=8000.0),
    1.07: dict(profile="Delft City",  network_m=15000.0),
    0.65: dict(profile="Delft Total", network_m=23000.0),
}
GD_GRID = list(DEMAND_CASES)     # ordering key, used throughout as GD_target

# --- The geothermal plant: FIXED, not optimised -----------------------------
FIXED_GEO_POWER_MW = 7.4         # [MW] paper 4.1.1 (320 m3/h of 75 C vs 55 C)

# The four configurations, in David's left-to-right order (+ GGAH).
# G and GG carry no ATES/HP and are only needed as CAC references; drop them
# from the list if you only want the storage configurations.
CONFIG_GRID = ["G", "GG", "GGA", "GGAH"]

# --- KNOB 1: ATES size ------------------------------------------------------
# max_V [m3/h], the aquifer flow rating. Drives both the ATES CAPEX
# (costperm3 * max_V) and, through ates_nominal_kW, the derived HP rating.
# The paper's own max_V sensitivity (4.3) spans 30-350 m3/h and its base case is
# 320; 15 is below that range and is kept only as a lower anchor.
ATES_MAXV_GRID = [15, 30, 70, 120, 160, 320]          # [m3/h]

# --- KNOB 2: ATES/HP sizing ratio ------------------------------------------
# ratio = ates_nominal_kW / HP_POWER_EL  [kW_th / kW_el].
# HIGH ratio -> small HP relative to the well; LOW ratio -> big HP.
# At max_V = 320 (ates_nominal 7431 kW): r=1 -> 7431 kW_el, r=6 -> 1239 kW_el.
# The 25.08 sweep used HP 3000 kW_el at max_V 320, i.e. r = 2.48.
RATIO_GRID = [2, 3, 3.5, 4, 4.5, 5, 6, 8]       # [-]

HOURS_PER_YEAR = 8760

# Water properties used for ates_nominal_kW. MUST match the RHO_CP constant in
# Test_File_Peter_17_08.run_case(), otherwise the achieved ratio drifts away
# from the target (the code below checks this per run and warns).
RHO_CP = 4180.0                  # [kJ/m3.K]

# --- Output folder + figure toggles ----------------------------------------
RESULTS_DIR      = "results optimisation delft"
SAVE_FIGURES     = True          # True -> save the graphs into RESULTS_DIR
MAKE_BAR_FIGURES = True          # False -> only the LCOH/CAC line figures
                                 #          (the bar figures get very wide here)

# Reduced ("best only") second version of every bar figure: GGA + the single
# best GGAH combination per demand case, instead of the full grid. Files get an
# extra '_bestonly' suffix.
MAKE_BEST_ONLY_FIGURES = True
# Keep the G and GG baselines in the reduced figures. False -> storage configs
# only, which makes the CI and CAC charts hard to read (no fossil reference).
BEST_ONLY_INCLUDE_G_GG = True
# Which column the optimum minimises. Also drives the 'Optimum' sheet.
# "system_lcoh_yang" or "CAC_eur_per_kg" (both are minimised).
OPT_COLUMN = "system_lcoh_yang"

# --- Per-run outputs: the two system_plot figures + full workbook ----------
RUN_MAKE_PLOTS  = False          # True -> each run makes its two system_plot figs
RUN_WRITE_EXCEL = False          # True -> each run writes its full workbook

# ================================================================== #


def _annual_demand_kWh(profile):
    """Annual DH demand [kWh] of one profile, for the banner and the G/D check."""
    dem = demand_class(T_in=DEMAND_T_IN, T_out=DEMAND_T_OUT,
                       example_demand=profile)
    return float(np.sum(dem.data))


def _ates_nominal_kW(max_V):
    """
    Peak direct-HX power of a freshly charged well [kW]:
        flow x rho*cp x (T_geo_supply - T_DHN_return)
    Mirrors the identical expression inside run_case(); keep the two in sync.
    """
    return (max_V / 3600.0) * RHO_CP * (GEO_T_OUT - DEMAND_T_OUT)


def _hp_power_for_ratio(max_V, ratio):
    """HP electrical rating [kW_el] that gives the target Ath/Hel ratio."""
    return _ates_nominal_kW(max_V) / ratio


def _build_run_grid():
    """
    Dimension-aware run list: each config only varies over the knobs it has.
    Returns (base_gd, cfg, max_V, ratio) with NaN where a knob does not apply.
    """
    grid = []
    for base_gd in GD_GRID:
        for cfg in CONFIG_GRID:
            if cfg in ("G", "GG"):                      # no ATES, no HP
                combos = [(np.nan, np.nan)]
            elif cfg == "GGA":                          # ATES, no HP
                combos = [(v, np.nan) for v in ATES_MAXV_GRID]
            elif cfg == "GGAH":                         # ATES + HP
                combos = [(v, r) for v in ATES_MAXV_GRID for r in RATIO_GRID]
            else:
                raise ValueError(f"Unknown config {cfg!r}")
            for max_V, ratio in combos:
                grid.append((base_gd, cfg, max_V, ratio))
    return grid


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, RESULTS_DIR)
    os.makedirs(results_dir, exist_ok=True)

    geo_power = FIXED_GEO_POWER_MW * 1000.0
    run_grid = _build_run_grid()

    print("=" * 70)
    print("OPTIMISATION SWEEP -- Delft demand cases, fixed geothermal")
    print("=" * 70)
    print(f"  geothermal (FIXED)  : {FIXED_GEO_POWER_MW} MW -> "
          f"{geo_power * HOURS_PER_YEAR / 1e6:.1f} GWh/yr")
    for base_gd, case in DEMAND_CASES.items():
        a = _annual_demand_kWh(case["profile"])
        print(f"  G/D = {base_gd:<5.2f} {case['profile']:<12} "
              f"{a / 1e6:7.2f} GWh/yr, {a / 1e6 / 8.760:6.2f} MW avg, "
              f"network {case['network_m'] / 1000:.0f} km "
              f"-> G/D = {geo_power * HOURS_PER_YEAR / a:.2f}")
    print(f"  ATES max_V grid     : {ATES_MAXV_GRID} m3/h")
    print(f"                        -> Ath "
          f"{[round(_ates_nominal_kW(v)) for v in ATES_MAXV_GRID]} kW")
    print(f"  Ath/Hel grid        : {RATIO_GRID}")
    print(f"  optimised on        : ATES max_V x Ath/Hel  (geothermal is fixed)")
    print(f"  TOTAL RUNS          : {len(run_grid)}")
    print("=" * 70)

    results = []

    for base_gd, cfg, max_V, ratio in run_grid:
        case = DEMAND_CASES[base_gd]

        # Build the tag from the knobs this config actually uses.
        tag = f"GD{base_gd:.2f}_{cfg}"
        if np.isfinite(max_V):
            tag += f"_V{max_V:g}"
        if np.isfinite(ratio):
            tag += f"_r{ratio:g}"

        # Derive the HP rating from the ATES size and the target ratio.
        hp_power = _hp_power_for_ratio(max_V, ratio) if np.isfinite(ratio) else np.nan

        print("\n" + "#" * 70)
        msg = (f"# RUN: {tag}   (GEO = {geo_power / 1000:.2f} MW fixed, "
               f"demand = {case['profile']}")
        if np.isfinite(max_V):
            msg += f", max_V = {max_V:g} m3/h -> Ath = {_ates_nominal_kW(max_V):.0f} kW"
        if np.isfinite(ratio):
            msg += f", r = {ratio:g} -> HP = {hp_power:.0f} kW_el"
        print(msg + ")")
        print("#" * 70)

        run_outfile = os.path.join(results_dir, f"timeseries_{tag}.xlsx")
        run_kwargs = dict(
            CONFIG=cfg,
            GEO_POWER=geo_power,
            DEMAND_EXAMPLE=case["profile"],
            NETWORK_LENGTH_M=case["network_m"],
            tag=tag,
            OUTFILE=run_outfile,
            make_plots=RUN_MAKE_PLOTS,
            write_excel=RUN_WRITE_EXCEL,
        )
        if np.isfinite(max_V):
            run_kwargs["ATES_MAX_V"] = float(max_V)
        if np.isfinite(hp_power):
            run_kwargs["HP_POWER_EL"] = float(hp_power)

        res = run_case(**run_kwargs)

        # Record the TARGETS too (the achieved values are in res["GD_ratio"] /
        # res["ATES_MAX_V"] / res["ratio_ATES_HP"]).
        res["GD_target"]    = base_gd          # the BASE label, not a target now
        res["demand_case"]  = f"G/D = {base_gd:.2f}"
        res["profile"]      = case["profile"]
        res["network_m"]    = case["network_m"]
        res["MAXV_target"]  = max_V
        res["ratio_target"] = ratio

        # Guard: the ratio run_case() reports back must equal the one we targeted.
        # A mismatch means RHO_CP / GEO_T_OUT / DEMAND_T_OUT drifted apart between
        # this file and run_case(), which would silently mislabel every GGAH bar.
        if np.isfinite(ratio):
            achieved = res.get("ratio_ATES_HP", np.nan)
            if not np.isfinite(achieved) or abs(achieved - ratio) > 1e-6:
                print(f"  WARNING: target Ath/Hel = {ratio:g} but run_case() reports "
                      f"{achieved} -- check RHO_CP / GEO_T_OUT / DEMAND_T_OUT.")

        results.append(res)

        # Save any per-run system_plot figures into results_dir (local numbering).
        open_nums = plt.get_fignums()
        if SAVE_FIGURES and RUN_MAKE_PLOTS:
            for local_i, num in enumerate(open_nums, start=1):
                fig = plt.figure(num)
                fig.savefig(os.path.join(results_dir, f"{tag}_fig{local_i}.png"),
                            dpi=150, bbox_inches="tight")
        for num in open_nums:
            plt.close(plt.figure(num))

    # --- Collect into a summary table -----------------------------------------
    summary_cols = [
        "tag", "demand_case", "profile", "GD_target", "GD_ratio", "config",
        "USE_HP", "GEO_POWER", "network_m",
        "MAXV_target", "ATES_MAX_V", "ratio_target", "ratio_ATES_HP",
        "HP_POWER_EL", "ates_nominal_kW",
        "Reff", "injected_volume_m3", "extracted_volume_m3",
        "demand_GWh", "geo_prod_GWh", "geo_to_demand_GWh", "geo_GWh",
        "ates_direct_GWh", "hp_GWh", "gas_GWh", "unmet_GWh",
        "system_lcoh_yang", "geo_lcoh", "ates_lcoh", "gas_lcoh",
        "hp_elec_GWh", "hp_elec_cost_eur", "hp_mean_COP", "hp_capex_Meur",
        "total_CO2_t", "total_CO2_cost_eur",
    ]
    df = pd.DataFrame([{k: r.get(k) for k in summary_cols} for r in results])
    if df["system_lcoh_yang"].isna().all():
        raise RuntimeError(
            "run_case() did not return 'system_lcoh_yang' -- add the LCOE_calc_Yang "
            "call and the return-dict entry in model_driver.py.")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    print("\n" + "=" * 70)
    print("OPTIMISATION SWEEP SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))

    # Sanity: the achieved G/D should equal the base label, since the doublet is
    # fixed and the demand profile is what defines the case.
    chk = df.groupby("GD_target")["GD_ratio"].first()
    print("\nAchieved G/D per demand case (should match the labels 1.62/1.07/0.65):")
    print(chk.to_string(float_format=lambda v: f"{v:.3f}"))

    out = os.path.join(results_dir, "optimisation_summary_delft.xlsx")
    fig_data_sheets = {}  # sheet name -> DataFrame, filled in the FIGURES section

    # ============================================================== #
    #  FIGURES                                                        #
    # ============================================================== #

    def _save(fig, name):
        if SAVE_FIGURES:
            path = os.path.join(results_dir, name + ".png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  saved figure -> {path}")
        plt.close(fig)

    # Stable ordering: demand case left->right, configs in CONFIG_GRID order
    # within each case, then max_V, then the ratio.
    df["_gd_order"]  = df["GD_target"].map({gd: i for i, gd in enumerate(GD_GRID)})
    df["_cfg_order"] = df["config"].map({c: i for i, c in enumerate(CONFIG_GRID)})
    df["_v_order"]   = df["MAXV_target"].fillna(-1)     # configs without ATES first
    df["_r_order"]   = df["ratio_target"].fillna(-1)    # configs without HP first
    df = df.sort_values(["_gd_order", "_cfg_order", "_v_order",
                         "_r_order"]).reset_index(drop=True)

    def _xlabel(row):
        base = f"{row['config']}\nG/D={row['GD_target']:.2f}"
        if np.isfinite(row.get("MAXV_target", np.nan)):
            base += f"\nV={row['MAXV_target']:g}"
        if np.isfinite(row.get("ratio_target", np.nan)):
            base += f"\nr={row['ratio_target']:g}"
        return base

    gas  = df["gas_lcoh"].values * 1000.0
    geo  = df["geo_lcoh"].values * 1000.0
    ates = df["ates_lcoh"].values * 1000.0
    syst = df["system_lcoh_yang"].values * 1000.0     # David's LCOE_calc_Yang

    fig_data_sheets["Fig LCOH data"] = pd.DataFrame({
        "tag": df["tag"].values,
        "demand_case": df["demand_case"].values,
        "config": df["config"].values,
        "GD_target": df["GD_target"].values,
        "MAXV_target": df["MAXV_target"].values,
        "ratio_target": df["ratio_target"].values,
        "HP_POWER_EL": df["HP_POWER_EL"].values,
        "Gas [euro/MWh]": gas,
        "Geo [euro/MWh]": geo,
        "ATES [euro/MWh]": ates,
        "System [euro/MWh]": syst,
    })

    # Exhaustive stacked share of DEMAND, summing to ~1.0. Geo uses
    # geo_to_demand_GWh (charging removed) so stored geo is not double-counted
    # here and again in the ATES segment. P_el is grid electricity, so it is
    # excluded from the renewable total.
    dem_arr   = df["demand_GWh"].values
    geo_f     = np.nan_to_num(df["geo_to_demand_GWh"].values) / dem_arr
    ates_f    = np.nan_to_num(df["ates_direct_GWh"].values) / dem_arr
    hp_el_f   = np.nan_to_num(df["hp_elec_GWh"].values) / dem_arr          # P_el
    hp_evap_f = np.nan_to_num(df["hp_GWh"].values) / dem_arr - hp_el_f     # Q_evap
    gas_f     = np.nan_to_num(df["gas_GWh"].values) / dem_arr
    res_total = geo_f + ates_f + hp_evap_f

    fig_data_sheets["Fig RES data"] = pd.DataFrame({
        "tag": df["tag"].values,
        "demand_case": df["demand_case"].values,
        "config": df["config"].values,
        "MAXV_target": df["MAXV_target"].values,
        "ratio_target": df["ratio_target"].values,
        "Geo frac": geo_f,
        "ATES direct frac": ates_f,
        "HP source heat frac": hp_evap_f,
        "HP electricity frac": hp_el_f,
        "Gas frac": gas_f,
        "Total renewable frac (excl. HP elec)": res_total,
        "Balance check": res_total + hp_el_f + gas_f,
    })

    # --- Carbon Abatement Cost (CAC) ------------------------------------------
    # CAC = (LCOH_config - LCOH_ref) / (CI_ref - CI_config), in euro/kgCO2.
    # Referenced to the gas-only baseline G WITHIN the same demand case, so this
    # is the CUMULATIVE abatement cost vs fossil, not the marginal cost of each
    # added component. Forced here: GGA is multi-row (one per max_V) and cannot
    # serve as a unique reference. See the header note.
    CAC_REF      = {"GG": "G", "GGA": "G", "GGAH": "G"}
    CONFIG_COLOR = {"GG": "tab:blue", "GGA": "tab:orange", "GGAH": "tab:green"}

    with np.errstate(divide="ignore", invalid="ignore"):
        df["CI_kg_per_kWh"] = (df["total_CO2_t"] * 1000.0) / (df["demand_GWh"] * 1e6)

    ref_index = {(r["GD_target"], r["config"]): r for _, r in df.iterrows()
                 if r["config"] in ("G", "GG")}

    cac_vals      = np.full(len(df), np.nan)
    lcoh_ref_vals = np.full(len(df), np.nan)
    ci_ref_vals   = np.full(len(df), np.nan)
    for pos, (_, row) in enumerate(df.iterrows()):
        ref_cfg = CAC_REF.get(row["config"])
        if ref_cfg is None:
            continue
        ref = ref_index.get((row["GD_target"], ref_cfg))
        if ref is None:
            print(f"  WARNING: no unique reference '{ref_cfg}' for {row['tag']}"
                  f" -> CAC = NaN")
            continue
        lcoh_ref_vals[pos] = ref["system_lcoh_yang"]
        ci_ref_vals[pos]   = ref["CI_kg_per_kWh"]
        d_lcoh = row["system_lcoh_yang"] - ref["system_lcoh_yang"]
        d_ci   = ref["CI_kg_per_kWh"] - row["CI_kg_per_kWh"]   # + = abatement
        cac_vals[pos] = d_lcoh / d_ci if abs(d_ci) > 1e-9 else np.nan

    df["CAC_eur_per_kg"] = cac_vals

    from matplotlib.patches import Patch

    fig_data_sheets["Fig CAC data"] = pd.DataFrame({
        "tag": df["tag"].values,
        "demand_case": df["demand_case"].values,
        "config": df["config"].values,
        "MAXV_target": df["MAXV_target"].values,
        "ratio_target": df["ratio_target"].values,
        "reference config": [CAC_REF.get(c, "") for c in df["config"].values],
        "LCOH [euro/kWh]": df["system_lcoh_yang"].values,
        "LCOH_ref [euro/kWh]": lcoh_ref_vals,
        "CI [kgCO2/kWh]": df["CI_kg_per_kWh"].values,
        "CI_ref [kgCO2/kWh]": ci_ref_vals,
        "delta_LCOH [euro/kWh]": df["system_lcoh_yang"].values - lcoh_ref_vals,
        "delta_CI [kgCO2/kWh]": ci_ref_vals - df["CI_kg_per_kWh"].values,
        "CAC [euro/kgCO2]": cac_vals,
    })

    CI_COLOR = {"G": "tab:red", "GG": "tab:blue",
                "GGA": "tab:orange", "GGAH": "tab:green"}
    ci_g = df["CI_kg_per_kWh"].values * 1000.0      # gCO2/kWh

    fig_data_sheets["Fig CI data"] = pd.DataFrame({
        "tag": df["tag"].values,
        "demand_case": df["demand_case"].values,
        "config": df["config"].values,
        "MAXV_target": df["MAXV_target"].values,
        "ratio_target": df["ratio_target"].values,
        "total_CO2_t": df["total_CO2_t"].values,
        "demand_GWh": df["demand_GWh"].values,
        "CI [kgCO2/kWh]": df["CI_kg_per_kWh"].values,
        "CI [gCO2/kWh]": ci_g,
    })

    # ============================================================== #
    #  BAR FIGURES  (drawn twice: full grid, then GGA + best GGAH)   #
    # ============================================================== #

    def _bar_figures(dsub, fname_suffix="", title_suffix=""):
        """
        The four bar/scatter figures for whatever subset of df is passed in.
        Everything is recomputed from dsub, so the reduced version is not a
        cropped copy of the full one -- same code, fewer rows.
        Requires CI_kg_per_kWh and CAC_eur_per_kg to be on dsub already.
        """
        if dsub.empty:
            return
        xs     = np.arange(len(dsub))
        labels = [_xlabel(r) for _, r in dsub.iterrows()]
        gdo    = dsub["_gd_order"].values
        width  = max(8, 0.9 * len(dsub))

        def _separators(ax):
            for i in range(1, len(dsub)):
                if gdo[i] != gdo[i - 1]:
                    ax.axvline(i - 0.5, color="0.8", linewidth=1, zorder=1)

        def _value_labels(ax, vals, fmt):
            finite = np.isfinite(vals)
            span = (np.nanmax(vals[finite]) - np.nanmin(vals[finite])) if finite.any() else 1.0
            pad = 0.02 * (span if span > 0 else 1.0)
            for xi, v in zip(xs, vals):
                if np.isfinite(v):
                    ax.text(xi, v + (pad if v >= 0 else -pad), format(v, fmt),
                            ha="center", va=("bottom" if v >= 0 else "top"),
                            fontsize=7)
            return finite

        # --- LCOH per combination (David Fig. 4 style) ------------------------
        g_ = dsub["gas_lcoh"].values * 1000.0
        e_ = dsub["geo_lcoh"].values * 1000.0
        a_ = dsub["ates_lcoh"].values * 1000.0
        s_ = dsub["system_lcoh_yang"].values * 1000.0

        fig, ax = plt.subplots(figsize=(width, 5))
        ax.scatter(xs, g_, color="tab:red",    label="Gas",  zorder=3)
        ax.scatter(xs, e_, color="tab:blue",   label="Geo",  zorder=3)
        ax.scatter(xs, a_, color="tab:orange", label="ATES", zorder=3)
        dash = 0.3
        for xi, s in zip(xs, s_):
            if np.isfinite(s):
                ax.hlines(s, xi - dash, xi + dash, color="tab:cyan", linewidth=2,
                          zorder=2, label="System (Yang)" if xi == 0 else None)
        _separators(ax)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("LCOH (\u20ac/MWh)")
        ax.set_title("LCOH per combination  (system = LCOE_calc_Yang, 60-yr horizon)"
                     + title_suffix)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        _save(fig, "lcoh_per_combination" + fname_suffix)

        # --- Cost split: additive decomposition of the system LCOH ------------
        # Segment = discounted_cost_i / pooled discounted heat, so the stack
        # height IS the system LCOH (black dash = the same number, as a check).
        SEG_ORDER = [
            ("Geothermal",             "#1f6fb4", ""),
            ("CO2 - geothermal",       "#9dc6e6", "xxx"),
            ("ATES well",              "#e8871a", ""),
            ("HP capex",               "#15703f", ""),
            ("HP opex (elec + fixed)", "#63c98d", "//"),
            ("CO2 - HP electricity",   "#b9e3ca", "xxx"),
            ("Gas boiler",             "#b3181f", ""),
            ("CO2 - gas",              "#f0a0a0", "xxx"),
            ("Network",                "#8c8c8c", ".."),
        ]
        plt.rcParams["hatch.linewidth"] = 0.6
        splits = [(res_by_tag.get(t) or {}).get("cost_split") or {}
                  for t in dsub["tag"]]

        if any(splits):
            fig, ax = plt.subplots(figsize=(width, 5.4))
            bot = np.zeros(len(dsub))
            for nm, colr, htch in SEG_ORDER:
                v = np.array([s.get(nm, 0.0) for s in splits], dtype=float)
                if not np.any(np.abs(v) > 1e-9):
                    continue
                ax.bar(xs, v, bottom=bot, color=colr, label=nm, hatch=htch,
                       edgecolor="white", linewidth=0.5)
                bot = bot + v
            for xi, s in zip(xs, s_):
                if np.isfinite(s):
                    ax.hlines(s, xi - 0.4, xi + 0.4, color="k", lw=1.4, zorder=4)
            _separators(ax)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_ylabel("Contribution to system LCOH (\u20ac/MWh)")
            ax.set_title("Cost split of the system LCOH  "
                         "(black dash = LCOE_calc_Yang)" + title_suffix)
            ax.legend(fontsize=7, ncol=3)
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            fig.tight_layout()
            _save(fig, "cost_split" + fname_suffix)

        # --- Same split as pies (shares). Skipped for the full grid: 147 pies
        #     is not a figure. Runs for the '_bestonly' subset.
        if any(splits) and len(dsub) <= 16:
            ncol = min(4, len(dsub))
            nrow = int(np.ceil(len(dsub) / ncol))
            figp, axp = plt.subplots(nrow, ncol,
                                     figsize=(4.1 * ncol, 4.9 * nrow),
                                     squeeze=False)
            for ax_, s, lbl, sysv in zip(axp.ravel(), splits, labels, s_):
                keep = [(n, c, h) for n, c, h in SEG_ORDER if s.get(n, 0.0) > 1e-9]
                if not keep:
                    ax_.axis("off")
                    continue
                wedges, _, autotexts = ax_.pie(
                    [s[n] for n, _, _ in keep],
                    colors=[c for _, c, _ in keep],
                    startangle=90, counterclock=False, radius=0.92,
                    autopct=lambda p: f"{p:.0f}%" if p >= 3 else "",
                    pctdistance=0.72,
                    wedgeprops=dict(edgecolor="white", linewidth=0.8))
                for w, (_, _, h) in zip(wedges, keep):
                    w.set_hatch(h)
                # Black on a translucent white plate: legible on every fill,
                # which white or black text alone is not.
                for t in autotexts:
                    t.set_fontsize(7.5)
                    t.set_color("black")
                    t.set_bbox(dict(facecolor="white", alpha=0.75,
                                    edgecolor="none", boxstyle="round,pad=0.12"))
                ax_.set_title(lbl.replace("\n", "  ") + f"\n{sysv:.1f} \u20ac/MWh",
                              fontsize=8.5, pad=10)
            for ax_ in axp.ravel()[len(dsub):]:
                ax_.axis("off")
            figp.legend(handles=[Patch(facecolor=c, hatch=h, edgecolor="white",
                                       label=n) for n, c, h in SEG_ORDER],
                        loc="lower center", ncol=3, fontsize=8.5,
                        frameon=False, handlelength=2.4, handleheight=1.4)
            figp.suptitle("Cost split shares of the system LCOH  "
                          "(absolute level in each title)" + title_suffix,
                          fontsize=11, y=0.985)
            # Explicit margins: tight_layout alone lets the suptitle and the
            # row titles collide once the titles are two lines.
            figp.subplots_adjust(top=0.90, bottom=0.13, hspace=0.38, wspace=0.06)
            _save(figp, "cost_split_pies" + fname_suffix)

        # --- Energy share per component --------------------------------------
        dem_  = dsub["demand_GWh"].values
        gf    = np.nan_to_num(dsub["geo_to_demand_GWh"].values) / dem_
        af    = np.nan_to_num(dsub["ates_direct_GWh"].values) / dem_
        hef   = np.nan_to_num(dsub["hp_elec_GWh"].values) / dem_          # P_el
        hvf   = np.nan_to_num(dsub["hp_GWh"].values) / dem_ - hef         # Q_evap
        gsf   = np.nan_to_num(dsub["gas_GWh"].values) / dem_
        rest_ = gf + af + hvf

        fig, ax = plt.subplots(figsize=(width, 5))
        ax.bar(xs, gf,  color="tab:blue",   label="Geo")
        ax.bar(xs, af,  bottom=gf,          color="tab:orange", label="ATES direct")
        b = gf + af
        ax.bar(xs, hvf, bottom=b,           color="tab:green",  label="HP source heat")
        b = b + hvf
        ax.bar(xs, hef, bottom=b,           color="tab:green",  hatch="//",
               edgecolor="white", label="HP electricity (grid)")
        b = b + hef
        ax.bar(xs, gsf, bottom=b,           color="tab:red",    label="Gas")

        def _seg_label(vals, bottoms):
            for xi, v, bo in zip(xs, vals, bottoms):
                if v > 0.015:
                    ax.text(xi, bo + v / 2.0, f"{v:.2f}",
                            ha="center", va="center", fontsize=7)
        _seg_label(gf,  np.zeros_like(gf))
        _seg_label(af,  gf)
        _seg_label(hvf, gf + af)
        _seg_label(hef, gf + af + hvf)
        _seg_label(gsf, gf + af + hvf + hef)

        for xi, r, tot in zip(xs, rest_, rest_ + hef + gsf):
            ax.text(xi, tot + 0.015, f"RES {r:.2f}", ha="center", va="bottom",
                    fontsize=7)

        _separators(ax)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("Share of demand met")
        ax.set_ylim(0, 1.12)
        ax.set_title("Energy share per component  "
                     "(HP split into source heat and grid electricity)" + title_suffix)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        _save(fig, "res_per_combination" + fname_suffix)

        # --- CAC per combination ---------------------------------------------
        cac_ = dsub["CAC_eur_per_kg"].values
        fig, ax = plt.subplots(figsize=(width, 5))
        ax.bar(xs, cac_, color=[CONFIG_COLOR.get(c, "0.5") for c in dsub["config"]])
        finite = _value_labels(ax, cac_, ".2f")
        ax.axhline(0.0, color="k", linewidth=0.8)
        _separators(ax)
        drawn = [c for c in CONFIG_GRID if c in set(dsub["config"][finite])]
        if drawn:
            ax.legend(handles=[Patch(facecolor=CONFIG_COLOR.get(c, "0.5"), label=c)
                               for c in drawn], title="Configuration")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("Carbon Abatement Cost (\u20ac/kgCO\u2082)")
        ax.set_title("CAC per combination  (reference = G, cumulative vs fossil)"
                     + title_suffix)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        _save(fig, "cac_per_combination" + fname_suffix)

        # --- Carbon intensity of delivered heat -------------------------------
        ci_ = dsub["CI_kg_per_kWh"].values * 1000.0      # gCO2/kWh
        fig, ax = plt.subplots(figsize=(width, 5))
        ax.bar(xs, ci_, color=[CI_COLOR.get(c, "0.5") for c in dsub["config"]])
        finite = _value_labels(ax, ci_, ".1f")
        ax.axhline(0.0, color="k", linewidth=0.8)
        _separators(ax)
        drawn = [c for c in CONFIG_GRID if c in set(dsub["config"][finite])]
        if drawn:
            ax.legend(handles=[Patch(facecolor=CI_COLOR.get(c, "0.5"), label=c)
                               for c in drawn], title="Configuration")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("Carbon intensity of delivered heat (gCO\u2082/kWh)")
        ax.set_title("Emissions per kWh delivered" + title_suffix)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        _save(fig, "ci_per_combination" + fname_suffix)

    res_by_tag = {r.get("tag"): r for r in results}

    if MAKE_BAR_FIGURES:
        _bar_figures(df)

    # --- Reduced: GGA + the best GGAH combination per demand case -------------
    # best_idx also feeds the 'Optimum' table below, so "best" is defined in
    # exactly one place. The GGA kept for each case is the one with the SAME
    # max_V as the winning GGAH -> like-for-like, same well, with and without HP.
    best_idx = []
    for gd in GD_GRID:
        sub = df[(df["GD_target"] == gd) & (df["config"] == "GGAH")]
        if sub.empty or sub[OPT_COLUMN].isna().all():
            print(f"  WARNING: no GGAH run with a finite {OPT_COLUMN} "
                  f"at G/D = {gd:.2f}")
            continue
        best_idx.append(sub[OPT_COLUMN].idxmin())

    if MAKE_BEST_ONLY_FIGURES and best_idx:
        keep = list(best_idx)
        for i in best_idx:
            gd, v = df.at[i, "GD_target"], df.at[i, "MAXV_target"]
            gga = df[(df["config"] == "GGA") & (df["GD_target"] == gd)
                     & (df["MAXV_target"] == v)]
            keep.extend(gga.index.tolist())
            if BEST_ONLY_INCLUDE_G_GG:
                base = df[(df["config"].isin(["G", "GG"])) & (df["GD_target"] == gd)]
                keep.extend(base.index.tolist())
        df_best = (df.loc[sorted(set(keep))]
                     .sort_values(["_gd_order", "_cfg_order", "_v_order", "_r_order"]))
        print(f"\nReduced figures: {len(df_best)} rows "
              f"({len(best_idx)} best GGAH + their GGA"
              f"{' + G/GG baselines' if BEST_ONLY_INCLUDE_G_GG else ''})")
        _bar_figures(df_best, fname_suffix="_bestonly",
                     title_suffix=f"\nGGA + best GGAH per demand case "
                                  f"(min {OPT_COLUMN})")

    # ============================================================== #
    #  EXTRACTION TEMPERATURE + DISPATCH MODE                        #
    # ============================================================== #
    # Hot-well extraction temperature over the year, each discharge hour
    # coloured by the dispatch mode calc_heat chose:
    #   A = direct HX only, HP idle
    #   B = HP on, T_extract still above the DHN return
    #   D = HP on, T_extract below the DHN return (HX delivers nothing, so the
    #       ATES forces the HP on regardless of the dispatch intent)
    # Hours where the ATES never ran are 'off' and are not plotted.
    # NB calc_heat iterates the second half of the year first, so the decline is
    # not left-to-right in hour order.
    MODE_COLOR = {"A": "tab:blue", "B": "tab:orange", "D": "tab:red"}
    MODE_LABEL = {"A": "A - direct HX only (HP idle)",
                  "B": "B - HP on, above DHN return",
                  "D": "D - HP on, below DHN return"}

    T_floor_plot = max(DEMAND_T_OUT - HP_DELTA_T_COLDSIDE, ATES_T_GROUND)

    if best_idx:
        fig, axes = plt.subplots(len(best_idx), 1,
                                 figsize=(11, 3.2 * len(best_idx)),
                                 sharex=True, sharey=True, squeeze=False)
        prof_rows = []
        for ax_i, idx in enumerate(best_idx):
            ax = axes[ax_i][0]
            tag = df.at[idx, "tag"]
            res = res_by_tag.get(tag)
            if res is None or res.get("ts_T_extract") is None:
                ax.text(0.5, 0.5,
                        f"no timeseries returned for {tag}\n"
                        "(add 'ts_T_extract' / 'ts_mode' to run_case)",
                        ha="center", va="center", transform=ax.transAxes)
                continue
            T_ex = np.asarray(res["ts_T_extract"], dtype=float)
            modes = np.asarray(res["ts_mode"], dtype=object)
            hours = np.arange(len(T_ex))

            counts = {}
            for m in ("A", "B", "D"):
                sel = (modes == m)
                counts[m] = int(sel.sum())
                if sel.any():
                    ax.scatter(hours[sel], T_ex[sel], s=4, zorder=3,
                               color=MODE_COLOR[m],
                               label=MODE_LABEL[m] if ax_i == 0 else None)
                    prof_rows.append(pd.DataFrame({
                        "tag": tag,
                        "demand_case": df.at[idx, "demand_case"],
                        "MAXV_target": df.at[idx, "MAXV_target"],
                        "ratio_target": df.at[idx, "ratio_target"],
                        "hour": hours[sel],
                        "T_extract [C]": T_ex[sel],
                        "mode": m,
                    }))

            ax.axhline(DEMAND_T_OUT, color="k", lw=0.9, ls="--", zorder=2,
                       label="DHN return (HX floor)" if ax_i == 0 else None)
            ax.axhline(T_floor_plot, color="0.4", lw=0.9, ls=":", zorder=2,
                       label="HP cold-side floor" if ax_i == 0 else None)
            ax.set_title(f"{df.at[idx, 'demand_case']}  "
                         f"({df.at[idx, 'profile']})   "
                         f"V = {df.at[idx, 'MAXV_target']:g} m3/h   "
                         f"r = {df.at[idx, 'ratio_target']:g}   "
                         f"(HP {df.at[idx, 'HP_POWER_EL']:.0f} kW_el)   "
                         f"hours A/B/D = {counts.get('A', 0)}/"
                         f"{counts.get('B', 0)}/{counts.get('D', 0)}", fontsize=9)
            ax.set_ylabel("T_extract (\u00b0C)")
            ax.grid(alpha=0.3)
        axes[-1][0].set_xlabel("Hour of the year")
        axes[0][0].set_xlim(0, 8760)
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            axes[0][0].legend(handles, labels, fontsize=8, markerscale=3,
                              loc="lower left")
        fig.suptitle(f"Hot-well extraction temperature and dispatch mode\n"
                     f"best GGAH per demand case (min {OPT_COLUMN})")
        fig.tight_layout()
        _save(fig, "T_extract_modes_best")
        if prof_rows:
            fig_data_sheets["Fig T_extract best"] = pd.concat(prof_rows,
                                                              ignore_index=True)

    # ============================================================== #
    #  OPTIMISATION FIGURES: metric vs the sizing knobs              #
    # ============================================================== #

    ggah = df[df["config"] == "GGAH"]

    def _sweep_figure(value_col, scale, ylabel, title, fname,
                      ref_col=None, ref_label=None,
                      x_col="ratio_target",
                      x_label="Ath/Hel  (ATES nominal kW_th / HP kW_el)",
                      line_col="MAXV_target", line_grid=None,
                      line_fmt="V={:g} m3/h"):
        """
        Line figure over the GGAH runs: <value_col> against <x_col>, one subplot
        per demand case and one line per value of <line_col>. The two sizing
        knobs are interchangeable, so the same code draws both views:
          * x = ratio, lines = max_V  (defaults)
          * x = max_V, lines = ratio  (pass x_col/line_col)

        GGA (no-HP) reference, if ref_col is given:
          * lines = max_V -> a dashed horizontal line per line, at the GGA value
            of that same max_V (GGA has no ratio, so it is flat in x).
          * x = max_V     -> ONE dashed GGA curve per panel, across max_V.
            Anything below it is a heat pump that pays for itself.
        """
        if ggah.empty:
            return None
        if line_grid is None:
            line_grid = ATES_MAXV_GRID if line_col == "MAXV_target" else RATIO_GRID
        gds = [gd for gd in GD_GRID if (ggah["GD_target"] == gd).any()]
        fig, axes = plt.subplots(1, len(gds), figsize=(5 * len(gds), 4.2),
                                 sharey=True, squeeze=False)
        rows = []
        for ax_i, gd in enumerate(gds):
            ax = axes[0][ax_i]
            sub_gd = ggah[ggah["GD_target"] == gd]
            for lv in line_grid:
                sub = sub_gd[sub_gd[line_col] == lv].sort_values(x_col)
                if sub.empty:
                    continue
                yv = sub[value_col].values * scale
                ax.plot(sub[x_col].values, yv, "o-", label=line_fmt.format(lv))
                for xv_, y_, hp_ in zip(sub[x_col].values, yv,
                                        sub["HP_POWER_EL"].values):
                    rows.append({"demand_case": f"G/D = {gd:.2f}", line_col: lv,
                                 x_col: xv_, "HP_POWER_EL": hp_, ylabel: y_})
                if ref_col is not None and line_col == "MAXV_target":
                    gga = df[(df["config"] == "GGA") & (df["GD_target"] == gd)
                             & (df["MAXV_target"] == lv)]
                    if not gga.empty and np.isfinite(gga[ref_col].values[0]):
                        ax.axhline(gga[ref_col].values[0] * scale, ls="--",
                                   lw=0.9, alpha=0.5,
                                   color=ax.lines[-1].get_color())
            if ref_col is not None and x_col == "MAXV_target":
                gga = (df[(df["config"] == "GGA") & (df["GD_target"] == gd)]
                       .sort_values("MAXV_target"))
                gga = gga[np.isfinite(gga[ref_col].values)]
                if not gga.empty:
                    ax.plot(gga["MAXV_target"].values,
                            gga[ref_col].values * scale, ls="--", lw=1.1,
                            marker="s", ms=4, color="0.4", alpha=0.8, zorder=1)
                    for xv_, y_ in zip(gga["MAXV_target"].values,
                                       gga[ref_col].values * scale):
                        rows.append({"demand_case": f"G/D = {gd:.2f}",
                                     line_col: "GGA (no HP)", x_col: xv_,
                                     "HP_POWER_EL": np.nan, ylabel: y_})
            ax.set_title(f"G/D = {gd:.2f}  ({DEMAND_CASES[gd]['profile']})",
                         fontsize=9)
            ax.set_xlabel(x_label)
            ax.grid(alpha=0.3)
            if ax_i == 0:
                ax.set_ylabel(ylabel)
        handles, labels = axes[0][0].get_legend_handles_labels()
        if ref_label is not None:
            handles.append(plt.Line2D([], [], ls="--", color="0.4", lw=0.9))
            labels.append(ref_label)
        axes[0][0].legend(handles, labels, fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        _save(fig, fname)
        return pd.DataFrame(rows)

    # --- View 1: x = Ath/Hel ratio, one line per ATES max_V -------------------
    lcoh_vs_r = _sweep_figure(
        "system_lcoh_yang", 1000.0, "System LCOH (\u20ac/MWh)",
        "System LCOH vs ATES/HP sizing ratio  (LCOE_calc_Yang, Delft cases)",
        "lcoh_vs_ratio",
        ref_col="system_lcoh_yang", ref_label="GGA (no HP), same V")
    if lcoh_vs_r is not None:
        fig_data_sheets["Fig LCOH vs ratio"] = lcoh_vs_r

    cac_vs_r = _sweep_figure(
        "CAC_eur_per_kg", 1.0, "CAC (\u20ac/kgCO\u2082)",
        "Carbon abatement cost vs ATES/HP sizing ratio  (ref = G)",
        "cac_vs_ratio",
        ref_col="CAC_eur_per_kg", ref_label="GGA (no HP), same V")
    if cac_vs_r is not None:
        fig_data_sheets["Fig CAC vs ratio"] = cac_vs_r

    # --- View 2: x = ATES max_V, one line per Ath/Hel ratio -------------------
    # The transpose of view 1. Reading a vertical slice here answers "how big
    # should the well be at this HP balance"; view 1 answers "how big should the
    # HP be at this well size". The dashed grey curve is GGA.
    MAXV_XLABEL = "ATES max_V (m3/h)"
    RATIO_FMT   = "r={:g}"

    lcoh_vs_v = _sweep_figure(
        "system_lcoh_yang", 1000.0, "System LCOH (\u20ac/MWh)",
        "System LCOH vs ATES size  (LCOE_calc_Yang, Delft cases)",
        "lcoh_vs_maxv",
        ref_col="system_lcoh_yang", ref_label="GGA (no HP)",
        x_col="MAXV_target", x_label=MAXV_XLABEL,
        line_col="ratio_target", line_fmt=RATIO_FMT)
    if lcoh_vs_v is not None:
        fig_data_sheets["Fig LCOH vs maxV"] = lcoh_vs_v

    cac_vs_v = _sweep_figure(
        "CAC_eur_per_kg", 1.0, "CAC (\u20ac/kgCO\u2082)",
        "Carbon abatement cost vs ATES size  (ref = G)",
        "cac_vs_maxv",
        ref_col="CAC_eur_per_kg", ref_label="GGA (no HP)",
        x_col="MAXV_target", x_label=MAXV_XLABEL,
        line_col="ratio_target", line_fmt=RATIO_FMT)
    if cac_vs_v is not None:
        fig_data_sheets["Fig CAC vs maxV"] = cac_vs_v

    # ============================================================== #
    #  OPTIMUM per demand case                                        #
    # ============================================================== #
    # ONE row per demand case: the best (max_V, ratio) combination of the GGAH
    # runs, i.e. the optimum over BOTH sizing knobs at once, with the geothermal
    # plant fixed. The objective is the system LCOH; CAC / CI / COP are reported
    # for the winning point but do not select it. Swap OPT_COLUMN to
    # "CAC_eur_per_kg" to optimise on abatement cost instead.
    #
    # These are the numbers to paste into OPTIMUM_SIZING in the cut-off script.
    opt_cols = ["tag", "demand_case", "profile", "config", "GD_target",
                "MAXV_target", "ratio_target", "HP_POWER_EL", "ates_nominal_kW",
                "system_lcoh_yang", "CAC_eur_per_kg", "CI_kg_per_kWh",
                "hp_mean_COP", "Reff"]
    opt_rows = []
    for i in best_idx:
        best = df.loc[i]
        row = {"objective": OPT_COLUMN}
        row.update({c: best.get(c) for c in opt_cols})
        # The GGA at the same max_V, for a like-for-like "does the HP pay" read.
        gga = df[(df["config"] == "GGA")
                 & (df["GD_target"] == best["GD_target"])
                 & (df["MAXV_target"] == best["MAXV_target"])]
        row["GGA same V [euro/MWh]"] = (gga["system_lcoh_yang"].values[0] * 1000.0
                                        if not gga.empty else np.nan)
        row["GGAH [euro/MWh]"] = best["system_lcoh_yang"] * 1000.0
        opt_rows.append(row)
    opt_df = pd.DataFrame(opt_rows)
    if not opt_df.empty:
        print("\n" + "=" * 70)
        print(f"OPTIMUM (max_V x Ath/Hel) PER DEMAND CASE  --  "
              f"objective: {OPT_COLUMN}")
        print("=" * 70)
        print(opt_df.to_string(index=False))
        print("\nPaste into OPTIMUM_SIZING of the cut-off script:")
        for _, r in opt_df.iterrows():
            print(f'    "{r["demand_case"]}": dict(max_V={r["MAXV_target"]:g}, '
                  f'ratio={r["ratio_target"]:g}),')
        fig_data_sheets["Optimum"] = opt_df

    # --- Write the workbook: Summary + one sheet per figure's data -------------
    df_out = df.drop(columns=["_gd_order", "_cfg_order", "_v_order", "_r_order"])
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="Summary", index=False)
        for sheet_name, fdf in fig_data_sheets.items():
            fdf.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"\nSaved optimisation summary -> {out}  ({len(df)} runs; "
          f"sheets: Summary, {', '.join(fig_data_sheets.keys())})")

    return df, results


if __name__ == "__main__":
    main()