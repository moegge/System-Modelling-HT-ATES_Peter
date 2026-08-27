# -*- coding: utf-8 -*-
"""
Test File Optimisation Peter Yang 26.08..py
==================================================================
Optimisation sweep over the ATES / HP sizing space, at the three G/D ratios of
David Geerts' paper. Derived from 'Test File Sweep Peter Yang 25.08..py'.

DIFFERENCE TO THE 25.08 SWEEP
-----------------------------
1. G/D is ALWAYS varied by adapting the geothermal plant capacity (the old
   GD_MODE='vary_geo'). The 'vary_demand' branch is not part of this file.
2. Two sizing knobs are swept instead of a fixed list of HP ratings:
     * ATES_MAXV_GRID  -- the ATES size, max_V [m3/h]
     * RATIO_GRID      -- the sizing ratio Ath/Hel [-]
   The HP rating is DERIVED, not given:
       ates_nominal_kW = (max_V/3600) * rho*cp * (GEO_T_OUT - DEMAND_T_OUT)
       HP_POWER_EL     = ates_nominal_kW / ratio
   This is the point of the file: the ATES size is an unknown here, so a fixed
   HP rating in kW would mean a different ATES/HP balance at every max_V.
3. The run grid is dimension-aware, so nothing is run more times than it varies:
     * G, GG   -> no ATES, no HP  -> one run per G/D
     * GGA     -> ATES, no HP     -> one run per G/D x max_V
     * GGAH    -> ATES + HP       -> one run per G/D x max_V x ratio
4. Four line figures show where the optimum sits, as two transposed views of the
   same GGAH grid: system LCOH and CAC vs the Ath/Hel ratio (one line per max_V),
   and the same two vs max_V (one line per ratio). The bar figures of the 25.08
   sweep are kept but can be switched off with MAKE_BAR_FIGURES, since they get
   very wide.
5. An 'Optimum' table/sheet reports the best GGAH point per G/D.

System LCOH is David's LCOE_calc_Yang figure (pooled discounted cost / pooled
discounted heat over a COMMON 60-year horizon with reinvestment), exactly as in
the 25.08 sweep. All saved files get a '_yang_optimisation' suffix, so they never
collide with the '_yang' files of the 25.08 sweep.

Requires run_case() to return 'system_lcoh_yang' (see Test_File_Peter_17_08.py).

Put this file in the SAME folder as Test_File_Peter_17_08.py, main2_Peter.py,
ATES_obj_Peter.py and the data files, then:

    python "Test File Optimisation Peter Yang 26.08..py"

Configurations:
  * G    = gas only
  * GG   = gas + geothermal
  * GGA  = gas + geothermal + HT-ATES        (no heat pump)
  * GGAH = gas + geothermal + HT-ATES + HP   (heat pump on the ATES)

G/D = annual geothermal production / annual heat demand, targeted by
back-solving the geothermal power:
    GEO_POWER = target_GD * annual_demand_kWh / 8760      (nameplate definition)
==================================================================
"""

import os

import matplotlib
matplotlib.use("Agg")          # headless: no windows even if something calls show()
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from Test_File_Peter_17_08 import (run_case, DEMAND_EXAMPLE, DEMAND_T_IN,
                                   DEMAND_T_OUT, GEO_T_OUT,
                                   HP_DELTA_T_COLDSIDE, ATES_T_GROUND)
from main2_Peter import demand_class

# ================================================================== #
#  DEFINE THE SWEEP GRID HERE                                        #
# ================================================================== #

# David's exact G/D values.
GD_GRID     = [1.62, 1.07, 0.65]
# The four configurations, in David's left-to-right order (+ GGAH).
# G and GG carry no ATES/HP and are only needed as CAC references; drop them
# from the list if you only want the storage configurations.
CONFIG_GRID = ["G", "GG", "GGA", "GGAH"]

# --- KNOB 1: ATES size ------------------------------------------------------
# max_V [m3/h], the aquifer flow rating. Drives both the ATES CAPEX
# (costperm3 * max_V) and, through ates_nominal_kW, the derived HP rating.
# Ranges are fine too, e.g.:
#     ATES_MAXV_GRID = list(np.arange(160, 641, 160))
#     ATES_MAXV_GRID = list(np.linspace(100, 700, 4))
ATES_MAXV_GRID = [30, 70, 120, 160, 320, 640]          # [m3/h]

# --- KNOB 2: ATES/HP sizing ratio ------------------------------------------
# ratio = ates_nominal_kW / HP_POWER_EL  [kW_th / kW_el].
# HIGH ratio -> small HP relative to the well; LOW ratio -> big HP.
# Anchors from the 25.08 sweep (max_V = 320 -> ates_nominal ~ 7431 kW):
#     HP  700 kW_el -> ratio ~ 10.6
#     HP 3000 kW_el -> ratio ~  2.5
# Ranges are fine too, e.g.:
#     RATIO_GRID = list(np.linspace(2, 16, 8))
RATIO_GRID = [1, 2, 3, 4, 5, 6]        # [-]

HOURS_PER_YEAR = 8760            # nameplate G/D basis (GEO_POWER x hours)

# Water properties used for ates_nominal_kW. MUST match the RHO_CP constant in
# Test_File_Peter_17_08.run_case(), otherwise the achieved ratio drifts away
# from the target (the code below checks this per run and warns).
RHO_CP = 4180.0                  # [kJ/m3.K]

# --- Output folder + figure toggles ----------------------------------------
RESULTS_DIR      = "results optimisation"  # all outputs go here
SAVE_FIGURES     = True          # True -> save the graphs into RESULTS_DIR
MAKE_BAR_FIGURES = True          # False -> only the LCOH/CAC-vs-ratio line figures
                                 #          (the bar figures get very wide here)

# Reduced ("best only") second version of every bar figure: GGA + the single
# best GGAH combination per G/D, instead of the full grid. Files get an extra
# '_bestonly' suffix. The LCOH/CAC-vs-ratio line figures have no reduced
# version -- there the whole point is the sweep across the ratio.
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


def _annual_demand_kWh():
    """Annual DH demand [kWh], to convert a target G/D into a geothermal power."""
    dem = demand_class(T_in=DEMAND_T_IN, T_out=DEMAND_T_OUT,
                       example_demand=DEMAND_EXAMPLE)
    return float(np.sum(dem.data))


def _geo_power_for_gd(target_gd, annual_demand_kWh):
    """Nameplate G/D: GEO_POWER [kW] = target_GD * annual_demand / hours."""
    return target_gd * annual_demand_kWh / HOURS_PER_YEAR


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
    Returns a list of (target_gd, cfg, max_V, ratio) with NaN where not applicable.
    """
    grid = []
    for target_gd in GD_GRID:
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
                grid.append((target_gd, cfg, max_V, ratio))
    return grid


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, RESULTS_DIR)
    os.makedirs(results_dir, exist_ok=True)

    annual_demand_kWh = _annual_demand_kWh()
    run_grid = _build_run_grid()

    print(f"Annual demand (fixed {DEMAND_EXAMPLE}): {annual_demand_kWh / 1e6:.3f} GWh "
          f"-> sizing geothermal for target G/D {GD_GRID}")
    print(f"ATES max_V grid : {ATES_MAXV_GRID} m3/h "
          f"-> ates_nominal {[round(_ates_nominal_kW(v)) for v in ATES_MAXV_GRID]} kW")
    print(f"Ath/Hel grid    : {RATIO_GRID}")
    print(f"TOTAL RUNS      : {len(run_grid)}")

    results = []

    for target_gd, cfg, max_V, ratio in run_grid:
        geo_power = _geo_power_for_gd(target_gd, annual_demand_kWh)

        # Build the tag from the knobs this config actually uses.
        tag = f"GD{target_gd:.2f}_{cfg}"
        if np.isfinite(max_V):
            tag += f"_V{max_V:g}"
        if np.isfinite(ratio):
            tag += f"_r{ratio:g}"

        # Derive the HP rating from the ATES size and the target ratio.
        hp_power = _hp_power_for_ratio(max_V, ratio) if np.isfinite(ratio) else np.nan

        print("\n" + "#" * 70)
        msg = f"# RUN: {tag}   (GEO_POWER = {geo_power:.1f} kW, demand = {DEMAND_EXAMPLE}"
        if np.isfinite(max_V):
            msg += f", max_V = {max_V:g} m3/h -> Ath = {_ates_nominal_kW(max_V):.0f} kW"
        if np.isfinite(ratio):
            msg += f", r = {ratio:g} -> HP = {hp_power:.0f} kW_el"
        print(msg + ")")
        print("#" * 70)

        run_outfile = os.path.join(results_dir, f"timeseries_{tag}_yang_optimisation.xlsx")
        run_kwargs = dict(
            CONFIG=cfg,
            GEO_POWER=geo_power,
            DEMAND_EXAMPLE=DEMAND_EXAMPLE,
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
        res["GD_target"]    = target_gd
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
                fig.savefig(os.path.join(results_dir, f"{tag}_fig{local_i}_yang_optimisation.png"),
                            dpi=150, bbox_inches="tight")
        for num in open_nums:
            plt.close(plt.figure(num))

    # --- Collect into a summary table -----------------------------------------
    summary_cols = [
        "tag", "GD_target", "GD_ratio", "config", "USE_HP", "GEO_POWER",
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
            "call and the return-dict entry in Test_File_Peter_17_08.py.")

    pd.set_option("display.width", 260)

    pd.set_option("display.max_columns", 40)
    print("\n" + "=" * 70)
    print("OPTIMISATION SWEEP SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))

    out = os.path.join(results_dir, "optimisation_summary_yang.xlsx")
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

    # x-axis: grouped by G/D, with the configs within each group. Build a stable
    # ordering (G/D groups left->right, configs in CONFIG_GRID order within each
    # group, then max_V, then the ratio) so the bars never move between runs.
    df["_gd_order"]  = df["GD_target"].map({gd: i for i, gd in enumerate(GD_GRID)})
    df["_cfg_order"] = df["config"].map({c: i for i, c in enumerate(CONFIG_GRID)})
    df["_v_order"]   = df["MAXV_target"].fillna(-1)     # configs without ATES first
    df["_r_order"]   = df["ratio_target"].fillna(-1)    # configs without HP first
    df = df.sort_values(["_gd_order", "_cfg_order", "_v_order", "_r_order"]).reset_index(drop=True)

    # x-labels: config + G/D, plus max_V for anything with an ATES and the
    # Ath/Hel ratio for anything with an HP.
    def _xlabel(row):
        base = f"{row['config']}\nG/D={row['GD_target']:.2f}"
        if np.isfinite(row.get("MAXV_target", np.nan)):
            base += f"\nV={row['MAXV_target']:g}"
        if np.isfinite(row.get("ratio_target", np.nan)):
            base += f"\nr={row['ratio_target']:g}"
        return base
    xlabels = [_xlabel(r) for _, r in df.iterrows()]
    x = np.arange(len(df))

    # --- Figure: LCOH per combination (David Fig. 4 style) --------------------
    # Component LCOHs (Gas / Geo / ATES) as coloured dots + System LCOH as a
    # cyan dash, grouped by G/D. LCOH stored in euro/kWh -> plotted euro/MWh.
    # The System dash is LCOE_calc_Yang (60-yr common horizon), matching David.
    gas  = df["gas_lcoh"].values * 1000.0
    geo  = df["geo_lcoh"].values * 1000.0
    ates = df["ates_lcoh"].values * 1000.0
    syst = df["system_lcoh_yang"].values * 1000.0     # David's LCOE_calc_Yang



    fig_data_sheets["Fig LCOH data"] = pd.DataFrame({
        "tag": df["tag"].values,
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

    # --- Figure: Energy share per component ----------------------------------
    # Exhaustive stacked share of DEMAND, summing to ~1.0:
    #   Geo (blue) + ATES direct (orange) + HP source heat (green)
    #   + HP electricity (green, hatched = grid) + Gas (red).
    # Geo uses geo_to_demand_GWh (charging removed) so stored geo is not
    # double-counted here and again in the ATES segment. The HP condenser output
    # is Q_evap + P_el; P_el is grid electricity, so it is drawn hatched and
    # excluded from the renewable total printed above each bar.
    dem_arr = df["demand_GWh"].values
    geo_f     = np.nan_to_num(df["geo_to_demand_GWh"].values) / dem_arr
    ates_f    = np.nan_to_num(df["ates_direct_GWh"].values) / dem_arr
    hp_el_f   = np.nan_to_num(df["hp_elec_GWh"].values) / dem_arr          # P_el
    hp_evap_f = np.nan_to_num(df["hp_GWh"].values) / dem_arr - hp_el_f     # Q_evap
    gas_f     = np.nan_to_num(df["gas_GWh"].values) / dem_arr
    res_total = geo_f + ates_f + hp_evap_f


    fig_data_sheets["Fig RES data"] = pd.DataFrame({
        "tag": df["tag"].values,
        "config": df["config"].values,
        "GD_target": df["GD_target"].values,
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

    # --- Carbon Abatement Cost (CAC) per combination --------------------------
    # CAC = (LCOH_config - LCOH_ref) / (CI_ref - CI_config), in euro/kgCO2.
    # LCOH   = CO2-INCLUSIVE system LCOH from LCOE_calc_Yang (euro/kWh) -> shifts
    #          CAC down by the carbon price vs a CO2-free CAC; intentional.
    # CI     = system carbon intensity of delivered heat = total_CO2 / demand
    #          [kgCO2/kWh]. Positive (CI_ref - CI_config) => real abatement.
    # Every config is referenced to the gas-only baseline G WITHIN the same G/D
    # group -> CAC is the cumulative abatement cost vs fossil, not the marginal
    # cost of each added component.
    # G is its own reference and is skipped (matches David's Fig. 4 caption).
    #
    # NB vs the 25.08 sweep: GGA is now MULTI-ROW (one per max_V), so it can no
    # longer serve as a reference -- the lookup below only admits the single-row
    # configs G and GG.
    CAC_REF      = {"GG": "G", "GGA": "G", "GGAH": "G"}
    CONFIG_COLOR = {"GG": "tab:blue", "GGA": "tab:orange", "GGAH": "tab:green"}

    # Carbon intensity of delivered heat [kgCO2/kWh] for every row.
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
            print(f"  WARNING: no unique reference '{ref_cfg}' for {row['tag']} -> CAC = NaN")
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
        "config": df["config"].values,
        "GD_target": df["GD_target"].values,
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

    # --- Carbon intensity of delivered heat (emissions per kWh) ---------------
    # CI = total system CO2 / heat delivered to demand -> plotted gCO2/kWh.
    # Same x-axis as RES/CAC, but every config has a value (G included).
    # CI_kg_per_kWh was already computed in the CAC block above -> reuse it.
    CI_COLOR = {"G": "tab:red", "GG": "tab:blue",
                "GGA": "tab:orange", "GGAH": "tab:green"}
    ci_g = df["CI_kg_per_kWh"].values * 1000.0      # gCO2/kWh



    fig_data_sheets["Fig CI data"] = pd.DataFrame({
        "tag": df["tag"].values,
        "config": df["config"].values,
        "GD_target": df["GD_target"].values,
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
            # Light vertical separators between G/D groups.
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
                            ha="center", va=("bottom" if v >= 0 else "top"), fontsize=7)
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
        _save(fig, "lcoh_per_combination_yang_opt" + fname_suffix)

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

        # Renewable total (P_el excluded) above each bar.
        for xi, r, tot in zip(xs, rest_, rest_ + hef + gsf):
            ax.text(xi, tot + 0.015, f"RES {r:.2f}", ha="center", va="bottom", fontsize=7)

        _separators(ax)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("Share of demand met")
        ax.set_ylim(0, 1.12)
        ax.set_title("Energy share per component  "
                     "(HP split into source heat and grid electricity)" + title_suffix)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        _save(fig, "res_per_combination_yang_opt" + fname_suffix)

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
        ax.set_title("CAC per combination  (LCOH = LCOE_calc_Yang)" + title_suffix)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        _save(fig, "cac_per_combination_yang_opt" + fname_suffix)

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
        _save(fig, "ci_per_combination_yang_opt" + fname_suffix)

    # --- Full grid ------------------------------------------------------------
    if MAKE_BAR_FIGURES:
        _bar_figures(df)

    # --- Reduced: GGA + the best GGAH combination per G/D ---------------------
    # best_idx also feeds the 'Optimum' table further down, so "best" is defined
    # in exactly one place. The GGA kept for each G/D is the one with the SAME
    # max_V as the winning GGAH -> like-for-like, same well, with and without HP.
    best_idx = []
    for gd in GD_GRID:
        sub = df[(df["GD_target"] == gd) & (df["config"] == "GGAH")]
        if sub.empty or sub[OPT_COLUMN].isna().all():
            print(f"  WARNING: no GGAH run with a finite {OPT_COLUMN} at G/D = {gd:.2f}")
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
                     title_suffix=f"\nGGA + best GGAH per G/D (min {OPT_COLUMN})")

    # ============================================================== #
    #  EXTRACTION TEMPERATURE + DISPATCH MODE                        #
    #  Best GGAH combination per G/D, one panel each.                #
    # ============================================================== #
    # Hot-well extraction temperature over the year, each discharge hour
    # coloured by the dispatch mode calc_heat chose:
    #   A = direct HX only, HP idle
    #   B = HP on, T_extract still above the DHN return
    #   D = HP on, T_extract below the DHN return (HX delivers nothing, so the
    #       ATES forces the HP on regardless of the dispatch intent)
    # Hours where the ATES never ran are 'off' and are not plotted (T_extract
    # is 0 there, which would drag the curve to the x-axis).
    # NB the run starts halfway through the year (calc_heat iterates the second
    # half first), so the temperature decline is not left-to-right in hour order.
    MODE_COLOR = {"A": "tab:blue", "B": "tab:orange", "D": "tab:red"}
    MODE_LABEL = {"A": "A - direct HX only (HP idle)",
                  "B": "B - HP on, above DHN return",
                  "D": "D - HP on, below DHN return"}

    # Link the winning rows back to the full run dicts (df was re-sorted and
    # re-indexed, so position is no longer the run order; the tag is unique).
    res_by_tag = {r.get("tag"): r for r in results}
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
                        "GD_target": df.at[idx, "GD_target"],
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
            ax.set_title(f"G/D = {df.at[idx, 'GD_target']:.2f}   "
                         f"V = {df.at[idx, 'MAXV_target']:g} m3/h   "
                         f"r = {df.at[idx, 'ratio_target']:g}   "
                         f"(HP {df.at[idx, 'HP_POWER_EL']:.0f} kW_el)   "
                         f"hours A/B/D = {counts.get('A', 0)}/{counts.get('B', 0)}/{counts.get('D', 0)}",
                         fontsize=9)
            ax.set_ylabel("T_extract (\u00b0C)")
            ax.grid(alpha=0.3)
        axes[-1][0].set_xlabel("Hour of the year")
        axes[0][0].set_xlim(0, 8760)
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            axes[0][0].legend(handles, labels, fontsize=8, markerscale=3,
                              loc="lower left")
        fig.suptitle(f"Hot-well extraction temperature and dispatch mode\n"
                     f"best GGAH per G/D (min {OPT_COLUMN})")
        fig.tight_layout()
        _save(fig, "T_extract_modes_best_yang_opt")
        if prof_rows:
            fig_data_sheets["Fig T_extract best"] = pd.concat(prof_rows,
                                                              ignore_index=True)

    # ============================================================== #
    #  OPTIMISATION FIGURES: metric vs the Ath/Hel ratio             #
    #  One panel per G/D, one line per ATES max_V. GGAH rows only.   #
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
        per G/D and one line per value of <line_col>. The two sizing knobs are
        interchangeable, so the same code draws both views:
          * x = ratio, lines = max_V  (defaults)
          * x = max_V, lines = ratio  (pass x_col/line_col)

        GGA (no-HP) reference, if ref_col is given:
          * lines = max_V -> a dashed horizontal line per line, at the GGA value
            of that same max_V (GGA has no ratio, so it is flat in x).
          * x = max_V     -> ONE dashed GGA curve per panel, across max_V. GGA
            does not depend on the ratio, so it is the same reference for every
            coloured line: anything below it is an HP that pays for itself.
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
                    rows.append({"GD_target": gd, line_col: lv, x_col: xv_,
                                 "HP_POWER_EL": hp_, ylabel: y_})
                # GGA reference at this line's max_V (flat in x).
                if ref_col is not None and line_col == "MAXV_target":
                    gga = df[(df["config"] == "GGA") & (df["GD_target"] == gd)
                             & (df["MAXV_target"] == lv)]
                    if not gga.empty and np.isfinite(gga[ref_col].values[0]):
                        ax.axhline(gga[ref_col].values[0] * scale, ls="--",
                                   lw=0.9, alpha=0.5,
                                   color=ax.lines[-1].get_color())
            # GGA reference as a curve, when max_V is the x-axis.
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
                        rows.append({"GD_target": gd, line_col: "GGA (no HP)",
                                     x_col: xv_, "HP_POWER_EL": np.nan,
                                     ylabel: y_})
            ax.set_title(f"G/D = {gd:.2f}")
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
        "System LCOH vs ATES/HP sizing ratio  (LCOE_calc_Yang)",
        "lcoh_vs_ratio_yang_opt",
        ref_col="system_lcoh_yang", ref_label="GGA (no HP), same V")
    if lcoh_vs_r is not None:
        fig_data_sheets["Fig LCOH vs ratio"] = lcoh_vs_r

    cac_vs_r = _sweep_figure(
        "CAC_eur_per_kg", 1.0, "CAC (\u20ac/kgCO\u2082)",
        "Carbon abatement cost vs ATES/HP sizing ratio  (ref = G)",
        "cac_vs_ratio_yang_opt",
        ref_col="CAC_eur_per_kg", ref_label="GGA (no HP), same V")
    if cac_vs_r is not None:
        fig_data_sheets["Fig CAC vs ratio"] = cac_vs_r

    # --- View 2: x = ATES max_V, one line per Ath/Hel ratio -------------------
    # The transpose of view 1. Reading down a vertical slice here answers "how
    # big should the well be at this HP balance", where view 1 answers "how big
    # should the HP be at this well size". The dashed grey curve is GGA, which
    # has no HP and therefore one value per max_V regardless of the ratio.
    MAXV_XLABEL = "ATES max_V (m3/h)"
    RATIO_FMT   = "r={:g}"

    lcoh_vs_v = _sweep_figure(
        "system_lcoh_yang", 1000.0, "System LCOH (\u20ac/MWh)",
        "System LCOH vs ATES size  (LCOE_calc_Yang)",
        "lcoh_vs_maxv_yang_opt",
        ref_col="system_lcoh_yang", ref_label="GGA (no HP)",
        x_col="MAXV_target", x_label=MAXV_XLABEL,
        line_col="ratio_target", line_fmt=RATIO_FMT)
    if lcoh_vs_v is not None:
        fig_data_sheets["Fig LCOH vs maxV"] = lcoh_vs_v

    cac_vs_v = _sweep_figure(
        "CAC_eur_per_kg", 1.0, "CAC (\u20ac/kgCO\u2082)",
        "Carbon abatement cost vs ATES size  (ref = G)",
        "cac_vs_maxv_yang_opt",
        ref_col="CAC_eur_per_kg", ref_label="GGA (no HP)",
        x_col="MAXV_target", x_label=MAXV_XLABEL,
        line_col="ratio_target", line_fmt=RATIO_FMT)
    if cac_vs_v is not None:
        fig_data_sheets["Fig CAC vs maxV"] = cac_vs_v

    # ============================================================== #
    #  OPTIMUM per G/D                                                #
    # ============================================================== #
    # ONE row per G/D: the best (max_V, ratio) combination of the GGAH runs,
    # i.e. the optimum over BOTH sizing knobs at once. The objective is the
    # system LCOH; CAC / CI / COP are reported for the winning point but do not
    # select it. Swap OPT_COLUMN to "CAC_eur_per_kg" to optimise on carbon
    # abatement cost instead (both are minimised).
    # OPT_COLUMN is set at the top of the file; best_idx was resolved in the
    # figures section above, so the table and the reduced figures can never
    # disagree about which combination is the optimum.
    opt_cols = ["tag", "config", "GD_target", "MAXV_target", "ratio_target",
                "HP_POWER_EL", "ates_nominal_kW", "system_lcoh_yang",
                "CAC_eur_per_kg", "CI_kg_per_kWh", "hp_mean_COP", "Reff"]
    opt_rows = []
    for i in best_idx:
        best = df.loc[i]
        row = {"objective": OPT_COLUMN}
        row.update({c: best.get(c) for c in opt_cols})
        opt_rows.append(row)
    opt_df = pd.DataFrame(opt_rows)
    if not opt_df.empty:
        print("\n" + "=" * 70)
        print(f"OPTIMUM (max_V x Ath/Hel) PER G/D  --  objective: {OPT_COLUMN}")
        print("=" * 70)
        print(opt_df.to_string(index=False))
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