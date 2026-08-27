# -*- coding: utf-8 -*-
"""
Test File Sweep Peter Yang 25.08..py
==================================================================
Drives Test_File_Peter_17_08.run_case() across a grid of G/D ratios x system
configurations (mirroring David Geerts' paper, plus a heat-pump variant), and
collects the headline results into one summary table + workbook + figures.

DIFFERENCE TO THE 17.08 SWEEP: the system LCOH used in the LCOH figure and in
the CAC denominator is David's LCOE_calc_Yang figure (pooled discounted cost /
pooled discounted heat over a COMMON 60-year horizon with reinvestment), not the
generation-weighted blend of component LCOHs. The blend is not reported here at
all. All saved files get a '_yang' suffix.

Requires run_case() to return 'system_lcoh_yang' (see Test_File_Peter_17_08.py).

Put this file in the SAME folder as Test_File_Peter_17_08.py, main2_Peter.py,
ATES_obj_Peter.py and the data files, then:

    python sweep.py

Configurations (David's three + one more):
  * G    = gas only
  * GG   = gas + geothermal
  * GGA  = gas + geothermal + HT-ATES        (no heat pump)
  * GGAH = gas + geothermal + HT-ATES + HP   (heat pump on the ATES)

G/D = annual geothermal production / annual heat demand. We target the paper's
values (0.65, 1.07, 1.62) by back-solving the geothermal power for each:
    GEO_POWER = target_GD * annual_demand_kWh / 8760      (nameplate definition)
All configs in one G/D group share that GEO_POWER (G ignores it -> no geo).

All outputs (summary workbook + figures) go into RESULTS_DIR, created next to
this file if absent.

Notes
-----
* SAVE_FIGURES toggles whether figures are written to RESULTS_DIR.
* RUN_MAKE_PLOTS / RUN_WRITE_EXCEL control the per-run system_plot figures and
  the per-run full workbook.
==================================================================
"""

import os

import matplotlib
matplotlib.use("Agg")          # headless: no windows even if something calls show()
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from Test_File_Peter_17_08 import run_case, DEMAND_EXAMPLE, DEMAND_T_IN, DEMAND_T_OUT
from main2_Peter import demand_class

# ================================================================== #
#  DEFINE THE SWEEP GRID HERE                                        #
# ================================================================== #

# David's exact G/D values.
GD_GRID     = [1.62, 1.07, 0.65]
# The four configurations, in David's left-to-right order (+ GGAH).
CONFIG_GRID = ["G", "GG", "GGA", "GGAH"]

# Heat-pump electrical ratings [kW] to run for the GGAH config. Each value
# produces its OWN GGAH bar (with its own ATES/HP ratio) in every G/D group.
# Configs without a heat pump (G / GG / GGA) ignore this and run once.
HP_POWER_GRID = [700, 3000]

HOURS_PER_YEAR = 8760            # nameplate G/D basis (GEO_POWER x hours)

# --- HOW to vary G/D -------------------------------------------------------
# "vary_geo"    : fixed demand profile (DEMAND_EXAMPLE in the test file), and
#                 GEO_POWER is back-solved per target G/D. (Your original mode.)
# "vary_demand" : fixed geothermal capacity (FIXED_GEO_POWER_MW, David's 7.4 MW),
#                 and each target G/D selects a DIFFERENT demand profile whose
#                 annual demand gives that ratio. This is David's approach for
#                 the Fig. 4 style charts.
GD_MODE = "vary_geo"

# Used only when GD_MODE == "vary_demand": David's fixed doublet size.
FIXED_GEO_POWER_MW = 7.4          # [MW] -> 7400 kW

# Used only when GD_MODE == "vary_demand": map each target G/D to the demand
# profile name that produces it against the fixed geo. You must first add these
# profiles inside demand_class (main2_Peter.py) and put their names here.
# G/D = geo_annual / demand_annual, so a LOWER G/D needs a LARGER demand profile.
GD_TO_DEMAND = {
    1.62: "Delft_162",     # e.g. avg ~4.5 MW  (replace with your profile name)
    1.07: "Delft_107",     # e.g. avg ~6.9 MW
    0.65: "Delft_065",     # e.g. avg ~11.4 MW
}

# --- Output folder + figure toggle ----------------------------------------
RESULTS_DIR  = "results sweep"   # all outputs (summary + figures) go here
SAVE_FIGURES = True              # True -> save the graphs into RESULTS_DIR

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


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, RESULTS_DIR)
    os.makedirs(results_dir, exist_ok=True)

    annual_demand_kWh = _annual_demand_kWh()
    print(f"GD_MODE = {GD_MODE!r}")
    if GD_MODE == "vary_geo":
        print(f"Annual demand (fixed {DEMAND_EXAMPLE}): {annual_demand_kWh / 1e6:.3f} GWh "
              f"-> sizing geothermal for target G/D {GD_GRID}")
    else:
        print(f"Fixed geothermal {FIXED_GEO_POWER_MW} MW "
              f"-> selecting demand profile per target G/D {GD_GRID}")

    results = []

    for target_gd in GD_GRID:
        # Decide the per-group knobs based on GD_MODE.
        if GD_MODE == "vary_geo":
            geo_power = _geo_power_for_gd(target_gd, annual_demand_kWh)
            demand_example = DEMAND_EXAMPLE                 # fixed demand profile
        elif GD_MODE == "vary_demand":
            geo_power = FIXED_GEO_POWER_MW * 1000.0         # fixed geo capacity [kW]
            if target_gd not in GD_TO_DEMAND:
                raise KeyError(f"GD_MODE='vary_demand' needs GD_TO_DEMAND[{target_gd}] "
                               f"set to a demand profile name.")
            demand_example = GD_TO_DEMAND[target_gd]        # per-G/D demand profile
        else:
            raise ValueError(f"GD_MODE must be 'vary_geo' or 'vary_demand', got {GD_MODE!r}")

        for cfg in CONFIG_GRID:
            # GGAH runs once per HP power in HP_POWER_GRID (each -> its own bar,
            # its own ATES/HP ratio). All other configs have no HP -> run once.
            hp_powers = HP_POWER_GRID if cfg == "GGAH" else [None]

            for hp_power in hp_powers:
                if cfg == "GGAH":
                    tag = f"GD{target_gd:.2f}_{cfg}_HP{int(hp_power)}"
                else:
                    tag = f"GD{target_gd:.2f}_{cfg}"
                print("\n" + "#" * 70)
                print(f"# RUN: {tag}   (GEO_POWER = {geo_power:.1f} kW, demand = {demand_example}"
                      + (f", HP = {int(hp_power)} kW" if hp_power is not None else "") + ")")
                print("#" * 70)
                run_outfile = os.path.join(results_dir, f"timeseries_{tag}_yang.xlsx")
                run_kwargs = dict(
                    CONFIG=cfg,
                    GEO_POWER=geo_power,
                    DEMAND_EXAMPLE=demand_example,
                    tag=tag,
                    OUTFILE=run_outfile,
                    make_plots=RUN_MAKE_PLOTS,
                    write_excel=RUN_WRITE_EXCEL,
                )
                if hp_power is not None:
                    run_kwargs["HP_POWER_EL"] = hp_power
                res = run_case(**run_kwargs)
                # Record the TARGET G/D too (the achieved one is in res["GD_ratio"]).
                res["GD_target"] = target_gd
                results.append(res)

                # Save any per-run system_plot figures into results_dir (local numbering).
                open_nums = plt.get_fignums()
                if SAVE_FIGURES and RUN_MAKE_PLOTS:
                    for local_i, num in enumerate(open_nums, start=1):
                        fig = plt.figure(num)
                        fig.savefig(os.path.join(results_dir, f"{tag}_fig{local_i}_yang.png"),
                                    dpi=150, bbox_inches="tight")
                for num in open_nums:
                    plt.close(plt.figure(num))

    # --- Collect into a summary table -----------------------------------------
    summary_cols = [
        "tag", "GD_target", "GD_ratio", "config", "USE_HP", "GEO_POWER", "ATES_MAX_V",
        "HP_POWER_EL", "ates_nominal_kW", "ratio_ATES_HP",
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
    print("SWEEP SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))

    out = os.path.join(results_dir, "sweep_summary_yang.xlsx")
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
    # ordering that matches David's layout (G/D groups left->right, configs in
    # CONFIG_GRID order within each group). GGAH may appear multiple times (one
    # per HP power); order those by HP power so the bars are stable.
    df["_gd_order"] = df["GD_target"].map({gd: i for i, gd in enumerate(GD_GRID)})
    df["_cfg_order"] = df["config"].map({c: i for i, c in enumerate(CONFIG_GRID)})
    df["_hp_order"] = df["HP_POWER_EL"].fillna(-1)      # non-HP configs sort first
    df = df.sort_values(["_gd_order", "_cfg_order", "_hp_order"]).reset_index(drop=True)

    # x-labels: config + G/D, plus the ATES/HP ratio for GGAH bars so each is
    # identified by its ratio (r = ATES nominal thermal / HP electrical).
    def _xlabel(row):
        base = f"{row['config']}\nG/D={row['GD_target']:.2f}"
        if row["config"] == "GGAH" and np.isfinite(row.get("ratio_ATES_HP", np.nan)):
            base += f"\nAth/Hel={row['ratio_ATES_HP']:.2f}"
        return base
    xlabels = [_xlabel(r) for _, r in df.iterrows()]
    x = np.arange(len(df))

    # --- Figure: LCOH per combination (David Fig. 4 style) --------------------
    # Component LCOHs (Gas / Geo / ATES) as coloured dots + System LCOH as a
    # cyan dash, grouped by G/D. LCOH stored in euro/kWh -> plotted euro/MWh.
    # The System dash is LCOE_calc_Yang (60-yr common horizon), matching David.
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(df)), 5))

    gas  = df["gas_lcoh"].values * 1000.0
    geo  = df["geo_lcoh"].values * 1000.0
    ates = df["ates_lcoh"].values * 1000.0
    syst = df["system_lcoh_yang"].values * 1000.0     # David's LCOE_calc_Yang

    ax.scatter(x, gas,  color="tab:red",    label="Gas",  zorder=3)
    ax.scatter(x, geo,  color="tab:blue",   label="Geo",  zorder=3)
    ax.scatter(x, ates, color="tab:orange", label="ATES", zorder=3)
    dash = 0.3
    for xi, s in zip(x, syst):
        if np.isfinite(s):
            ax.hlines(s, xi - dash, xi + dash, color="tab:cyan", linewidth=2,
                      zorder=2, label="System (Yang)" if xi == 0 else None)
    # Light vertical separators between G/D groups.
    for i in range(1, len(df)):
        if df["_gd_order"].iloc[i] != df["_gd_order"].iloc[i - 1]:
            ax.axvline(i - 0.5, color="0.8", linewidth=1, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("LCOH (\u20ac/MWh)")
    ax.set_title("LCOH per combination  (system = LCOE_calc_Yang, 60-yr horizon)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "lcoh_per_combination_yang")

    fig_data_sheets["Fig LCOH data"] = pd.DataFrame({
        "tag": df["tag"].values,
        "config": df["config"].values,
        "GD_target": df["GD_target"].values,
        "HP_POWER_EL": df["HP_POWER_EL"].values,
        "ratio_ATES_HP": df["ratio_ATES_HP"].values,
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
    # is Q_evap + P_el; P_el is grid electricity (same CO2 intensity as the gas
    # boiler per kWh input), so it is drawn hatched and excluded from the
    # renewable total printed above each bar.
    dem_arr = df["demand_GWh"].values
    geo_f     = np.nan_to_num(df["geo_to_demand_GWh"].values) / dem_arr
    ates_f    = np.nan_to_num(df["ates_direct_GWh"].values) / dem_arr
    hp_el_f   = np.nan_to_num(df["hp_elec_GWh"].values) / dem_arr          # P_el
    hp_evap_f = np.nan_to_num(df["hp_GWh"].values) / dem_arr - hp_el_f     # Q_evap
    gas_f     = np.nan_to_num(df["gas_GWh"].values) / dem_arr

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(df)), 5))
    ax.bar(x, geo_f,     color="tab:blue",   label="Geo")
    ax.bar(x, ates_f,    bottom=geo_f,       color="tab:orange", label="ATES direct")
    b = geo_f + ates_f
    ax.bar(x, hp_evap_f, bottom=b,           color="tab:green",  label="HP source heat")
    b = b + hp_evap_f
    ax.bar(x, hp_el_f,   bottom=b,           color="tab:green",  hatch="//",
           edgecolor="white", label="HP electricity (grid)")
    b = b + hp_el_f
    ax.bar(x, gas_f,     bottom=b,           color="tab:red",    label="Gas")

    def _seg_label(vals, bottoms):
        for xi, v, bo in zip(x, vals, bottoms):
            if v > 0.015:
                ax.text(xi, bo + v / 2.0, f"{v:.2f}",
                        ha="center", va="center", fontsize=8)
    _seg_label(geo_f,     np.zeros_like(geo_f))
    _seg_label(ates_f,    geo_f)
    _seg_label(hp_evap_f, geo_f + ates_f)
    _seg_label(hp_el_f,   geo_f + ates_f + hp_evap_f)
    _seg_label(gas_f,     geo_f + ates_f + hp_evap_f + hp_el_f)

    # Renewable total (P_el excluded) above each bar.
    res_total = geo_f + ates_f + hp_evap_f
    for xi, r, tot in zip(x, res_total, res_total + hp_el_f + gas_f):
        ax.text(xi, tot + 0.015, f"RES {r:.2f}", ha="center", va="bottom", fontsize=8)

    for i in range(1, len(df)):
        if df["_gd_order"].iloc[i] != df["_gd_order"].iloc[i - 1]:
            ax.axvline(i - 0.5, color="0.8", linewidth=1, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Share of demand met")
    ax.set_ylim(0, 1.12)
    ax.set_title("Energy share per component  (HP split into source heat and grid electricity)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save(fig, "res_per_combination_yang")

    fig_data_sheets["Fig RES data"] = pd.DataFrame({
        "tag": df["tag"].values,
        "config": df["config"].values,
        "GD_target": df["GD_target"].values,
        "HP_POWER_EL": df["HP_POWER_EL"].values,
        "ratio_ATES_HP": df["ratio_ATES_HP"].values,
        "Geo frac": geo_f,
        "ATES direct frac": ates_f,
        "HP source heat frac": hp_evap_f,
        "HP electricity frac": hp_el_f,
        "Gas frac": gas_f,
        "Total renewable frac (excl. HP elec)": res_total,
        "Balance check": res_total + hp_el_f + gas_f,
    })


    # --- Figure: Carbon Abatement Cost (CAC) per combination ------------------
    # CAC = (LCOH_config - LCOH_ref) / (CI_ref - CI_config), in euro/kgCO2.
    # LCOH   = CO2-INCLUSIVE system LCOH from LCOE_calc_Yang (euro/kWh) -> shifts
    #          CAC down by the carbon price vs a CO2-free CAC; intentional.
    # CI     = system carbon intensity of delivered heat = total_CO2 / demand
    #          [kgCO2/kWh]. Positive (CI_ref - CI_config) => real abatement.
    # Every config is referenced to the gas-only baseline G WITHIN the same G/D
    # group -> CAC is the cumulative abatement cost vs fossil, not the marginal
    # cost of each added component. GGAH gives one CAC per HP power.
    # G is its own reference and is skipped (matches David's Fig. 4 caption).

    CAC_REF      = {"GG": "G", "GGA": "G", "GGAH": "G"}
    CONFIG_COLOR = {"GG": "tab:blue", "GGA": "tab:orange", "GGAH": "tab:green"}

    # Carbon intensity of delivered heat [kgCO2/kWh] for every row.
    with np.errstate(divide="ignore", invalid="ignore"):
        df["CI_kg_per_kWh"] = (df["total_CO2_t"] * 1000.0) / (df["demand_GWh"] * 1e6)

    # Reference lookup: only single-row configs (G/GG/GGA) are ever references,
    # so exactly one row each per G/D group.
    ref_index = {(r["GD_target"], r["config"]): r for _, r in df.iterrows()
                 if r["config"] in ("G", "GG", "GGA")}

    cac_vals      = np.full(len(df), np.nan)
    lcoh_ref_vals = np.full(len(df), np.nan)
    ci_ref_vals   = np.full(len(df), np.nan)
    for pos, (_, row) in enumerate(df.iterrows()):
        ref_cfg = CAC_REF.get(row["config"])
        if ref_cfg is None:
            continue
        ref = ref_index.get((row["GD_target"], ref_cfg))
        if ref is None:
            continue
        lcoh_ref_vals[pos] = ref["system_lcoh_yang"]
        ci_ref_vals[pos]   = ref["CI_kg_per_kWh"]
        d_lcoh = row["system_lcoh_yang"] - ref["system_lcoh_yang"]
        d_ci   = ref["CI_kg_per_kWh"] - row["CI_kg_per_kWh"]   # + = abatement
        cac_vals[pos] = d_lcoh / d_ci if abs(d_ci) > 1e-9 else np.nan

    df["CAC_eur_per_kg"] = cac_vals

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(df)), 5))
    ax.bar(x, cac_vals, color=[CONFIG_COLOR.get(c, "0.5") for c in df["config"]])

    finite = np.isfinite(cac_vals)
    span = (np.nanmax(cac_vals[finite]) - np.nanmin(cac_vals[finite])) if finite.any() else 1.0
    pad = 0.02 * (span if span > 0 else 1.0)
    for xi, v in zip(x, cac_vals):
        if np.isfinite(v):
            ax.text(xi, v + (pad if v >= 0 else -pad), f"{v:.2f}",
                    ha="center", va=("bottom" if v >= 0 else "top"), fontsize=8)

    ax.axhline(0.0, color="k", linewidth=0.8)
    for i in range(1, len(df)):
        if df["_gd_order"].iloc[i] != df["_gd_order"].iloc[i - 1]:
            ax.axvline(i - 0.5, color="0.8", linewidth=1, zorder=1)


    from matplotlib.patches import Patch
    drawn = [c for c in CONFIG_GRID if c in set(df["config"][finite])]
    if drawn:
        ax.legend(handles=[Patch(facecolor=CONFIG_COLOR.get(c, "0.5"), label=c)
                           for c in drawn], title="Configuration")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Carbon Abatement Cost (\u20ac/kgCO\u2082)")
    ax.set_title("CAC per combination  (LCOH = LCOE_calc_Yang)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "cac_per_combination_yang")

    fig_data_sheets["Fig CAC data"] = pd.DataFrame({
        "tag": df["tag"].values,
        "config": df["config"].values,
        "GD_target": df["GD_target"].values,
        "HP_POWER_EL": df["HP_POWER_EL"].values,
        "ratio_ATES_HP": df["ratio_ATES_HP"].values,
        "reference config": [CAC_REF.get(c, "") for c in df["config"].values],
        "LCOH [euro/kWh]": df["system_lcoh_yang"].values,
        "LCOH_ref [euro/kWh]": lcoh_ref_vals,
        "CI [kgCO2/kWh]": df["CI_kg_per_kWh"].values,
        "CI_ref [kgCO2/kWh]": ci_ref_vals,
        "delta_LCOH [euro/kWh]": df["system_lcoh_yang"].values - lcoh_ref_vals,
        "delta_CI [kgCO2/kWh]": ci_ref_vals - df["CI_kg_per_kWh"].values,
        "CAC [euro/kgCO2]": cac_vals,
    })
    # --- Figure: Carbon intensity of delivered heat (emissions per kWh) -------
    # CI = total system CO2 / heat delivered to demand -> plotted gCO2/kWh.
    # Same x-axis as RES/CAC, but every config has a value (G included).
    # CI_kg_per_kWh was already computed in the CAC block above -> reuse it.
    CI_COLOR = {"G": "tab:red", "GG": "tab:blue",
                "GGA": "tab:orange", "GGAH": "tab:green"}
    ci_g = df["CI_kg_per_kWh"].values * 1000.0      # gCO2/kWh

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(df)), 5))
    ax.bar(x, ci_g, color=[CI_COLOR.get(c, "0.5") for c in df["config"]])

    finite = np.isfinite(ci_g)
    span = (np.nanmax(ci_g[finite]) - np.nanmin(ci_g[finite])) if finite.any() else 1.0
    pad = 0.02 * (span if span > 0 else 1.0)
    for xi, v in zip(x, ci_g):
        if np.isfinite(v):
            ax.text(xi, v + (pad if v >= 0 else -pad), f"{v:.1f}",
                    ha="center", va=("bottom" if v >= 0 else "top"), fontsize=8)

    ax.axhline(0.0, color="k", linewidth=0.8)
    for i in range(1, len(df)):
        if df["_gd_order"].iloc[i] != df["_gd_order"].iloc[i - 1]:
            ax.axvline(i - 0.5, color="0.8", linewidth=1, zorder=1)

    drawn = [c for c in CONFIG_GRID if c in set(df["config"][finite])]
    if drawn:
        ax.legend(handles=[Patch(facecolor=CI_COLOR.get(c, "0.5"), label=c)
                           for c in drawn], title="Configuration")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Carbon intensity of delivered heat (gCO\u2082/kWh)")
    ax.set_title("Emissions per kWh delivered")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "ci_per_combination_yang")

    fig_data_sheets["Fig CI data"] = pd.DataFrame({
        "tag": df["tag"].values,
        "config": df["config"].values,
        "GD_target": df["GD_target"].values,
        "HP_POWER_EL": df["HP_POWER_EL"].values,
        "ratio_ATES_HP": df["ratio_ATES_HP"].values,
        "total_CO2_t": df["total_CO2_t"].values,
        "demand_GWh": df["demand_GWh"].values,
        "CI [kgCO2/kWh]": df["CI_kg_per_kWh"].values,
        "CI [gCO2/kWh]": ci_g,
    })

    # --- Write the workbook: Summary + one sheet per figure's data -------------
    df_out = df.drop(columns=["_gd_order", "_cfg_order", "_hp_order"])
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="Summary", index=False)
        for sheet_name, fdf in fig_data_sheets.items():
            fdf.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"\nSaved sweep summary -> {out}  ({len(df)} runs; "
          f"sheets: Summary, {', '.join(fig_data_sheets.keys())})")

    return df, results


if __name__ == "__main__":
    main()