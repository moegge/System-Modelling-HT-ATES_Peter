# -*- coding: utf-8 -*-
"""
Result Test David Original LCOH vs GD and LCOH vs maxv.py
Producing Fig 6 & Fig 7
==================================================================
REFERENCE RUNS of three of David Geerts' paper figures against his UNTOUCHED
published code -- 'main2_publish.py' and 'ATES_obj_publish.py' -- so they can be
laid next to the equivalents produced by 'Test File GD Curve Peter.py'.

  STUDY A -- Fig. 6(a): system LCOH vs the G/D ratio
  STUDY A -- Fig. 6(b): system LCOH vs the renewable energy share (RES)
                        (SAME runs as 6a, only the x-axis changes)
  STUDY B -- Fig. 7   : HT-ATES LCOH, system LCOH and RES vs max_V

Every sweep setting is deliberately identical to the patched-code script: same
demand profiles, same G/D and max_V grids, same 7.4 MW doublet, same network
lengths, same aquifer parameters, same discount rate, same 60-year horizon. The
only thing that differs is which model code is imported, so any gap between the
two sets of figures is attributable to the code, not to the scenario.

WHY THIS FILE EXISTS SEPARATELY
-------------------------------
David's API differs from the patched one, so run_case() cannot be reused:
  * system()            has no hp_on / hp_dynamic_dispatch arguments
  * economic_analysis() has no len_timestep argument
  * calc_heat()         takes HP=... positionally, not hp_on=...
  * ATES_obj            has no discharge-side heat pump at all
So this file rebuilds the same component chain directly against his API.
GG and GGA only -- his code has no heat pump, so GGAH cannot be run here.

DEMAND DATA
-----------
His demand_class hardcodes OneDrive paths that will not exist on your machine,
and its `if demand_array == None` test raises on a numpy array. Rather than
editing his file, the parquet is read here and the demand object is populated
directly (see _make_demand). The column selection is copied verbatim from his
demand_class, except that the wanted column is picked by NAME instead of by
position -- his `[0,:]` works only because the stray 'Unnamed: 3' column happens
to sort last.

TWO PARAMETERS THAT DISAGREE WITH THE PAPER'S OWN TABLE 1
----------------------------------------------------------
In main2_publish.py the class defaults are:
    geothermal CO2_kg    = 27     but Table 1 says 12.5 kgCO2/MWh
    gas_boiler gas_price = 0.1    but Table 1 says 55 EUR/MWh = 0.055
His driver script (not shared) presumably overrides both. They are exposed here
as GEO_CO2_KG / GAS_PRICE and default to the TABLE 1 values, so this reference
run is comparable to the patched run. Set USE_DAVID_CLASS_DEFAULTS = True to see
what the untouched defaults do -- gas at 0.10 instead of 0.055 alone is worth
roughly 25-30 EUR/MWh on the gas component, so run it BOTH ways before drawing
any conclusion about the model code.

A NOTE ON THE ML MODEL
----------------------
predict_reff loads 'Predict_REFF_boostedregression.pkl' on EVERY call. On a
newer XGBoost than the one that wrote it, loading goes through the pickle
compatibility path and XGBoost warns about it -- see the printed banner for the
version actually in use. That warning applies equally to the patched run, so it
cannot explain a gap BETWEEN the two, but it is a live candidate for a gap
against the PUBLISHED figures. CACHE_ML_MODEL below loads the pickle once
instead of ~10^2 times, which speeds the sweep up and reduces the warning to a
single line; it does not change any prediction.

    python "Test File Reference David.py"
==================================================================
"""

import os
import io
import contextlib
import traceback
from functools import lru_cache
import sys

# main2_publish.py / ATES_obj_publish.py live one level up, in
# System-Modelling-HT-ATES_Peter; this file sits in the "Delft Case" subfolder.
_MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODEL_ROOT not in sys.path:
    sys.path.insert(0, _MODEL_ROOT)
os.chdir(_MODEL_ROOT)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd

# David's untouched modules.
import ATES_obj_publish as _ates_module
from main2_publish import (demand_class, geothermal, gas_boiler, system,
                           economic_analysis, LCOE_calc_Yang, CO2_emissions_calc)
from ATES_obj_publish import ATES_obj

# ================================================================== #
#  WHICH STUDIES TO RUN                                              #
# ================================================================== #

RUN_GD_STUDY   = True      # Fig. 6(a) and 6(b)
RUN_MAXV_STUDY = True      # Fig. 7

# ================================================================== #
#  SHARED SETUP -- identical to the patched-code script              #
# ================================================================== #

# The parquet holding the three Delft demand columns. Point this at your copy.
DEMAND_PARQUET = (r"C:\Users\0527831\PycharmProjects\System-Modelling-HT-ATES_Peter"
                  r"\Data_and_scripts\Demand_data\Warmtevraag_Delft_parquet")

# label -> (column to keep, columns to drop, network length [m], colour)
# The drop lists are copied verbatim from David's demand_class. 8 km / 15 km are
# from paper 4.2; 23 km for "Delft Total" is an assumption (= TU + City), the
# paper does not state it.
DEMAND_CASES = {
    "G/D = 1.62": dict(keep="Demand TUD",   drop=["Demand Total", "Demand OWD"],
                       network_m=8000.0,  color="tab:blue"),
    "G/D = 1.07": dict(keep="Demand OWD",   drop=["Demand Total", "Demand TUD"],
                       network_m=15000.0, color="tab:orange"),
    "G/D = 0.65": dict(keep="Demand Total", drop=["Demand OWD", "Demand TUD"],
                       network_m=23000.0, color="tab:green"),
}

# --- Configurations ---------------------------------------------------------
# label -> (component set, matplotlib linestyle)
CONFIG_VARIANTS = {
    "GG":  dict(use_ates=False, linestyle="-"),
    "GGA": dict(use_ates=True,  linestyle="--"),
}

# --- Temperatures and geothermal -------------------------------------------
DEMAND_T_IN  = 75.0              # [C] DHN supply
DEMAND_T_OUT = 55.0              # [C] DHN return / ATES cut-off
GEO_T_OUT    = 75.0              # [C]
FIXED_GEO_POWER_MW = 7.4         # [MW] the paper's doublet (study B only)

# --- ATES aquifer (paper 4.1.1) --------------------------------------------
ATES_MAX_V_BASE = 320.0          # [m3/h] base size, used by study A
ATES_THICKNESS  = 55.0           # [m]
ATES_KH         = 10.0           # [m/day]
ATES_ANI        = 5.0            # [-]  (kh 10 / kv 2)
ATES_T_GROUND   = 15.0           # [C]
ATES_POROSITY   = 0.3            # [-]
ATES_LIFETIME   = 30             # [yr]         # [-]

# --- Economics --------------------------------------------------------------
CO2_PRICE       = 150.0          # [euro/tonne]
DISC_RATE       = 0.05
LIFETIME_SYSTEM = 60             # [yr]
NETWORK_EUR_PER_M = 1157.0
NETWORK_OPEX_PERC = 0.02
LIFETIME_NETWORK  = 60

# ATES variable-opex convention. True = Table 1 method (1.389 kWh/m3 x
# electricity price x mean of injected and extracted volume). False = the
# Thiem-equation pumping model inside ATES_obj.calc_opex. NOTE his
# economic_analysis defaults to False; the patched run uses True, so True is set
# here for a like-for-like comparison.
OPEX_ATES_FIXED = True

# See the header note: David's class defaults disagree with his Table 1.
USE_DAVID_CLASS_DEFAULTS = False
GEO_CO2_KG = 12.5                # [kgCO2/MWh]  Table 1  (class default: 27)
GAS_PRICE  = 0.055               # [euro/kWh]   Table 1  (class default: 0.1)

# Load the recovery-efficiency pickle once instead of on every predict_reff call.
CACHE_ML_MODEL = True

# --- Output / behaviour -----------------------------------------------------
RESULTS_DIR  = "results reference david"
SAVE_FIGURES = True
QUIET_RUNS   = True
OPT_TOL      = 0.01

TIMESTEP = 3600                  # [s] hourly, per paper 3 section 6
HOURS_PER_YEAR = 8760

# ================================================================== #
#  STUDY A SETTINGS -- G/D sweep (Fig. 6a and 6b)                    #
# ================================================================== #

GD_SWEEP = [0.25, 0.5, 0.75, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
            1.6, 1.8, 2.0, 2.25, 2.5, 2.75, 3.0]
GD_CACHE_CSV = "gd_reference_data.csv"

# Paper axis ranges, so the replicas overlay the published figures directly.
FIG6A_XLIM = (0.2, 3.0)
FIG6A_YLIM = (80.0, 127.0)
FIG6B_XLIM = (0.15, 1.02)
FIG6B_YLIM = (80.0, 127.0)
# Set any to None to autoscale that axis instead.

# ================================================================== #
#  STUDY B SETTINGS -- max_V sweep (Fig. 7)                          #
# ================================================================== #

# Paper 4.3: "The size is varied between 30 and 350 m3/h."
MAXV_SWEEP = list(range(30, 351, 10))   # [m3/h]
MAXV_CACHE_CSV = "maxv_reference_data.csv"

MAXV_LCOH_YLIM = (70.0, 165.0)
MAXV_RES_YLIM  = (0.40, 1.02)

# ================================================================== #


def _install_model_cache():
    """
    Make ATES_obj_publish.predict_reff reuse one loaded model instead of
    re-reading the pickle on every call. Predictions are unchanged -- predict()
    does not mutate the estimator -- but the sweep runs markedly faster and the
    XGBoost pickle-compatibility warning is printed once instead of hundreds of
    times. Set CACHE_ML_MODEL = False to restore the original behaviour exactly.
    """
    import joblib as _joblib

    _orig_load = _joblib.load

    @lru_cache(maxsize=8)
    def _cached_load(path):
        return _orig_load(path)

    class _JoblibProxy:
        """Delegates everything to joblib except load(), which is memoised."""
        load = staticmethod(_cached_load)

        def __getattr__(self, name):
            return getattr(_joblib, name)

    _ates_module.joblib = _JoblibProxy()


def _make_demand(case):
    """
    Build a demand_class instance holding one Delft column.

    Constructed directly rather than through __init__, because his __init__ only
    reads its hardcoded paths and its `demand_array == None` test raises on a
    numpy array. Every attribute __init__ would set is set here.
    """
    df = pd.read_parquet(DEMAND_PARQUET)
    df = df.drop(columns=[c for c in case["drop"] if c in df.columns])
    if case["keep"] not in df.columns:
        raise KeyError(f"column {case['keep']!r} not in {DEMAND_PARQUET}; "
                       f"found {list(df.columns)}")
    data = np.asarray(df[case["keep"]].values, dtype=float) * 1000.0   # MWh -> kWh
    if len(data) != 8760:
        raise ValueError(f"expected 8760 hourly values, got {len(data)}")

    dem = demand_class.__new__(demand_class)
    dem.T_in = DEMAND_T_IN
    dem.T_out = DEMAND_T_OUT
    dem.len_timestep = 3600
    dem.type = "demand"
    dem.heat_cap = 4186
    dem.density = 997
    dem.data = data
    return dem


@lru_cache(maxsize=8)
def _annual_demand_kWh(case_label):
    """Annual demand [kWh] of one case. Cached: the parquet read is not free."""
    return float(np.sum(_make_demand(DEMAND_CASES[case_label]).data))


def _geo_power_for_gd(gd, annual_demand_kWh):
    """Nameplate G/D: GEO_POWER [kW] = gd * annual_demand / 8760."""
    return gd * annual_demand_kWh / HOURS_PER_YEAR


def _simulate(case, config_label, geo_power, max_V):
    """
    One simulation against David's code. Returns a flat result row, or None.
    config_label selects GG (geo + gas) or GGA (geo + ATES + gas).
    """
    variant = CONFIG_VARIANTS[config_label]
    demand = _make_demand(case)

    gas_kwargs = {} if USE_DAVID_CLASS_DEFAULTS else dict(gas_price=GAS_PRICE)
    geo_kwargs = {} if USE_DAVID_CLASS_DEFAULTS else dict(CO2_kg=GEO_CO2_KG)

    gas = gas_boiler(**gas_kwargs)
    geo = geothermal(power=geo_power, T_out=GEO_T_OUT, **geo_kwargs)

    if variant["use_ates"]:
        ates = ATES_obj([geo], max_V=float(max_V), thickness=ATES_THICKNESS,
                        porosity=ATES_POROSITY, kh=ATES_KH, ani=ATES_ANI,
                        T_ground=ATES_T_GROUND, lifetime=ATES_LIFETIME)
        supply = [geo, ates, gas]
    else:
        ates = None
        supply = [geo, gas]

    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(buf) if QUIET_RUNS else contextlib.nullcontext()
    try:
        with ctx:
            result, df_flow = system(demand, supply, len_timestep=3600)
            df_eco = economic_analysis(result, supply, disc_rate=DISC_RATE,
                                       incorporate_CO2=True, CO2_price=CO2_PRICE,
                                       opex_ATES_fixed=OPEX_ATES_FIXED)
            lcoh_sys = LCOE_calc_Yang(
                result, supply, df_eco, disc_rate=DISC_RATE,
                lifetime_system=LIFETIME_SYSTEM,
                capex_network=case["network_m"] * NETWORK_EUR_PER_M,
                opex_network_perc=NETWORK_OPEX_PERC,
                lifetime_network=LIFETIME_NETWORK)
            co2 = CO2_emissions_calc(result, supply, CO2_price=CO2_PRICE)
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        return None

    GWh = 1e6
    n = len(result)

    def _col(name):
        return np.nan_to_num(result[name].values) if name in result else np.zeros(n)

    # Geothermal heat routed into storage, so it is not double counted in RES.
    charge_kWh = np.zeros(n)
    if ates is not None:
        for s in ates.supplier:
            pct, prod = s.name + " percentage to storage", s.name + " production"
            if pct in result and prod in result:
                charge_kWh += np.nan_to_num((result[pct] * result[prod]).values)

    demand_GWh = result["Demand"].sum() / GWh
    geo_corr = _col("Geothermal well corrected")
    geo_to_demand_GWh = (geo_corr - charge_kWh).sum() / GWh
    ates_GWh = _col("ATES corrected").sum() / GWh
    gas_GWh = _col("Gas boiler corrected").sum() / GWh
    geo_prod_GWh = _col("Geothermal well production").sum() / GWh

    def _lcoe(name):
        return df_eco.at[name, "LCOE"] if name in df_eco.index else np.nan

    def _opex(name):
        return df_eco.at[name, "opex"] / 1e6 if name in df_eco.index else np.nan

    return {
        "demand_case":  None,           # filled by the caller
        "config":       config_label,
        "GEO_POWER_kW": geo_power,
        "ATES_MAX_V":   float(max_V) if ates is not None else np.nan,
        "ATES_LIFETIME": ATES_LIFETIME if ates is not None else np.nan,
        "network_m":    case["network_m"],
        "demand_GWh":   demand_GWh,
        "GD_achieved":  geo_prod_GWh / demand_GWh if demand_GWh else np.nan,
        "system_lcoh_yang": lcoh_sys,
        "geo_lcoh":     _lcoe("Geothermal well"),
        "ates_lcoh":    _lcoe("ATES"),
        "gas_lcoh":     _lcoe("Gas boiler"),
        "Reff":         float(getattr(ates, "Reff", np.nan)) if ates is not None else np.nan,
        "injected_volume_m3":  float(getattr(ates, "volume", np.nan))
                               if ates is not None else np.nan,
        "extracted_volume_m3": float(np.nansum(getattr(ates, "flow_extracted", np.nan)))
                               if ates is not None else np.nan,
        "geo_prod_GWh":      geo_prod_GWh,
        "geo_to_demand_GWh": geo_to_demand_GWh,
        "ates_direct_GWh":   ates_GWh,
        "gas_GWh":      gas_GWh,
        "unmet_GWh":    float(np.clip(result["Demand"] - result["Total production"],
                                      0, None).sum() / GWh),
        "total_CO2_t":  float(co2["CO2_emission [kg]"].sum() / 1000.0),
        # RES = renewable heat delivered to demand / total delivered. No heat
        # pump in this code, so it is geothermal-to-demand plus ATES.
        "RES":          ((geo_to_demand_GWh + ates_GWh) / demand_GWh
                         if demand_GWh else np.nan),
        "ates_capex_Meur": (ates.capex / 1e6) if ates is not None else np.nan,
        "ates_opex_Meur":  _opex("ATES"),
    }


def _load_cache(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  cache unreadable ({type(e).__name__}: {e}) -- starting fresh")
        return pd.DataFrame()


def _append_row(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", index=False,
                               header=not os.path.exists(path))


def _save(fig, results_dir, name):
    if SAVE_FIGURES:
        path = os.path.join(results_dir, name + ".png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved figure -> {path}")
    plt.close(fig)


def _optimum_table(df, x_col, value_col, value_label, group_cols, tol=OPT_TOL):
    """Per group: the x minimising `value_col`, plus the span within `tol`."""
    rows = []
    for key, sub in df.groupby(group_cols, sort=False):
        sub = sub.dropna(subset=[value_col]).sort_values(x_col)
        if sub.empty:
            continue
        y = sub[value_col].values * 1000.0
        x = sub[x_col].values
        i = int(np.argmin(y))
        within = x[y <= y[i] * (1.0 + tol)]
        row = dict(zip(group_cols if isinstance(group_cols, list) else [group_cols],
                       key if isinstance(key, tuple) else (key,)))
        row[f"optimal {x_col}"] = x[i]
        row[f"{x_col} within {tol:.0%} of min"] = f"{within.min():g} - {within.max():g}"
        row[f"min {value_label} [euro/MWh]"] = y[i]
        rows.append(row)
    return pd.DataFrame(rows)


# ================================================================== #
#  STUDY A -- G/D SWEEP  (paper Fig. 6a and 6b)                      #
# ================================================================== #

def run_gd_study(results_dir):
    cache_path = os.path.join(results_dir, GD_CACHE_CSV)
    n_runs = len(GD_SWEEP) * len(DEMAND_CASES) * len(CONFIG_VARIANTS)

    print("=" * 70)
    print("STUDY A -- system LCOH vs G/D and vs RES   (paper Fig. 6a, 6b)")
    print("=" * 70)
    print(f"  G/D points          : {len(GD_SWEEP)}  ({min(GD_SWEEP)} .. {max(GD_SWEEP)})")
    print(f"  configurations      : {', '.join(CONFIG_VARIANTS)}")
    print(f"  ATES max_V (fixed)  : {ATES_MAX_V_BASE:g} m3/h")
    print(f"  TOTAL RUNS          : {n_runs}")
    print(f"  cache               : {cache_path}")
    for case_label, case in DEMAND_CASES.items():
        a = _annual_demand_kWh(case_label)
        print(f"  {case_label:<12} {case['keep']:<14} {a / 1e6:7.2f} GWh/yr, "
              f"{a / 1e6 / 8.760:6.2f} MW avg, network {case['network_m'] / 1000:.0f} km")
    print("=" * 70)

    cache = _load_cache(cache_path)
    done = set()
    if not cache.empty:
        done = {(r["demand_case"], r["config"], round(float(r["GD_target"]), 4))
                for _, r in cache.iterrows()}
        print(f"  resuming: {len(done)} of {n_runs} runs already cached\n")

    counter = 0
    for case_label, case in DEMAND_CASES.items():
        annual = _annual_demand_kWh(case_label)
        for config_label in CONFIG_VARIANTS:
            for gd in GD_SWEEP:
                counter += 1
                if (case_label, config_label, round(float(gd), 4)) in done:
                    continue
                print(f"[{counter:>4}/{n_runs}] {case_label} | {config_label:<4} "
                      f"| G/D = {gd:.2f}", flush=True)
                row = _simulate(case, config_label,
                                _geo_power_for_gd(gd, annual), ATES_MAX_V_BASE)
                if row is None:
                    continue
                row["demand_case"] = case_label
                row["GD_target"] = gd
                _append_row(cache_path, row)
                print(f"          -> system {row['system_lcoh_yang'] * 1000:.1f} "
                      f"EUR/MWh | RES {row['RES']:.3f}", flush=True)

    df = _load_cache(cache_path)
    if df.empty:
        print("\nStudy A: no results -- nothing to plot.")
        return df

    df["_case_order"] = df["demand_case"].map({c: i for i, c in enumerate(DEMAND_CASES)})
    df["_cfg_order"] = df["config"].map({c: i for i, c in enumerate(CONFIG_VARIANTS)})
    df = (df.dropna(subset=["_case_order", "_cfg_order"])
            .sort_values(["_case_order", "_cfg_order", "GD_target"])
            .reset_index(drop=True))

    drift = (df["GD_achieved"] - df["GD_target"]).abs()
    if drift.max() > 1e-6:
        print(f"\n  WARNING: achieved G/D drifts from target by up to {drift.max():.4f}")

    print(f"\nStudy A: collected {len(df)} runs")
    _plot_fig6(df, results_dir, x_col="GD_target",
               xlabel=r"$G/D$", xlim=FIG6A_XLIM, ylim=FIG6A_YLIM,
               title="Fig. 6(a) replica -- David's published code",
               fname="reference_david_fig6a")
    _plot_fig6(df, results_dir, x_col="RES",
               xlabel="Renewable Energy Share (RES)",
               xlim=FIG6B_XLIM, ylim=FIG6B_YLIM,
               title="Fig. 6(b) replica -- David's published code",
               fname="reference_david_fig6b")

    opt = _optimum_table(df, "GD_target", "system_lcoh_yang", "system LCOH",
                         ["demand_case", "config"])
    if not opt.empty:
        print("\nOPTIMAL G/D PER CURVE (minimum of the system LCOH curve)")
        print(opt.to_string(index=False))
        print("  Paper reference: GG optimum between 1.4 and 1.6; "
              "GGA optimum between 1.1 and 1.3.")

    # Paper 5.3.2 quotes the GGA cost of the last few RES points at G/D = 1.07:
    # 81 -> 85 -> 104 EUR/MWh at 95%, 99% and 100% RES.
    print("\nSystem LCOH at RES targets (GGA), paper quotes 81 / 85 / 104 "
          "EUR/MWh at 95% / 99% / 100% for G/D = 1.07:")
    for case_label in DEMAND_CASES:
        sub = (df[(df["demand_case"] == case_label) & (df["config"] == "GGA")]
               .dropna(subset=["RES", "system_lcoh_yang"]).sort_values("RES"))
        if sub.empty:
            continue
        vals = [np.interp(t, sub["RES"].values,
                          sub["system_lcoh_yang"].values * 1000.0)
                for t in (0.95, 0.99, 1.00)]
        print(f"  {case_label:<12} " + " / ".join(f"{v:.1f}" for v in vals)
              + f"   (RES range covered: {sub['RES'].min():.2f} - "
                f"{sub['RES'].max():.2f})")

    wide_gd = df.pivot_table(index="GD_target", columns=["demand_case", "config"],
                             values="system_lcoh_yang") * 1000.0
    wide_gd.columns = [f"{c} | {v} [euro/MWh]" for c, v in wide_gd.columns]
    wide_res = df.pivot_table(index="GD_target", columns=["demand_case", "config"],
                              values="RES")
    wide_res.columns = [f"{c} | {v} [-]" for c, v in wide_res.columns]

    out = os.path.join(results_dir, "gd_reference_david_summary.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.drop(columns=["_case_order", "_cfg_order"]).to_excel(
            writer, sheet_name="All runs", index=False)
        wide_gd.reset_index().to_excel(writer, sheet_name="System LCOH", index=False)
        wide_res.reset_index().to_excel(writer, sheet_name="RES", index=False)
        if not opt.empty:
            opt.to_excel(writer, sheet_name="Optimum", index=False)
    print(f"Saved -> {out}")
    return df


def _plot_fig6(df, results_dir, x_col, xlabel, xlim, ylim, title, fname):
    """
    Fig. 6(a) and 6(b) share one plotting routine: identical curves, different
    x-axis (G/D or RES). Colour = demand case, style = configuration.

    Saved twice -- cropped to the paper's axis ranges (for shape) and autoscaled
    (for level), because a cropped axis silently hides curves leaving the frame,
    which is exactly what the GGA lines do at high G/D.
    """
    fig, ax = plt.subplots(figsize=(5.6, 4.8))

    for case_label, case in DEMAND_CASES.items():
        for config_label, variant in CONFIG_VARIANTS.items():
            sub = (df[(df["demand_case"] == case_label)
                      & (df["config"] == config_label)]
                   .dropna(subset=[x_col, "system_lcoh_yang"])
                   .sort_values(x_col))
            if sub.empty:
                continue
            ax.plot(sub[x_col].values, sub["system_lcoh_yang"].values * 1000.0,
                    color=case["color"], linestyle=variant["linestyle"], lw=1.4)

    handles = [Line2D([], [], color=c["color"], lw=1.4, label=lbl)
               for lbl, c in DEMAND_CASES.items()]
    handles += [Line2D([], [], color="k", lw=1.4, linestyle=v["linestyle"], label=lbl)
                for lbl, v in CONFIG_VARIANTS.items()]
    ax.legend(handles=handles, loc="upper left" if x_col == "RES" else "upper right",
              fontsize=8, framealpha=1.0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("System LCOH (\u20ac/MWh)")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()

    if SAVE_FIGURES:
        path = os.path.join(results_dir, fname + ".png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved figure -> {path}")
        ax.set_xlim(auto=True)
        ax.set_ylim(auto=True)
        ax.relim()
        ax.autoscale_view()
        path = os.path.join(results_dir, fname + "_uncropped.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved figure -> {path}")
    plt.close(fig)


# ================================================================== #
#  STUDY B -- max_V SWEEP  (paper Fig. 7)                            #
# ================================================================== #

def run_maxv_study(results_dir):
    cache_path = os.path.join(results_dir, MAXV_CACHE_CSV)
    geo_power = FIXED_GEO_POWER_MW * 1000.0
    n_runs = len(MAXV_SWEEP) * len(DEMAND_CASES)

    print("\n" + "=" * 70)
    print("STUDY B -- LCOH + RES vs HT-ATES max_V   (paper Fig. 7)")
    print("=" * 70)
    print(f"  max_V points        : {len(MAXV_SWEEP)}  "
          f"({min(MAXV_SWEEP)} .. {max(MAXV_SWEEP)} m3/h)")
    print(f"  configuration       : GGA")
    print(f"  geothermal (fixed)  : {FIXED_GEO_POWER_MW} MW -> base G/D per case")
    print(f"  TOTAL RUNS          : {n_runs}")
    print(f"  cache               : {cache_path}")
    print("=" * 70)

    cache = _load_cache(cache_path)
    done = set()
    if not cache.empty:
        done = {(r["demand_case"], round(float(r["MAXV_target"]), 4))
                for _, r in cache.iterrows()}
        print(f"  resuming: {len(done)} of {n_runs} runs already cached\n")

    counter = 0
    for case_label, case in DEMAND_CASES.items():
        for max_V in MAXV_SWEEP:
            counter += 1
            if (case_label, round(float(max_V), 4)) in done:
                continue
            print(f"[{counter:>3}/{n_runs}] {case_label} | max_V = {max_V:g} m3/h",
                  flush=True)
            row = _simulate(case, "GGA", geo_power, max_V)
            if row is None:
                continue
            row["demand_case"] = case_label
            row["MAXV_target"] = float(max_V)
            _append_row(cache_path, row)
            print(f"          -> system {row['system_lcoh_yang'] * 1000:.1f} | "
                  f"ATES {row['ates_lcoh'] * 1000:.1f} EUR/MWh | "
                  f"RES {row['RES']:.3f} | Reff {row['Reff']:.3f}", flush=True)

    df = _load_cache(cache_path)
    if df.empty:
        print("\nStudy B: no results -- nothing to plot.")
        return df

    df["_case_order"] = df["demand_case"].map({c: i for i, c in enumerate(DEMAND_CASES)})
    df = (df.dropna(subset=["_case_order"])
            .sort_values(["_case_order", "MAXV_target"]).reset_index(drop=True))

    print(f"\nStudy B: collected {len(df)} runs")
    print("Base G/D achieved per demand case (should be ~1.63 / 1.07 / 0.65):")
    print(df.groupby("demand_case")["GD_achieved"].first().to_string())

    _plot_fig7(df, results_dir)
    _plot_maxv_diagnostics(df, results_dir)

    opt_ates = _optimum_table(df, "MAXV_target", "ates_lcoh", "ATES LCOH",
                              ["demand_case"])
    opt_sys = _optimum_table(df, "MAXV_target", "system_lcoh_yang", "system LCOH",
                             ["demand_case"])
    if not opt_ates.empty:
        print("\nOPTIMAL max_V (minimum HT-ATES component LCOH)")
        print(opt_ates.to_string(index=False))
        print("  Paper reference: 100-200 m3/h for G/D = 1.07 and 0.65; "
              "a relatively low rate for G/D = 1.62.")

    wide_sys = df.pivot_table(index="MAXV_target", columns="demand_case",
                              values="system_lcoh_yang") * 1000.0
    wide_ates = df.pivot_table(index="MAXV_target", columns="demand_case",
                               values="ates_lcoh") * 1000.0
    wide_res = df.pivot_table(index="MAXV_target", columns="demand_case", values="RES")

    out = os.path.join(results_dir, "maxv_reference_david_summary.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.drop(columns=["_case_order"]).to_excel(writer, sheet_name="All runs",
                                                  index=False)
        wide_sys.reset_index().to_excel(writer, sheet_name="System LCOH", index=False)
        wide_ates.reset_index().to_excel(writer, sheet_name="ATES LCOH", index=False)
        wide_res.reset_index().to_excel(writer, sheet_name="RES", index=False)
        if not opt_sys.empty:
            opt_sys.to_excel(writer, sheet_name="Optimum system", index=False)
        if not opt_ates.empty:
            opt_ates.to_excel(writer, sheet_name="Optimum ATES", index=False)
    print(f"Saved -> {out}")
    return df


def _plot_fig7(df, results_dir):
    """Paper Fig. 7 layout, same styling as the patched-code figure."""
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax_res = ax.twinx()

    for case_label, case in DEMAND_CASES.items():
        sub = df[df["demand_case"] == case_label].sort_values("MAXV_target")
        if sub.empty:
            continue
        x = sub["MAXV_target"].values
        ax.plot(x, sub["system_lcoh_yang"].values * 1000.0,
                color=case["color"], linestyle="--", lw=1.5)
        ax.plot(x, sub["ates_lcoh"].values * 1000.0,
                color=case["color"], linestyle="-.", lw=1.5)
        ax_res.plot(x, sub["RES"].values, color=case["color"], linestyle="-", lw=1.5)

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
    ax.set_title("Fig. 7 replica -- GGA, David's published code", fontsize=9)
    fig.tight_layout()
    _save(fig, results_dir, "reference_david_fig7")


def _plot_maxv_diagnostics(df, results_dir):
    """
    Reff and injected volume vs max_V. Paper 5.3.2 blames the erratic low-rate
    behaviour on the recovery-efficiency model, so this panel says whether a
    wobble is physics or extrapolation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for case_label, case in DEMAND_CASES.items():
        sub = df[df["demand_case"] == case_label].sort_values("MAXV_target")
        if sub.empty:
            continue
        axes[0].plot(sub["MAXV_target"], sub["Reff"], "o-", ms=3,
                     color=case["color"], label=case_label)
        axes[1].plot(sub["MAXV_target"], sub["injected_volume_m3"] / 1e3, "o-",
                     ms=3, color=case["color"], label=case_label)
    axes[0].set_xlabel("max_V (m\u00b3/h)")
    axes[0].set_ylabel("Recovery efficiency Reff (-)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("max_V (m\u00b3/h)")
    axes[1].set_ylabel("Annual injected volume (thousand m\u00b3)")
    axes[1].grid(alpha=0.3)
    fig.suptitle("HT-ATES diagnostics -- David's code  "
                 "(paper: Reff 86.6% at G/D 1.62, 83.9% at 0.65, max_V = 320)",
                 fontsize=9)
    fig.tight_layout()
    _save(fig, results_dir, "reference_david_fig7_diagnostics")


# ================================================================== #

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, RESULTS_DIR)
    os.makedirs(results_dir, exist_ok=True)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)

    if CACHE_ML_MODEL:
        _install_model_cache()

    # Provenance banner: the recovery-efficiency pickle was written by an older
    # XGBoost, so record what is actually loading it.
    versions = []
    for mod in ("xgboost", "sklearn", "joblib", "numpy", "pandas"):
        try:
            versions.append(f"{mod} {__import__(mod).__version__}")
        except Exception:
            versions.append(f"{mod} ?")
    print("environment: " + " | ".join(versions))
    print("model code : main2_publish.py / ATES_obj_publish.py (untouched)")
    print("parameters : "
          + ("David's class defaults (geo CO2 27, gas 0.10 EUR/kWh)"
             if USE_DAVID_CLASS_DEFAULTS
             else f"Table 1 (geo CO2 {GEO_CO2_KG}, gas {GAS_PRICE} EUR/kWh)"))
    print("ATES opex  : "
          + ("Table 1 fixed method" if OPEX_ATES_FIXED else "Thiem pumping model")
          + "\n")

    df_gd = run_gd_study(results_dir) if RUN_GD_STUDY else pd.DataFrame()
    df_v = run_maxv_study(results_dir) if RUN_MAXV_STUDY else pd.DataFrame()
    return df_gd, df_v


if __name__ == "__main__":
    main()