# -*- coding: utf-8 -*-
"""
Test_Sweep_Peter.py
==================================================================
Parameter sweep of the UPDATED (main2_Peter / ATES_obj_Peter) model, NO heat pump.

Runs an identical set of one-parameter-at-a-time scenarios around a base case,
sums each run to annual totals [GWh], and writes:
    test_sweep_Peter.csv

Run BOTH this and Test_Sweep_David.py from the repo root (the folder with
main2_*.py, ATES_obj_*.py, results_AXI_V2, Predict_REFF_boostedregression.pkl).
Whichever you run second automatically prints a per-column comparison so you can
confirm the two models agree.

NOTE: the two no-HP engines are NOT expected to be bit-identical. Peter's
calc_heat recomputes the extraction temperature inside the over-delivery
correction (David re-reads a fixed index) and handles the depletion edge with
`continue` instead of advancing T_start. Small differences on the ATES / Gas /
Total columns are the refactor, not a bug; differences on the Demand or
Geothermal columns WOULD indicate a real problem. The comparison reports
magnitudes so you can judge.

    python Test_Sweep_Peter.py
==================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from main2_Peter import demand_class, geothermal, gas_boiler, ATES_obj, system, economic_analysis

MODEL_LABEL = "Peter"
OTHER_LABEL = "David"
REQUIRED_DATA = ["results_AXI_V2", "Predict_REFF_boostedregression.pkl"]

LEN_TIMESTEP = 3600      # [s]
N_HOURS      = 8760

# ------------------------------------------------------------------ #
#  Base case + one-at-a-time sweep values  (KEEP IDENTICAL in both files)
# ------------------------------------------------------------------ #
BASE = dict(
    annual_demand   = 50_000_000.0,  # [kWh/yr]
    demand_T_supply = 70.0,          # [C]
    demand_T_return = 40.0,          # [C]
    geo_power       = 5000.0,        # [kW]
    geo_T_out       = 90.0,          # [C]
    ates_max_V      = 100.0,         # [m3/h]
    ates_thickness  = 50.0,          # [m]
    ates_porosity   = 0.2,           # [-]
    ates_kh         = 5.0,           # [m/day]
    ates_ani        = 1.0,           # [-]
    ates_T_ground   = 25.0,          # [C]
)

SWEEP = {
    "geo_power":       [3000.0, 4000.0, 6000.0, 7000.0],
    "ates_max_V":      [50.0, 150.0, 200.0],
    "ates_T_ground":   [15.0, 20.0, 30.0],
    "geo_T_out":       [80.0, 100.0],
    "ates_thickness":  [30.0, 70.0],
    "ates_porosity":   [0.15, 0.30],
    "ates_kh":         [2.0, 10.0],
    "annual_demand":   [40_000_000.0, 60_000_000.0],
    "demand_T_return": [35.0, 45.0],
}

PARAM_COLS = ["annual_demand", "demand_T_supply", "demand_T_return", "geo_power",
              "geo_T_out", "ates_max_V", "ates_thickness", "ates_porosity",
              "ates_kh", "ates_ani", "ates_T_ground"]

OUTPUT_COLS = ["Demand", "Total production",
               "Geothermal well production", "Geothermal well corrected",
               "ATES production", "ATES corrected",
               "Gas boiler production", "Gas boiler corrected",
               "Heat pump production", "Unmet",
               "ATES Reff", "ATES injected volume", "ATES extracted volume"]


def build_scenarios():
    """Base case + each single-parameter override. Deterministic and identical."""
    scen = [("base", "-", dict(BASE))]
    for param, values in SWEEP.items():
        for v in values:
            cfg = dict(BASE)
            cfg[param] = v
            scen.append((f"{param}={v:g}", param, cfg))
    return scen


def build_demand_profile(annual_demand):
    """Deterministic winter-peaked hourly demand summing to annual_demand [kWh/yr]."""
    hours    = np.arange(N_HOURS)
    seasonal = 0.5 * (1.0 + np.cos(2.0 * np.pi * hours / N_HOURS))
    daily    = 0.15 * np.sin(2.0 * np.pi * (hours % 24) / 24.0 - np.pi / 2.0)
    shape    = np.clip(0.25 + seasonal + daily, 0.0, None)
    return (shape / shape.sum() * annual_demand).tolist()


def build_system(cfg):
    """Fresh demand + geo + ATES(no HP) + gas for one scenario."""
    demand = demand_class(T_in=cfg["demand_T_supply"], T_out=cfg["demand_T_return"],
                          demand_array=build_demand_profile(cfg["annual_demand"]))
    demand.data = np.asarray(demand.data, dtype=float)   # avoid list-repetition in calc_flow

    geo = geothermal(power=cfg["geo_power"], T_out=cfg["geo_T_out"])

    ates = ATES_obj(supplier=[geo],
                    max_V=cfg["ates_max_V"], thickness=cfg["ates_thickness"],
                    porosity=cfg["ates_porosity"], kh=cfg["ates_kh"], ani=cfg["ates_ani"],
                    T_ground=cfg["ates_T_ground"], HP=None)

    gas = gas_boiler()
    return demand, [geo, ates, gas]


def run_one(cfg):
    """Run system() once and return annual totals [GWh] + ATES diagnostics."""
    demand, supply = build_system(cfg)

    result, df_flow = system(demand, supply,
                             len_timestep=LEN_TIMESTEP,
                             time_horizon=N_HOURS,
                             control=None,
                             hp_on=None)   # no HP intent

    GWh = 1e6
    col = lambda name: (result[name].sum() / GWh) if name in result else np.nan
    ates = next(i for i in supply if i.name == "ATES")

    return {
        "Demand":                     col("Demand"),
        "Total production":           col("Total production"),
        "Geothermal well production": col("Geothermal well production"),
        "Geothermal well corrected":  col("Geothermal well corrected"),
        "ATES production":            col("ATES production"),
        "ATES corrected":             col("ATES corrected"),
        "Gas boiler production":      col("Gas boiler production"),
        "Gas boiler corrected":       col("Gas boiler corrected"),
        "Heat pump production":       col("Heat pump production"),
        "Unmet":                      np.clip(result["Demand"] - result["Total production"], 0, None).sum() / GWh,
        "ATES Reff":                  float(getattr(ates, "Reff", np.nan)),
        "ATES injected volume":       float(getattr(ates, "volume", np.nan)),
        "ATES extracted volume":      float(np.nansum(getattr(ates, "flow_extracted", np.nan))),
    }


def compare():
    """If the other model's sweep CSV exists, print a per-output-column diff."""
    this_csv  = f"test_sweep_{MODEL_LABEL}.csv"
    other_csv = f"test_sweep_{OTHER_LABEL}.csv"
    if not os.path.exists(other_csv):
        print(f"\n[compare] {other_csv} not found yet -- run Test_Sweep_{OTHER_LABEL}.py too;")
        print(f"[compare] whichever you run second prints the comparison.")
        return

    a = pd.read_csv(this_csv).set_index("scenario")
    b = pd.read_csv(other_csv).set_index("scenario")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    # 1) sanity check: the swept inputs must be identical across both files
    bad = [c for c in PARAM_COLS
           if c in a and c in b and not np.allclose(a[c].astype(float), b[c].astype(float))]
    if bad:
        print("\n[compare] WARNING: input parameters differ between the two sweeps:", bad)
        print("[compare] The systems are NOT identical; fix before trusting the diff.")
    else:
        print(f"\n[compare] input parameters identical across both sweeps ({len(common)} scenarios). OK")

    # 2) per-output-column differences
    print(f"\n[compare] {MODEL_LABEL} vs {OTHER_LABEL}   (annual totals [GWh], Reff [-], volumes [m3])")
    print(f"{'column':<30}{'max|abs|':>14}{'max|rel|':>12}   worst-scenario")
    print("-" * 84)
    for c in OUTPUT_COLS:
        if c not in a or c not in b:
            continue
        va, vb = a[c].astype(float), b[c].astype(float)
        adiff = (va - vb).abs()
        rdiff = adiff / va.abs().clip(lower=1e-9)
        worst = adiff.idxmax()
        print(f"{c:<30}{adiff.max():>14.6f}{rdiff.max():>11.3%}   {worst}")
    print("-" * 84)
    print("Demand / Geothermal columns should be ~0 diff. Small ATES / Gas / Total")
    print("differences are the calc_heat refactor (see header). Large diffs = investigate.")


def main():
    missing = [f for f in REQUIRED_DATA if not os.path.exists(f)]
    if missing:
        print("ERROR: required data files not found in the current directory:")
        for f in missing:
            print("   -", f)
        print("Run this from the repository root where those files live.")
        sys.exit(1)

    scenarios = build_scenarios()
    print(f"[{MODEL_LABEL}] running {len(scenarios)} scenarios (no HP)...")
    rows = []
    for label, swept, cfg in scenarios:
        row = {"scenario": label, "swept": swept}
        row.update({k: cfg[k] for k in PARAM_COLS})
        try:
            row.update(run_one(cfg))
            print(f"[{MODEL_LABEL}] {label:<24} ok")
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"[{MODEL_LABEL}] {label:<24} ERROR: {row['error']}")
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = f"test_sweep_{MODEL_LABEL}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[{MODEL_LABEL}] wrote {out_csv} ({len(df)} rows)")

    compare()
    return df


if __name__ == "__main__":
    main()