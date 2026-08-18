# -*- coding: utf-8 -*-
"""
Test_File_Peter.py
==================================================================
One-year production run of the *updated* model (main2_Peter + ATES_obj_Peter)
with the HEAT PUMP DISABLED, so it can be compared 1:1 against the original
model (Test_File_David.py builds the identical system).

WHY NO HP OBJECT IS ATTACHED
----------------------------
"HP off" is only truly equivalent to the old model when NO heat-pump object is
attached to the ATES. If an HP object were present, calc_heat's mode-D override
(hp_override_below_cutoff=True) would still force-run the HP whenever the mean
extraction temperature drops below the DHN return, even with hp_on all-False.
So for a clean baseline we pass HP=None here.

HOW TO RUN
----------
Run from the repository root, i.e. the folder that also contains:
    main2_Peter.py, ATES_obj_Peter.py,
    results_AXI_V2                       (parquet with the DOE curves)
    Predict_REFF_boostedregression.pkl   (the Reff surrogate)
Those data files are NOT created here; they must already be present.

    python Test_File_Peter.py

Prints per-source annual totals and writes:
    test_totals_Peter.csv     (annual totals, for diffing)
    test_result_Peter.parquet (full hourly result)
==================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

# Headless-safe plotting (main2 imports matplotlib.pyplot at import time)
import matplotlib
matplotlib.use("Agg")

# `ATES_obj` is re-exported by main2_Peter (`from ATES_obj_Peter import ATES_obj`),
# so importing it from main2_Peter guarantees the test uses the SAME ATES class.
from main2_Peter import demand_class, geothermal, gas_boiler, ATES_obj, system, economic_analysis

MODEL_LABEL = "Peter"
REQUIRED_DATA = ["results_AXI_V2", "Predict_REFF_boostedregression.pkl"]

# ------------------------------------------------------------------ #
#  Deterministic shared inputs  (KEEP IDENTICAL in Test_File_David.py)
# ------------------------------------------------------------------ #
LEN_TIMESTEP   = 3600          # [s]
N_HOURS        = 8760
ANNUAL_DEMAND  = 50_000_000.0  # [kWh/yr] ~50 GWh, like the paper demo cases

DEMAND_T_SUPPLY = 70.0   # demand.T_in  -> DHN supply temperature (HP condenser sink)
DEMAND_T_RETURN = 40.0   # demand.T_out -> DHN return temperature (= ATES T_cutoff)

GEO_POWER   = 5000.0     # [kW] constant geothermal baseload
GEO_T_OUT   = 90.0       # [C]

ATES_MAX_V     = 100.0   # [m3/h]
ATES_THICKNESS = 50.0    # [m]
ATES_POROSITY  = 0.2     # [-]
ATES_KH        = 5.0     # [m/day]
ATES_ANI       = 1.0     # [-]
ATES_T_GROUND  = 25.0    # [C]


def build_demand_profile():
    """Deterministic winter-peaked hourly heat demand summing to ANNUAL_DEMAND.

    No randomness -> byte-identical in both test files. Returns a python list
    (demand_class does `demand_array == None`, fine for a list, crashes for ndarray).
    """
    hours    = np.arange(N_HOURS)
    seasonal = 0.5 * (1.0 + np.cos(2.0 * np.pi * hours / N_HOURS))   # 1 winter, 0 summer
    daily    = 0.15 * np.sin(2.0 * np.pi * (hours % 24) / 24.0 - np.pi / 2.0)
    shape    = np.clip(0.25 + seasonal + daily, 0.0, None)
    profile  = shape / shape.sum() * ANNUAL_DEMAND                   # [kWh/hour]
    return profile.tolist()


def build_system():
    """Construct the identical demand + supply chain used by both models."""
    demand = demand_class(T_in=DEMAND_T_SUPPLY, T_out=DEMAND_T_RETURN,
                          demand_array=build_demand_profile())
    # calc_flow() later does `self.data * 3600000`; a list would be *repeated*,
    # so force a numpy array now (constructor stored the list we passed).
    demand.data = np.asarray(demand.data, dtype=float)

    geo = geothermal(power=GEO_POWER, T_out=GEO_T_OUT)

    ates = ATES_obj(supplier=[geo],
                    max_V=ATES_MAX_V, thickness=ATES_THICKNESS,
                    porosity=ATES_POROSITY, kh=ATES_KH, ani=ATES_ANI,
                    T_ground=ATES_T_GROUND,
                    HP=None)                        # <-- HP disabled

    gas = gas_boiler()

    supply = [geo, ates, gas]                       # order matters for the "corrected" cascade
    return demand, supply


def summarise(result, supply):
    """Collapse the hourly result into annual totals [GWh]."""
    GWh = 1e6
    rows = {}
    rows["Demand"]               = result["Demand"].sum() / GWh
    rows["Total production"]     = result["Total production"].sum() / GWh
    if "Heat pump production" in result:
        rows["Heat pump production"] = result["Heat pump production"].sum() / GWh
    for i in supply:
        for col in (i.name + " production", i.name + " corrected"):
            if col in result:
                rows[col] = result[col].sum() / GWh
    rows["Unmet (Demand-Total, clipped)"] = \
        np.clip(result["Demand"] - result["Total production"], 0, None).sum() / GWh
    return pd.Series(rows, name=MODEL_LABEL)


def main():
    missing = [f for f in REQUIRED_DATA if not os.path.exists(f)]
    if missing:
        print("ERROR: required data files not found in the current directory:")
        for f in missing:
            print("   -", f)
        print("Run this script from the repository root where those files live.")
        sys.exit(1)

    demand, supply = build_system()

    result, df_flow = system(demand, supply,
                             len_timestep=LEN_TIMESTEP,
                             time_horizon=N_HOURS,
                             control=None,
                             hp_on=None)             # HP intent: none

    totals = summarise(result, supply)

    ates = next(i for i in supply if i.name == "ATES")
    diag = {
        "ATES Reff":                  getattr(ates, "Reff", np.nan),
        "ATES injected volume [m3]":  float(getattr(ates, "volume", np.nan)),
        "ATES extracted volume [m3]": float(np.nansum(getattr(ates, "flow_extracted", np.nan))),
    }

    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print("=" * 64)
    print(f"  MODEL: {MODEL_LABEL}  (HP disabled)   annual totals [GWh]")
    print("=" * 64)
    print(totals.to_string())
    print("-" * 64)
    for k, v in diag.items():
        print(f"  {k:<28}: {v:,.4f}")
    print("=" * 64)

    totals.to_csv(f"test_totals_{MODEL_LABEL}.csv", header=True)
    try:
        result.to_parquet(f"test_result_{MODEL_LABEL}.parquet")
    except Exception as e:
        result.to_csv(f"test_result_{MODEL_LABEL}.csv", index=False)
        print(f"(parquet unavailable, wrote CSV instead: {e})")

    # Optional economics; guarded so a failure never hides the production result.
    try:
        eco = economic_analysis(result, supply)
        print("\nEconomic analysis:")
        print(eco.to_string())
        eco.to_csv(f"test_eco_{MODEL_LABEL}.csv")
    except Exception as e:
        print(f"\n(economic_analysis skipped: {type(e).__name__}: {e})")

    return result, df_flow, totals


if __name__ == "__main__":
    main()