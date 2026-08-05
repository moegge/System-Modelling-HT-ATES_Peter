# -*- coding: utf-8 -*-
"""
sweep_HTATES.py
==================================================================
Small parameter sweep built on the *proven* Test_File_David pattern.

The demand + supply chain are constructed exactly as in Test_File_David.py
(synthetic demand via `demand_array`, geothermal + ATES(no HP) + gas boiler),
so this runs against main2_publish / ATES_obj_publish without touching the
broken hard-coded demand paths.

It loops over a grid of (geo power, ATES max_V) and records annual totals,
ATES Reff, renewable share, and (guarded) LCOH for each combination.

HOW TO RUN
----------
Run from the repository root, i.e. the folder that also contains:
    main2_publish.py, ATES_obj_publish.py,
    results_AXI_V2                       (parquet with the DOE curves)
    Predict_REFF_boostedregression.pkl   (the Reff surrogate)

    python sweep_HTATES.py

Writes:
    sweep_results.csv   (one row per (geo power, ATES max_V) combination)
==================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")   # keep before importing the model (main2 imports pyplot)

# ATES_obj is re-exported by main2_publish (`from ATES_obj_publish import ATES_obj`).
from main2_publish import (
    demand_class,
    geothermal,
    gas_boiler,
    ATES_obj,
    system,
    economic_analysis,
    LCOE_calc_Yang,
)

REQUIRED_DATA = ["results_AXI_V2", "Predict_REFF_boostedregression.pkl"]

# ------------------------------------------------------------------ #
#  Fixed inputs (identical to Test_File_David.py)
# ------------------------------------------------------------------ #
LEN_TIMESTEP    = 3600           # [s]
N_HOURS         = 8760
ANNUAL_DEMAND   = 50_000_000.0   # [kWh/yr] ~50 GWh

DEMAND_T_SUPPLY = 70.0           # demand.T_in
DEMAND_T_RETURN = 40.0           # demand.T_out (= ATES T_cutoff)

GEO_T_OUT       = 90.0           # [C]

ATES_THICKNESS  = 50.0           # [m]
ATES_POROSITY   = 0.2            # [-]
ATES_KH         = 5.0            # [m/day]
ATES_ANI        = 1.0            # [-]
ATES_T_GROUND   = 25.0           # [C]

# ------------------------------------------------------------------ #
#  Sweep axes  (the "few values" to run over -- edit freely)
# ------------------------------------------------------------------ #
GEO_POWERS = [3000.0, 5000.0, 7000.0]   # [kW]  geothermal baseload
ATES_SIZES = [50.0, 100.0, 150.0]       # [m3/h] ATES max_V

# District-heating network, only used for the system LCOH (LCOE_calc_Yang)
NETWORK_LENGTH_KM  = 23
NETWORK_COST_PER_M = 1157        # euro / m
NETWORK_OPEX_PERC  = 0.02
NETWORK_LIFETIME   = 60
SYSTEM_LIFETIME    = 60
DISC_RATE          = 0.05

RES_LIB = ["Geothermal well corrected", "ATES corrected",
           "Heat pump corrected", "Solar boiler corrected"]

GWh = 1e6


def build_demand_profile():
    """Deterministic winter-peaked hourly demand summing to ANNUAL_DEMAND (list, len 8760)."""
    hours    = np.arange(N_HOURS)
    seasonal = 0.5 * (1.0 + np.cos(2.0 * np.pi * hours / N_HOURS))
    daily    = 0.15 * np.sin(2.0 * np.pi * (hours % 24) / 24.0 - np.pi / 2.0)
    shape    = np.clip(0.25 + seasonal + daily, 0.0, None)
    profile  = shape / shape.sum() * ANNUAL_DEMAND
    return profile.tolist()


def build_system(geo_power, ates_max_v):
    """Construct the demand + supply chain for one grid point (David pattern)."""
    demand = demand_class(T_in=DEMAND_T_SUPPLY, T_out=DEMAND_T_RETURN,
                          demand_array=build_demand_profile())
    demand.data = np.asarray(demand.data, dtype=float)   # avoid list-repetition in calc_flow

    geo = geothermal(power=geo_power, T_out=GEO_T_OUT)

    ates = ATES_obj(supplier=[geo],
                    max_V=ates_max_v, thickness=ATES_THICKNESS,
                    porosity=ATES_POROSITY, kh=ATES_KH, ani=ATES_ANI,
                    T_ground=ATES_T_GROUND,
                    HP=None)                              # HP disabled

    gas = gas_boiler()
    return demand, [geo, ates, gas]


def res_share(result, demand):
    """Renewable share of total demand, discounting energy charged into storage."""
    res = 0.0
    for col in result.columns:
        if any(tag in col for tag in RES_LIB):
            res += result[col].sum()
            try:
                to_storage = (result[col.replace("corrected", "percentage to storage")]
                              * result[col.replace(" corrected", " production")]).sum()
                if not np.isnan(to_storage):
                    res -= to_storage
            except Exception:
                pass
    return res / np.sum(demand.data)


def evaluate(geo_power, ates_max_v):
    """Run one grid point; return a dict of results (never raises)."""
    row = {"Geo_power_kW": geo_power, "ATES_max_V": ates_max_v}
    try:
        demand, supply = build_system(geo_power, ates_max_v)

        result, _ = system(demand, supply, len_timestep=LEN_TIMESTEP,
                            time_horizon=N_HOURS, control=None)

        # --- physical results (these always exist if system() ran) ---
        row["Demand_GWh"]     = result["Demand"].sum() / GWh
        row["Total_prod_GWh"] = result["Total production"].sum() / GWh
        for i in supply:
            col = i.name + " corrected"
            if col in result:
                row[f"{i.name}_GWh"] = result[col].sum() / GWh
        row["Unmet_GWh"] = np.clip(result["Demand"] - result["Total production"],
                                   0, None).sum() / GWh

        ates = next(i for i in supply if i.name == "ATES")
        row["ATES_Reff"]  = float(getattr(ates, "Reff", np.nan))
        row["RES_share"]  = res_share(result, demand)

        # --- economics (guarded: record NaN rather than crash the sweep) ---
        try:
            df_eco = economic_analysis(result, supply, disc_rate=DISC_RATE)
            for i in supply:
                if i.name in df_eco.index:
                    row[f"LCOE_{i.name}"] = df_eco.at[i.name, "LCOE"]

            try:
                capex_network = NETWORK_LENGTH_KM * NETWORK_COST_PER_M * 1000
                row["System_LCOH"] = LCOE_calc_Yang(
                    result, supply, df_eco, disc_rate=DISC_RATE,
                    lifetime_system=SYSTEM_LIFETIME, capex_network=capex_network,
                    opex_network_perc=NETWORK_OPEX_PERC, lifetime_network=NETWORK_LIFETIME)
            except Exception as e:
                row["System_LCOH"] = np.nan
                print(f"    LCOE_calc_Yang skipped: {type(e).__name__}: {e}", flush=True)

        except Exception as e:
            print(f"    economic_analysis skipped: {type(e).__name__}: {e}", flush=True)

        row["status"] = "ok"

    except Exception as e:
        row["status"] = f"FAILED: {type(e).__name__}: {e}"
        print(f"    run FAILED: {type(e).__name__}: {e}", flush=True)

    return row


def main():
    missing = [f for f in REQUIRED_DATA if not os.path.exists(f)]
    if missing:
        print("ERROR: required data files not found in the current directory:")
        for f in missing:
            print("   -", f)
        print("Run this script from the repository root where those files live.")
        sys.exit(1)

    rows = []
    total = len(GEO_POWERS) * len(ATES_SIZES)
    n = 0
    for gp in GEO_POWERS:
        for av in ATES_SIZES:
            n += 1
            print(f"[{n}/{total}] geo_power={gp:.0f} kW  ATES_max_V={av:.0f} m3/h", flush=True)
            rows.append(evaluate(gp, av))

    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print("\n" + "=" * 72)
    print(df.to_string(index=False))
    print("=" * 72)

    df.to_csv("sweep_results.csv", index=False)
    print("\nwrote sweep_results.csv")
    return df


if __name__ == "__main__":
    # To parallelise later: move `evaluate` to take a (gp, av) tuple and replace the
    # loop in main() with  mp.Pool(mp.cpu_count()).starmap(evaluate, product(...)).
    main()