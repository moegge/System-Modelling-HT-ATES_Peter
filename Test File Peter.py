# -*- coding: utf-8 -*-
"""
Test_File_Peter_HPidle.py
==================================================================
Same system as Test_File_Peter.py, but with an HP OBJECT ATTACHED to the ATES
and hp_on = all False. This is NOT a clean "off": it exposes the two couplings
flagged earlier.

  * CHARGING side: `Factor_due_HP` and the cold-well init in system() fire on the
    mere PRESENCE of an HP object (storage_obj.HP != None), so the injected volume
    is divided by (geoT - (returnT - delta))/(geoT - returnT)  (= 1.4 with the
    values below) and the cold well is initialised at returnT - delta. This happens
    even though hp_on is all False.

  * DISCHARGE side: calc_heat's mode-D override (hp_override_below_cutoff=True,
    hard-coded via the default in system()) still force-runs the HP whenever the
    mean extraction temperature drops below the DHN return. So you WILL see
    nonzero HP production and mode-D timesteps despite hp_on = all False.

Compare this run against:
  * Test_File_Peter.py   (HP=None, the true no-HP baseline)
  * Test_File_David.py   (original model, no HP)

HOW TO RUN
----------
Run from the repository root (needs main2_Peter.py, ATES_obj_Peter.py,
results_AXI_V2, Predict_REFF_boostedregression.pkl):

    python Test_File_Peter_HPidle.py

Writes test_totals_Peter_HPidle.csv and test_result_Peter_HPidle.parquet.
==================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from main2_Peter import (demand_class, geothermal, gas_boiler, ATES_obj,
                         heat_pump_ATES, system, economic_analysis)

MODEL_LABEL = "Peter_HPidle"     # HP attached, but hp_on = all False
REQUIRED_DATA = ["results_AXI_V2", "Predict_REFF_boostedregression.pkl"]

# ------------------------------------------------------------------ #
#  Deterministic shared inputs  (KEEP IDENTICAL to the other test files)
# ------------------------------------------------------------------ #
LEN_TIMESTEP   = 3600          # [s]
N_HOURS        = 8760
ANNUAL_DEMAND  = 50_000_000.0  # [kWh/yr]

DEMAND_T_SUPPLY = 70.0   # demand.T_in  -> DHN supply temperature (HP condenser sink)
DEMAND_T_RETURN = 40.0   # demand.T_out -> DHN return temperature (= ATES T_cutoff)

GEO_POWER   = 5000.0     # [kW]
GEO_T_OUT   = 90.0       # [C]

ATES_MAX_V     = 100.0   # [m3/h]
ATES_THICKNESS = 50.0    # [m]
ATES_POROSITY  = 0.2     # [-]
ATES_KH        = 5.0     # [m/day]
ATES_ANI       = 1.0     # [-]
ATES_T_GROUND  = 25.0    # [C]

# HP parameters (illustrative - change to taste)
HP_POWER_EL     = 1000.0  # [kW_el] compressor rating (caps Q_evap per hour)
HP_DELTA_T_COLD = 20.0    # [K] cold-side cooling below the DHN return


def build_demand_profile():
    """Deterministic winter-peaked hourly heat demand summing to ANNUAL_DEMAND."""
    hours    = np.arange(N_HOURS)
    seasonal = 0.5 * (1.0 + np.cos(2.0 * np.pi * hours / N_HOURS))
    daily    = 0.15 * np.sin(2.0 * np.pi * (hours % 24) / 24.0 - np.pi / 2.0)
    shape    = np.clip(0.25 + seasonal + daily, 0.0, None)
    profile  = shape / shape.sum() * ANNUAL_DEMAND
    return profile.tolist()


def build_system():
    """Identical demand + supply chain, but with an HP object attached to the ATES."""
    demand = demand_class(T_in=DEMAND_T_SUPPLY, T_out=DEMAND_T_RETURN,
                          demand_array=build_demand_profile())
    demand.data = np.asarray(demand.data, dtype=float)

    geo = geothermal(power=GEO_POWER, T_out=GEO_T_OUT)

    hp = heat_pump_ATES(power_el=HP_POWER_EL, delta_T_coldside=HP_DELTA_T_COLD)

    ates = ATES_obj(supplier=[geo],
                    max_V=ATES_MAX_V, thickness=ATES_THICKNESS,
                    porosity=ATES_POROSITY, kh=ATES_KH, ani=ATES_ANI,
                    T_ground=ATES_T_GROUND,
                    HP=hp)                          # <-- HP object attached (idle)

    gas = gas_boiler()

    supply = [geo, ates, gas]
    return demand, supply


def summarise(result, supply):
    GWh = 1e6
    rows = {}
    rows["Demand"]           = result["Demand"].sum() / GWh
    rows["Total production"] = result["Total production"].sum() / GWh
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

    hp_on = np.zeros(N_HOURS, dtype=bool)            # HP intent: OFF every hour

    result, df_flow = system(demand, supply,
                             len_timestep=LEN_TIMESTEP,
                             time_horizon=N_HOURS,
                             control=None,
                             hp_on=hp_on)

    totals = summarise(result, supply)

    ates = next(i for i in supply if i.name == "ATES")
    diag = {
        "ATES Reff":                  getattr(ates, "Reff", np.nan),
        "ATES injected volume [m3]":  float(getattr(ates, "volume", np.nan)),
        "ATES extracted volume [m3]": float(np.nansum(getattr(ates, "flow_extracted", np.nan))),
    }
    # HP / mode diagnostics -- this is where the "idle HP" effect shows up.
    mode = np.asarray(getattr(ates, "mode", []), dtype=object)
    diag.update({
        "HP production [GWh]": np.nansum(getattr(ates, "output_HP", 0.0)) / 1e6,
        "HP elec P_el [GWh]":  np.nansum(getattr(ates, "P_el", 0.0)) / 1e6,
        "timesteps mode A":    int(np.sum(mode == 'A')) if mode.size else 0,
        "timesteps mode B":    int(np.sum(mode == 'B')) if mode.size else 0,
        "timesteps mode D":    int(np.sum(mode == 'D')) if mode.size else 0,
    })

    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print("=" * 64)
    print(f"  MODEL: {MODEL_LABEL}  (HP attached, hp_on=all False)  totals [GWh]")
    print("=" * 64)
    print(totals.to_string())
    print("-" * 64)
    for k, v in diag.items():
        print(f"  {k:<28}: {v:,.4f}" if isinstance(v, float) else f"  {k:<28}: {v}")
    print("=" * 64)
    print("NOTE: HP production > 0 and mode D > 0 are EXPECTED here despite hp_on=all False,")
    print("      because of the mode-D override; and 'ATES injected volume' is smaller than")
    print("      the HP=None run because Factor_due_HP (=1.4) shrinks the charge volume.")

    totals.to_csv(f"test_totals_{MODEL_LABEL}.csv", header=True)
    try:
        result.to_parquet(f"test_result_{MODEL_LABEL}.parquet")
    except Exception as e:
        result.to_csv(f"test_result_{MODEL_LABEL}.csv", index=False)
        print(f"(parquet unavailable, wrote CSV instead: {e})")

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