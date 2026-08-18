# -*- coding: utf-8 -*-
"""
Test_File_Peter_HPoff.py
==================================================================
Companion to Test_File_Peter_HPon.py: the SAME system (same demand profile, same
geothermal / ATES / gas) but with NO heat pump, so you can compare on vs off.

"HP off" here means HP=None -- no heat-pump object at all. That is the clean
baseline: the HP contributes nothing on discharge, AND the charging side reverts
to the plain return temperature (T_cold = demand.T_out, Factor_due_HP = 1). So the
difference between this run and the HP-on run is the FULL effect of the heat pump,
on both charging and discharging.

(If instead you wanted to keep the HP object but idle it, that is hp_on = all
False -- but note the mode-D override still force-runs the HP below the return
temperature, so that is not a truly "off" discharge. HP=None is the clean off.)

Run from the repo root (needs main2_Peter.py, ATES_obj_Peter.py, results_AXI_V2,
Predict_REFF_boostedregression.pkl, and the Amsterdam demand parquet).

    python Test_File_Peter_HPoff.py
==================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main2_Peter import (geothermal, demand_class, gas_boiler,
                         system, economic_analysis, system_plot)
from ATES_obj_Peter import ATES_obj

if __name__ == "__main__":

    timestep = 3600  # [s]

    # --- District-heating components (IDENTICAL to the HP-on file) -------------
    demand = demand_class(T_in=75, T_out=55, example_demand="Amsterdam")
    gas    = gas_boiler()
    geo    = geothermal(power=5000, T_out=75)

    # --- ATES with NO heat pump ------------------------------------------------
    ATES = ATES_obj([geo], max_V=300, thickness=40, kh=5, ani=4, T_ground=15, HP=None)

    # Preferred order for plotting: sustainable source, storage, back-up.
    supply = [geo, ATES, gas]

    # --- Run the simulation (no hp_on needed; HP is absent) --------------------
    result, df_flow = system(demand, supply, len_timestep=timestep)

    # --- Plots -----------------------------------------------------------------
    system_plot(result, supply, demand, len_timestep=timestep, setting="demand_met")
    system_plot(result, supply, demand, len_timestep=timestep, setting="ordered")

    # --- Economics (LCOH per component) ----------------------------------------
    df_eco = economic_analysis(result, supply, incorporate_CO2=True)
    print("\nLCOH per component:")
    for i in range(len(df_eco)):
        print(f"  {df_eco.iloc[i, 0]:<16} = {round(df_eco.iloc[i].loc['LCOE'], 3)} euro/kWh")

    # --- Same summary block as the HP-on file (HP terms are 0 here) ------------
    GWh = 1e6
    mode = np.asarray(getattr(ATES, "mode", []), dtype=object)
    cop  = np.asarray(getattr(ATES, "COP", []), dtype=float)
    cop_active = cop[np.isfinite(cop) & (cop > 0)] if cop.size else np.array([])
    print("\nHeat-pump summary (HP OFF):")
    print(f"  HP heat delivered (Q_evap+P_el) : {np.nansum(getattr(ATES, 'output_HP', 0.0)) / GWh:8.3f} GWh")
    print(f"  HP electricity  (P_el)          : {np.nansum(getattr(ATES, 'P_el', 0.0)) / GWh:8.3f} GWh")
    print(f"  ATES direct HX (Q_dir)          : {result['ATES production'].sum() / GWh:8.3f} GWh")
    print(f"  Total heat to demand            : {result['Total production'].sum() / GWh:8.3f} GWh")
    if mode.size:
        print(f"  discharge hours -> mode A/B/D   : "
              f"{int((mode=='A').sum())} / {int((mode=='B').sum())} / {int((mode=='D').sum())}")
    if cop_active.size:
        print(f"  mean COP while running          : {cop_active.mean():.2f}")

    # --- Extra: a few annual totals, handy for the on/off comparison -----------
    print("\nAnnual totals [GWh]:")
    for c in ["Demand", "Total production", "Geothermal well corrected",
              "ATES corrected", "Gas boiler corrected"]:
        if c in result:
            print(f"  {c:<28}: {result[c].sum() / GWh:8.3f}")
    print(f"  {'Unmet (Demand-Total, clip)':<28}: "
          f"{np.clip(result['Demand'] - result['Total production'], 0, None).sum() / GWh:8.3f}")

    # --- Timeseries export: per-source heat to demand, every timestep ----------
    hp_ts = result["Heat pump production"] if "Heat pump production" in result \
        else pd.Series(0.0, index=result.index)
    # No HP here -> hp_ts is all zeros, so ATES direct == ATES corrected.
    ates_direct = result["ATES corrected"] - hp_ts

    ts = pd.DataFrame({
        "Time (hours)": result["Time (hours)"].values,
        "Demand [kWh]": result["Demand"].values,
        "Geothermal [kWh]": result["Geothermal well corrected"].values,
        "ATES direct [kWh]": ates_direct.values,
        "Heat pump [kWh]": hp_ts.values,
        "Gas boiler [kWh]": result["Gas boiler corrected"].values,
    }).fillna(0.0)

    # Sanity check: the four sources should reconstruct demand on covered hours.
    ts["Sum sources [kWh]"] = (ts["Geothermal [kWh]"] + ts["ATES direct [kWh]"]
                               + ts["Heat pump [kWh]"] + ts["Gas boiler [kWh]"])

    with pd.ExcelWriter("timeseries_HPoff.xlsx", engine="openpyxl") as writer:
        ts.to_excel(writer, sheet_name="Per-source heat", index=False)
        # writer.sheets["Per-source heat"].freeze_panes = "A2"
    print(f"\nSaved per-source timeseries -> timeseries_HPoff.xlsx "
          f"({len(ts)} timesteps)")
    plt.show()