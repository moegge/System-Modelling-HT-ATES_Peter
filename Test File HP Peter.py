# -*- coding: utf-8 -*-
"""
Test_File_Peter_HPon.py
==================================================================
One simulation of the updated model (main2_Peter / ATES_obj_Peter) with the
heat pump ON at all times -- i.e. active on EVERY discharge hour.

Why "all times" == "every discharge hour": in calc_heat the loop only visits
timesteps where missing_energy > 0 (the ATES is being drawn on), and the whole
HP dispatch + output lives inside that loop. On charging/surplus hours the HP
code is never reached, so hp_on there does nothing. Charge-side HP is not
implemented. Passing hp_on = all True therefore runs the HP on every discharge
hour (mode B above the return temperature, mode D below it).

Run from the repo root (needs main2_Peter.py, ATES_obj_Peter.py, results_AXI_V2,
Predict_REFF_boostedregression.pkl, and the Amsterdam demand parquet).

    python Test_File_Peter_HPon.py
==================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main2_Peter import (geothermal, demand_class, gas_boiler, heat_pump_ATES,
                         system, economic_analysis, system_plot)
from ATES_obj_Peter import ATES_obj

if __name__ == "__main__":

    timestep = 3600  # [s]

    # --- District-heating components (same setup as the example script) --------
    demand = demand_class(T_in=75, T_out=55, example_demand="Amsterdam")
    gas    = gas_boiler()
    geo    = geothermal(power=5000, T_out=75)

    # --- Heat pump on the ATES discharge side ----------------------------------
    # power_el = compressor rating [kW_el]; delta_T_coldside = cooling below the
    # DHN return [K]. Both fixed -> constant COP. Tweak to taste.
    hp = heat_pump_ATES(power_el=1500, delta_T_coldside=20)

    ATES = ATES_obj([geo], max_V=300, thickness=40, kh=5, ani=4, T_ground=15, HP=hp)

    # Preferred order for plotting: sustainable source, storage, back-up.
    supply = [geo, ATES, gas]

    # --- HP dispatch: ON every hour (only the discharge hours actually use it) --
    hp_on = np.ones(len(demand.data), dtype=bool)

    # --- Run the simulation ----------------------------------------------------
    result, df_flow = system(demand, supply, len_timestep=timestep, hp_on=hp_on)

    # --- Plots -----------------------------------------------------------------
    system_plot(result, supply, demand, len_timestep=timestep, setting="demand_met")
    system_plot(result, supply, demand, len_timestep=timestep, setting="ordered")

    # --- Economics (LCOH per component) ----------------------------------------
    df_eco = economic_analysis(result, supply, incorporate_CO2=True)
    print("\nLCOH per component:")
    for i in range(len(df_eco)):
        print(f"  {df_eco.iloc[i, 0]:<16} = {round(df_eco.iloc[i].loc['LCOE'], 3)} euro/kWh")

    # --- Heat-pump diagnostics -------------------------------------------------
    GWh = 1e6
    mode = np.asarray(getattr(ATES, "mode", []), dtype=object)
    cop  = np.asarray(getattr(ATES, "COP", []), dtype=float)
    cop_active = cop[np.isfinite(cop) & (cop > 0)] if cop.size else np.array([])
    print("\nHeat-pump summary (HP on every discharge hour):")
    print(f"  HP heat delivered (Q_evap+P_el) : {np.nansum(getattr(ATES, 'output_HP', 0.0)) / GWh:8.3f} GWh")
    print(f"  HP electricity  (P_el)          : {np.nansum(getattr(ATES, 'P_el', 0.0)) / GWh:8.3f} GWh")
    print(f"  ATES direct HX (Q_dir)          : {result['ATES production'].sum() / GWh:8.3f} GWh")
    print(f"  Total heat to demand            : {result['Total production'].sum() / GWh:8.3f} GWh")
    if mode.size:
        print(f"  discharge hours -> mode A/B/D   : "
              f"{int((mode=='A').sum())} / {int((mode=='B').sum())} / {int((mode=='D').sum())}")
    if cop_active.size:
        print(f"  mean COP while running          : {cop_active.mean():.2f}")

    plt.show()