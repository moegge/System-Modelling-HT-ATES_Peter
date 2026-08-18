# -*- coding: utf-8 -*-
"""
Test_File_Peter_17_08.py
==================================================================
One simulation of the updated model (main2_Peter / ATES_obj_Peter), wrapped in
run_case() so it can be run directly (single run) OR imported and called from a
sweep script (see sweep.py).

Toggle use_hp selects between two configurations:
  * use_hp = True  -> heat pump on the ATES discharge side, active on EVERY
                      discharge hour (mode B above the return temperature,
                      mode D below it). No HP object is created when False.
  * use_hp = False -> NO heat-pump object is attached to the ATES at all.
                      This is a clean baseline equivalent to the original model:
                      calc_heat's mode-D override is guarded by 'self.HP is not
                      None', so with no HP nothing can turn it on.

Why "all times" == "every discharge hour": in calc_heat the loop only visits
timesteps where missing_energy > 0 (the ATES is being drawn on), and the whole
HP dispatch + output lives inside that loop. On charging/surplus hours the HP
code is never reached, so hp_on there does nothing. Charge-side HP is not
implemented. Passing hp_on = all True therefore runs the HP on every discharge
hour.

Run from the repo root (needs main2_Peter.py, ATES_obj_Peter.py, results_AXI_V2,
Predict_REFF_boostedregression.pkl, and the Amsterdam demand parquet).

    python Test_File_Peter_17_08.py            # single run, plots + Excel
    from Test_File_Peter_17_08 import run_case  # driven by sweep.py
==================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main2_Peter import (geothermal, demand_class, gas_boiler, heat_pump_ATES,
                         system, economic_analysis, system_plot, CO2_emissions_calc)
from ATES_obj_Peter import ATES_obj

# ================================================================== #
#  DEFAULT CONFIGURATION  --  the single-run defaults.               #
#  A sweep overrides any of these by passing them to run_case().     #
# ================================================================== #

# --- Master toggle: run WITH or WITHOUT the discharge-side heat pump -------
USE_HP = True            # True  -> HP created and attached to the ATES
                         # False -> NO HP object created at all (clean baseline)

# --- Simulation ------------------------------------------------------------
TIMESTEP = 3600          # [s]

# --- District-heating demand ----------------------------------------------
DEMAND_EXAMPLE = "Amsterdam"
DEMAND_T_IN    = 75      # [C] DHN supply temperature (HP condenser sink)
DEMAND_T_OUT   = 55      # [C] DHN return temperature (= ATES cutoff / HX floor)

# --- Geothermal baseload (also the ATES charging source) ------------------
GEO_POWER = 5000         # [kW]
GEO_T_OUT = 75           # [C]

# --- ATES aquifer ----------------------------------------------------------
ATES_MAX_V     = 300     # [m3/h]
ATES_THICKNESS = 40      # [m]
ATES_KH        = 5       # [m/day]
ATES_ANI       = 4       # [-]
ATES_T_GROUND  = 15      # [C]

# --- Heat pump (only used if USE_HP is True) -------------------------------
HP_POWER_EL         = 1500   # [kW_el] fixed compressor rating
HP_DELTA_T_COLDSIDE = 20     # [K] cooling below the DHN return -> fixed cold-well T

# --- CO2 price for the economics -------------------------------------------
CO2_PRICE = 70           # [euro/ton]

# ================================================================== #


def run_case(USE_HP=USE_HP, TIMESTEP=TIMESTEP,
             DEMAND_EXAMPLE=DEMAND_EXAMPLE, DEMAND_T_IN=DEMAND_T_IN, DEMAND_T_OUT=DEMAND_T_OUT,
             GEO_POWER=GEO_POWER, GEO_T_OUT=GEO_T_OUT,
             ATES_MAX_V=ATES_MAX_V, ATES_THICKNESS=ATES_THICKNESS, ATES_KH=ATES_KH,
             ATES_ANI=ATES_ANI, ATES_T_GROUND=ATES_T_GROUND,
             HP_POWER_EL=HP_POWER_EL, HP_DELTA_T_COLDSIDE=HP_DELTA_T_COLDSIDE,
             CO2_PRICE=CO2_PRICE,
             OUTFILE=None, tag="",
             make_plots=False, write_excel=True):
    """
    Run one configuration. Any argument left at its default reproduces the
    single-run config; a sweep passes only the knobs it varies.

    OUTFILE     : Excel path. None -> auto 'timeseries_HPon[/off][_tag].xlsx'.
    tag         : suffix on the auto filename so sweep runs don't overwrite.
    make_plots  : show the two system_plot figures (keep False in a sweep).
    write_excel : write the 5-sheet workbook (False in a sweep that only wants
                  the returned numbers).

    Returns a dict of headline results so a sweep can collect rows.
    """

    # --- Output filename (auto-suffixed by config) -----------------------------
    if OUTFILE is None:
        OUTFILE = f"timeseries_{'HPon' if USE_HP else 'HPoff'}{('_' + tag) if tag else ''}.xlsx"

    timestep = TIMESTEP

    # --- District-heating components ------------------------------------------
    demand = demand_class(T_in=DEMAND_T_IN, T_out=DEMAND_T_OUT,
                          example_demand=DEMAND_EXAMPLE)
    gas    = gas_boiler()
    geo    = geothermal(power=GEO_POWER, T_out=GEO_T_OUT)

    # --- Heat pump on the ATES discharge side (only if enabled) ---------------
    # power_el = compressor rating [kW_el]; delta_T_coldside = cooling below the
    # DHN return [K]. Both fixed -> constant COP.
    if USE_HP:
        hp = heat_pump_ATES(power_el=HP_POWER_EL, delta_T_coldside=HP_DELTA_T_COLDSIDE)
    else:
        hp = None            # NO heat-pump object is created at all

    ATES = ATES_obj([geo], max_V=ATES_MAX_V, thickness=ATES_THICKNESS,
                    kh=ATES_KH, ani=ATES_ANI, T_ground=ATES_T_GROUND, HP=hp)

    # Preferred order for plotting: sustainable source, storage, back-up.
    supply = [geo, ATES, gas]

    # --- HP dispatch: ON every hour (only the discharge hours actually use it) --
    # With no HP, hp_on=None -> calc_heat keeps the HP off and the mode-D override
    # is itself guarded by 'self.HP is not None', so nothing can turn it on.
    hp_on = np.ones(len(demand.data), dtype=bool) if USE_HP else None

    # --- Run the simulation ----------------------------------------------------
    result, df_flow = system(demand, supply, len_timestep=timestep, hp_on=hp_on)

    # --- Plots -----------------------------------------------------------------
    if make_plots:
        system_plot(result, supply, demand, len_timestep=timestep, setting="demand_met")
        system_plot(result, supply, demand, len_timestep=timestep, setting="ordered")

    # --- Economics (LCOH per component) ----------------------------------------
    df_eco = economic_analysis(result, supply, incorporate_CO2=True)
    print("\nLCOH per component:")
    for i in range(len(df_eco)):
        print(f"  {df_eco.iloc[i, 0]:<16} = {round(df_eco.iloc[i].loc['LCOE'], 3)} euro/kWh")
    # --- Heat-pump diagnostics (console) ---------------------------------------
    GWh = 1e6
    mode = np.asarray(getattr(ATES, "mode", []), dtype=object)
    cop  = np.asarray(getattr(ATES, "COP", []), dtype=float)
    cop_active = cop[np.isfinite(cop) & (cop > 0)] if cop.size else np.array([])
    mean_cop = float(cop_active.mean()) if cop_active.size else np.nan

    print("\nHeat-pump summary (HP on every discharge hour):" if USE_HP
          else "\nBaseline summary (no heat pump):")
    print(f"  HP condenser (Q_evap+P_el)      : {np.nansum(getattr(ATES, 'output_HP', 0.0)) / GWh:8.3f} GWh")
    print(f"  HP electricity  (P_el)          : {np.nansum(getattr(ATES, 'P_el', 0.0)) / GWh:8.3f} GWh")
    print(f"  ATES subsystem (direct+HP)      : {result['ATES production'].sum() / GWh:8.3f} GWh")
    print(f"  Total heat to demand            : {result['Total production'].sum() / GWh:8.3f} GWh")
    if mode.size:
        print(f"  discharge hours -> mode A/B/D   : "
              f"{int((mode=='A').sum())} / {int((mode=='B').sum())} / {int((mode=='D').sum())}")
    if cop_active.size:
        print(f"  mean COP while running          : {mean_cop:.2f}")

    # --- Timeseries export: per-source heat + HP/ATES diagnostics --------------
    # __main__ level (NOT nested under the COP block), so it always runs.
    n = len(result)

    hp_ts = result["Heat pump production"] if "Heat pump production" in result \
            else pd.Series(0.0, index=result.index)
    ates_direct = result["ATES corrected"] - hp_ts

    def _arr(attr, fill=0.0):
        a = np.asarray(getattr(ATES, attr, np.full(n, fill)), dtype=float)
        return a if a.size == n else np.full(n, fill, dtype=float)

    mode_arr = np.asarray(getattr(ATES, "mode", np.full(n, "off", dtype=object)),
                          dtype=object)
    if mode_arr.size != n:
        mode_arr = np.full(n, "off", dtype=object)

    T_extract = _arr("T_extract")
    T_inject  = _arr("T_inject", fill=demand.T_out)
    cop_arr   = _arr("COP", fill=np.nan)
    P_el      = _arr("P_el")
    out_evap  = _arr("output_evap")
    out_dir   = _arr("output_dir")
    out_HP    = _arr("output_HP")
    flow_ext  = _arr("flow_extracted")
    flow_inj  = _arr("flow_injected")

    charge_kWh = np.zeros(n)
    for s in ATES.supplier:
        c_pct, c_prod = s.name + " percentage to storage", s.name + " production"
        if c_pct in result and c_prod in result:
            charge_kWh += np.nan_to_num((result[c_pct] * result[c_prod]).values)

    ates_prod = (result["ATES production"].values
                 if "ATES production" in result else np.zeros(n))
    with np.errstate(divide="ignore", invalid="ignore"):
        cop_check = np.where(P_el > 0, out_HP / P_el, np.nan)

    geo_corr      = np.nan_to_num(result["Geothermal well corrected"].values)
    ates_dir_corr = np.nan_to_num(ates_direct.values)
    hp_corr       = np.nan_to_num(hp_ts.values)
    gas_corr      = np.nan_to_num(result["Gas boiler corrected"].values)
    sum_src       = geo_corr + ates_dir_corr + hp_corr + gas_corr

    ts = pd.DataFrame({
        "Time (hours)": result["Time (hours)"].values,
        "Demand [kWh]": result["Demand"].values,
        "Geothermal [kWh]": geo_corr,
        "ATES direct [kWh]": ates_dir_corr,
        "ATES direct split [kWh]": out_dir,
        "Heat pump [kWh]": hp_corr,
        "Gas boiler [kWh]": gas_corr,
        "Sum sources [kWh]": sum_src,
        "Mode": mode_arr,
        "T_extract [C]": T_extract,
        "T_inject [C]": T_inject,
        "COP [-]": cop_arr,
        "COP from HP/P_el [-]": cop_check,
        "HP evap/source [kWh]": out_evap,
        "P_el compressor [kWh]": P_el,
        "HP condenser [kWh]": out_HP,
        "HP identity resid [kWh]": out_HP - (out_evap + P_el),
        "ATES production [kWh]": ates_prod,
        "Split sum [kWh]": out_dir + out_evap + P_el,
        "Split residual [kWh]": (out_dir + out_evap + P_el) - ates_prod,
        "Flow extracted [m3]": flow_ext,
        "Charge to ATES [kWh]": charge_kWh,
        "Charge volume [m3]": flow_inj,
    })

    # --- Consistency checks: cycle totals that should reconcile to ~0 ----------
    def _sum(x):
        return float(np.nansum(x))

    check_defs = [
        ("Split sum vs ATES production",
         out_dir + out_evap + P_el, ates_prod, True,
         "Energy balance on the ATES subsystem (direct HX + HP condenser)."),
        ("HP condenser vs evap + P_el",
         out_HP, out_evap + P_el, True,
         "First law on the heat pump (per-hour = 'HP identity resid' column)."),
        ("HP condenser vs P_el x COP",
         out_HP, np.where(np.isfinite(cop_arr), P_el * cop_arr, 0.0), True,
         "COP consistency, energy-weighted (COP is a ratio, summed via P_el x COP)."),
        ("HP condenser vs 'Heat pump production'",
         out_HP, hp_corr, True,
         "Diagnostics array equals the result column used by plots/economics."),
        ("Total Sum of sources vs Demand",
         sum_src, result["Demand"].values, True,
         "Backup fills demand; residual = unmet/reconstruction gap over the year. Total Geo production included"),
        ("Sum of sources vs Demand",
         (geo_corr - charge_kWh) + ates_dir_corr + hp_corr + gas_corr,
         result["Demand"].values, True,
         "Sources-to-demand (geo charging removed) vs demand; residual = unmet demand. Geo production directly going to Demand included(should be ~0)."),
        ("ATES direct: corrected vs raw split",
         ates_dir_corr, out_dir, False,
         "INFO, not zero: system() demand-clipping vs raw _energy_split output."),
        ("'HP identity resid' column sum",
         out_HP - (out_evap + P_el), np.zeros(n), True,
         "Direct sum of the 'HP identity resid [kWh]' output column; should be ~0."),
        ("'Split residual' column sum",
         (out_dir + out_evap + P_el) - ates_prod, np.zeros(n), True,
         "Direct sum of the 'Split residual [kWh]' output column; should be ~0."),
    ]

    rows = []
    for name, a, b, is_zero, note in check_defs:
        a = np.nan_to_num(np.asarray(a, dtype=float))
        b = np.nan_to_num(np.asarray(b, dtype=float))
        resid = a - b
        sa, sb = _sum(a), _sum(b)
        sum_resid = _sum(resid)
        sum_abs = _sum(np.abs(resid))
        status = ("PASS" if sum_abs < 1e-3 else "CHECK") if is_zero else "info"
        rows.append({
            "Check": name,
            "Sum A [GWh]": sa / 1e6,
            "Sum B [GWh]": sb / 1e6,
            "Difference A-B [kWh]": sum_resid,
            "Sum |per-hour resid| [kWh]": sum_abs,
            "Status": status,
            "Note": note,
        })

    cop_dev = cop_check - cop_arr
    cop_abs = float(np.nansum(np.abs(cop_dev)))
    cop_max = float(np.nanmax(np.abs(cop_dev))) if np.isfinite(cop_dev).any() else 0.0
    rows.append({
        "Check": "COP column vs HP/P_el reconstruction",
        "Sum A [GWh]": np.nan, "Sum B [GWh]": np.nan, "Difference A-B [kWh]": np.nan,
        "Sum |per-hour resid| [kWh]": cop_abs,
        "Status": "PASS" if cop_abs < 1e-6 else "CHECK",
        "Note": f"Dimensionless (COP units). max |dev| = {cop_max:.2e}.",
    })

    cons = pd.DataFrame(rows, columns=[
        "Check", "Sum A [GWh]", "Sum B [GWh]", "Difference A-B [kWh]",
        "Sum |per-hour resid| [kWh]", "Status", "Note"])

    # --- Economics: pull the key numbers out of df_eco -------------------------
    # df_eco (from economic_analysis above, incorporate_CO2=True) has per-component
    # capex, opex (CO2 cost already folded in), generated-discounted and LCOE.
    # NOTE: with the HP enabled, the HP's own capex/opex are folded INTO the ATES
    # row by economic_analysis -> the HP-detail rows below are broken out for
    # visibility and are already counted inside the ATES component figures.
    hp_obj         = getattr(ATES, "HP", None)
    P_el_total_kWh = float(np.nansum(getattr(ATES, "P_el", 0.0)))
    hp_rating_kW   = ((getattr(hp_obj, "rated_power", None) or getattr(hp_obj, "power_el", np.nan))
                      if hp_obj is not None else np.nan)
    hp_elec_price  = getattr(hp_obj, "elec_price", np.nan) if hp_obj is not None else np.nan
    hp_capex_eur   = (hp_rating_kW * hp_obj.capex)      if hp_obj is not None else 0.0
    hp_fixopex_eur = (hp_obj.fixed_opex * hp_rating_kW) if hp_obj is not None else 0.0
    hp_elec_cost   = P_el_total_kWh * hp_elec_price     if hp_obj is not None else 0.0

    try:
        co2_df = CO2_emissions_calc(result, supply, CO2_price=CO2_PRICE)
    except Exception as e:
        print(f"(CO2 breakdown skipped: {type(e).__name__}: {e})")
        co2_df = None

    def _co2_kg(k):  return co2_df.at[k, "CO2_emission [kg]"] if (co2_df is not None and k in co2_df.index) else np.nan
    def _co2_eur(k): return co2_df.at[k, "Cost_CO2"]         if (co2_df is not None and k in co2_df.index) else np.nan

    eco_tbl = pd.DataFrame({
        "Component": list(df_eco.index),
        "CAPEX [Meuro]":              [df_eco.at[k, "capex"] / 1e6 for k in df_eco.index],
        "OPEX (incl. CO2) [Meuro/yr]":[df_eco.at[k, "opex"]  / 1e6 for k in df_eco.index],
        "Generated discounted [GWh]": [(df_eco.at[k, "generated discounted"] / 1e6)
                                       if pd.notna(df_eco.at[k, "generated discounted"]) else np.nan
                                       for k in df_eco.index],
        "LCOH [euro/kWh]":            [df_eco.at[k, "LCOE"] for k in df_eco.index],
        "CO2 [t/yr]":                 [_co2_kg(k) / 1000 for k in df_eco.index],
        "CO2 cost [euro/yr]":         [_co2_eur(k) for k in df_eco.index],
    })

    # --- Heat-pump broken out as its own row (for visibility) ------------------
    if hp_obj is not None:
        hp_row = pd.DataFrame([{
            "Component": "Heat pump (in ATES)",
            "CAPEX [Meuro]": hp_capex_eur / 1e6,
            "OPEX (incl. CO2) [Meuro/yr]": (hp_elec_cost + hp_fixopex_eur) / 1e6,
            "Generated discounted [GWh]": np.nan,
            "LCOH [euro/kWh]": np.nan,
            "CO2 [t/yr]": np.nan,
            "CO2 cost [euro/yr]": np.nan,
        }])
        eco_tbl = pd.concat([eco_tbl, hp_row], ignore_index=True)

    # System LCOH: generation-weighted blend of the component LCOHs (approximate;
    # the rigorous system figure is LCOE_calc_Yang, not called here).
    gens = np.array([df_eco.at[k, "generated discounted"] for k in df_eco.index], dtype=float)
    lcoh = np.array([df_eco.at[k, "LCOE"] for k in df_eco.index], dtype=float)
    m = np.isfinite(gens) & np.isfinite(lcoh) & (gens > 0)
    system_lcoh = float(np.nansum(gens[m] * lcoh[m]) / np.nansum(gens[m])) if m.any() else np.nan

    summary_tbl = pd.DataFrame({
        "Metric": [
            "System LCOH (generation-weighted) [euro/kWh]",
            "Total CO2 [t/yr]",
            "Total CO2 cost [euro/yr]",
            "HP rated power [kW]",
            "HP electricity consumed [GWh/yr]",
            "HP electricity price [euro/kWh]",
            "HP electricity cost [euro/yr]",
            "HP CAPEX [Meuro]  (already inside ATES)",
            "HP fixed OPEX [euro/yr]  (already inside ATES)",
            "HP mean COP (running)",
        ],
        "Value": [
            system_lcoh,
            (co2_df["CO2_emission [kg]"].sum() / 1000) if co2_df is not None else np.nan,
            co2_df["Cost_CO2"].sum() if co2_df is not None else np.nan,
            hp_rating_kW,
            P_el_total_kWh / 1e6,
            hp_elec_price,
            hp_elec_cost,
            hp_capex_eur / 1e6,
            hp_fixopex_eur,
            mean_cop,
        ],
    })

    glossary = pd.DataFrame({
        "Variable": [
            "Time (hours)", "Demand [kWh]", "Geothermal [kWh]",
            "ATES direct [kWh]", "ATES direct split [kWh]", "Heat pump [kWh]",
            "Gas boiler [kWh]", "Sum sources [kWh]", "Mode", "T_extract [C]",
            "T_inject [C]", "COP [-]", "COP from HP/P_el [-]",
            "HP evap/source [kWh]", "P_el compressor [kWh]", "HP condenser [kWh]",
            "HP identity resid [kWh]", "ATES production [kWh]", "Split sum [kWh]",
            "Split residual [kWh]", "Flow extracted [m3]", "Charge to ATES [kWh]",
            "Charge volume [m3]",
        ],
        "Description": [
            "Hour of the year (simulation timestamp).",
            "DHN heat demand in this hour.",
            "Geothermal heat delivered to demand (corrected, demand-clipped).",
            "ATES direct-HX heat to demand from system() (demand-clipped, HP heat removed).",
            "Raw direct-HX heat (output_dir) from _energy_split, before system clipping. Compare with 'ATES direct [kWh]'.",
            "HP condenser heat delivered to demand this hour (= evap + P_el).",
            "Back-up gas-boiler heat to demand.",
            "Geothermal + ATES direct + Heat pump + Gas boiler. Should reconstruct Demand on covered hours.",
            "Dispatch state: off (no discharge), A (HX only, HP idle), B (HP on, T_extract above return), D (HP on, T_extract below return).",
            "Mean hot-well extraction temperature this hour.",
            "Realized cold-well reinjection temperature: T_return (HP off) or T_floor (HP active).",
            "HP coefficient of performance from Calculate_COP. NaN when HP off.",
            "COP reconstructed as HP condenser / P_el. Should equal 'COP [-]' wherever HP runs (verification).",
            "Heat pulled from the aquifer through the evaporator (Q_evap).",
            "Electricity consumed by the compressor.",
            "Total heat leaving the HP condenser (Q_evap + P_el).",
            "HP condenser - (evap + P_el). First law on the HP; should be ~0.",
            "Total ATES-subsystem heat (direct HX + HP condenser), raw from calc_heat.",
            "output_dir + output_evap + P_el. Should equal ATES production.",
            "Split sum - ATES production. Energy balance on the ATES; should be ~0.",
            "Volume drawn from the hot well this hour.",
            "Heat charged into storage this hour, summed over storage suppliers.",
            "Volume injected into storage this hour (flow_injected).",
        ],
    })

    if write_excel:
        with pd.ExcelWriter(OUTFILE, engine="openpyxl") as writer:
            ts.to_excel(writer, sheet_name="Per-source heat", index=False)

            params = pd.DataFrame({
                "Parameter": [
                    "Configuration (USE_HP)",
                    "DHN supply T_in [C]", "DHN return T_out [C]",
                    "ATES T_return / HX floor [C]", "ATES T_floor / HP cold side [C]",
                    "Ground temp T_g [C]", "Recovery efficiency Reff [-]",
                    "Annual injected volume [m3]", "max_V [m3/h]",
                    "HP power_el [kW]", "HP delta_T_coldside [K]",
                    "HP COP_max [-]", "HP elec_price [euro/kWh]",
                ],
                "Value": [
                    "HP on" if USE_HP else "HP off (no HP object)",
                    demand.T_in, demand.T_out,
                    getattr(ATES, "T_return", demand.T_out),
                    getattr(ATES, "T_floor", np.nan),
                    ATES.T_g, getattr(ATES, "Reff", np.nan),
                    getattr(ATES, "volume", np.nan), ATES.max_V,
                    (hp.power_el         if USE_HP else np.nan),
                    (hp.delta_T_coldside if USE_HP else np.nan),
                    (hp.COP_max          if USE_HP else np.nan),
                    (hp.elec_price       if USE_HP else np.nan),
                ],
            })
            params.to_excel(writer, sheet_name="Parameters", index=False)

            # Economics sheet: two stacked tables with titles.
            eco_tbl.to_excel(writer, sheet_name="Economics", index=False, startrow=1)
            ws_eco = writer.sheets["Economics"]
            ws_eco.cell(row=1, column=1, value="Per-component economics")
            sum_start = len(eco_tbl) + 4                       # 0-indexed startrow
            summary_tbl.to_excel(writer, sheet_name="Economics", index=False, startrow=sum_start)
            ws_eco.cell(row=sum_start, column=1, value="Heat-pump & system summary")

            cons.to_excel(writer, sheet_name="Consistency checks", index=False)
            glossary.to_excel(writer, sheet_name="Glossary", index=False)

        print(f"\nSaved -> {OUTFILE} "
              f"({len(ts)} timesteps, {ts.shape[1]} columns; "
              f"sheets: Per-source heat, Parameters, Economics, Consistency checks, Glossary)")
    print("\nConsistency checks (Sum |resid| is the strict test):")
    for _, r in cons.iterrows():
        print(f"  [{r['Status']:>5}] {r['Check']:<40} "
              f"\u03a3resid = {r['Difference A-B [kWh]']:+.3e}   "
              f"\u03a3|resid| = {r['Sum |per-hour resid| [kWh]']:.3e}")

    # --- Console summary: annual totals + economics (HP-aware) -----------------
    LABEL = "Peter (HP on)" if USE_HP else "Peter (no HP)"

    print("=" * 64)
    print(f"  MODEL: {LABEL}   annual totals [GWh]")
    print("=" * 64)

    totals = {}
    totals["Demand"] = result["Demand"].sum() / GWh
    totals["Total production"] = result["Total production"].sum() / GWh
    totals["Heat pump production"] = hp_ts.sum() / GWh
    for i in supply:
        for col in (i.name + " production", i.name + " corrected"):
            if col in result:
                totals[col] = result[col].sum() / GWh
    totals["ATES direct only"] = ates_dir_corr.sum() / GWh
    totals["Unmet (Demand-Total, clipped)"] = \
        np.clip(result["Demand"] - result["Total production"], 0, None).sum() / GWh
    for k, v in totals.items():
        print(f"  {k:<30}: {v:>10.4f}")

    print("-" * 64)
    print(f"  {'ATES Reff':<30}: {getattr(ATES, 'Reff', np.nan):>10.4f}")
    print(f"  {'ATES injected volume [m3]':<30}: {float(getattr(ATES, 'volume', np.nan)):>14,.1f}")
    print(f"  {'ATES extracted volume [m3]':<30}: {float(np.nansum(getattr(ATES, 'flow_extracted', np.nan))):>14,.1f}")
    print("-" * 64)
    print("  Heat-pump specifics")
    print(f"  {'HP condenser heat [GWh]':<30}: {out_HP.sum() / GWh:>10.4f}")
    print(f"  {'HP source heat / evap [GWh]':<30}: {out_evap.sum() / GWh:>10.4f}")
    print(f"  {'HP electricity P_el [GWh]':<30}: {P_el.sum() / GWh:>10.4f}")
    print(f"  {'HP mean COP (running)':<30}: {mean_cop:>10.4f}")
    print(f"  {'HP rated power [kW]':<30}: {hp_rating_kW:>10.1f}")
    print(f"  {'HP elec price [euro/kWh]':<30}: {hp_elec_price:>10.4f}")
    print(f"  {'HP elec cost [euro/yr]':<30}: {hp_elec_cost:>14,.0f}")
    print(f"  {'HP CAPEX [euro] (in ATES)':<30}: {hp_capex_eur:>14,.0f}")
    a_b_d = (int((mode == 'A').sum()), int((mode == 'B').sum()), int((mode == 'D').sum())) \
        if mode.size else (0, 0, 0)
    print(f"  {'discharge hours A/B/D':<30}: {a_b_d[0]:>4} / {a_b_d[1]} / {a_b_d[2]}")
    print("=" * 64)

    print("\nEconomic analysis:")
    print(df_eco.to_string())
    print(f"\n  System LCOH (generation-weighted) : {system_lcoh:.4f} euro/kWh")
    if co2_df is not None:
        print(f"  Total CO2                         : "
              f"{co2_df['CO2_emission [kg]'].sum() / 1000:,.1f} t/yr "
              f"-> {co2_df['Cost_CO2'].sum():,.0f} euro/yr")
    print("=" * 64)

    if make_plots:
        plt.show()

    # --- Return headline results so a sweep can collect rows -------------------
    return {
        "USE_HP": USE_HP,
        "tag": tag,
        "outfile": OUTFILE if write_excel else None,
        "ATES_MAX_V": ATES_MAX_V,
        "HP_POWER_EL": HP_POWER_EL if USE_HP else np.nan,
        "HP_DELTA_T_COLDSIDE": HP_DELTA_T_COLDSIDE if USE_HP else np.nan,
        "DEMAND_T_IN": DEMAND_T_IN,
        "DEMAND_T_OUT": DEMAND_T_OUT,
        "Reff": float(getattr(ATES, "Reff", np.nan)),
        "injected_volume_m3": float(getattr(ATES, "volume", np.nan)),
        "extracted_volume_m3": float(np.nansum(getattr(ATES, "flow_extracted", np.nan))),
        "demand_GWh": result["Demand"].sum() / GWh,
        "geo_GWh": geo_corr.sum() / GWh,
        "ates_direct_GWh": ates_dir_corr.sum() / GWh,
        "hp_GWh": hp_corr.sum() / GWh,
        "gas_GWh": gas_corr.sum() / GWh,
        "unmet_GWh": float(np.clip(result["Demand"] - result["Total production"], 0, None).sum() / GWh),
        "system_lcoh": system_lcoh,
        "geo_lcoh": df_eco.at["Geothermal well", "LCOE"] if "Geothermal well" in df_eco.index else np.nan,
        "ates_lcoh": df_eco.at["ATES", "LCOE"] if "ATES" in df_eco.index else np.nan,
        "gas_lcoh": df_eco.at["Gas boiler", "LCOE"] if "Gas boiler" in df_eco.index else np.nan,
        "hp_elec_GWh": P_el_total_kWh / GWh,
        "hp_elec_cost_eur": hp_elec_cost,
        "hp_mean_COP": mean_cop,
        "hp_capex_Meur": hp_capex_eur / 1e6,
        "total_CO2_t": (co2_df["CO2_emission [kg]"].sum() / 1000) if co2_df is not None else np.nan,
        "total_CO2_cost_eur": (co2_df["Cost_CO2"].sum()) if co2_df is not None else np.nan,
        "df_eco": df_eco,
    }


# ================================================================== #
#  Single-run behaviour when executed directly (unchanged output)    #
# ================================================================== #
if __name__ == "__main__":
    run_case(make_plots=True, write_excel=True)