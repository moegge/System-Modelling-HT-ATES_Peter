# -*- coding: utf-8 -*-
"""
model_driver.py
==================================================================
One simulation of the updated model (main2_Peter / ATES_obj_Peter), wrapped in
run_case() so it can be run directly (single run) OR imported and called from a
sweep script (see sweep.py).

CONFIG selects which components are built (mirrors David Geerts' configurations,
plus a heat-pump variant):
  * "G"    = gas only
  * "GG"   = gas + geothermal
  * "GGA"  = gas + geothermal + HT-ATES        (no heat pump)
  * "GGAH" = gas + geothermal + HT-ATES + HP   (heat pump on the ATES)

The legacy USE_HP toggle still works and maps onto CONFIG:
  * USE_HP=True  -> "GGAH"      * USE_HP=False -> "GGA"
If CONFIG is given explicitly it takes precedence over USE_HP.

Why "HP on all times" == "every discharge hour": in calc_heat the loop only
visits timesteps where missing_energy > 0 (the ATES is being drawn on), and the
whole HP dispatch + output lives inside that loop. Charge-side HP is not
implemented. Passing hp_on = all True therefore runs the HP on every discharge
hour.

SITE selects the site preset and with it the storage technology
(site_config_<SITE>.py, PARAMS dict):
  * "Delft"     -> HT-ATES  (ATES_obj_Peter)   = the defaults below
  * "Bochum"    -> MTES     (MTES_obj_Peter)
  * "Darmstadt" -> BTES     (BTES_obj_Peter, needs pygfunction)
In the config names the "A" means "storage" of the site's type.
Precedence:  run_case(...) argument  >  site preset  >  object default.

Run from the repo root (needs main2_Peter.py, ATES_obj_Peter.py, results_AXI_V2,
Predict_REFF_boostedregression.pkl, and the Amsterdam demand parquet).

    python model_driver.py            # single run, plots + Excel
    from Test_File_Peter_17_08 import run_case  # driven by sweep.py
==================================================================
"""
import os
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main2_Peter import (geothermal, demand_class, gas_boiler, heat_pump_ATES,
                         system, economic_analysis, system_plot, CO2_emissions_calc,
                         LCOE_calc_Yang)
from ATES_obj_Peter import ATES_obj
from site_config_Delft import PARAMS as _DELFT

# ================================================================== #
#  DEFAULT CONFIGURATION  --  the single-run defaults.               #
#  A sweep overrides any of these by passing them to run_case().     #
#  Site values (demand, source, storage, HP, prices) are the Delft   #
#  preset; they stay module constants because the Delft Case        #
#  scripts import them. Other sites: site_config_<SITE>.py.         #
# ================================================================== #

# --- Site (selects the preset in site_config_<SITE>.py) ---------------------
SITE = "Delft"           # "Delft" (ATES), "Bochum" (MTES), "Darmstadt" (BTES)

# --- Master toggle: run WITH or WITHOUT the discharge-side heat pump -------
USE_HP = True            # True  -> HP created and attached to the ATES
                         # False -> NO HP object created at all (clean baseline)

# --- System configuration (takes precedence over USE_HP if not None) -------
CONFIG = None            # None -> derive from USE_HP; else one of
                         # "G", "GG", "GGA", "GGAH"

# --- Simulation ------------------------------------------------------------
TIMESTEP = 3600          # [s]

# --- District-heating demand ----------------------------------------------
DEMAND_EXAMPLE = _DELFT["DEMAND_EXAMPLE"]  # Amsterdam; Delft Peter; TU Delft
DEMAND_T_IN    = _DELFT["DEMAND_T_IN"]     # [C] DHN supply temperature (HP condenser sink)
DEMAND_T_OUT   = _DELFT["DEMAND_T_OUT"]    # [C] DHN return temperature (= storage cutoff / HX floor)

# --- Geothermal baseload (also the storage charging source) ---------------
GEO_POWER = _DELFT["GEO_POWER"]            # [kW]
GEO_T_OUT = _DELFT["GEO_T_OUT"]            # [C]

# --- ATES aquifer (Delft) --------------------------------------------------
ATES_MAX_V     = _DELFT["STORAGE_KW"]["max_V"]      # [m3/h]
ATES_THICKNESS = _DELFT["STORAGE_KW"]["thickness"]  # [m]
ATES_KH        = _DELFT["STORAGE_KW"]["kh"]         # [m/day]
ATES_ANI       = _DELFT["STORAGE_KW"]["ani"]        # [-]
ATES_T_GROUND  = _DELFT["STORAGE_KW"]["T_ground"]   # [C]
ATES_LIFETIME  = _DELFT["STORAGE_KW"]["lifetime"]

# --- Heat pump (only used if the config includes the HP) -------------------
HP_POWER_EL         = _DELFT["HP_POWER_EL"]          # [kW_el] fixed compressor rating
HP_DELTA_T_COLDSIDE = _DELFT["HP_DELTA_T_COLDSIDE"]  # [K] cooling below the DHN return -> fixed cold-well T

# --- Fuel and CO2 prices for the economics ---------------------------------
GAS_PRICE = _DELFT["GAS_PRICE"]            # [euro/kWh_gas]
CO2_PRICE = _DELFT["CO2_PRICE"]            # [euro/ton]
ELEC_PRICE = _DELFT["ELEC_PRICE"]          # [euro/kWh]

# ================================================================== #

# Output folder: '<folder of this file>/results'. Anchored on __file__, not the
# working directory, so a sweep launched from elsewhere writes to the same place.
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
RESULTS_DIR = os.path.join(_HERE, "results")


def _config_from_args(CONFIG, USE_HP):
    """Resolve the effective config string. CONFIG wins if given; else USE_HP."""
    if CONFIG is not None:
        c = str(CONFIG).upper()
        if c not in ("G", "GG", "GGA", "GGAH"):
            raise ValueError(f"CONFIG must be one of G/GG/GGA/GGAH, got {CONFIG!r}")
        return c
    return "GGAH" if USE_HP else "GGA"

# Delft defaults; per site they come from the preset and are passed on to
# LCOE_calc_Yang and _cost_split, so the two always use the same values.
DISC_RATE         = _DELFT["DISC_RATE"]
LIFETIME_SYSTEM   = _DELFT["LIFETIME_SYSTEM"]
LIFETIME_NETWORK  = _DELFT["LIFETIME_NETWORK"]
OPEX_NETWORK_PERC = _DELFT["OPEX_NETWORK_PERC"]
NETWORK_EUR_PER_M = _DELFT["NETWORK_EUR_PER_M"]
# than redeclaring, so the reported ratio_ATES_HP can never drift from the target.
RHO_CP = 4180.0                  # [kJ/m3.K]

# run_case arguments that a site preset can fill (default None = take the preset).
_SITE_KEYS = ("STORAGE_TYPE", "DEMAND_EXAMPLE", "DEMAND_T_IN", "DEMAND_T_OUT",
              "GEO_POWER", "GEO_T_OUT", "GEO_COSTPERKW", "GEO_FIXED_OPEX",
              "GEO_VAR_OPEX", "GEO_LIFETIME", "GEO_CO2", "GAS_PRICE", "GAS_CO2",
              "CO2_PRICE", "ELEC_PRICE", "DISC_RATE", "LIFETIME_SYSTEM",
              "LIFETIME_NETWORK", "NETWORK_LENGTH_M", "NETWORK_EUR_PER_M",
              "OPEX_NETWORK_PERC", "HP_POWER_EL", "HP_DELTA_T_COLDSIDE", "HP_CAPEX",
              "HP_FIXED_OPEX", "HP_LIFETIME", "HP_CO2_EL", "STORAGE_ELEC_KWH_PER_M3")

# Legacy ATES_* run_case arguments -> storage-object keyword. max_V, T_ground and
# lifetime apply to every storage type; the aquifer ones only to the ATES.
_ATES_ARG_TO_KW = {"ATES_MAX_V": "max_V", "ATES_T_GROUND": "T_ground",
                   "ATES_LIFETIME": "lifetime", "ATES_THICKNESS": "thickness",
                   "ATES_KH": "kh", "ATES_ANI": "ani"}


def load_site(site):
    """PARAMS dict of site_config_<site>.py (a copy, safe to modify)."""
    try:
        mod = importlib.import_module(f"site_config_{site}")
    except ModuleNotFoundError as e:
        raise ValueError(f"No site preset 'site_config_{site}.py' for SITE={site!r}") from e
    p = dict(mod.PARAMS)
    p["STORAGE_KW"] = dict(p.get("STORAGE_KW", {}))
    return p


def _build_storage(kind, supplier, hp, storage_kw, elec_price, elec_kWh_per_m3):
    """
    Storage object for the site. ATES / MTES / BTES all expose the same
    interface to main2 (see the MTES_obj_Peter / BTES_obj_Peter docstrings).
    BTES_obj is imported here, so an ATES or MTES run does not need pygfunction.
    """
    kw = dict(storage_kw)
    if kind == "ATES":
        S = ATES_obj(supplier, HP=hp, elec_price=elec_price, **kw)
    elif kind == "MTES":
        from MTES_obj_Peter import MTES_obj
        S = MTES_obj(supplier, HP=hp, elec_price=elec_price, **kw)
    elif kind == "BTES":
        from BTES_obj_Peter import BTES_obj
        S = BTES_obj(supplier, HP=hp, elec_price=elec_price, **kw)
    else:
        raise ValueError(f"STORAGE_TYPE must be ATES / MTES / BTES, got {kind!r}")
    # Read by main2.economic_analysis (David's pumping OPEX formula).
    S.elec_kWh_per_m3 = elec_kWh_per_m3
    # Read by main2.system(): the HP charging-volume factor needs a cold well,
    # which only the ATES has. MTES / BTES are closed loops -> factor 1.
    S.charge_factor_HP = (kind == "ATES")
    return S


def _cost_split(result, df_eco, supply, co2_df, generated_disc, capex_network,
                hp_co2_eur, disc_rate=DISC_RATE, lifetime_system=LIFETIME_SYSTEM,
                lifetime_network=LIFETIME_NETWORK, opex_network_perc=OPEX_NETWORK_PERC):
    """
    Additive decomposition of the LCOE_calc_Yang system LCOH into [EUR/MWh].

    Reproduces LCOE_calc_Yang's discounting component by component:
      capex -> discounted at every reinvestment year (j % lifetime == 0)
      opex  -> discounted every year j = 0 .. LIFETIME_SYSTEM-1
    then divides each stream by the POOLED discounted heat, so the segments sum
    to the system LCOH. This is NOT df_eco["LCOE"], which divides each component
    by its OWN output and therefore sums to nothing.

    NETWORK IS NOT DISCOUNTED here, because LCOE_calc_Yang adds the network
    terms at face value while discounting everything else. Replicated on purpose
    so the segments reconcile with the published figure.
    """
    if not generated_disc or not np.isfinite(generated_disc) or generated_disc <= 0:
        return {}

    r = disc_rate
    af = sum(1.0 / (1.0 + r) ** j for j in range(lifetime_system))

    def _cap(capex, lifetime):
        return sum(capex / (1.0 + r) ** j
                   for j in range(0, lifetime_system, max(int(lifetime), 1)))

    def _val(name, col):
        if col not in df_eco.columns or name not in df_eco.index:
            return 0.0
        return float(np.nan_to_num(df_eco.at[name, col]))

    def _co2(name):
        if co2_df is None or name not in co2_df.index:
            return 0.0
        return float(np.nan_to_num(co2_df.at[name, "Cost_CO2"]))

    # Storage segment label: "ATES well" (as in the Delft figures), "MTES well", "BTES well".
    stor_seg = next((s.name + " well" for s in supply if s.control == "storage"), "ATES well")

    seg, co2_geo = {}, 0.0
    for i in supply:
        nm = i.name
        if nm not in df_eco.index:
            continue
        capex, opex, co2 = _val(nm, "capex"), _val(nm, "opex"), _co2(nm)

        if i.control == "storage":
            # A NaN LCOE means LCOE_calc skipped it (nothing extracted), and
            # LCOE_calc_Yang skipped it too -> no cost in the system figure.
            if not np.isfinite(_val(nm, "LCOE")):
                continue
            hp_cap, hp_el = _val(nm, "hp_capex"), _val(nm, "hp_elec_cost")
            hp_fix = _val(nm, "hp_fixopex")
            seg[stor_seg] = (seg.get(stor_seg, 0.0)
                             + _cap(capex - hp_cap, i.lifetime)
                             + (opex - co2 - hp_el - hp_fix) * af)
            if hp_cap or hp_el:
                seg["HP capex"] = _cap(hp_cap, i.lifetime)
                seg["HP opex (elec + fixed)"] = (hp_el + hp_fix) * af
                seg["CO2 - HP electricity"] = hp_co2_eur * af
            # The ATES CO2 row carries the geothermal heat it stored PLUS the HP
            # grid emissions; only the geothermal part belongs in the geo bucket.
            co2_geo += max(co2 - hp_co2_eur, 0.0)
        elif nm == "Geothermal well":
            # Paper convention: the variable opex of the heat this plant sent to
            # storage belongs to the ATES, not here. LCOE_calc reassigns it via
            # add_opex_ATES; LCOE_calc_Yang does not, so do it in the split.
            # Moves cost between two segments -- the total is unchanged.
            pct, prod = nm + " percentage to storage", nm + " production"
            stored = (float(np.nan_to_num((result[pct] * result[prod]).sum()))
                      if pct in result and prod in result else 0.0)
            to_ates = getattr(i, "var_opex", 0.0) * stored
            seg["Geothermal"] = _cap(capex, i.lifetime) + (opex - co2 - to_ates) * af
            seg[stor_seg] = seg.get(stor_seg, 0.0) + to_ates * af
            co2_geo += co2
        elif nm == "Gas boiler":
            seg["Gas boiler"] = _cap(capex, i.lifetime) + (opex - co2) * af
            seg["CO2 - gas"] = co2 * af
        else:
            seg[nm] = _cap(capex, i.lifetime) + (opex - co2) * af
    if co2_geo:
        seg["CO2 - geothermal"] = co2_geo * af

    if capex_network:
        seg["Network"] = (len(range(0, lifetime_system, lifetime_network))
                          * capex_network
                          + lifetime_system * opex_network_perc * capex_network)

    return {k: v / generated_disc * 1000.0 for k, v in seg.items()}

def run_case(USE_HP=USE_HP, CONFIG=CONFIG, TIMESTEP=TIMESTEP, SITE=SITE,
             STORAGE_TYPE=None, STORAGE_KW=None, STORAGE_ELEC_KWH_PER_M3=None,
             DEMAND_EXAMPLE=None, DEMAND_T_IN=None, DEMAND_T_OUT=None,
             GEO_POWER=None, GEO_T_OUT=None, GEO_COSTPERKW=None, GEO_FIXED_OPEX=None,
             GEO_VAR_OPEX=None, GEO_LIFETIME=None, GEO_CO2=None,
             ATES_MAX_V=None, ATES_THICKNESS=None, ATES_KH=None,
             ATES_ANI=None, ATES_T_GROUND=None,
             ATES_LIFETIME=None,
             HP_POWER_EL=None, HP_DELTA_T_COLDSIDE=None, HP_CAPEX=None,
             HP_FIXED_OPEX=None, HP_LIFETIME=None, HP_CO2_EL=None,
             HP_DYNAMIC_DISPATCH=False, HP_THRESHOLD_EUR_MWH=60.0,
             GAS_PRICE=None, GAS_CO2=None, CO2_PRICE=None, ELEC_PRICE=None,
             DISC_RATE=None, LIFETIME_SYSTEM=None, LIFETIME_NETWORK=None,
             NETWORK_LENGTH_M=None, NETWORK_EUR_PER_M=None, OPEX_NETWORK_PERC=None,
             OUTFILE=None, tag="",
             make_plots=False, write_excel=True):
    """
    Run one configuration. Any argument left at its default reproduces the
    single-run config; a sweep passes only the knobs it varies.

    SITE        : "Delft" / "Bochum" / "Darmstadt" -> preset site_config_<SITE>.py.
                  Every site argument left at None is taken from that preset.
    STORAGE_KW  : dict of extra storage-object keywords, merged over the preset's
                  STORAGE_KW (e.g. {"N_boreholes": 19} for the BTES).
    ATES_*      : legacy names. ATES_MAX_V / ATES_T_GROUND / ATES_LIFETIME set
                  max_V / T_ground / lifetime of ANY storage type.
    CONFIG      : "G" / "GG" / "GGA" / "GGAH". If None, derived from USE_HP.
                  "A" = the site's storage (ATES, MTES or BTES).
    OUTFILE     : Excel path. None -> auto 'timeseries_<CONFIG>[_tag].xlsx'.
    tag         : suffix on the auto filename so sweep runs don't overwrite.
    make_plots  : show the two system_plot figures (keep False in a sweep).
    write_excel : write the workbook (False in a sweep that only wants numbers).

    Returns a dict of headline results so a sweep can collect rows.
    """
    _args = dict(locals())
    cfg = _config_from_args(CONFIG, USE_HP)
    use_geo  = cfg in ("GG", "GGA", "GGAH")
    use_ates = cfg in ("GGA", "GGAH")      # "ates" = the site's storage
    use_hp   = cfg == "GGAH"

    # --- Resolve parameters: argument > site preset > object default ----------
    p = load_site(SITE)
    p.update({k: _args[k] for k in _SITE_KEYS if _args.get(k) is not None})
    storage_kw = p["STORAGE_KW"]
    for arg, kw in _ATES_ARG_TO_KW.items():
        if _args[arg] is not None:
            storage_kw[kw] = _args[arg]
    if STORAGE_KW:
        storage_kw.update(STORAGE_KW)
    storage_type = p["STORAGE_TYPE"]

    DEMAND_EXAMPLE, DEMAND_T_IN, DEMAND_T_OUT = (p["DEMAND_EXAMPLE"], p["DEMAND_T_IN"],
                                                 p["DEMAND_T_OUT"])
    GEO_POWER, GEO_T_OUT = p["GEO_POWER"], p["GEO_T_OUT"]
    HP_POWER_EL, HP_DELTA_T_COLDSIDE = p["HP_POWER_EL"], p["HP_DELTA_T_COLDSIDE"]
    GAS_PRICE, CO2_PRICE, ELEC_PRICE = p["GAS_PRICE"], p["CO2_PRICE"], p["ELEC_PRICE"]
    NETWORK_LENGTH_M = p["NETWORK_LENGTH_M"]

    # No dynamic pricing outside Delft: the dispatch signal and Eq. 2 use NL
    # day-ahead prices and Stedin/NL tariffs.
    if HP_DYNAMIC_DISPATCH and storage_type != "ATES":
        raise ValueError(f"HP_DYNAMIC_DISPATCH is only set up for the Delft ATES "
                         f"(NL prices); {SITE} / {storage_type} uses the flat ELEC_PRICE.")

    # --- Output filename (auto-suffixed by config) -----------------------------
    # A bare filename (auto-generated or passed in) is placed in RESULTS_DIR
    # (Delft) or RESULTS_DIR/<SITE> (other sites).
    # An absolute path, or one that already carries a directory, is left alone.
    results_dir = RESULTS_DIR if SITE == "Delft" else os.path.join(RESULTS_DIR, SITE)
    if OUTFILE is None:
        OUTFILE = f"timeseries_{cfg}{('_' + tag) if tag else ''}.xlsx"
    if not os.path.isabs(OUTFILE) and not os.path.dirname(OUTFILE):
        OUTFILE = os.path.join(results_dir, OUTFILE)

    timestep = TIMESTEP

    # --- District-heating components ------------------------------------------
    demand = demand_class(T_in=DEMAND_T_IN, T_out=DEMAND_T_OUT,
                          example_demand=DEMAND_EXAMPLE)
    gas    = gas_boiler(gas_price=GAS_PRICE, CO2_kg=p["GAS_CO2"])

    # Geothermal is present in GG / GGA / GGAH (charging source for the storage too).
    # Bochum / Darmstadt: zero CAPEX / fixed OPEX, heat bought via var_opex.
    geo = (geothermal(power=GEO_POWER, T_out=GEO_T_OUT, costperkW=p["GEO_COSTPERKW"],
                      fixed_opex=p["GEO_FIXED_OPEX"], var_opex=p["GEO_VAR_OPEX"],
                      lifetime=p["GEO_LIFETIME"], CO2_kg=p["GEO_CO2"])
           if use_geo else None)

    # Heat pump only in GGAH.
    hp = (heat_pump_ATES(power_el=HP_POWER_EL, delta_T_coldside=HP_DELTA_T_COLDSIDE,
                         costperkW=p["HP_CAPEX"], fixed_opex=p["HP_FIXED_OPEX"],
                         elec_price=ELEC_PRICE, lifetime=p["HP_LIFETIME"],
                         CO2_kg_el=p["HP_CO2_EL"])
          if use_hp else None)

    # Storage only in GGA / GGAH; it charges from the geothermal supplier.
    # Variable kept as ATES (small diff); it is the site's storage object.
    if use_ates:
        ATES = _build_storage(storage_type, [geo], hp, storage_kw, ELEC_PRICE,
                              p["STORAGE_ELEC_KWH_PER_M3"])
    else:
        ATES = None
    sname = ATES.name if ATES is not None else storage_type   # result column prefix

    # Preferred order: sustainable source, storage, back-up. Only include the
    # components that exist for this config.
    supply = []
    if geo is not None:
        supply.append(geo)
    if ATES is not None:
        supply.append(ATES)
    supply.append(gas)

    # --- HP dispatch -----------------------------------------------------------
    # Dynamic -> hp_on=None so system() builds the signal from the spot price.
    # Static  -> ON every hour (only discharge hours actually use it).
    hp_on = (None if (HP_DYNAMIC_DISPATCH or not use_hp)
             else np.ones(len(demand.data), dtype=bool))

    result, df_flow = system(demand, supply, len_timestep=timestep, hp_on=hp_on,
                             hp_dynamic_dispatch=(HP_DYNAMIC_DISPATCH and use_hp),
                             hp_threshold_eur_mwh=HP_THRESHOLD_EUR_MWH)

    # --- Plots -----------------------------------------------------------------
    if make_plots:
        system_plot(result, supply, demand, len_timestep=timestep, setting="demand_met")
        system_plot(result, supply, demand, len_timestep=timestep, setting="ordered")

    # --- Economics (LCOH per component) ----------------------------------------
    # Site economics (resolved above); _cost_split below gets the same values.
    eco_kw = dict(disc_rate=p["DISC_RATE"], lifetime_system=p["LIFETIME_SYSTEM"],
                  lifetime_network=p["LIFETIME_NETWORK"],
                  opex_network_perc=p["OPEX_NETWORK_PERC"])
    df_eco = economic_analysis(result, supply, disc_rate=eco_kw["disc_rate"],
                               incorporate_CO2=True, CO2_price=CO2_PRICE,
                               len_timestep=timestep)
    capex_network_eur = NETWORK_LENGTH_M * p["NETWORK_EUR_PER_M"]
    system_lcoh_yang, generated_disc = LCOE_calc_Yang(
        result, supply, df_eco, capex_network=capex_network_eur, **eco_kw)
    print("\nLCOH per component:")
    for i in range(len(df_eco)):
        print(f"  {df_eco.iloc[i, 0]:<16} = {round(df_eco.iloc[i].loc['LCOE'], 3)} euro/kWh")

    # --- Heat-pump diagnostics (console) ---------------------------------------
    GWh = 1e6
    n = len(result)

    # Safe access to a result column: returns zeros if the column is absent
    # (a component missing in this config -> system() never created its column).
    def _col(name):
        return (np.nan_to_num(result[name].values)
                if name in result else np.zeros(n))

    # Safe access to a per-timestep array written on the ATES object.
    def _arr(attr, fill=0.0):
        if ATES is None:
            return np.full(n, fill, dtype=float)
        a = np.asarray(getattr(ATES, attr, np.full(n, fill)), dtype=float)
        return a if a.size == n else np.full(n, fill, dtype=float)

    mode = np.asarray(getattr(ATES, "mode", []), dtype=object) if ATES is not None \
        else np.array([], dtype=object)
    cop  = np.asarray(getattr(ATES, "COP", []), dtype=float) if ATES is not None \
        else np.array([], dtype=float)
    cop_active = cop[np.isfinite(cop) & (cop > 0)] if cop.size else np.array([])
    mean_cop = float(cop_active.mean()) if cop_active.size else np.nan

    print(f"\nConfig {cfg} summary:")
    print(f"  HP condenser (Q_evap+P_el)      : {np.nansum(_arr('output_HP')) / GWh:8.3f} GWh")
    print(f"  HP electricity  (P_el)          : {np.nansum(_arr('P_el')) / GWh:8.3f} GWh")
    ates_prod_col = _col(sname + " production")
    print(f"  {sname} subsystem (direct+HP)      : {ates_prod_col.sum() / GWh:8.3f} GWh")
    print(f"  Total heat to demand            : {_col('Total production').sum() / GWh:8.3f} GWh")
    if mode.size:
        print(f"  discharge hours -> mode A/B/D   : "
              f"{int((mode=='A').sum())} / {int((mode=='B').sum())} / {int((mode=='D').sum())}")
    if cop_active.size:
        print(f"  mean COP while running          : {mean_cop:.2f}")

    # --- Timeseries build: per-source heat + HP/ATES diagnostics ---------------
    hp_ts = result["Heat pump production"] if "Heat pump production" in result \
            else pd.Series(0.0, index=result.index)
    # "<storage> corrected" includes HP condenser heat -> strip it for the HX-only part.
    if sname + " corrected" in result:
        ates_direct = result[sname + " corrected"] - hp_ts
    else:
        ates_direct = pd.Series(0.0, index=result.index)

    mode_arr = np.asarray(getattr(ATES, "mode", np.full(n, "off", dtype=object)),
                          dtype=object) if ATES is not None \
        else np.full(n, "off", dtype=object)
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

    # #5 Heat charged into the ATES each hour, summed over storage suppliers.
    charge_kWh = np.zeros(n)
    if ATES is not None:
        for s in ATES.supplier:
            c_pct, c_prod = s.name + " percentage to storage", s.name + " production"
            if c_pct in result and c_prod in result:
                charge_kWh += np.nan_to_num((result[c_pct] * result[c_prod]).values)

    ates_prod = _col(sname + " production")
    with np.errstate(divide="ignore", invalid="ignore"):
        cop_check = np.where(P_el > 0, out_HP / P_el, np.nan)

    geo_corr      = _col("Geothermal well corrected")
    ates_dir_corr = np.nan_to_num(ates_direct.values)
    hp_corr       = np.nan_to_num(hp_ts.values)
    gas_corr      = _col("Gas boiler corrected")
    sum_src       = geo_corr + ates_dir_corr + hp_corr + gas_corr

    ts = pd.DataFrame({
        "Time (hours)": result["Time (hours)"].values,
        "Demand [kWh]": result["Demand"].values,
        "Geothermal [kWh]": geo_corr,
        "Geothermal to demand [kWh]": geo_corr - charge_kWh,
        f"{sname} direct [kWh]": ates_dir_corr,
        f"{sname} direct split [kWh]": out_dir,
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
        f"{sname} production [kWh]": ates_prod,
        "Split sum [kWh]": out_dir + out_evap + P_el,
        "Split residual [kWh]": (out_dir + out_evap + P_el) - ates_prod,
        "Flow extracted [m3]": flow_ext,
        f"Charge to {sname} [kWh]": charge_kWh,
        "Charge volume [m3]": flow_inj,
    })

    # --- Consistency checks: cycle totals that should reconcile to ~0 ----------
    def _sum(x):
        return float(np.nansum(x))

    check_defs = [
        (f"Split sum vs {sname} production",
         out_dir + out_evap + P_el, ates_prod, True,
         f"Energy balance on the {sname} subsystem (direct HX + HP condenser)."),
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
        (f"{sname} direct: corrected vs raw split",
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
    # With the HP enabled, the HP's own capex/opex are folded INTO the ATES row
    # by economic_analysis -> the HP-detail rows below are broken out for
    # visibility and are already counted inside the ATES component figures.
    hp_obj         = getattr(ATES, "HP", None) if ATES is not None else None
    P_el_total_kWh = float(np.nansum(_arr("P_el")))
    hp_rating_kW   = ((getattr(hp_obj, "rated_power", None) or getattr(hp_obj, "power_el", np.nan))
                      if hp_obj is not None else np.nan)
    hp_elec_price = getattr(hp_obj, "elec_price", np.nan) if hp_obj is not None else np.nan
    hp_capex_eur = (hp_rating_kW * hp_obj.capex) if hp_obj is not None else 0.0
    hp_fixopex_eur = (hp_obj.fixed_opex * hp_rating_kW) if hp_obj is not None else 0.0
    # Eq. 2 when an hourly spot series is attached, flat elec_price otherwise.
    hp_elec_cost = hp_obj.elec_cost(len_timestep=timestep) if hp_obj is not None else 0.0
    hp_hourly = (getattr(hp_obj, "elec_spot_series", None) is not None
                 if hp_obj is not None else False)
    hp_breakdown = getattr(hp_obj, "elec_cost_breakdown", None) if hp_obj is not None else None
    # All-in price actually paid, and the flat-price counterfactual.
    hp_price_paid = (hp_elec_cost / P_el_total_kWh) if P_el_total_kWh > 0 else np.nan
    hp_flat_cost = P_el_total_kWh * hp_elec_price if hp_obj is not None else 0.0

    try:
        co2_df = CO2_emissions_calc(result, supply, CO2_price=CO2_PRICE)
    except Exception as e:
        print(f"(CO2 breakdown skipped: {type(e).__name__}: {e})")
        co2_df = None

    # --- Additive cost split of the system LCOH [EUR/MWh] ----------------------
    hp_co2_eur = 0.0
    if hp_obj is not None:
        try:    # calc_emissions returns grams; CO2_PRICE is EUR/tonne
            hp_co2_eur = float(hp_obj.calc_emissions(result)) / 1e6 * CO2_PRICE
        except Exception:
            hp_co2_eur = 0.0
    cost_split = _cost_split(result, df_eco, supply, co2_df, generated_disc,
                             capex_network_eur, hp_co2_eur, **eco_kw)
    if cost_split:
        print(f"  cost split residual: "
              f"{sum(cost_split.values()) - system_lcoh_yang * 1000.0:+.4f} "
              f"EUR/MWh (should be ~0)")

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
    # These figures are the HP's OWN capex/opex. They are ALREADY folded into the
    # "ATES" row above by economic_analysis -> this line is a breakout, NOT an
    # addition. Do not sum the column with this row included (it would double-count).
    if hp_obj is not None:
        hp_row = pd.DataFrame([{
            "Component": f"Heat pump (in {sname})",
            "CAPEX [Meuro]": hp_capex_eur / 1e6,
            "OPEX (incl. CO2) [Meuro/yr]": (hp_elec_cost + hp_fixopex_eur) / 1e6,
            "Generated discounted [GWh]": np.nan,
            "LCOH [euro/kWh]": np.nan,
            "CO2 [t/yr]": np.nan,
            "CO2 cost [euro/yr]": np.nan,
        }])
        eco_tbl = pd.concat([eco_tbl, hp_row], ignore_index=True)

    # System LCOH = LCOE_calc_Yang (pooled discounted cost / heat, 60-yr horizon),
    # computed above. The old generation-weighted blend has been removed.

    summary_tbl = pd.DataFrame({
        "Metric": [
            "Config",
            f"System LCOH (Yang, {p['LIFETIME_SYSTEM']}-yr horizon) [euro/kWh]",
            "Total CO2 [t/yr]",
            "Total CO2 cost [euro/yr]",
            "HP rated power [kW]",
            "HP electricity consumed [GWh/yr]",
            "HP pricing mode",
            "HP mean price paid [euro/kWh]",
            "HP electricity cost [euro/yr]",
            "HP electricity cost at flat price [euro/yr]",
            f"HP CAPEX [Meuro]  (already inside {sname})",
            f"HP fixed OPEX [euro/yr]  (already inside {sname})",
            "HP mean COP (running)",
        ],
        "Value": [
            cfg,
            system_lcoh_yang,
            (co2_df["CO2_emission [kg]"].sum() / 1000) if co2_df is not None else np.nan,
            co2_df["Cost_CO2"].sum() if co2_df is not None else np.nan,
            hp_rating_kW,
            P_el_total_kWh / 1e6,
            ("hourly (Eq. 2)" if hp_hourly else "flat"),
            hp_price_paid,
            hp_elec_cost,
            hp_flat_cost,
            hp_capex_eur / 1e6,
            hp_fixopex_eur,
            mean_cop,
        ],
    })

    glossary = pd.DataFrame({
        "Variable": [
            "Time (hours)", "Demand [kWh]", "Geothermal [kWh]",
            "Geothermal to demand [kWh]",
            f"{sname} direct [kWh]", f"{sname} direct split [kWh]", "Heat pump [kWh]",
            "Gas boiler [kWh]", "Sum sources [kWh]", "Mode", "T_extract [C]",
            "T_inject [C]", "COP [-]", "COP from HP/P_el [-]",
            "HP evap/source [kWh]", "P_el compressor [kWh]", "HP condenser [kWh]",
            "HP identity resid [kWh]", f"{sname} production [kWh]", "Split sum [kWh]",
            "Split residual [kWh]", "Flow extracted [m3]", f"Charge to {sname} [kWh]",
            "Charge volume [m3]",
        ],
        "Description": [
            "Hour of the year (simulation timestamp).",
            "DHN heat demand in this hour.",
            "Geothermal heat delivered to demand (corrected, demand-clipped).",
            f"Geothermal heat that went straight to demand, excluding heat routed into {sname} charging (= 'Geothermal [kWh]' minus 'Charge to {sname} [kWh]').",
            f"{sname} direct-HX heat to demand from system() (demand-clipped, HP heat removed).",
            f"Raw direct-HX heat (output_dir) from _energy_split, before system clipping. Compare with '{sname} direct [kWh]'.",
            "HP condenser heat delivered to demand this hour (= evap + P_el).",
            "Back-up gas-boiler heat to demand.",
            f"Geothermal + {sname} direct + Heat pump + Gas boiler. Should reconstruct Demand on covered hours.",
            "Dispatch state: off (no discharge), A (HX only, HP idle), B (HP on, T_extract above return), D (HP on, T_extract below return).",
            "Mean hot-well extraction temperature this hour.",
            "Realized cold-well reinjection temperature: T_return (HP off) or T_floor (HP active).",
            "HP coefficient of performance from Calculate_COP. NaN when HP off.",
            "COP reconstructed as HP condenser / P_el. Should equal 'COP [-]' wherever HP runs (verification).",
            "Heat pulled from the aquifer through the evaporator (Q_evap).",
            "Electricity consumed by the compressor.",
            "Total heat leaving the HP condenser (Q_evap + P_el).",
            "HP condenser - (evap + P_el). First law on the HP; should be ~0.",
            f"Total {sname}-subsystem heat (direct HX + HP condenser), raw from calc_heat.",
            f"output_dir + output_evap + P_el. Should equal {sname} production.",
            f"Split sum - {sname} production. Energy balance on the {sname}; should be ~0.",
            "Volume drawn from the hot well this hour.",
            "Heat charged into storage this hour, summed over storage suppliers.",
            "Volume injected into storage this hour (flow_injected).",
        ],
    })

    if write_excel:
        os.makedirs(os.path.dirname(OUTFILE) or ".", exist_ok=True)
        with pd.ExcelWriter(OUTFILE, engine="openpyxl") as writer:
            ts.to_excel(writer, sheet_name="Per-source heat", index=False)

            params = pd.DataFrame({
                "Parameter": [
                    "Configuration (CONFIG)",
                    "DHN supply T_in [C]", "DHN return T_out [C]",
                    f"{sname} T_return / HX floor [C]", f"{sname} T_floor / HP cold side [C]",
                    "Ground temp T_g [C]", "Recovery efficiency Reff [-]",
                    "Annual injected volume [m3]", "max_V [m3/h]",
                    "HP power_el [kW]", "HP delta_T_coldside [K]",
                    "HP COP_max [-]", "HP elec_price flat [euro/kWh]",
                    "HP pricing mode", "HP mean price paid [euro/kWh]",
                ],
                "Value": [
                    cfg,
                    demand.T_in, demand.T_out,
                    (getattr(ATES, "T_return", demand.T_out) if ATES is not None else np.nan),
                    (getattr(ATES, "T_floor", np.nan) if ATES is not None else np.nan),
                    (ATES.T_g if ATES is not None else np.nan),
                    (getattr(ATES, "Reff", np.nan) if ATES is not None else np.nan),
                    (getattr(ATES, "volume", np.nan) if ATES is not None else np.nan),
                    (ATES.max_V if ATES is not None else np.nan),
                    (hp.power_el         if hp is not None else np.nan),
                    (hp.delta_T_coldside if hp is not None else np.nan),
                    (hp.COP_max if hp is not None else np.nan),
                    (hp.elec_price if hp is not None else np.nan),
                    ("hourly (Eq. 2)" if hp_hourly else "flat"),
                    hp_price_paid,
                ],
            })
            params.to_excel(writer, sheet_name="Parameters", index=False)

            # Economics sheet: two stacked tables with titles.
            eco_tbl.to_excel(writer, sheet_name="Economics", index=False, startrow=1)
            ws_eco = writer.sheets["Economics"]
            ws_eco.cell(row=1, column=1, value="Per-component economics")
            sum_start = len(eco_tbl) + 4                   # 0-indexed startrow
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

    # --- Console summary: annual totals + economics ----------------------------
    LABEL = f"Peter ({cfg})"

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
    totals[f"{sname} direct only"] = ates_dir_corr.sum() / GWh
    totals["Unmet (Demand-Total, clipped)"] = \
        np.clip(result["Demand"] - result["Total production"], 0, None).sum() / GWh
    for k, v in totals.items():
        print(f"  {k:<30}: {v:>10.4f}")

    print("-" * 64)
    print(f"  {sname + ' Reff':<30}: {(getattr(ATES, 'Reff', np.nan) if ATES is not None else np.nan):>10.4f}")
    if ATES is not None and storage_type != "ATES":
        print(f"  {sname + ' utilisation':<30}: {float(getattr(ATES, 'utilisation', np.nan)):>10.4f}")
    print(f"  {sname + ' injected volume [m3]':<30}: {float(getattr(ATES, 'volume', np.nan) if ATES is not None else np.nan):>14,.1f}")
    print(f"  {sname + ' extracted volume [m3]':<30}: {float(np.nansum(flow_ext)):>14,.1f}")
    print("-" * 64)
    print("  Heat-pump specifics")
    print(f"  {'HP condenser heat [GWh]':<30}: {out_HP.sum() / GWh:>10.4f}")
    print(f"  {'HP source heat / evap [GWh]':<30}: {out_evap.sum() / GWh:>10.4f}")
    print(f"  {'HP electricity P_el [GWh]':<30}: {P_el.sum() / GWh:>10.4f}")
    print(f"  {'HP mean COP (running)':<30}: {mean_cop:>10.4f}")
    print(f"  {'HP rated power [kW]':<30}: {hp_rating_kW:>10.1f}")
    print(f"  {'HP pricing mode':<30}: {('hourly (Eq. 2)' if hp_hourly else 'flat'):>10}")
    print(f"  {'HP mean price paid [eur/kWh]':<30}: {hp_price_paid:>10.4f}")
    print(f"  {'HP elec cost [euro/yr]':<30}: {hp_elec_cost:>14,.0f}")
    print(f"  {'HP elec cost flat [euro/yr]':<30}: {hp_flat_cost:>14,.0f}")
    if hp_breakdown:
        for k, v in hp_breakdown.items():
            print(f"    {k:<28}: {v:>14,.0f}")
    print(f"  {'HP CAPEX [euro] (in ' + sname + ')':<30}: {hp_capex_eur:>14,.0f}")
    a_b_d = (int((mode == 'A').sum()), int((mode == 'B').sum()), int((mode == 'D').sum())) \
        if mode.size else (0, 0, 0)
    print(f"  {'discharge hours A/B/D':<30}: {a_b_d[0]:>4} / {a_b_d[1]} / {a_b_d[2]}")
    print("=" * 64)

    print("\nEconomic analysis:")
    print(df_eco.to_string())
    print(f"\n  System LCOH (Yang, {p['LIFETIME_SYSTEM']}-yr)         : {system_lcoh_yang:.4f} euro/kWh")
    if co2_df is not None:
        print(f"  Total CO2                         : "
              f"{co2_df['CO2_emission [kg]'].sum() / 1000:,.1f} t/yr "
              f"-> {co2_df['Cost_CO2'].sum():,.0f} euro/yr")
    print("=" * 64)

    if make_plots:
        plt.show()

    # --- Geo-to-demand (charging removed) for RES; annual geo output for G/D ----
    geo_to_demand_GWh = (geo_corr - charge_kWh).sum() / GWh
    geo_prod_GWh = _col("Geothermal well production").sum() / GWh
    demand_GWh   = result["Demand"].sum() / GWh
    gd_ratio = (geo_prod_GWh / demand_GWh) if demand_GWh > 0 else np.nan

    # --- Storage nominal thermal capacity vs HP electrical rating --------------
    # ates_nominal = flow x rho*cp x (T_geo_supply - T_DHN_return): peak direct-HX
    # power of a freshly-charged well (charged to the geo supply temperature,
    # delivering down to the DHN return). Compared against the HP electrical
    # rating -> a fixed, COP-independent sizing metric. Uses the object's max_V,
    # because the BTES derives it from the number of boreholes.
    storage_max_V = ATES.max_V if ATES is not None else np.nan
    if use_ates:
        ates_nominal_kW = (storage_max_V / 3600.0) * RHO_CP * (GEO_T_OUT - DEMAND_T_OUT)
    else:
        ates_nominal_kW = np.nan
    ratio_ATES_HP = (ates_nominal_kW / HP_POWER_EL) if use_hp else np.nan
    storage_lcoh = df_eco.at[sname, "LCOE"] if sname in df_eco.index else np.nan

    # --- Return headline results so a sweep can collect rows -------------------
    # The ates_* / ATES_* keys are kept for the Delft Case scripts; they describe
    # the site's storage, whatever its type. storage_* are the same numbers.
    return {
        "config": cfg,
        "USE_HP": use_hp,
        "tag": tag,
        "site": SITE,
        "storage_type": storage_type if use_ates else None,
        "outfile": OUTFILE if write_excel else None,
        "GEO_POWER": GEO_POWER if use_geo else 0.0,
        "GD_ratio": gd_ratio,
        "ATES_MAX_V": storage_max_V if use_ates else np.nan,
        "ATES_LIFETIME": ATES.lifetime if use_ates else np.nan,
        "storage_max_V": storage_max_V if use_ates else np.nan,
        "storage_lifetime": ATES.lifetime if use_ates else np.nan,
        "storage_utilisation": float(getattr(ATES, "utilisation", np.nan)) if ATES is not None else np.nan,
        "storage_yearly_Reff": getattr(ATES, "yearly_Reff", None) if ATES is not None else None,
        "HP_POWER_EL": HP_POWER_EL if use_hp else np.nan,
        "HP_DELTA_T_COLDSIDE": HP_DELTA_T_COLDSIDE if use_hp else np.nan,
        "dynamic_dispatch": bool(HP_DYNAMIC_DISPATCH and use_hp),
        "threshold_eur_mwh": HP_THRESHOLD_EUR_MWH if (HP_DYNAMIC_DISPATCH and use_hp) else np.nan,
        "ates_nominal_kW": ates_nominal_kW,
        "ratio_ATES_HP": ratio_ATES_HP,
        "Reff": float(getattr(ATES, "Reff", np.nan)) if ATES is not None else np.nan,
        "injected_volume_m3": float(getattr(ATES, "volume", np.nan)) if ATES is not None else np.nan,
        "extracted_volume_m3": float(np.nansum(flow_ext)),
        "demand_GWh": demand_GWh,
        "geo_to_demand_GWh": geo_to_demand_GWh,
        "geo_GWh": geo_corr.sum() / GWh,
        "geo_prod_GWh": geo_prod_GWh,
        "ates_direct_GWh": ates_dir_corr.sum() / GWh,
        "hp_GWh": hp_corr.sum() / GWh,
        "gas_GWh": gas_corr.sum() / GWh,
        "unmet_GWh": float(np.clip(result["Demand"] - result["Total production"], 0, None).sum() / GWh),
        "system_lcoh_yang": system_lcoh_yang,
        "cost_split": cost_split,
        "geo_lcoh": df_eco.at["Geothermal well", "LCOE"] if "Geothermal well" in df_eco.index else np.nan,
        "ates_lcoh": storage_lcoh,
        "storage_lcoh": storage_lcoh,
        "storage_direct_GWh": ates_dir_corr.sum() / GWh,
        "gas_lcoh": df_eco.at["Gas boiler", "LCOE"] if "Gas boiler" in df_eco.index else np.nan,
        "hp_elec_GWh": P_el_total_kWh / GWh,
        "hp_elec_cost_eur": hp_elec_cost,
        "hp_elec_cost_flat_eur": hp_flat_cost,
        "hp_price_paid_eur_kwh": hp_price_paid,
        "hp_hourly_pricing": hp_hourly,
        "hp_mean_COP": mean_cop,
        "hp_capex_Meur": hp_capex_eur / 1e6,
        "total_CO2_t": (co2_df["CO2_emission [kg]"].sum() / 1000) if co2_df is not None else np.nan,
        "total_CO2_cost_eur": (co2_df["Cost_CO2"].sum()) if co2_df is not None else np.nan,
        "df_eco": df_eco,
        # Per-timestep arrays for the dispatch plots in the optimisation sweep.
        # T_extract is 0 and mode is 'off' on hours the ATES never ran, so plot
        # only where mode != 'off'.
        "ts_T_extract": T_extract,
        "ts_mode": mode_arr,
    }


# ================================================================== #
#  Single-run behaviour when executed directly (unchanged output)    #
# ================================================================== #
if __name__ == "__main__":
    run_case(make_plots=True, write_excel=True)