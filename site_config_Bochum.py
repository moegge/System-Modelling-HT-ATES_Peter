# -*- coding: utf-8 -*-
"""
site_config_Bochum.py
==================================================================
Site preset for model_driver.run_case(SITE="Bochum"): mine thermal energy
storage (MTES, MTES_obj_Peter) charged by a CHP.

Precedence in run_case:  run_case(...) argument  >  this preset  >  object default.
STORAGE_KW goes straight into MTES_obj(...); everything not listed there is the
MTES_obj default (rock properties, t_max_days, N_YEARS, ...).

Source column: [SHEET] Parameters PUSH-IT.xlsx (System Inputs Bochum /
Econ_Inputs, mostly Thesis Till), [DELFT] Delft value reused, [OBJ] MTES_obj
default, [DERIVED] computed here from sheet values.
==================================================================
"""

# HP sizing: the sheet gives the HP in MWth; main2 sizes it in kW_el. Converted
# with the model's own COP (heat_pump_ATES.Calculate_COP) at a 90 C sink and a
# 50 C mean source (60 C return -> 40 C HP floor): fc = 0.35 + 0.6/200 * 40
# = 0.47 -> COP = 0.47 * 363 / 40 = 4.27.
_HP_COP_DESIGN = 4.27
_HP_MWTH       = 3.5            # [MWth] median of 1 / 3.5 / 7 MWth              [SHEET]
_HP_CAPEX_KWTH = 1500           # [euro/kWth] #P: sheet says euro/MWth, assumed kWth [SHEET]
_MTES_CAPEX    = 1351672 / 150  # [euro/(m3/h)] 1,351,672 euro at 150 m3/h, incl.
                                # 651,672 drilling + pump replacements            [SHEET]

PARAMS = dict(
    SITE_NAME       = "Bochum",
    STORAGE_TYPE    = "MTES",

    # --- District-heating demand ------------------------------------------
    DEMAND_EXAMPLE  = "Bochum",     # [-]  #P: demand file still to be added to demand_class
    DEMAND_T_IN     = 90,           # [C]  DHN supply                                [SHEET]
    DEMAND_T_OUT    = 60,           # [C]  DHN return                                [SHEET]

    # --- Heat source: CHP, modelled as a zero-cost geothermal plant ---------
    GEO_POWER       = 7500,         # [kW] #P: sheet says 7500 MWth, assumed kWth    [SHEET]
    GEO_T_OUT       = 90,           # [C]  #P: CHP supply = DHN supply assumed
    GEO_COSTPERKW   = 0,            # [euro/kW] CHP CAPEX not considered             [SHEET]
    GEO_FIXED_OPEX  = 0,            # [euro/kW/yr] not considered                    [SHEET]
    GEO_VAR_OPEX    = 0.15,         # [euro/kWh] #P: heat price 150 euro/MWh, check later [SHEET]
    GEO_LIFETIME    = 40,           # [yr] no CAPEX, so only sets the LCOE horizon  [SHEET]
    GEO_CO2         = 0,            # [gCO2/kWh] #P: CHP CO2 not in the sheet

    # --- Back-up gas boiler -------------------------------------------------
    GAS_PRICE       = 0.055,        # [euro/kWh_gas] #P: '?' in the sheet            [DELFT]
    GAS_CO2         = 233.5,        # [gCO2/kWh]                                     [SHEET]

    # --- Prices, economics, network ---------------------------------------
    CO2_PRICE         = 55,         # [euro/t] #P: sheet gives 25 / 55 / 65          [SHEET]
    ELEC_PRICE        = 0.08,       # [euro/kWh] flat (HP + storage pumps)           [SHEET]
    DISC_RATE         = 0.07,       # [-] #P: sheet gives 3 / 7 / 10 %               [SHEET]
    LIFETIME_SYSTEM   = 40,         # [yr] #P: sheet gives 30 / 40 / 50              [SHEET]
    LIFETIME_NETWORK  = 40,         # [yr]                                           [SHEET]
    NETWORK_LENGTH_M  = 0.0,        # [m] #P: not in the sheet
    NETWORK_EUR_PER_M = 1157.0,     # [euro/m] #P: not in the sheet                  [DELFT]
    OPEX_NETWORK_PERC = 0.02,       # [1/yr] #P: not in the sheet                    [DELFT]

    # --- Heat pump on the storage discharge side (GGAH only) --------------
    HP_POWER_EL         = round(_HP_MWTH * 1000 / _HP_COP_DESIGN),   # [kW_el] ~820  [DERIVED]
    HP_DELTA_T_COLDSIDE = 20,       # [K] #P: not in the sheet (sheet: HP leave T 50 C) [DELFT]
    HP_CAPEX            = round(_HP_CAPEX_KWTH * _HP_COP_DESIGN),    # [euro/kW_el] ~6400 [DERIVED]
    HP_FIXED_OPEX       = round(0.03 * _HP_CAPEX_KWTH * _HP_COP_DESIGN),  # [euro/kW_el/yr] 3 % of CAPEX [SHEET]
    HP_LIFETIME         = 40,       # [yr] 30 / 40 / 50                              [SHEET]
    HP_CO2_EL           = 434,      # [gCO2/kWh_el] grid                             [SHEET]

    # --- Storage: MTES ----------------------------------------------------------
    STORAGE_ELEC_KWH_PER_M3 = 0.233,   # [kWh/m3] pumping #P: sheet source '?'       [SHEET]
    STORAGE_KW = dict(                 # -> MTES_obj(...)
        V_tank          = 7000,        # [m3] mine water volume                      [SHEET]
        L               = (207 + 150) / 2,  # [m] mean chamber length               [OBJ]
        max_V           = 150,         # [m3/h] pump rating                          [SHEET]
        T_ground        = 10,          # [C]  undisturbed mine temperature           [OBJ]
        mine_depth_m    = 100.0,       # [m]  sets the far-field loss #P: check      [OBJ]
        recovery_factor = 1.0,         # [-]  flat derating off                      [OBJ]
        pump_head_m     = 20.0,        # [m]  loop friction head (pump control)      [OBJ]
        costperm3       = _MTES_CAPEX, # [euro/(m3/h)] ~9011                         [SHEET]
        fixed_opex      = 0.03 * _MTES_CAPEX,  # [euro/(m3/h)/yr] 3 % of CAPEX       [SHEET]
        lifetime        = 40,          # [yr] 30 / 40 / 50                           [SHEET]
    ),
)
