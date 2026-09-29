# -*- coding: utf-8 -*-
"""
site_config_Darmstadt.py
==================================================================
Site preset for model_driver.run_case(SITE="Darmstadt"): borehole thermal
energy storage (BTES, BTES_obj_Peter), SKEWS field at TU Darmstadt.

Precedence in run_case:  run_case(...) argument  >  this preset  >  object default.
STORAGE_KW goes straight into BTES_obj(...); everything not listed there is the
BTES_obj default (pipe radii, grout, fluid, g-function numerics, N_YEARS, ...).
max_V is NOT set: BTES_obj derives it as max_V_per_borehole * N_boreholes.

Source column: [SHEET] Parameters PUSH-IT.xlsx (System Inputs Darmstadt /
Econ_Inputs, TU Darmstadt), [DELFT] Delft value reused, [OBJ] BTES_obj default
(its own source tags: DATA / SKEWS / ASSUMED), [DERIVED] computed from those.
==================================================================
"""

PARAMS = dict(
    SITE_NAME       = "Darmstadt",
    STORAGE_TYPE    = "BTES",

    # --- District-heating demand ------------------------------------------
    DEMAND_EXAMPLE  = "Darmstadt",  # [-]  25 GWh/yr #P: demand file still to be added to demand_class
    DEMAND_T_IN     = 70,           # [C]  DHN supply #P: sheet 60-70 C (HP leave T 70 C) [SHEET]
    DEMAND_T_OUT    = 50,           # [C]  DHN return #P: sheet 45-55 C               [SHEET]

    # --- Heat source: modelled as a zero-cost geothermal plant --------------
    GEO_POWER       = 9000,         # [kW]                                           [SHEET]
    GEO_T_OUT       = 80,           # [C]  charging T #P: measured max 80-84 C (BTES_obj [DATA])
    GEO_COSTPERKW   = 0,            # [euro/kW] #P: not in the sheet
    GEO_FIXED_OPEX  = 0,            # [euro/kW/yr] #P: not in the sheet
    GEO_VAR_OPEX    = 0.15,         # [euro/kWh] #P: heat price 150 euro/MWh, check later [SHEET]
    GEO_LIFETIME    = 30,           # [yr] no CAPEX, so only sets the LCOE horizon  [DELFT]
    GEO_CO2         = 0,            # [gCO2/kWh] #P: not in the sheet

    # --- Back-up gas boiler -------------------------------------------------
    GAS_PRICE       = 0.055,        # [euro/kWh_gas] 55 euro/MWh                     [SHEET]
    GAS_CO2         = 200,          # [gCO2/kWh]                                     [SHEET]

    # --- Prices, economics, network ---------------------------------------
    CO2_PRICE         = 65,         # [euro/t]                                       [SHEET]
    ELEC_PRICE        = 0.2,        # [euro/kWh] flat (HP + storage pumps)           [SHEET]
    DISC_RATE         = 0.07,       # [-]                                            [SHEET]
    LIFETIME_SYSTEM   = 60,         # [yr]                                           [SHEET]
    LIFETIME_NETWORK  = 60,         # [yr]                                           [SHEET]
    NETWORK_LENGTH_M  = 4200.0,     # [m]                                            [SHEET]
    NETWORK_EUR_PER_M = 1157.0,     # [euro/m] #P: not in the sheet                  [DELFT]
    OPEX_NETWORK_PERC = 0.02,       # [1/yr] #P: not in the sheet                    [DELFT]

    # --- Heat pump on the storage discharge side (GGAH only) --------------
    HP_POWER_EL         = 1000,     # [kW_el] #P: not in the sheet, placeholder
    HP_DELTA_T_COLDSIDE = 20,       # [K] -> HP floor 30 C (> T_ground ~22 C) #P: not in the sheet [DELFT]
    HP_CAPEX            = 1200,     # [euro/kW_el] #P: not in the sheet             [DELFT]
    HP_FIXED_OPEX       = 50,       # [euro/kW_el/yr] #P: not in the sheet          [DELFT]
    HP_LIFETIME         = 20,       # [yr]                                           [SHEET]
    HP_CO2_EL           = 200,      # [gCO2/kWh_el] #P: not in the sheet            [DELFT]

    # --- Storage: BTES ----------------------------------------------------------
    STORAGE_ELEC_KWH_PER_M3 = 0.163,   # [kWh/m3] #P: not in the sheet; = BTES_obj pump
                                       # 30 m head / eta 0.5 (997*9.81*30/0.5/3.6e6)  [DERIVED]
    STORAGE_KW = dict(                 # -> BTES_obj(...)
        layout             = "triangular",
        N_boreholes        = 37,       # [-] #P: 3 built; 19 / 37 planned stages      [OBJ]
        B                  = 2.5,      # [m] spacing #P: built 8.6 m                  [OBJ]
        H                  = 750.0,    # [m] borehole length                          [OBJ]
        T_ground           = None,     # [C] None -> T_surface + gradient * mid-depth ~22.3 C
        T_surface          = 11.0,     # [C]                                          [OBJ]
        geo_gradient       = 0.03,     # [K/m]                                        [OBJ]
        k_s                = 2.8,      # [W/mK] ground conductivity                   [OBJ]
        pipe_type          = "coaxial",
        max_V_per_borehole = 9.25,     # [m3/h] measured max per BHE                  [OBJ]
        cost_per_m         = 1000.0,   # [euro/m drilled] #P: placeholder, not in the sheet [OBJ]
        fixed_opex_frac    = 0.01,     # [1/yr] share of CAPEX #P: not in the sheet   [OBJ]
        lifetime           = 50,       # [yr]                                         [SHEET]
    ),
)
