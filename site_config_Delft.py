# -*- coding: utf-8 -*-
"""
site_config_Delft.py
==================================================================
Site preset for model_driver.run_case(SITE="Delft"): HT-ATES.

Reproduces the model_driver defaults as they were before the site presets
(= the Delft Case results). Precedence in run_case:
    run_case(...) argument  >  this preset  >  object default.

STORAGE_KW goes straight into ATES_obj(...). The ATES_* arguments of run_case
(ATES_MAX_V, ATES_T_GROUND, ...) still override the matching entries.
Source column: [DAVID] paper David / Parameters PUSH-IT.xlsx, [DRIVER] previous
model_driver default.
==================================================================
"""

PARAMS = dict(
    SITE_NAME       = "Delft",
    STORAGE_TYPE    = "ATES",

    # --- District-heating demand ------------------------------------------
    DEMAND_EXAMPLE  = "Amsterdam",  # [-]  #P: driver default; Delft profiles: "Delft Peter", "TU Delft", ...
    DEMAND_T_IN     = 75,           # [C]  DHN supply (HP condenser sink)            [DAVID]
    DEMAND_T_OUT    = 55,           # [C]  DHN return (= storage cutoff / HX floor)  [DAVID]

    # --- Heat source: geothermal baseload (also charges the storage) ------
    GEO_POWER       = 5000,         # [kW]                                           [DRIVER]
    GEO_T_OUT       = 75,           # [C]                                            [DAVID]
    GEO_COSTPERKW   = 1909,         # [euro/kW]                                      [DAVID]
    GEO_FIXED_OPEX  = 69,           # [euro/kW/yr]                                   [DAVID]
    GEO_VAR_OPEX    = 0.0072,       # [euro/kWh]                                     [DAVID]
    GEO_LIFETIME    = 30,           # [yr]                                           [DAVID]
    GEO_CO2         = 12.5,         # [gCO2/kWh]                                     [DAVID]

    # --- Back-up gas boiler -------------------------------------------------
    GAS_PRICE       = 0.055,        # [euro/kWh_gas]                                 [DAVID]
    GAS_CO2         = 200,          # [gCO2/kWh]                                     [DAVID]

    # --- Prices, economics, network ---------------------------------------
    CO2_PRICE         = 150,        # [euro/t]                                       [DAVID]
    ELEC_PRICE        = 0.2,        # [euro/kWh] flat (HP + storage pumps)           [DAVID]
    DISC_RATE         = 0.05,       # [-]                                            [DAVID]
    LIFETIME_SYSTEM   = 60,         # [yr]                                           [DAVID]
    LIFETIME_NETWORK  = 60,         # [yr]                                           [DAVID]
    NETWORK_LENGTH_M  = 0.0,        # [m]  sweeps pass 8/15/23 km per G/D case       [DRIVER]
    NETWORK_EUR_PER_M = 1157.0,     # [euro/m]                                       [DAVID]
    OPEX_NETWORK_PERC = 0.02,       # [1/yr] share of network CAPEX                  [DAVID]

    # --- Heat pump on the storage discharge side (GGAH only) --------------
    HP_POWER_EL         = 1500,     # [kW_el] compressor rating                      [DRIVER]
    HP_DELTA_T_COLDSIDE = 20,       # [K] cooling below the DHN return               [DRIVER]
    HP_CAPEX            = 1200,     # [euro/kW_el]                                   [DRIVER]
    HP_FIXED_OPEX       = 50,       # [euro/kW_el/yr]                                [DRIVER]
    HP_LIFETIME         = 15,       # [yr]                                           [DRIVER]
    HP_CO2_EL           = 200,      # [gCO2/kWh_el] grid                             [DRIVER]

    # --- Storage: HT-ATES ----------------------------------------------------
    STORAGE_ELEC_KWH_PER_M3 = 1.389,   # [kWh/m3] pumping, David's OPEX formula      [DAVID]
    STORAGE_KW = dict(                 # -> ATES_obj(...)
        max_V     = 320,               # [m3/h] well rating                          [DAVID]
        thickness = 55,                # [m]  aquifer thickness                      [DRIVER]
        kh        = 10,                # [m/day] horizontal conductivity             [DRIVER]
        ani       = 5,                 # [-]  anisotropy                             [DRIVER]
        T_ground  = 15,                # [C]  undisturbed aquifer temperature        [DRIVER]
        lifetime  = 30,                # [yr]                                        [DAVID]
        # costperm3 = 3400000/320 and fixed_opex = 765.6: ATES_obj defaults
    ),
)
