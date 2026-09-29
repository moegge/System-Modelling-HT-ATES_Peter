# -*- coding: utf-8 -*-
"""
MTES_obj_Peter.py
==================================================================
Mine Thermal Energy Storage (MTES) object for main2_Peter.system().

Drop-in replacement for ATES_obj_Peter.ATES_obj when the seasonal storage is a
flooded mine (water in old mine chambers, surrounded by rock) instead of an
aquifer. It exposes the SAME attributes and methods that main2_Peter.system(),
economic_analysis(), LCOE_calc() and system_plot() use on the storage object,
so the rest of the model does not care which technology sits in the storage slot.

PHYSICS
-------
Lumped two-node model from Till Spengler's thesis ("Co-Simulation of a District
Heating and Cooling System in Combination with Mine Thermal Energy Storage"),
exactly as implemented in MTES/mtes_class.py (MTES.do_step) and driven in
"MTES/MTES Bochum.py":

  node 1  water in the mine chambers, fully mixed ("tank")  : T_tank
  node 2  rock buffer around the chambers, one lumped node  : T_rock

  per timestep dt:
    (1) mixing      dT_tank = (m_in / m_tank) * (T_in - T_tank)          CSTR, forward Euler
    (2) conduction  Q_cond  = 2 pi L lambda_rock (T_tank - T_rock) dt / ln(r_th_max / r_tank)
                    T_tank -= Q_cond / (m_tank cp_water)                 eqs. 4.32 - 4.34
                    T_rock += Q_cond / (m_rock cp_rock)

  r_th_max is FIXED from an assumed maximum cycle length t_max (eq. 4.31,
  Sec. 4.2.4.4) and m_rock is the cylindrical rock shell between r_tank and
  r_th_max -- identical to "MTES Bochum.py" / execute_fmu_adaptable.py.
  (MTES.py, the Fig. 4.15 reproduction, lets the thermal front grow with
  sqrt(t); that is not usable for a multi-year run, so the fixed radius is used.)

HOW IT MAPS ONTO THE ATES INTERFACE
-----------------------------------
ATES_obj works from a pre-computed "extraction temperature vs. extracted volume"
curve (MODFLOW data, year 8) and walks that curve in calc_heat(). A lumped mine
has no such curve: its temperature is a STATE. Therefore

  initialize(volume, T_in, len_timestep)
      stores the annual charging volume and the charging temperature (= T_av
      from main2) and resets tank and rock to the undisturbed mine temperature.
  calc_heat(T_cutoff, T_demand_out, storage_extraction, missing_energy, hp_on=...)
      time-steps ONE FULL YEAR chronologically -- charging with
      self.flow_injected (set by main2 just before initialize()) at T_charge,
      discharging wherever missing_energy > 0 -- and repeats that identical year
      n_spinup_years times (default: the N_YEARS toggle at the top of this file;
      1 = a single year from the undisturbed mine at T_ground, 8 = ATES-like,
      rock in periodic steady state).
      The per-timestep arrays report the LAST year; the per-year
      totals fill total_heat_extracted_vs_T_ground_kWh_first_8_years, the
      8-year ramp main2's LCOE uses (for the ATES that array holds the 8 MODFLOW
      years).

Discharge in a timestep (same split as ATES_obj._energy_split, with the tank
temperature in place of the well curve). f = RECOVERY_FACTOR scales what ARRIVES,
applied after the ideal HX and BEFORE the heat pump:
  Q_dir  = f * C * max(0, T_tank - T_return)                    direct HX to the DHN
  Q_evap = min( f * C * max(0, min(T_tank, T_return) - T_floor), HP source heat,
                P_el,max * (COP - 1) * dt )                      capped by the compressor
  P_el   = Q_evap / (COP - 1);   delivered = Q_dir + Q_evap + P_el
  Q_water = (Q_dir + Q_evap) / f                                out of the mine water
The extracted water goes BACK INTO THE TANK at T_tank - Q_water/C: an MTES has no
cold well, and that return is what cools the mine down. Note Q_water, not the
delivered heat -- the (1 - f) recovery loss still cools the mine, which is what
keeps Reff at ~f instead of drifting back to 1.
Modes: 'A' HX only, 'B' HX + HP (T_tank >= T_return), 'D' HP only (T_tank < T_return).

INTERFACE CONTRACT (what main2_Peter reads / calls on the storage object)
-------------------------------------------------------------------------
  attributes : name, type='supply', control='storage', supplier, HP, T_g, max_V,
               capex, fix_opex, var_opex, lifetime, elec_price, volume,
               flow_injected, flow_extracted, output_dir, output_evap, output_HP,
               P_el, COP, T_extract, T_inject, mode, Reff,
               total_heat_extracted_vs_T_ground_kWh_first_8_years
  methods    : initialize(), init_cold_well(), calc_heat(), calc_opex(),
               calc_emissions(), set_reff()

NOTE ON main2_Peter.py (NOT changed by this file)
-------------------------------------------------
main2_Peter.py tests `i.name == "ATES"` in LCOE_calc(), LCOE_calc_Yang() and
economic_analysis(), and heat_pump_ATES.init() tests `ATES.name != 'ATES'`.
With name="MTES" the economics fall through to the generic per-technology
branch (works, but does NOT fold the HP CAPEX into the storage row) and
HP.init() raises. Widen those tests to `i.control == "storage"` (and
`result["ATES corrected"]` to `result[i.name + " corrected"]`) to run the MTES
through main2 with a heat pump.

Values from the thesis are flagged [THESIS]; choices copied from ATES_obj_Peter
for comparability [ATES-PARITY]; everything else is [ASSUMED].
==================================================================
"""
import os
import time
import warnings

import numpy as np
import pandas as pd

# Single definition of the cold-side temperature, shared with main2_Peter.system()
# and ATES_obj, so the charging Factor_due_HP and the discharge T_floor agree.
from ATES_obj_Peter import cold_well_T

_HERE = os.path.dirname(os.path.abspath(__file__))

# ================================================================== #
#  TOGGLE: number of identical years simulated back-to-back, and     #
#  therefore which year the results describe.                        #
#    1 -> one year from the undisturbed mine at T_ground: the fill-up #
#         year, most of the charge goes into warming water and rock. #
#    8 -> ATES-like: the 8th year of a repeating annual cycle, rock  #
#         in periodic steady state (ATES_obj reports year 8 of its   #
#         MODFLOW data; main2's LCOE ramp array also has 8 entries). #
#  A run can still override this with MTES_obj(..., n_spinup_years=).#
# ================================================================== #
N_YEARS = 10

# ================================================================== #
#  TOGGLE: recovery factor [-], 0 < f <= 1.                          #
#  A FLAT derating of the storage: the mine water still gives up the  #
#  full heat (so it cools at the same rate and the energy balance     #
#  closes), but only this share ARRIVES -- at the DHN through the HX, #
#  and at the HP evaporator. The rest is booked as a recovery loss.   #
#  Applied AFTER the ideal HX and BEFORE the heat pump, so the HP     #
#  sees the derated source heat and its COP/electricity follow.       #
#                                                                    #
#  DEFAULT IS NOW 1.0 (no flat derating), because the far-field loss  #
#  below is the physical mechanism and is temperature-dependent: the  #
#  HP case runs the mine colder, so it loses less, automatically.     #
#  A flat factor cannot do that -- it takes the same share whether    #
#  the mine is at 74 C or 55.2 C, which is exactly the HP/no-HP       #
#  distinction we want to keep. Use f < 1 only for losses that are    #
#  NOT conduction to the ground (HX effects, short-circuiting).       #
#  A run can override with MTES_obj(..., recovery_factor=).           #
# ================================================================== #
RECOVERY_FACTOR = 1.0

# ================================================================== #
#  TOGGLE: mine depth [m] below the ground surface.                  #
#  Only used to DERIVE the far-field heat loss when loss_W_per_K is  #
#  left at None (the default), from the standard conduction shape    #
#  factor for a horizontal cylinder buried below an isothermal        #
#  surface held at T_ground (Incropera, S = 2 pi L / arccosh(2z/D)):  #
#                                                                    #
#      G = 2 pi L lambda_rock / arccosh(z / r_th_max)     [W/K]       #
#                                                                    #
#  This replaces the flat RECOVERY_FACTOR as the loss mechanism. It   #
#  is TEMPERATURE-DEPENDENT (Q = G * (T_rock - T_ground)), so a mine  #
#  run colder -- e.g. the HP case, which pulls it to the 35 C floor   #
#  instead of 55 C -- loses proportionally less, which is the whole   #
#  point. Reff then comes out of the physics rather than a knob.      #
#  CAVEAT: this is the STEADY-STATE loss. The real loss is higher in  #
#  the first decades while the far field is still warming, so this is #
#  a lower bound. The rigorous alternative is a time-dependent ground #
#  response (g-function + load aggregation, as BTES_obj uses), which  #
#  would also remove the t_max_days assumption behind r_th_max.       #
#  Valid for z > 1.5 * r_th_max; warns and clips otherwise.           #
# ================================================================== #
MINE_DEPTH_M = 100.0

# ATES_obj keywords that have no meaning for a mine. Accepted and ignored (with a
# warning) so a driver can swap ATES_obj -> MTES_obj without touching every call.
_AQUIFER_ONLY_KWARGS = {"thickness", "porosity", "kh", "ani", "N_wells",
                        "HX_eta", "start_full_volume"}


class MTES_obj:
    """
    Mine Thermal Energy Storage: lumped water volume + lumped rock buffer.

    Parameters
    ----------
    supplier : list
        Supply objects that charge the storage (e.g. [geothermal]). main2 uses it
        to route surplus heat; calc_emissions uses it for the embodied CO2.
    V_tank : float, optional
        Water volume of the flooded mine chambers [m3] (default 7000, Bochum).
    L : float, optional
        Average chamber length [m] (default (207+150)/2 = 178.5). The chambers are
        modelled as ONE cylinder of volume V_tank and length L, so
        r_tank = sqrt(V_tank / (pi L)). [THESIS L, ASSUMED cylinder, as in MTES.py]
    lambda_rock, rho_rock, cp_rock : float, optional
        Rock conductivity [W/mK], density [kg/m3], heat capacity [J/kgK]
        (sandstone: 3.5, 2469.84, (800+2000)/2). [THESIS Table 4.2]
    t_max_days : float, optional
        Assumed maximum cycle length that FIXES the thermal buffer radius
        r_th_max = r_tank + 1.5 sqrt(alpha t_max) (eq. 4.31; default 100 d). [THESIS]
    density_fluid, heat_capacity_fluid : float, optional
        Water [kg/m3], [J/kgK] (997, 4180). [THESIS Table 4.2]
    T_ground : float, optional
        Undisturbed mine / rock temperature [C]: initial T_tank and T_rock
        (default 10, Bochum). Read by main2 as .T_g for the HP cold side.
    max_V : float, optional
        Pump rating [m3/h]: the most water that can be moved into or out of the
        mine per hour (default 150, pump 3 in the thesis). Also the sizing base
        for capex / fix_opex, exactly as in ATES_obj.
    min_dT_extract : float, optional
        Minimum useful temperature difference [K] between the mine water and the
        temperature it is returned at, below which the discharge pump does not
        start. Default 0.0 = IDEAL heat exchanger with no approach temperature:
        the mine is usable right down to the DHN return (or the HP floor), the
        same assumption ATES_obj makes. Raise it to represent a real HX approach
        temperature (e.g. 3 K -> the mine stops at 58 C against a 55 C return),
        which also acts as a deadband on the pump. [ASSUMED]
    min_heat_per_elec : float, optional
        Minimum useful heat per unit of pumping electricity [kWh_th / kWh_el],
        i.e. a COP for the circulation pump (default 5). The discharge pump does
        not start unless a FULL-flow pass would clear it, so the mine stops when
        circulating the water costs more than the heat is worth. Because both
        sides scale with flow this is a temperature deadband DERIVED from the
        pumping economics instead of guessed, and it follows pump_head_m,
        pump_efficiency, recovery_factor and the DHN levels automatically.
        Break-even is 1.0; below that the pump burns more electricity than it
        delivers heat. Set 0 to disable. Separates pump CONTROL from the HX
        physics in min_dT_extract. [ASSUMED]
    recovery_factor : float, optional
        Share of the extracted heat that actually arrives, 0 < f <= 1. None
        (default) -> the RECOVERY_FACTOR toggle at the top of this file. See the
        comment there: the water gives up the full heat, only f reaches the DHN /
        HP evaporator, and Reff settles at ~f instead of 1. [ASSUMED]
    pump_head_m : float, optional
        FRICTION head of the circulation loop [m] (default 20 m ~ 2 bar), used for
        the pumping electricity in calc_opex and for the min_heat_per_elec test.
        NOT the mine depth: in a closed doublet the water rises in one borehole and
        falls in the other, so the static head cancels and the pump only fights
        pipe / borehole / HX pressure drop. Replace with the real loop pressure
        drop if it is known -- it scales the pumping OPEX linearly and sets the
        temperature at which discharge stops. [ASSUMED]
    loss_W_per_K : float or None, optional
        Conductance from the rock node to the undisturbed far field at T_ground
        [W/K]. None (default) DERIVES it from the buried-cylinder conduction shape
        factor, G = 2 pi L lambda / arccosh(mine_depth_m / r_th_max) -- see the
        MINE_DEPTH_M block at the top. Because the loss is then proportional to
        (T_rock - T_ground), a mine run colder (the HP case) loses less, and Reff
        falls out of the physics instead of the flat RECOVERY_FACTOR. Pass a number
        to override; pass 0.0 to restore mtes_class.MTES exactly, which is lossless
        in the long run -- the rock only buffers -- so Reff tends to 1 after
        spin-up. Whether it was derived is recorded in loss_is_derived. [ASSUMED]
    mine_depth_m : float, optional
        Depth of the mine below the ground surface [m]. None (default) -> the
        MINE_DEPTH_M toggle at the top. ONLY used to derive loss_W_per_K; it is
        not the pumping head (see pump_head_m). [ASSUMED]
    costperm3 : float, optional
        CAPEX per m3/h of pump capacity [euro/(m3/h)] (default 3400000/320, the
        ATES value; the example files contain no MTES cost data). [ATES-PARITY]
    capex_fixed : float, optional
        Lump-sum CAPEX [euro] (shafts, access, HX) on top (default 0). [ASSUMED]
    cost_per_m3_tank : float, optional
        CAPEX per m3 of mine water volume [euro/m3] (default 0). [ASSUMED]
    fixed_opex : float, optional
        Fixed OPEX per m3/h of pump capacity [euro/(m3/h)/yr] (765.6). [ATES-PARITY]
    var_opex : float, optional
        Variable OPEX [euro/kWh] (default 2/40), read by main2's LCOE_calc. [ATES-PARITY]
    lifetime : int, optional
        Lifetime [yr] (default 25). [ATES-PARITY]
    elec_price : float, optional
        Flat electricity price for the pumps [euro/kWh] (default 0.2).
    pump_efficiency : float, optional
        Overall pump efficiency [-] (default 0.5). [ATES-PARITY]
    HP : heat_pump_ATES or None, optional
        Discharge-side heat pump, the same object as for the ATES.
    n_spinup_years : int, optional
        Identical years simulated back-to-back. None (default) -> the N_YEARS
        toggle at the top of this file. 1 = one year from the undisturbed mine at
        T_ground; 8 = ATES-like (length of main2's LCOE ramp array, rock in
        periodic steady state).
    name : str, optional
        Column prefix in main2's result DataFrame (default "MTES").
    timing, verbose : bool, optional
        Print runtime / a per-year table from calc_heat().

    Attributes (after calc_heat)
    ----------------------------
    T_tank_hist, T_rock_hist : np.ndarray
        Tank / rock temperature per timestep, last (steady-state) year [C].
    T_tank_hist_all, T_rock_hist_all : np.ndarray
        Same over the whole spin-up (n_spinup_years * n + 1 points).
    yearly_delivered_kWh, yearly_extracted_kWh, yearly_useful_kWh,
    yearly_charged_kWh, yearly_offered_kWh, yearly_loss_kWh,
    yearly_rec_loss_kWh : np.ndarray
        Per spin-up year: heat to the DHN (incl. HP electricity), heat taken out
        of the mine WATER (raw, what cools the mine), the part of that which
        arrived (= raw * recovery_factor), heat the tank actually absorbed, heat
        the supply side sent (main2's accounting: flow * (T_charge - T_floor)),
        far-field loss, and the recovery loss (raw - useful).
    yearly_Reff : np.ndarray
        Recovery efficiency of each spin-up year, useful / absorbed. Rises from
        well below its final value (year 1 fills cold water AND cold rock) to
        ~recovery_factor once the year is periodic. yearly_Reff[-1] is Reff.
    Reff : float
        Recovery efficiency of the mature year, useful / absorbed: the share of
        the heat that went into the mine that came back out AND arrived. Settles
        at ~recovery_factor (further reduced by loss_W_per_K if that is set).
        Comparable to the ATES well recovery efficiency.
    utilisation : float
        absorbed / offered: the share of the surplus main2 booked to the storage
        that the mine could actually take. Drops well below 1 when the mine is
        small compared to the summer surplus (the tank sits at T_charge).
    """

    def __init__(self, supplier,
                 # --- mine geometry ------------------------------------------------
                 V_tank=7000.0, L=(207 + 150) / 2,
                 # --- rock [THESIS Table 4.2] --------------------------------------
                 lambda_rock=3.5, rho_rock=2469.84, cp_rock=(800 + 2000) / 2,
                 t_max_days=100.0,
                 # --- water [THESIS Table 4.2] -------------------------------------
                 density_fluid=997.0, heat_capacity_fluid=4180.0,
                 # --- site ---------------------------------------------------------
                 T_ground=10.0, max_V=150.0, min_dT_extract=0.0, min_heat_per_elec=5.0,
                 pump_head_m=20.0, loss_W_per_K=None, mine_depth_m=None,
                 recovery_factor=None,
                 # --- economics ----------------------------------------------------
                 costperm3=651672/150, capex_fixed=0.0, cost_per_m3_tank=0.0,
                 fixed_opex=765.6, var_opex=2 / 40, lifetime=25,
                 elec_price=0.2, pump_efficiency=0.5,
                 # --- coupling / run control ---------------------------------------
                 HP=None, n_spinup_years=None, name="MTES",
                 timing=False, verbose=False,
                 **aquifer_kwargs):
        # Identity for main2_Peter.system()
        self.supplier = supplier
        self.name = name
        self.control = 'storage'
        self.type = 'supply'

        if aquifer_kwargs:
            unknown = set(aquifer_kwargs) - _AQUIFER_ONLY_KWARGS
            if unknown:
                raise TypeError(f"MTES_obj: unexpected keyword argument(s) {sorted(unknown)}")
            warnings.warn(f"MTES_obj: ignoring aquifer-only argument(s) "
                          f"{sorted(aquifer_kwargs)}", RuntimeWarning, stacklevel=2)

        # --- Geometry: one cylinder of volume V_tank and length L ---------------
        self.V_tank = float(V_tank)          # m3 water
        self.L = float(L)                    # m
        self.r_tank = np.sqrt(self.V_tank / (np.pi * self.L))          # m

        # --- Rock buffer: fixed thermal radius (eq. 4.31) and shell mass --------
        self.lambda_rock = lambda_rock       # W/mK
        self.rho_rock = rho_rock             # kg/m3
        self.cp_rock = cp_rock               # J/kgK
        self.alpha_rock = lambda_rock / (rho_rock * cp_rock)             # m2/s
        self.t_max = t_max_days * 24 * 3600                              # s
        self.r_th_max = 1.5 * np.sqrt(self.alpha_rock * self.t_max) + self.r_tank
        self.V_rock = np.pi * self.L * (self.r_th_max ** 2 - self.r_tank ** 2)
        self.m_rock = self.V_rock * rho_rock                             # kg
        self.ln_ratio = np.log(self.r_th_max / self.r_tank)
        # Conductance tank <-> rock (eq. 4.32 without dT and dt)
        self.G_cond = 2 * np.pi * self.L * lambda_rock / self.ln_ratio   # W/K
        # --- Far-field loss: rock buffer -> undisturbed ground -------------------
        # None -> derive from the buried-cylinder conduction shape factor (see the
        # MINE_DEPTH_M block at the top). A number overrides it; 0.0 restores
        # Till's original lossless model, where the rock only buffers.
        self.mine_depth_m = float(MINE_DEPTH_M if mine_depth_m is None else mine_depth_m)
        if loss_W_per_K is None:
            z_over_r = self.mine_depth_m / self.r_th_max
            if z_over_r <= 1.5:
                warnings.warn(
                    f"MTES_obj: mine_depth_m / r_th_max = {z_over_r:.2f} is outside the "
                    f"shape-factor's validity range (z > 1.5 r); clipped to 1.5. Give an "
                    f"explicit loss_W_per_K, or a larger mine_depth_m.",
                    RuntimeWarning, stacklevel=2)
                z_over_r = 1.5
            self.loss_W_per_K = (2 * np.pi * self.L * self.lambda_rock
                                 / np.arccosh(z_over_r))
            self.loss_is_derived = True
        else:
            self.loss_W_per_K = float(loss_W_per_K)
            self.loss_is_derived = False

        # --- Water --------------------------------------------------------------
        self.density = density_fluid         # kg/m3
        self.heat_cap = heat_capacity_fluid  # J/kgK
        self.m_tank = self.V_tank * density_fluid                        # kg
        self.C_tank_kWh_per_K = self.m_tank * self.heat_cap / 3.6e6      # kWh/K
        self.C_rock_kWh_per_K = self.m_rock * self.cp_rock / 3.6e6       # kWh/K

        # --- Site / operation ---------------------------------------------------
        self.T_g = float(T_ground)           # C, undisturbed mine temperature
        self.max_V = float(max_V)            # m3/h pump rating
        self.min_dT_extract = float(min_dT_extract)   # K
        self.min_heat_per_elec = float(min_heat_per_elec)   # [-] kWh_th per kWh_el
        self.recovery_factor = float(RECOVERY_FACTOR if recovery_factor is None
                                     else recovery_factor)
        if not 0.0 < self.recovery_factor <= 1.0:
            raise ValueError(f"recovery_factor must be in (0, 1], got "
                             f"{self.recovery_factor}")
        self.pump_head_m = float(pump_head_m)   # m, FRICTION head of the loop
        self.pump_efficiency = pump_efficiency
        # Pump electricity per m3 circulated: rho g h_fric / eta -> kWh/m3.
        # h_fric is the loop FRICTION head, NOT the mine depth: in a closed doublet
        # the water rises in one borehole and falls in the other, so the static head
        # cancels and the pump only fights pipe / borehole / HX pressure drop.
        # (ATES_obj charges rho g dh from the Thiem equation instead, which IS a real
        # dissipative loss -- pushing water through porous rock. Not applicable here.)
        self.pump_kWh_per_m3 = (density_fluid * 9.81 * self.pump_head_m
                                / pump_efficiency / 3.6e6)

        # --- Economics (same structure as ATES_obj so the LCOE maths is shared) --
        #P: CHECK THE CAPEX/OPEX AGAIN. All cost numbers here are carried over from
        #P: ATES_obj or assumed; none come from an MTES source. Open points:
        #P:  - capex scales with max_V (pump rating) only; cost_per_m3_tank and
        #P:    capex_fixed default to 0, so mine volume and shaft/access costs are
        #P:    currently free. Is max_V the right cost driver for a mine?
        #P:  - fixed_opex is still David's 765.6 euro/(m3/h)/yr, which against the
        #P:    current costperm3 is ~18 %/yr of capex (it was ~7 % for the ATES).
        #P:    Either rescale it with the same source or put it on a % basis.
        #P:  - var_opex = 2/40 is inherited and NEVER USED (calc_opex ignores it,
        #P:    main2 only reads var_opex on the suppliers). Delete or wire it up.
        #P:  - lifetime 25 yr is the ATES value; a mine reuse project is likely longer.
        #P:  - pump_head_m = 20 m friction is a guess -> scales pumping opex linearly.
        self.capex = capex_fixed + costperm3 * self.max_V + cost_per_m3_tank * self.V_tank
        self.fix_opex = fixed_opex * self.max_V   # euro/yr
        self.var_opex = var_opex                  # euro/kWh
        self.lifetime = lifetime
        self.elec_price = elec_price              # euro/kWh

        # --- Heat pump (same check as ATES_obj) ---------------------------------
        if HP is None:
            self.HP = None
        elif getattr(HP, "name", None) == "Heat pump":
            self.HP = HP
        else:
            print("MTES connected Heat pump is not recognised, set to no Heat pump")
            self.HP = None

        self.n_spinup_years = int(N_YEARS if n_spinup_years is None else n_spinup_years)
        self.timing = timing
        self.verbose = verbose

        # --- State and results (filled by initialize / calc_heat) ---------------
        self.T_tank = self.T_g
        self.T_rock = self.T_g
        self.volume = 0.0                    # m3/yr charged (set by initialize)
        self.T_charge = np.nan               # C   charging temperature
        self.len_timestep = 3600
        self.flow_injected = None            # m3 per timestep, set by main2
        self.flow_extracted = None
        self.Reff = np.nan
        self.utilisation = np.nan
        self.Reff_set = False
        self.total_heat_extracted_vs_T_ground_kWh_first_8_years = np.zeros(8)
        self._euler_warned = False

    # ------------------------------------------------------------------ #
    #  Till's two physics steps (mtes_class.MTES.do_step, split in two)   #
    # ------------------------------------------------------------------ #
    def _mix(self, T_in, v_m3):
        """
        Step 1 of MTES.do_step: v_m3 of water at T_in enters the fully mixed tank
        (the same volume leaves at T_tank). Forward Euler, so v_m3 <= V_tank is
        required; larger flows are clipped with a one-time warning.
        """
        if v_m3 <= 0.0:
            return
        if v_m3 > self.V_tank:
            if not self._euler_warned:
                warnings.warn(
                    f"MTES_obj: {v_m3:.0f} m3 per timestep exceeds the mine volume "
                    f"({self.V_tank:.0f} m3); flow clipped to one tank volume per step. "
                    f"Use a shorter timestep or a larger V_tank.", RuntimeWarning, stacklevel=3)
                self._euler_warned = True
            v_m3 = self.V_tank
        m_in = v_m3 * self.density
        Q_in = m_in * self.heat_cap * (T_in - self.T_tank)     # J, per step
        self.T_tank += Q_in / (self.m_tank * self.heat_cap)

    def _conduct(self, dt):
        """
        Step 2 of MTES.do_step: conduction tank <-> rock buffer (eqs. 4.32-4.34),
        plus the optional far-field loss from the rock node. Returns the far-field
        loss of this step [kWh] (0 with the default loss_W_per_K = 0).
        """
        Q_cond = self.G_cond * (self.T_tank - self.T_rock) * dt   # J
        self.T_tank -= Q_cond / (self.m_tank * self.heat_cap)
        self.T_rock += Q_cond / (self.m_rock * self.cp_rock)
        if self.loss_W_per_K > 0.0:
            Q_loss = self.loss_W_per_K * (self.T_rock - self.T_g) * dt   # J
            self.T_rock -= Q_loss / (self.m_rock * self.cp_rock)
            return Q_loss / 3.6e6
        return 0.0

    # ------------------------------------------------------------------ #
    #  ATES_obj interface                                                 #
    # ------------------------------------------------------------------ #
    def initialize(self, volume, T_in, len_timestep):
        """
        Called by main2 after it has set self.flow_injected (per-timestep flow to
        storage, m3) with volume = sum(flow_injected) and T_in = the flow-weighted
        supply temperature. Stores both and resets the mine to T_ground; the
        actual time-stepping happens in calc_heat(), which needs missing_energy.

        Standalone use: if flow_injected is not set (or has the wrong length),
        calc_heat() spreads `volume` over the summer with a half-cosine profile.
        """
        if volume < 1:
            print("Volume smaller than 1 m^3 per year, set to 0")
            volume = 0.0
        self.volume = float(volume)
        self.T_charge = float(T_in)
        self.len_timestep = len_timestep
        self.T_tank = self.T_g
        self.T_rock = self.T_g

    def init_cold_well(self, T_in, volume):
        """
        ATES parity only. An MTES has no separate cold well: the water that is
        cooled in the HX / HP evaporator flows straight back into the same mine
        volume. main2 calls this in the cold-well-loss loop for non-geothermal
        suppliers and reads cold_well_T_ave, so report 'no loss'.
        """
        self.cold_well_reff = 1.0
        self.cold_well_T_ave = float(T_in)

    def set_reff(self, Reff):
        """Force the reported recovery efficiency (only used by calc_emissions)."""
        if Reff > 1:
            raise ValueError("Reff higher than 100%, don't do that")
        self.Reff = Reff
        self.Reff_set = True

    def _injection_profile(self, n, dt):
        """Per-timestep charging flow [m3]: main2's flow_injected, else synthetic."""
        fi = self.flow_injected
        if fi is not None:
            fi = np.nan_to_num(np.asarray(fi, dtype=float))
            if len(fi) == n:
                return np.clip(fi, 0.0, None)
            warnings.warn(f"MTES_obj: flow_injected has {len(fi)} entries, expected "
                          f"{n}; using a synthetic summer profile instead.",
                          RuntimeWarning, stacklevel=3)
        if self.volume <= 0.0:
            return np.zeros(n)
        # Summer-peaking half cosine (positive half of -cos over the year): the
        # same shape ATES_obj.calculate_flow uses, at timestep resolution.
        shape = np.clip(-np.cos(2 * np.pi * np.arange(n) / n), 0.0, None)
        prof = shape / shape.sum() * self.volume
        cap = self.max_V * dt / 3600.0
        if prof.max() > cap:
            prof = np.minimum(prof, cap)
            warnings.warn(f"MTES_obj: synthetic charging profile capped at max_V = "
                          f"{self.max_V:g} m3/h; injected volume reduced to "
                          f"{prof.sum():.0f} m3/yr (asked {self.volume:.0f}).",
                          RuntimeWarning, stacklevel=3)
        self.flow_injected = prof
        return prof

    def _energy_split(self, flow_m3, T_tank, T_inj, T_demand_out, hp_running):
        """
        Split the heat delivered in one timestep into its physical components
        (identical to ATES_obj._energy_split, with the mixed tank temperature in
        place of the well curve's mean extraction temperature).

        The recovery factor f scales what ARRIVES: the mine water gives up the
        full heat (so it cools at the same rate and the energy balance closes),
        but only f of it reaches the DHN through the HX and the HP evaporator.
        The (1 - f) remainder is a recovery loss.

        flow_m3  : volume pumped out of the mine this timestep [m3]
        T_tank   : mine water temperature at the start of the step [C]
        T_inj    : target temperature of the water going back [C]
                   (= self.T_return if HP off, = self.T_floor if HP on)
        Returns (Q_dir, Q_evap, P_el, Q_tot, COP, Q_water) in kWh. The first four
        are USEFUL heat (post-factor); Q_water is what actually left the mine
        water (= (Q_dir + Q_evap) / f) and is what sets T_back.
        """
        f = self.recovery_factor
        C_A = flow_m3 * self.density * self.heat_cap / 3.6e6   # kWh/K

        # (a) Direct HX: only the part of the mine water ABOVE the DHN return.
        Q_dir_water = C_A * max(0.0, T_tank - self.T_return)   # out of the water
        Q_dir = f * Q_dir_water                                # into the DHN

        if not hp_running or self.HP is None:
            return Q_dir, 0.0, 0.0, Q_dir, np.nan, Q_dir_water

        # (b) Source heat available if this step's water is cooled to T_inj.
        #     Mode B: glide T_return -> T_floor.   Mode D: glide T_tank -> T_floor.
        #     f is applied HERE, before the HP, so the compressor sees the
        #     derated source heat and its COP / electricity follow from it.
        T_evap_in = min(T_tank, self.T_return)
        Q_evap_avail = f * C_A * max(0.0, T_evap_in - T_inj)
        if Q_evap_avail <= 0.0:
            return Q_dir, 0.0, 0.0, Q_dir, np.nan, Q_dir_water

        # (c) COP from the actual temperatures this step (varies in mode D).
        T_source = 0.5 * (T_evap_in + T_inj)
        COP = self.HP.Calculate_COP(T_demand_out, T_source)

        # (d) Fixed compressor power caps how much source heat can be moved.
        Q_evap_cap = self.HP.power_el * (COP - 1.0) * (self.len_timestep / 3600.0)
        Q_evap = min(Q_evap_avail, Q_evap_cap)
        P_el = Q_evap / (COP - 1.0)

        # Heat that left the water: the evaporator only received f of its share.
        Q_water = Q_dir_water + Q_evap / f
        return Q_dir, Q_evap, P_el, Q_dir + Q_evap + P_el, COP, Q_water

    def calc_heat(self, T_cutoff, T_demand_out, storage_extraction, missing_energy,
                  hp_on=None, hp_override_below_cutoff=True,
                  HP=None, len_timestep=3600, firstyear=False, control=None):
        """
        Time-step the mine through n_spinup_years identical years and return the
        heat delivered to the DHN per timestep in the last year [kWh] (= MTES + HP).

        Parameters
        ----------
        T_cutoff : float
            DHN return temperature. Below this the mine water cannot heat the
            network passively; it is the HX floor and the mode-A return temperature.
        T_demand_out : float
            DHN supply temperature (the HP condenser sink).
        storage_extraction : array
            Mask defining when the storage is allowed to discharge.
        missing_energy : pd.Series or array
            Heat still to be covered, per timestep [kWh].
        hp_on : array of bool, optional
            Per-timestep HP dispatch intent from main2. None -> HP never runs
            unless the mode-D override triggers.
        hp_override_below_cutoff : bool, optional
            Force the HP on whenever T_tank < T_cutoff (mode D), since the HX
            delivers nothing there.
        HP, firstyear : unused, kept for signature parity with ATES_obj.
        len_timestep : int
            Length of each timestep in seconds.
        control : str, optional
            "Peak shaving" is accepted but not implemented (warns).

        Per-timestep results of the last year are stored on the object:
        flow_extracted, output_dir, output_evap, output_HP, P_el, COP, T_extract,
        T_inject, mode, T_tank_hist, T_rock_hist. Per-year totals in yearly_*.
        """
        if self.timing:
            t0 = time.time()
        self.len_timestep = len_timestep
        dt = float(len_timestep)

        missing = np.nan_to_num(np.asarray(missing_energy, dtype=float)
                                * np.asarray(storage_extraction, dtype=float))
        n = len(missing)
        if control == "Peak shaving":
            warnings.warn("MTES_obj: control='Peak shaving' is not implemented; "
                          "discharging whatever the demand deficit allows.",
                          RuntimeWarning, stacklevel=2)

        flow_inj = self._injection_profile(n, dt)
        max_flow_step = min(self.max_V * dt / 3600.0, self.V_tank)     # m3 per step
        C_per_m3 = self.density * self.heat_cap / 3.6e6                # kWh/(m3 K)

        # NOTE: no "extracted <= injected" cap here, unlike ATES_obj. The aquifer's
        # injected volume IS its stored heat -- a hot bubble of finite extent, so
        # extracting past it would draw ambient groundwater at T_ground. The mine is
        # a CLOSED DOUBLET: a fixed inventory of V_tank m3 circulates from one
        # borehole through the surface HX/HP and back into the other. flow_injected
        # and flow_extracted are THROUGHPUT, not stock, and nothing runs out.
        # The only limits are the pump rating (max_flow_step) and the temperature
        # floor (min_dT_extract above T_return without HP, above T_floor with one).

        # --- Temperature levels (set once, never mutated) -----------------------
        self.T_return = float(T_cutoff)                                # HX floor
        self.T_floor = cold_well_T(T_cutoff, self.T_g, self.HP)        # HP cold side

        # --- HP dispatch intent from main2 --------------------------------------
        if hp_on is None or self.HP is None:
            hp_on = np.zeros(n, dtype=bool)
        else:
            if len(hp_on) != n:
                raise ValueError("hp_on must be one value per timestep")
            hp_on = np.asarray(hp_on).astype(bool)

        # --- Result arrays (last year) ------------------------------------------
        output              = np.zeros(n)   # total heat to the DHN [kWh]
        self.flow_extracted = np.zeros(n)
        self.output_dir     = np.zeros(n)   # via HX
        self.output_evap    = np.zeros(n)   # via HP source side
        self.output_HP      = np.zeros(n)   # HP condenser output = evap + P_el
        self.P_el           = np.zeros(n)   # compressor electricity
        self.COP            = np.full(n, np.nan)
        self.T_extract      = np.zeros(n)   # tank T when discharging
        self.T_inject       = np.full(n, float(T_cutoff))   # realised return T
        self.mode           = np.full(n, 'off', dtype=object)
        self.heat_charged   = np.zeros(n)   # heat absorbed by the tank [kWh]
        self.T_tank_hist    = np.zeros(n)
        self.T_rock_hist    = np.zeros(n)

        n_years = max(1, self.n_spinup_years)
        self.T_tank_hist_all = np.zeros(n_years * n + 1)
        self.T_rock_hist_all = np.zeros(n_years * n + 1)
        self.yearly_delivered_kWh = np.zeros(n_years)
        self.yearly_extracted_kWh = np.zeros(n_years)
        self.yearly_charged_kWh   = np.zeros(n_years)
        self.yearly_offered_kWh   = np.zeros(n_years)
        self.yearly_loss_kWh      = np.zeros(n_years)
        self.yearly_rec_loss_kWh  = np.zeros(n_years)   # (1-f) share, never arrives
        self.yearly_T_tank_start  = np.zeros(n_years)
        self.yearly_T_rock_start  = np.zeros(n_years)

        # Start from the undisturbed mine (calc_heat may be called more than once).
        self.T_tank = self.T_g
        self.T_rock = self.T_g
        self.T_tank_hist_all[0] = self.T_tank
        self.T_rock_hist_all[0] = self.T_rock

        if self.verbose:
            print(f"{'year':>4} {'T_tank Jan1':>11} {'T_rock Jan1':>11} {'offered':>12} "
                  f"{'charged':>12} {'extracted':>12} {'delivered':>12}   [kWh]")

        # --- Spin-up: identical years back-to-back -----------------------------
        for year in range(n_years):
            last = (year == n_years - 1)
            self.yearly_T_tank_start[year] = self.T_tank
            self.yearly_T_rock_start[year] = self.T_rock
            Q_deliv = Q_extr = Q_ch = Q_off = Q_loss = Q_rec_loss = 0.0

            for t in range(n):
                # ---- 1. Charging: supply water at T_charge into the mine ---------
                v_in = flow_inj[t]
                if v_in > 0.0:
                    # What the supply side booked as sent to storage (main2 divides
                    # the surplus volume by Factor_due_HP = (T_sup - T_cold)/(T_sup - T_ret)).
                    Q_off += v_in * C_per_m3 * max(0.0, self.T_charge - self.T_floor)
                    if self.T_charge > self.T_tank:     # HX cannot heat a warmer tank
                        T_before = self.T_tank
                        self._mix(self.T_charge, v_in)
                        q_abs = (self.T_tank - T_before) * self.C_tank_kWh_per_K
                        Q_ch += q_abs
                        if last:
                            self.heat_charged[t] = q_abs

                # ---- 2. Discharging: cover the deficit from the mine -------------
                if missing[t] > 0.0:
                    T_now = self.T_tank
                    hp_running = bool(hp_on[t])
                    # Mode-D override: below the DHN return the HX delivers nothing,
                    # so the HP is the only way to get heat out. Overrules main2.
                    if (self.HP is not None and hp_override_below_cutoff
                            and T_now < self.T_return):
                        hp_running = True
                    T_inj = self.T_floor if hp_running else self.T_return

                    # Mine down to the return level (no HP) or to the HP floor:
                    # nothing left to extract. With the default min_dT_extract = 0
                    # the HX is ideal, so the mine is usable right down to T_inj.
                    # This temperature limit -- together with the pump rating --
                    # is the ONLY thing that stops discharge.
                    flow = max_flow_step
                    Q_water = 0.0
                    if T_now - T_inj < self.min_dT_extract:
                        Q_tot = 0.0
                    else:
                        Q_dir, Q_evap, P_el, Q_tot, COP, Q_water = self._energy_split(
                            flow, T_now, T_inj, T_demand_out, hp_running)
                        # Pump control, as a "COP for the pump": run only if the heat
                        # that arrives is worth the electricity to circulate it.
                        # Both sides scale with flow, so this is a temperature
                        # deadband DERIVED from the pumping economics rather than
                        # guessed, and it re-derives itself when pump_head_m,
                        # pump_efficiency, recovery_factor or the DHN levels change.
                        # Tested on the capability (full flow), not on the demand-
                        # matched flow below, so a small deficit is still served.
                        if Q_tot < self.min_heat_per_elec * flow * self.pump_kWh_per_m3:
                            Q_tot = 0.0

                    # Don't over-deliver: shrink the flow to match missing_energy.
                    # Q_dir and Q_evap_avail are linear in flow, the compressor cap
                    # is not, hence the short fixed-point loop (as in ATES_obj).
                    if Q_tot > missing[t]:
                        for _ in range(20):
                            factor = missing[t] / Q_tot
                            if 0.995 <= factor <= 1.0:
                                break
                            flow *= factor
                            Q_dir, Q_evap, P_el, Q_tot, COP, Q_water = self._energy_split(
                                flow, T_now, T_inj, T_demand_out, hp_running)
                            if Q_tot <= 0.0:
                                break

                    # Mode D with the compressor capping Q_evap: there is no HX heat
                    # to gain from extra flow, so pump only what the evaporator can
                    # cool to T_floor. (In mode B the full flow stays: it feeds the HX,
                    # and the part the evaporator cannot take returns at T_return.)
                    # Uses the WATER-side heat, so the reduced flow still lands on T_inj.
                    if Q_tot > 0.0 and Q_dir <= 0.0 and Q_evap > 0.0:
                        flow = min(flow, Q_water / (C_per_m3 * (T_now - T_inj)))

                    if Q_tot > 0.0:
                        # The pumped water goes back into the mine at the temperature
                        # it leaves the HX / evaporator: that is what cools the tank.
                        # Q_water, not Q_tot: the recovery loss still cools the mine.
                        C_A = flow * C_per_m3
                        T_back = T_now - Q_water / C_A
                        self._mix(T_back, flow)
                        Q_deliv += Q_tot
                        Q_extr += Q_water
                        Q_rec_loss += Q_water - (Q_dir + Q_evap)

                        if last:
                            hp_active = (Q_evap > 0.0)
                            self.flow_extracted[t] = flow
                            self.output_dir[t]     = Q_dir
                            self.output_evap[t]    = Q_evap
                            self.output_HP[t]      = Q_evap + P_el
                            self.P_el[t]           = P_el
                            self.COP[t]            = COP
                            self.T_extract[t]      = T_now
                            self.T_inject[t]       = T_back
                            self.mode[t] = ('A' if not hp_active
                                            else ('B' if T_now >= self.T_return else 'D'))
                            output[t] = Q_tot

                # ---- 3. Conduction with the rock buffer (+ optional loss) --------
                Q_loss += self._conduct(dt)

                k = year * n + t + 1
                self.T_tank_hist_all[k] = self.T_tank
                self.T_rock_hist_all[k] = self.T_rock
                if last:
                    self.T_tank_hist[t] = self.T_tank
                    self.T_rock_hist[t] = self.T_rock

            self.yearly_delivered_kWh[year] = Q_deliv
            self.yearly_extracted_kWh[year] = Q_extr
            self.yearly_charged_kWh[year]   = Q_ch
            self.yearly_offered_kWh[year]   = Q_off
            self.yearly_loss_kWh[year]      = Q_loss
            self.yearly_rec_loss_kWh[year]  = Q_rec_loss
            if self.verbose:
                print(f"{year + 1:>4} {self.yearly_T_tank_start[year]:>11.2f} "
                      f"{self.yearly_T_rock_start[year]:>11.2f} {Q_off:>12,.0f} "
                      f"{Q_ch:>12,.0f} {Q_extr:>12,.0f} {Q_deliv:>12,.0f}")

        # --- Aggregates main2 reads ---------------------------------------------
        # 8-year ramp for LCOE_calc: years 1..7 as simulated, entry 8 = the mature
        # (last) year, which is also what the per-timestep arrays report.
        y = self.yearly_delivered_kWh
        if len(y) >= 8:
            ramp = np.concatenate([y[:7], [y[-1]]])
        else:
            ramp = np.concatenate([y, np.full(8 - len(y), y[-1])])
        self.total_heat_extracted_vs_T_ground_kWh_first_8_years = ramp

        # Heat that actually ARRIVED at the DHN/HP from the mine, per year: the
        # raw heat removed from the water minus the recovery loss.
        self.yearly_useful_kWh = self.yearly_extracted_kWh - self.yearly_rec_loss_kWh

        # Recovery efficiency of EVERY simulated year, so the spin-up can be
        # inspected/plotted. Depressed further while the rock is still warming up
        # (the mine absorbs more than it gives back); -> recovery_factor once the
        # year is periodic.
        with np.errstate(divide="ignore", invalid="ignore"):
            self.yearly_Reff = np.where(self.yearly_charged_kWh > 0,
                                        self.yearly_useful_kWh / self.yearly_charged_kWh,
                                        np.nan)

        # Mature-year efficiencies. Reff = what ARRIVED of what went in (the ATES
        # meaning), so it settles at ~recovery_factor rather than 1; utilisation =
        # what went in of what main2 booked to the storage.
        absorbed = self.yearly_charged_kWh[-1]
        offered = self.yearly_offered_kWh[-1]
        if not self.Reff_set:
            self.Reff = float(self.yearly_useful_kWh[-1] / absorbed) if absorbed > 0 else 0.0
        self.utilisation = float(absorbed / offered) if offered > 0 else 0.0

        self.heat_offered_kWh   = float(self.yearly_offered_kWh[-1])
        self.heat_charged_kWh   = float(self.yearly_charged_kWh[-1])
        self.heat_extracted_kWh = float(self.yearly_extracted_kWh[-1])
        self.heat_useful_kWh    = float(self.yearly_useful_kWh[-1])
        self.heat_rec_loss_kWh  = float(self.yearly_rec_loss_kWh[-1])
        self.heat_delivered_kWh = float(self.yearly_delivered_kWh[-1])

        # Backward compatibility: old main2 reads storage_obj.HP.COP
        if self.HP is not None:
            self.HP.COP = self.COP

        if self.timing:
            print(f"MTES calc_heat: {n_years} x {n} steps took {time.time() - t0:.2f}s")
        return output

    def energy_balance_kWh(self):
        """
        Closure of the last simulated year [kWh]:
            (dE_tank + dE_rock) - (charged - extracted - loss)
        Should be ~0; a quick self-test that mixing, discharge and conduction agree.
        """
        n = len(self.T_tank_hist)
        i0 = (self.n_spinup_years - 1) * n
        dE_tank = (self.T_tank_hist_all[-1] - self.T_tank_hist_all[i0]) * self.C_tank_kWh_per_K
        dE_rock = (self.T_rock_hist_all[-1] - self.T_rock_hist_all[i0]) * self.C_rock_kWh_per_K
        return (dE_tank + dE_rock) - (self.heat_charged_kWh - self.heat_extracted_kWh
                                      - float(self.yearly_loss_kWh[-1]))

    # ------------------------------------------------------------------ #
    #  Economics / emissions (called by main2.economic_analysis)          #
    # ------------------------------------------------------------------ #
    def calc_opex(self, kWh_generated):
        """Fixed OPEX + pump electricity for every m3 moved in and out of the mine."""
        #P: CHECK THIS AGAIN. kWh_generated is accepted but ignored (no variable
        #P: opex term), the electricity is priced flat at elec_price rather than on
        #P: the spot series the HP uses, and pumping is rho*g*h_fric/eta on EVERY m3
        #P: circulated in both directions -- no part-load, so the pump is charged at
        #P: full rate whenever it runs. Also note main2 only calls this when the
        #P: storage is NOT named "ATES"; the ATES branch uses its own fixed formula.
        try:
            pumped = 0.0
            if self.flow_injected is not None:
                pumped += float(np.nansum(self.flow_injected))
            if self.flow_extracted is not None:
                pumped += float(np.nansum(self.flow_extracted))
            opex = self.fix_opex + pumped * self.pump_kWh_per_m3 * self.elec_price
        except Exception:
            opex = self.fix_opex
        return opex

    def calc_emissions(self, result):
        """
        Embodied CO2 of the recovered stored heat, same construction as ATES_obj
        (extracted kWh inflated back to the supply heat it took), plus the HP
        compressor electricity. For the mine the supply heat is what main2 booked
        to the storage, i.e. extracted / (Reff * utilisation).
        """
        try:
            sum_CO2 = 0
            for i in self.supplier:
                sum_CO2 = i.CO2_kg + sum_CO2
            mine_kWh = float(np.nansum(self.output_dir) + np.nansum(self.output_evap))
            return_value = mine_kWh * (sum_CO2 / (self.Reff * self.utilisation)
                                       / len(self.supplier))
        except Exception:
            return_value = 0
        if self.HP is not None:
            return_value = return_value + self.HP.calc_emissions(result)
        return return_value

    def summary(self):
        """Print the sizing and the mature-year balance."""
        print("-" * 64)
        print(f"MTES '{self.name}': V_tank = {self.V_tank:,.0f} m3, L = {self.L:.1f} m, "
              f"r_tank = {self.r_tank:.2f} m, r_th_max = {self.r_th_max:.2f} m")
        print(f"  m_tank = {self.m_tank / 1e6:.2f} kt  ({self.C_tank_kWh_per_K / 1e3:.1f} MWh/K), "
              f"m_rock = {self.m_rock / 1e6:.1f} kt  ({self.C_rock_kWh_per_K / 1e3:.1f} MWh/K), "
              f"G_cond = {self.G_cond / 1e3:.1f} kW/K")
        print(f"  max_V = {self.max_V:g} m3/h, T_ground = {self.T_g:g} C, "
              f"capex = {self.capex / 1e6:.2f} Meuro, fix_opex = {self.fix_opex / 1e3:.1f} keuro/yr")
        # Deadband the min_heat_per_elec test works out to, for transparency.
        _c = self.density * self.heat_cap / 3.6e6 * self.recovery_factor  # kWh/(m3.K)
        _dT = (self.min_heat_per_elec * self.pump_kWh_per_m3 / _c) if _c > 0 else np.nan
        print(f"  far-field loss {self.loss_W_per_K / 1e3:.2f} kW/K "
              f"({'derived: z = %g m, r_th_max = %.2f m' % (self.mine_depth_m, self.r_th_max)
                 if self.loss_is_derived else 'given'}), "
              f"recovery factor {self.recovery_factor:g}")
        print(f"  pump head {self.pump_head_m:g} m friction "
              f"({self.pump_kWh_per_m3:.4f} kWh_el/m3), min heat/elec "
              f"{self.min_heat_per_elec:g} -> stops {_dT:.2f} K above the return/floor")
        if hasattr(self, "heat_delivered_kWh"):
            print(f"  charge {self.T_charge:g} C, {self.volume:,.0f} m3/yr | "
                  f"return {self.T_return:g} C, floor {self.T_floor:g} C")
            print(f"  mature year: offered {self.heat_offered_kWh / 1e3:,.0f} MWh, "
                  f"absorbed {self.heat_charged_kWh / 1e3:,.0f} MWh, "
                  f"out of the water {self.heat_extracted_kWh / 1e3:,.0f} MWh, "
                  f"useful {self.heat_useful_kWh / 1e3:,.0f} MWh, "
                  f"delivered {self.heat_delivered_kWh / 1e3:,.0f} MWh")
            print(f"  recovery factor {self.recovery_factor:.2f} -> "
                  f"recovery loss {self.heat_rec_loss_kWh / 1e3:,.0f} MWh "
                  f"({'HX + HP source side' if self.HP is not None else 'HX side'})")
            print(f"  Reff (useful/absorbed) = {self.Reff:.3f}, "
                  f"utilisation (absorbed/offered) = {self.utilisation:.3f}, "
                  f"pumped {np.nansum(self.flow_injected) + np.nansum(self.flow_extracted):,.0f} m3")
            modes = pd.Series(self.mode).value_counts()
            print(f"  hours by mode: " + ", ".join(f"{k}: {v}" for k, v in modes.items()))
            print(f"  energy balance closure: {self.energy_balance_kWh():.3f} kWh")
        print("-" * 64)


# ====================================================================== #
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------------ #
    #  Standalone test, no main2 needed: Bochum-size mine, synthetic year  #
    # ------------------------------------------------------------------ #
    dt = 3600
    n = 8760
    hours = np.arange(n)
    T_supply, T_return = 75.0, 55.0      # DHN, as in model_driver.py
    T_charge = 75.0                       # geothermal outlet = charging temperature

    # Winter deficit the storage should cover [kWh per hour]: peak on Jan 1 / Dec 31.
    winter = np.clip(np.cos(2 * np.pi * hours / n), 0.0, None)
    missing_energy = pd.Series(1200.0 * winter)          # 1.2 MW peak, ~3.3 GWh/yr
    storage_extraction = np.ones(n)

    # Charging volume for the summer: initialize() builds the half-cosine profile
    # itself when flow_injected is not set (main2 would supply the real one).
    Volume = 300_000                                      # m3/yr, peak ~108 m3/h < max_V

    WITH_HP = True
    hp = None
    if WITH_HP:
        try:
            from main2_Peter import heat_pump_ATES
            hp = heat_pump_ATES(power_el=250, delta_T_coldside=20)   # kW_el, K
        except Exception as e:                        # main2 pulls in the ATES stack
            print(f"Heat pump demo skipped ({e!r}); running without HP.")

    runs = {}
    cases = [("no HP", None), ("with HP", hp)]
    for k, (label, HP) in enumerate(cases, start=1):
        if label == "with HP" and HP is None:
            continue
        print("\n" + "#" * 70)
        print(f"#  RUN {k}/{len(cases)}: {label.upper()}   "
              f"(V_tank = 7000 m3, max_V = 150 m3/h, N_YEARS = {N_YEARS}"
              + (f", HP = {HP.power_el:g} kW_el, dT_cold = {HP.delta_T_coldside:g} K)" if HP is not None else ")"))
        print("#" * 70)
        mtes = MTES_obj([], V_tank=7000, max_V=150, T_ground=10, HP=HP,
                        verbose=True, timing=True)          # years: N_YEARS toggle at the top
        mtes.initialize(Volume, T_charge, dt)
        hp_on = np.ones(n, dtype=bool) if HP is not None else None   # HP allowed all year
        out = mtes.calc_heat(T_return, T_supply, storage_extraction, missing_energy,
                             hp_on=hp_on, len_timestep=dt)
        mtes.summary()
        print(f"  covers {out.sum() / missing_energy.sum() * 100:.1f} % of the winter deficit; "
              f"8-year ramp [MWh]: {np.round(mtes.total_heat_extracted_vs_T_ground_kWh_first_8_years / 1e3, 1)}")
        if HP is not None:
            cop = mtes.COP[np.isfinite(mtes.COP)]
            print(f"  HP: {np.nansum(mtes.P_el) / 1e3:,.0f} MWh_el, mean COP {cop.mean():.2f}")
        runs[label] = (mtes, out)

    # --- Figure 1: spin-up of tank and rock, one panel per run -----------------
    # Styled like the MTES_Rock_Temperatures plot in MTES/execute_fmu_adaptable.py:
    # blue water / orange rock at lw 1.8, grey dashed year separators, integer year
    # ticks, large fonts, legend below the axes.
    fig, axes = plt.subplots(len(runs), 1, figsize=(12, 5.2 * len(runs)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, (label, (m, _)) in zip(axes, runs.items()):
        yrs = np.arange(len(m.T_tank_hist_all)) / n

        ax.plot(yrs, m.T_tank_hist_all, label="Water Temperature [°C]",
                color="tab:blue", linewidth=1.8)
        ax.plot(yrs, m.T_rock_hist_all, label="Rock Temperature [°C]",
                color="tab:orange", linewidth=1.8)

        ax.set_title(f"MTES Water & Rock Temperature Over Time  ({label})", fontsize=18)
        ax.set_ylabel("Temperature [°C]", fontsize=15)
        ax.grid(True, alpha=0.5)

        # Integer year ticks + grey year separators, as in the reference figure.
        year_ticks = np.arange(1, int(round(yrs[-1])) + 1)
        ax.set_xticks(year_ticks)
        ax.set_xticklabels([str(y) for y in year_ticks], fontsize=12)
        ax.set_xlim(0, yrs[-1])
        for x in year_ticks:
            ax.axvline(x, color="gray", linestyle="--", linewidth=1)

    axes[-1].set_xlabel("Time [years]", fontsize=15)
    # One legend for the whole window: both panels carry the same two series.
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.10 if len(runs) > 1 else 0.20)
    fig.legend(*axes[0].get_legend_handles_labels(),
               loc="lower center", ncol=2, fontsize=13)

    # --- Figure 2: mature year, per run ----------------------------------------
    for label, (m, out) in runs.items():
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        days = hours / 24
        a1.plot(days, m.T_tank_hist, label="Tank", lw=1.2)
        a1.plot(days, m.T_rock_hist, label="Rock", lw=1.2, ls="--")
        a1.axhline(m.T_return, color="k", lw=0.6, ls=":", label="DHN return")
        if m.HP is not None:
            a1.axhline(m.T_floor, color="grey", lw=0.6, ls=":", label="HP floor")
        a1.set_ylabel("Temperature [C]")
        a1.set_title(f"MTES mature year ({label})")
        a1.grid(alpha=0.4)
        a1.legend(fontsize=8)

        kW = 3600 / dt
        a2.fill_between(days, 0, m.output_dir * kW, label="Direct (HX)", alpha=0.7)
        a2.fill_between(days, m.output_dir * kW, (m.output_dir + m.output_HP) * kW,
                        label="Heat pump", alpha=0.7)
        a2.fill_between(days, 0, -m.heat_charged * kW, label="Charging (absorbed)", alpha=0.5)
        a2.plot(days, missing_energy * kW, color="k", lw=0.6, label="Deficit")
        a2.set_xlabel("Day of year")
        a2.set_ylabel("Power [kW]")
        a2.grid(alpha=0.4)
        a2.legend(fontsize=8, loc="upper center")
        fig.tight_layout()

    # --- Figure 3: recovery efficiency per simulated year ----------------------
    # Top panel is the answer, bottom panel is the reason: while the rock is still
    # warming up the mine absorbs far more than it gives back, so Reff < 1. The two
    # curves close as the year becomes periodic, and Reff -> 1 (the model is
    # lossless unless loss_W_per_K > 0).
    from matplotlib.ticker import MaxNLocator

    fig, (b1, b2) = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True,
                                 gridspec_kw={"height_ratios": [2.0, 1.3]})
    colours = {"no HP": "tab:blue", "with HP": "tab:red"}
    for label, (m, _) in runs.items():
        yrs = np.arange(1, len(m.yearly_Reff) + 1)
        c = colours.get(label)

        b1.plot(yrs, m.yearly_Reff, "o-", ms=4.5, lw=1.5, color=c, label=label)
        b1.annotate(f"{m.yearly_Reff[-1]:.3f}", (yrs[-1], m.yearly_Reff[-1]),
                    textcoords="offset points", xytext=(7, -4),
                    fontsize=9, fontweight="bold", color=c)

        # First year within 0.5 % of the converged value -> "spun up" from here on.
        # Labels staggered so two runs converging in the same year stay readable.
        settled = np.where(np.abs(m.yearly_Reff - m.yearly_Reff[-1]) < 0.005)[0]
        if len(settled) and len(yrs) > 1:
            b1.axvline(yrs[settled[0]], color=c, lw=0.8, ls="--", alpha=0.5)
            b1.annotate(f"converged year {yrs[settled[0]]} ({label})",
                        (yrs[settled[0]], 0.08 + 0.09 * list(runs).index(label)),
                        textcoords="offset points", xytext=(5, 0),
                        fontsize=8, color=c, ha="left",
                        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))

        # Absorbed vs USEFUL (not the raw water-side heat, which equals absorbed
        # once periodic): the gap between the two curves IS the recovery loss, so
        # their ratio is the number plotted above.
        b2.plot(yrs, m.yearly_charged_kWh / 1e3, "o--", ms=3, lw=1.0,
                color=c, alpha=0.55, label=f"absorbed ({label})")
        b2.plot(yrs, m.yearly_useful_kWh / 1e3, "o-", ms=3, lw=1.5,
                color=c, label=f"useful ({label})")

    _f = runs[list(runs)[0]][0].recovery_factor
    b1.axhline(1.0, color="k", lw=0.8, ls=":", alpha=0.5,
               label="lossless model limit")
    if _f < 1.0:
        b1.axhline(_f, color="k", lw=0.9, ls="--", alpha=0.7,
                   label=f"RECOVERY_FACTOR = {_f:g}")
    b1.set_ylabel("Recovery efficiency\nuseful / absorbed  [-]")
    _m0 = runs[list(runs)[0]][0]
    b1.set_title(f"MTES recovery efficiency per simulated year "
                 f"(V_tank = {_m0.V_tank:,.0f} m3, "
                 f"RECOVERY_FACTOR = {_f:g}, "
                 f"far-field loss = {_m0.loss_W_per_K / 1e3:.2f} kW/K"
                 + (f" derived at z = {_m0.mine_depth_m:g} m)" if _m0.loss_is_derived else ")"))
    b1.set_ylim(0, 1.15)
    b1.grid(alpha=0.35)
    # "best" so the box dodges the curves, which now plateau anywhere from ~0.2
    # (small lossy mine) to ~1.0 (loss_W_per_K = 0) depending on the settings.
    b1.legend(fontsize=8.5, loc="best")

    b2.set_xlabel("Simulated year")
    b2.set_ylabel("Heat [MWh/yr]")
    b2.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Right margin scales with the run length so the end-value label always fits.
    _ny = max(len(m.yearly_Reff) for m, _ in runs.values())
    b2.set_xlim(0.5, _ny + max(0.6, 0.06 * _ny))
    b2.grid(alpha=0.35)
    b2.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    plt.show()
