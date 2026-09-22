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
      n_spinup_years times (default 1: a single year starting from the undisturbed
      mine at T_ground; 8 lets the rock warm up to its periodic steady state).
      The per-timestep arrays report the LAST year; the per-year
      totals fill total_heat_extracted_vs_T_ground_kWh_first_8_years, the
      8-year ramp main2's LCOE uses (for the ATES that array holds the 8 MODFLOW
      years).

Discharge in a timestep (same split as ATES_obj._energy_split, with the tank
temperature in place of the well curve):
  Q_dir  = C * max(0, T_tank - T_return)                        direct HX to the DHN
  Q_evap = min( C * max(0, min(T_tank, T_return) - T_floor),     HP source heat,
                P_el,max * (COP - 1) * dt )                      capped by the compressor
  P_el   = Q_evap / (COP - 1);   delivered = Q_dir + Q_evap + P_el
The extracted water goes BACK INTO THE TANK at T_tank - (Q_dir + Q_evap)/C: an
MTES has no cold well, and that return is what cools the mine down.
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
        The discharge pump is not started unless the mine water is at least this
        much [K] above the temperature it would be returned at (default 0.5).
        Stops the pump running at full flow for a few kWh once the mine has
        cooled to the return / HP floor. [ASSUMED]
    depth : float, optional
        Pumping lift [m] for the opex electricity (default 100). [ASSUMED]
    loss_W_per_K : float, optional
        Optional conductance from the rock node to the undisturbed far field at
        T_ground [W/K]. 0 (default) reproduces mtes_class.MTES exactly. NOTE that
        the model is then lossless in the long run -- the rock only buffers --
        so the recovery efficiency tends to 1 after spin-up. [ASSUMED]
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
        Identical years simulated back-to-back (default 1: one year from the
        undisturbed mine at T_ground). 8 = length of main2's LCOE ramp array and
        enough for the rock to reach its periodic steady state.
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
    yearly_delivered_kWh, yearly_extracted_kWh, yearly_charged_kWh,
    yearly_offered_kWh, yearly_loss_kWh : np.ndarray
        Per spin-up year: heat to the DHN (incl. HP electricity), heat taken out
        of the tank (Q_dir + Q_evap), heat the tank actually absorbed, heat the
        supply side sent (main2's accounting: flow * (T_charge - T_floor)), and
        far-field loss.
    Reff : float
        Recovery efficiency of the mature year, extracted / absorbed: the share
        of the heat that went into the mine that came back out (1.0 with the
        default lossless rock buffer; < 1 with loss_W_per_K > 0). Comparable to
        the ATES well recovery efficiency.
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
                 T_ground=10.0, max_V=150.0, min_dT_extract=0.5, depth=100.0,
                 loss_W_per_K=0.0,
                 # --- economics ----------------------------------------------------
                 costperm3=3400000 / 320, capex_fixed=0.0, cost_per_m3_tank=0.0,
                 fixed_opex=765.6, var_opex=2 / 40, lifetime=25,
                 elec_price=0.2, pump_efficiency=0.5,
                 # --- coupling / run control ---------------------------------------
                 HP=None, n_spinup_years=1, name="MTES",
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
        self.loss_W_per_K = float(loss_W_per_K)

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
        self.depth = depth                   # m pumping lift
        self.pump_efficiency = pump_efficiency
        # Pump electricity per m3 moved: rho g h / eta  ->  kWh/m3
        self.pump_kWh_per_m3 = density_fluid * 9.81 * depth / pump_efficiency / 3.6e6

        # --- Economics (same structure as ATES_obj so the LCOE maths is shared) --
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

        self.n_spinup_years = int(n_spinup_years)
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

        flow_m3  : volume pumped out of the mine this timestep [m3]
        T_tank   : mine water temperature at the start of the step [C]
        T_inj    : target temperature of the water going back [C]
                   (= self.T_return if HP off, = self.T_floor if HP on)
        Returns (Q_dir, Q_evap, P_el, Q_tot, COP) in kWh.
        """
        C_A = flow_m3 * self.density * self.heat_cap / 3.6e6   # kWh/K

        # (a) Direct HX: only the part of the mine water ABOVE the DHN return
        Q_dir = C_A * max(0.0, T_tank - self.T_return)

        if not hp_running or self.HP is None:
            return Q_dir, 0.0, 0.0, Q_dir, np.nan

        # (b) Source heat available if this step's water is cooled to T_inj.
        #     Mode B: glide T_return -> T_floor.   Mode D: glide T_tank -> T_floor.
        T_evap_in = min(T_tank, self.T_return)
        Q_evap_avail = C_A * max(0.0, T_evap_in - T_inj)
        if Q_evap_avail <= 0.0:
            return Q_dir, 0.0, 0.0, Q_dir, np.nan

        # (c) COP from the actual temperatures this step (varies in mode D).
        T_source = 0.5 * (T_evap_in + T_inj)
        COP = self.HP.Calculate_COP(T_demand_out, T_source)

        # (d) Fixed compressor power caps how much source heat can be moved.
        Q_evap_cap = self.HP.power_el * (COP - 1.0) * (self.len_timestep / 3600.0)
        Q_evap = min(Q_evap_avail, Q_evap_cap)
        P_el = Q_evap / (COP - 1.0)

        return Q_dir, Q_evap, P_el, Q_dir + Q_evap + P_el, COP

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
            Q_deliv = Q_extr = Q_ch = Q_off = Q_loss = 0.0
            V_extr = 0.0            # ATES-PARITY: annual extracted <= annual injected

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

                    # Mine (almost) at the return level: nothing worth pumping for.
                    # Or the year's injected volume has all been extracted (ATES-PARITY,
                    # same cap as ATES_obj.calc_heat's max_flow).
                    flow = min(max_flow_step, flow_inj.sum() - V_extr)
                    if T_now - T_inj < self.min_dT_extract or flow <= 0.0:
                        Q_tot = 0.0
                    else:
                        Q_dir, Q_evap, P_el, Q_tot, COP = self._energy_split(
                            flow, T_now, T_inj, T_demand_out, hp_running)

                    # Don't over-deliver: shrink the flow to match missing_energy.
                    # Q_dir and Q_evap_avail are linear in flow, the compressor cap
                    # is not, hence the short fixed-point loop (as in ATES_obj).
                    if Q_tot > missing[t]:
                        for _ in range(20):
                            factor = missing[t] / Q_tot
                            if 0.995 <= factor <= 1.0:
                                break
                            flow *= factor
                            Q_dir, Q_evap, P_el, Q_tot, COP = self._energy_split(
                                flow, T_now, T_inj, T_demand_out, hp_running)
                            if Q_tot <= 0.0:
                                break

                    # Mode D with the compressor capping Q_evap: there is no HX heat
                    # to gain from extra flow, so pump only what the evaporator can
                    # cool to T_floor. (In mode B the full flow stays: it feeds the HX,
                    # and the part the evaporator cannot take returns at T_return.)
                    if Q_tot > 0.0 and Q_dir <= 0.0 and Q_evap > 0.0:
                        flow = min(flow, Q_evap / (C_per_m3 * (T_now - T_inj)))

                    if Q_tot > 0.0:
                        # The pumped water goes back into the mine at the temperature
                        # it leaves the HX / evaporator: that is what cools the tank.
                        C_A = flow * C_per_m3
                        T_back = T_now - (Q_dir + Q_evap) / C_A
                        self._mix(T_back, flow)
                        Q_deliv += Q_tot
                        Q_extr += Q_dir + Q_evap
                        V_extr += flow

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

        # Mature-year efficiencies. Reff = what came back of what went in (the ATES
        # meaning); utilisation = what went in of what main2 booked to the storage.
        absorbed = self.yearly_charged_kWh[-1]
        offered = self.yearly_offered_kWh[-1]
        if not self.Reff_set:
            self.Reff = float(self.yearly_extracted_kWh[-1] / absorbed) if absorbed > 0 else 0.0
        self.utilisation = float(absorbed / offered) if offered > 0 else 0.0

        self.heat_offered_kWh   = float(self.yearly_offered_kWh[-1])
        self.heat_charged_kWh   = float(self.yearly_charged_kWh[-1])
        self.heat_extracted_kWh = float(self.yearly_extracted_kWh[-1])
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
        if hasattr(self, "heat_delivered_kWh"):
            print(f"  charge {self.T_charge:g} C, {self.volume:,.0f} m3/yr | "
                  f"return {self.T_return:g} C, floor {self.T_floor:g} C")
            print(f"  mature year: offered {self.heat_offered_kWh / 1e3:,.0f} MWh, "
                  f"absorbed {self.heat_charged_kWh / 1e3:,.0f} MWh, "
                  f"extracted {self.heat_extracted_kWh / 1e3:,.0f} MWh, "
                  f"delivered {self.heat_delivered_kWh / 1e3:,.0f} MWh")
            print(f"  Reff (extracted/absorbed) = {self.Reff:.3f}, "
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
    for label, HP in [("no HP", None), ("with HP", hp)]:
        if label == "with HP" and HP is None:
            continue
        mtes = MTES_obj([], V_tank=7000, max_V=150, T_ground=10, HP=HP,
                        n_spinup_years=1, verbose=True, timing=True)
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

    # --- Figure 1: spin-up of tank and rock ---------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for label, (m, _) in runs.items():
        yrs = np.arange(len(m.T_tank_hist_all)) / n
        ax.plot(yrs, m.T_tank_hist_all, lw=1.2, label=f"Tank ({label})")
        ax.plot(yrs, m.T_rock_hist_all, lw=1.2, ls="--", label=f"Rock ({label})")
    ax.axhline(T_return, color="k", lw=0.6, ls=":", label="DHN return")
    ax.set_xlabel("Time [years]")
    ax.set_ylabel("Temperature [C]")
    ax.set_title("MTES spin-up: mine water and rock buffer")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

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

    plt.show()
