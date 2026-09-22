# -*- coding: utf-8 -*-
"""
BTES_obj_Peter.py
==================================================================
Borehole Thermal Energy Storage (BTES) object for main2_Peter.system().

Drop-in replacement for ATES_obj_Peter.ATES_obj / MTES_obj_Peter.MTES_obj when
the seasonal storage is a field of closed-loop borehole heat exchangers (U-tubes
in grouted boreholes) instead of an aquifer or a flooded mine. It exposes the
SAME attributes and methods that main2_Peter.system(), economic_analysis(),
LCOE_calc() and system_plot() use on the storage object, so the rest of the
model does not care which technology sits in the storage slot.

PHYSICS  (pygfunction, https://github.com/MassimoCimmino/pygfunction)
-------
  ground   g-function of the field (pygfunction.gfunction.gFunction), computed
           ONCE per geometry and cached to disk (BTES/gfunction_cache). It is the
           dimensionless step response of the mean borehole wall temperature:
               T_b(t) = T_g - q' / (2 pi k_s) * g(t / t_s),   t_s = H^2 / (9 alpha)
           Hourly loads are superposed with Claesson-Javed / Kallstrom load
           aggregation (implemented here, vectorised, same scheme as
           pygfunction.load_aggregation.ClaesKallstrom), so a multi-year hourly
           run stays fast. aggregation_error() checks it against the exact
           superposition.
  borehole pygfunction pipe model (SingleUTube / MultipleUTube, multipole
           method) -> heat-exchanger effectiveness of ONE borehole as a function
           of its flow, tabulated once in __init__:
               eps(m_b) = (T_in - T_out) / (T_in - T_b)
           All boreholes are in PARALLEL and share the same wall temperature
           (boundary condition UBWT), so the field effectiveness equals the
           borehole effectiveness at m_b = m_field / N_b.
  coupling The wall temperature responds to the CURRENT step's load too
           (dT = g(dt)/(2 pi k_s) * q'), and the load depends on the wall
           temperature. Both are linear, so each step is solved explicitly:
               Q = C eps (T_b0 - T_in) / (1 + C eps g(dt) / (2 pi k_s H_tot))
           with T_b0 = wall temperature from the history alone, C = m cp.

  Unlike the lumped MTES model, the BTES has REAL losses: heat conducts away
  from the field into the surrounding ground and to the surface (which the
  g-function holds at T_g). No RECOVERY_FACTOR knob is needed; Reff comes out of
  the physics and rises over the spin-up years as the surrounding ground warms.

HOW IT MAPS ONTO THE ATES INTERFACE
-----------------------------------
A BTES is CLOSED-LOOP: it stores HEAT, not water. main2 hands the storage a
charging VOLUME (flow_injected, m3 per timestep) at T_av, which is aquifer
thinking; here that flow is THROUGHPUT through the field (like the MTES), and
the heat absorbed follows from the field's effectiveness and wall temperature:

  initialize(volume, T_in, len_timestep)
      stores the annual charging volume and the charging temperature (= T_av
      from main2).
  calc_heat(T_cutoff, T_demand_out, storage_extraction, missing_energy, hp_on=...)
      time-steps ONE FULL YEAR chronologically -- charging with
      self.flow_injected at T_charge, discharging wherever missing_energy > 0 --
      and repeats that identical year n_spinup_years times (the N_YEARS toggle
      below). The ground keeps its full thermal history across the years.
      The per-timestep arrays report the LAST year; the per-year delivered
      totals fill total_heat_extracted_vs_T_ground_kWh_first_8_years, the
      8-year ramp main2's LCOE uses.

Charging in a timestep (supply water at T_charge through the field):
  Q_abs  = C eps' (T_charge - T_b0)            only if T_charge > T_b0
  the water returns to the supplier at T_charge - Q_abs / C (T_charge_return).

Discharge in a timestep (same split as ATES_obj._energy_split, with the fluid
leaving the field, T_out, in place of the well temperature). The loop water
returns to the field at T_back (= T_return without HP, T_floor with HP):
  Q_ground = C eps' (T_b0 - T_back)             heat out of the ground
  T_out    = T_back + Q_ground / C
  Q_dir    = C * max(0, T_out - T_return)                         direct HX to the DHN
  Q_evap   = min( C * (min(T_out, T_return) - T_floor),           HP source heat,
                  P_el,max * (COP - 1) * dt )                     capped by the compressor
  P_el     = Q_evap / (COP - 1);   delivered = Q_dir + Q_evap + P_el
  When the compressor caps Q_evap, the water goes back warmer than T_floor and
  the ground gives less; that is solved exactly (still linear), so
  Q_ground = Q_dir + Q_evap always holds. The COP is evaluated on the target
  glide (min(T_out, T_return) -> T_floor), as in ATES_obj / MTES_obj.
Modes: 'A' HX only, 'B' HX + HP (T_out >= T_return), 'D' HP only (T_out < T_return).

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
Same situation as the MTES: main2_Peter.py tests `i.name == "ATES"` in
LCOE_calc(), LCOE_calc_Yang() and economic_analysis(), and
heat_pump_ATES.init() tests `ATES.name != 'ATES'`. With name="BTES" the
economics fall through to the generic per-technology branch (works, but does
NOT fold the HP CAPEX/OPEX into the storage row) and HP.init() raises. Widen
those tests to `i.control == "storage"` (and `result["ATES corrected"]` to
`result[i.name + " corrected"]`) to run the BTES through main2 with a heat pump.

MODEL LIMITS (built into the g-function method)
-----------------------------------------------
  - homogeneous ground (one k_s, one rho*c_s), no layers
  - NO groundwater flow (advection). Relevant in Dutch sandy layers.
  - ground surface held at T_g: a top insulation layer is NOT modelled
    (only approximated through the buried depth D)
  - constant properties, no moisture migration / drying out at high T
  - borehole and grout thermal capacity neglected (fine at hourly steps)
  - all boreholes in parallel, one mean wall temperature (no centre-to-edge
    stratification, no series-connected strings)

Values typical for shallow Dutch ground / standard BHE design are flagged
[TYPICAL]; choices copied from ATES_obj_Peter / MTES_obj_Peter for
comparability [ATES-PARITY]; everything else is [ASSUMED].
==================================================================
"""
import hashlib
import json
import math
import os
import time
import warnings

import numpy as np
import pandas as pd

try:
    import pygfunction as gt
except ImportError as _e:                                   # pragma: no cover
    raise ImportError(
        "BTES_obj_Peter needs pygfunction. Install it into the project venv:\n"
        "    .venv\\Scripts\\python.exe -m pip install pygfunction") from _e

# Single definition of the cold-side temperature, shared with main2_Peter.system()
# and ATES_obj / MTES_obj, so the charging Factor_due_HP and the discharge T_floor agree.
from ATES_obj_Peter import cold_well_T

_HERE = os.path.dirname(os.path.abspath(__file__))

# ====================================================================== #
#  BTES PARAMETERS -- every default of BTES_obj lives here.              #
#  A driver / sweep overrides any of them with BTES_obj(..., name=value).#
#  Parameters marked (g) change the g-function -> a new (cached) g-calc; #
#  all others reuse the cached g-function and are cheap to sweep.        #
# ====================================================================== #

# --- Run control ------------------------------------------------------------
#  Identical years simulated back-to-back; the results describe the LAST one.
#    1 -> first year from the undisturbed ground (most heat warms the ground)
#    8 -> ATES-like (ATES_obj reports year 8; main2's LCOE ramp has 8 entries)
#  A BTES needs several years to settle; the Reff figure shows when it does.
N_YEARS = 10

# --- Borefield geometry (g) -------------------------------------------------
N_X        = 20       # [-]  boreholes along x           -> N_b = N_X * N_Y   [ASSUMED]
N_Y        = 20       # [-]  boreholes along y                                  [ASSUMED]
B_SPACING  = 3.0      # [m]  borehole spacing (storage: 2-4 m)                  [TYPICAL]
H_BOREHOLE = 50.0     # [m]  active borehole length (storage: 30-100 m)         [TYPICAL]
D_BURIED   = 2.0      # [m]  buried depth of the borehole top                   [TYPICAL]
R_BOREHOLE = 0.075    # [m]  borehole radius (150 mm drill diameter)            [TYPICAL]

# --- Ground -----------------------------------------------------------------
K_GROUND      = 2.0     # [W/mK]   thermal conductivity (sat. sand/clay 1.5-2.5) (g) [TYPICAL]
RHO_CP_GROUND = 2.4e6   # [J/m3K]  volumetric heat capacity (2.0-2.8e6)          (g) [TYPICAL]
T_GROUND      = 10.0    # [C]      undisturbed ground temperature (NL ~10-12)        [ATES-PARITY]

# --- Borehole internals (set R_b / effectiveness, NOT the g-function) -------
PIPE_TYPE      = "single_U"   # "single_U" or "double_U" (double: 2 U-tubes in parallel)
R_PIPE_OUT     = 0.016        # [m]    pipe outer radius (32 mm pipe)             [TYPICAL]
R_PIPE_IN      = 0.0131       # [m]    pipe inner radius (SDR 11 -> 2.9 mm wall)  [TYPICAL]
D_SHANK        = 0.035        # [m]    pipe centre to borehole centre             [TYPICAL]
K_PIPE         = 0.40         # [W/mK] pipe wall (PE-RT / PE-X for high T)        [TYPICAL]
K_GROUT        = 1.5          # [W/mK] thermally enhanced grout                    [TYPICAL]
PIPE_ROUGHNESS = 1.0e-6       # [m]    pipe roughness (smooth plastic)             [TYPICAL]

# --- Heat-transfer fluid (water) ----------------------------------------------
RHO_FLUID = 997.0      # [kg/m3]  density                                         [ATES-PARITY]
CP_FLUID  = 4186.0     # [J/kgK]  heat capacity                                   [ATES-PARITY]
MU_FLUID  = 5.5e-4     # [Pa s]   dynamic viscosity (water ~50 C)                 [TYPICAL]
K_FLUID   = 0.64       # [W/mK]   thermal conductivity (water ~50 C)              [TYPICAL]

# --- Operation --------------------------------------------------------------
MAX_V           = 150.0  # [m3/h] field pump rating: most water through the field per hour [ATES-PARITY]
MIN_DT_EXTRACT  = 0.0    # [K]  wall must be this far above the return T to discharge [ASSUMED]
MIN_USEFUL_KW   = 20.0   # [kW] pump only starts if a full-flow pass delivers this much [ASSUMED]
PUMP_HEAD       = 30.0   # [m]  pressure head of field + headers, for pump electricity  [ASSUMED]
PUMP_EFFICIENCY = 0.5    # [-]  overall pump efficiency                              [ATES-PARITY]

# --- Economics of the BTES itself (HP, prices, LCOE maths stay in main2) -----
COST_PER_M      = 70.0   # [euro/m drilled] drilling + U-tube + grout             [ASSUMED]
CAPEX_FIXED     = 0.0    # [euro] headers, manifolds, top insulation, connection    [ASSUMED]
FIXED_OPEX_FRAC = 0.01   # [1/yr] fixed OPEX as a share of CAPEX                   [ASSUMED]
VAR_OPEX        = 2 / 40 # [euro/kWh] kept for interface parity                    [ATES-PARITY]
LIFETIME        = 50     # [yr] borehole heat exchangers                           [ASSUMED]
ELEC_PRICE      = 0.2    # [euro/kWh] flat price for the circulation pumps         [ATES-PARITY]

# --- g-function / numerics ----------------------------------------------------
GFUNC_METHOD     = "equivalent"  # pygfunction method: 'equivalent' (fast), 'similarities', 'detailed' (g)
GFUNC_BC         = "UBWT"        # uniform borehole wall temperature (g)
GFUNC_NSEGMENTS  = 8             # segments per borehole (g)
GFUNC_T_MIN      = 300.0         # [s]  first time of the g-function grid            (g)
GFUNC_T_MAX_YR   = 200.0         # [yr] last time of the g-function grid             (g)
GFUNC_N_TIMES    = 60            # [-]  log-spaced grid points (g interpolated in ln t) (g)
USE_GFUNC_CACHE  = True          # re-use g-functions stored in BTES/gfunction_cache
CELLS_PER_LEVEL  = 5             # load aggregation: cells per level (5 = pygfunction default)
N_EPS_TABLE      = 30            # flow points of the effectiveness table
# ====================================================================== #

_GFUNC_CACHE_DIR = os.path.join(_HERE, "BTES", "gfunction_cache")

# ATES_obj keywords that have no meaning for a borefield. Accepted and ignored
# (with a warning) so a driver can swap ATES_obj -> BTES_obj without touching
# every call.
_AQUIFER_ONLY_KWARGS = {"thickness", "porosity", "kh", "ani", "N_wells",
                        "HX_eta", "start_full_volume", "depth"}

# Discharge result when nothing can be extracted:
# (Q_dir, Q_evap, P_el, Q_tot, COP, Q_ground, T_out, T_back, Q_evap_avail, capped)
_NO_DISCHARGE = (0.0, 0.0, 0.0, 0.0, np.nan, 0.0, np.nan, np.nan, 0.0, False)


class BTES_obj:
    """
    Borehole Thermal Energy Storage: pygfunction g-function + load aggregation
    for the ground, pygfunction pipe model for the boreholes.

    Parameters (defaults: the BTES PARAMETERS block at the top of this file)
    ----------
    supplier : list
        Supply objects that charge the storage (e.g. [geothermal]). main2 uses it
        to route surplus heat; calc_emissions uses it for the embodied CO2.
    N_x, N_y : int
        Rectangular field of N_x * N_y boreholes.
    B, H, D, r_b : float
        Spacing, active length, buried depth and radius of the boreholes [m].
    k_s, rho_cp_s : float
        Ground conductivity [W/mK] and volumetric heat capacity [J/m3K].
    T_ground : float
        Undisturbed ground temperature [C]. Read by main2 as .T_g for the HP
        cold side.
    pipe_type : str
        "single_U" or "double_U".
    r_in, r_out, D_s, k_p, k_g, epsilon : float
        Pipe radii [m], shank spacing (pipe centre to borehole centre) [m], pipe
        and grout conductivity [W/mK], pipe roughness [m].
    density_fluid, heat_capacity_fluid, mu_fluid, k_fluid : float
        Water properties [kg/m3], [J/kgK], [Pa s], [W/mK].
    max_V : float
        Field pump rating [m3/h]: the most water through the field per hour,
        charging or discharging. main2 caps flow_injected with it.
    min_dT_extract : float
        Discharge only if the wall is at least this much [K] above the
        temperature the loop water returns at. 0 = ideal surface HX.
    min_useful_kW : float
        The discharge pump does not start unless a FULL-flow pass would deliver
        at least this much heat. Tested on the capability, not on the demand-
        matched flow, so small deficits are still served. 0 disables.
    pump_head, pump_efficiency : float
        Pressure head [m] and efficiency [-] for the circulation-pump electricity.
    cost_per_m, capex_fixed, fixed_opex_frac, var_opex, lifetime, elec_price :
        capex = capex_fixed + cost_per_m * N_b * H;  fix_opex = fixed_opex_frac * capex.
    gfunc_method, gfunc_bc, gfunc_nSegments : pygfunction g-function options.
    use_cache : bool
        Load / store the g-function in BTES/gfunction_cache.
    cells_per_level : int
        Load-aggregation resolution (more = more accurate, slower).
    HP : heat_pump_ATES or None
        Discharge-side heat pump, the same object as for the ATES.
    n_spinup_years : int or None
        None -> the N_YEARS toggle.
    name : str
        Column prefix in main2's result DataFrame (default "BTES").
    timing, verbose : bool
        Print runtime / a per-year table from calc_heat().

    Attributes (after calc_heat)
    ----------------------------
    T_b_hist, T_fluid_hist : np.ndarray
        Mean borehole wall temperature, and the temperature of the fluid leaving
        the field (discharge: T_out; charging: T_charge_return; idle: T_b), per
        timestep of the last year [C].
    T_b_hist_all, T_fluid_hist_all : np.ndarray
        Same over the whole spin-up (n_spinup_years * n + 1 points).
    q_hist_all : np.ndarray
        Net ground load per timestep over the whole run [W/m], extraction > 0.
    yearly_delivered_kWh, yearly_extracted_kWh, yearly_charged_kWh,
    yearly_offered_kWh, yearly_net_to_ground_kWh : np.ndarray
        Per spin-up year: heat to the DHN (incl. HP electricity), heat taken out
        of the ground, heat the ground absorbed, heat the supply side sent
        (main2's accounting: flow * (T_charge - T_floor)), and absorbed - extracted
        (losses to the surroundings + change in stored heat).
    yearly_Reff : np.ndarray
        extracted / absorbed per year. Low in year 1 (the surrounding ground is
        cold), rising as the field and its surroundings warm up.
    Reff : float
        Recovery efficiency of the mature (last) year, extracted / absorbed.
    utilisation : float
        absorbed / offered: the share of the surplus main2 booked to the storage
        that the field could actually take. Low when the field is small compared
        to the summer surplus (the wall is already near T_charge).
    """

    def __init__(self, supplier,
                 # --- borefield geometry -------------------------------------------
                 N_x=N_X, N_y=N_Y, B=B_SPACING, H=H_BOREHOLE, D=D_BURIED, r_b=R_BOREHOLE,
                 # --- ground -------------------------------------------------------
                 k_s=K_GROUND, rho_cp_s=RHO_CP_GROUND, T_ground=T_GROUND,
                 # --- borehole internals -------------------------------------------
                 pipe_type=PIPE_TYPE, r_in=R_PIPE_IN, r_out=R_PIPE_OUT, D_s=D_SHANK,
                 k_p=K_PIPE, k_g=K_GROUT, epsilon=PIPE_ROUGHNESS,
                 # --- fluid --------------------------------------------------------
                 density_fluid=RHO_FLUID, heat_capacity_fluid=CP_FLUID,
                 mu_fluid=MU_FLUID, k_fluid=K_FLUID,
                 # --- operation ----------------------------------------------------
                 max_V=MAX_V, min_dT_extract=MIN_DT_EXTRACT, min_useful_kW=MIN_USEFUL_KW,
                 pump_head=PUMP_HEAD, pump_efficiency=PUMP_EFFICIENCY,
                 # --- economics ----------------------------------------------------
                 cost_per_m=COST_PER_M, capex_fixed=CAPEX_FIXED,
                 fixed_opex_frac=FIXED_OPEX_FRAC, var_opex=VAR_OPEX,
                 lifetime=LIFETIME, elec_price=ELEC_PRICE,
                 # --- g-function / numerics ----------------------------------------
                 gfunc_method=GFUNC_METHOD, gfunc_bc=GFUNC_BC,
                 gfunc_nSegments=GFUNC_NSEGMENTS, use_cache=USE_GFUNC_CACHE,
                 cells_per_level=CELLS_PER_LEVEL,
                 # --- coupling / run control ---------------------------------------
                 HP=None, n_spinup_years=None, name="BTES",
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
                raise TypeError(f"BTES_obj: unexpected keyword argument(s) {sorted(unknown)}")
            warnings.warn(f"BTES_obj: ignoring aquifer-only argument(s) "
                          f"{sorted(aquifer_kwargs)}", RuntimeWarning, stacklevel=2)

        self.timing = timing
        self.verbose = verbose
        if timing:
            t0 = time.time()

        # --- Borefield geometry ---------------------------------------------------
        self.N_x, self.N_y = int(N_x), int(N_y)
        self.N_b = self.N_x * self.N_y                   # number of boreholes
        self.B, self.H, self.D, self.r_b = float(B), float(H), float(D), float(r_b)
        self.H_tot = self.N_b * self.H                   # m drilled (active)
        # Storage volume: every borehole "owns" a B x B column of ground.
        self.V_ground = (self.N_x * self.B) * (self.N_y * self.B) * self.H   # m3

        # --- Ground -----------------------------------------------------------------
        self.k_s = float(k_s)                            # W/mK
        self.rho_cp_s = float(rho_cp_s)                  # J/m3K
        self.alpha = self.k_s / self.rho_cp_s            # m2/s
        self.t_s = self.H ** 2 / (9 * self.alpha)        # s, characteristic time
        self.T_g = float(T_ground)                       # C
        self.C_ground_kWh_per_K = self.V_ground * self.rho_cp_s / 3.6e6

        # --- Borehole internals and fluid ---------------------------------------------
        if pipe_type not in ("single_U", "double_U"):
            raise ValueError(f"pipe_type must be 'single_U' or 'double_U', got {pipe_type!r}")
        self.pipe_type = pipe_type
        self.r_in, self.r_out, self.D_s = float(r_in), float(r_out), float(D_s)
        self.k_p, self.k_g, self.epsilon = float(k_p), float(k_g), float(epsilon)
        self.density = float(density_fluid)              # kg/m3
        self.heat_cap = float(heat_capacity_fluid)       # J/kgK
        self.mu_f = float(mu_fluid)                      # Pa s
        self.k_f = float(k_fluid)                        # W/mK
        self._C_per_m3 = self.density * self.heat_cap / 3.6e6   # kWh/(m3 K)

        # --- Operation --------------------------------------------------------------
        self.max_V = float(max_V)                        # m3/h
        self.min_dT_extract = float(min_dT_extract)      # K
        self.min_useful_kW = float(min_useful_kW)        # kW
        self.pump_head = float(pump_head)                # m
        self.pump_efficiency = float(pump_efficiency)
        # Pump electricity per m3 circulated: rho g h / eta  ->  kWh/m3
        self.pump_kWh_per_m3 = self.density * 9.81 * self.pump_head / self.pump_efficiency / 3.6e6

        # --- Economics (same structure as ATES_obj so the LCOE maths is shared) --------
        self.cost_per_m = float(cost_per_m)
        self.capex = float(capex_fixed) + self.cost_per_m * self.H_tot   # euro
        self.fix_opex = float(fixed_opex_frac) * self.capex              # euro/yr
        self.var_opex = var_opex                                         # euro/kWh
        self.lifetime = lifetime
        self.elec_price = elec_price                                     # euro/kWh

        # --- Heat pump (same check as ATES_obj) -----------------------------------------
        if HP is None:
            self.HP = None
        elif getattr(HP, "name", None) == "Heat pump":
            self.HP = HP
        else:
            print("BTES connected Heat pump is not recognised, set to no Heat pump")
            self.HP = None

        self.n_spinup_years = int(N_YEARS if n_spinup_years is None else n_spinup_years)
        self.cells_per_level = int(cells_per_level)

        # --- g-function (the expensive part: once per geometry, cached) ------------------
        self.gfunc_method = gfunc_method
        self.gfunc_bc = gfunc_bc
        self.gfunc_nSegments = int(gfunc_nSegments)
        self.use_cache = use_cache
        self.g_time, self.g_values = self._load_or_compute_gfunction()
        self._ln_g_time = np.log(self.g_time)

        # --- Borehole effectiveness vs flow (pygfunction pipe model) -----------------------
        self._build_eps_table()

        # --- State and results (filled by initialize / calc_heat) ------------------------
        self.volume = 0.0                    # m3/yr charged (set by initialize)
        self.T_charge = np.nan               # C   charging temperature
        self.len_timestep = 3600
        self.flow_injected = None            # m3 per timestep, set by main2
        self.flow_extracted = None
        self.Reff = np.nan
        self.utilisation = np.nan
        self.Reff_set = False
        self.total_heat_extracted_vs_T_ground_kWh_first_8_years = np.zeros(8)

        if timing:
            print(f"BTES __init__ (g-function + effectiveness table): {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------ #
    #  g-function: compute once per geometry, cache to disk               #
    # ------------------------------------------------------------------ #
    def _gfunc_spec(self):
        """Everything the g-function depends on. Hashed into the cache file name."""
        return {"layout": "rectangle", "N_x": self.N_x, "N_y": self.N_y,
                "B": self.B, "H": self.H, "D": self.D, "r_b": self.r_b,
                "alpha": float(f"{self.alpha:.10e}"),
                "method": self.gfunc_method, "bc": self.gfunc_bc,
                "nSegments": self.gfunc_nSegments,
                "grid": [GFUNC_T_MIN, GFUNC_T_MAX_YR, GFUNC_N_TIMES],
                "pygfunction": getattr(gt, "__version__", "?")}

    def _load_or_compute_gfunction(self):
        spec = self._gfunc_spec()
        key = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
        path = os.path.join(_GFUNC_CACHE_DIR, f"g_{self.N_x}x{self.N_y}_H{self.H:g}"
                                              f"_B{self.B:g}_{key}.npz")
        if self.use_cache and os.path.isfile(path):
            data = np.load(path)
            self.gfunc_source = f"cache ({os.path.basename(path)})"
            return data["t"], data["g"]

        t = np.geomspace(GFUNC_T_MIN, GFUNC_T_MAX_YR * 365 * 24 * 3600.0, GFUNC_N_TIMES)
        t0 = time.time()
        g = self._compute_gfunction(t)
        self.gfunc_source = f"computed in {time.time() - t0:.1f}s"
        if self.use_cache:
            try:
                os.makedirs(_GFUNC_CACHE_DIR, exist_ok=True)
                np.savez(path, t=t, g=g, spec=json.dumps(spec))
            except OSError as e:
                warnings.warn(f"BTES_obj: could not write g-function cache ({e})",
                              RuntimeWarning, stacklevel=2)
        return t, g

    def _compute_gfunction(self, t):
        """pygfunction g-function of the rectangular field at times t [s]."""
        field = gt.boreholes.rectangle_field(self.N_x, self.N_y, self.B, self.B,
                                             self.H, self.D, self.r_b)
        gfunc = gt.gfunction.gFunction(
            field, self.alpha, time=t, method=self.gfunc_method,
            boundary_condition=self.gfunc_bc,
            options={"nSegments": self.gfunc_nSegments, "disp": False})
        return np.asarray(gfunc.gFunc, dtype=float)

    def g_of_t(self, t):
        """g-function at time(s) t [s], interpolated in ln(t). g(0) = 0."""
        t = np.asarray(t, dtype=float)
        if np.any(t > self.g_time[-1] * 1.0001):
            warnings.warn("BTES_obj: run longer than the g-function grid "
                          f"({GFUNC_T_MAX_YR:g} yr); g held constant beyond it.",
                          RuntimeWarning, stacklevel=2)
        return np.interp(np.log(np.maximum(t, 1e-9)), self._ln_g_time, self.g_values,
                         left=self.g_values[0], right=self.g_values[-1])

    # ------------------------------------------------------------------ #
    #  Borehole: effectiveness of one borehole vs its flow                #
    # ------------------------------------------------------------------ #
    def _make_pipe(self, borehole, R_p):
        """pygfunction pipe object for one borehole with pipe resistance R_p [mK/W]."""
        if self.pipe_type == "single_U":
            pos = [(-self.D_s, 0.0), (self.D_s, 0.0)]
            return gt.pipes.SingleUTube(pos, self.r_in, self.r_out, borehole,
                                        self.k_s, self.k_g, R_p)
        # Double U: inlets first, then outlets, each inlet opposite its outlet.
        pos = [(-self.D_s, 0.0), (0.0, -self.D_s), (self.D_s, 0.0), (0.0, self.D_s)]
        return gt.pipes.MultipleUTube(pos, self.r_in, self.r_out, borehole,
                                      self.k_s, self.k_g, R_p, nPipes=2,
                                      config="parallel")

    def _pipe_at_flow(self, m_b):
        """Pipe object at a flow of m_b [kg/s] per borehole (sets convection)."""
        n_u = 1 if self.pipe_type == "single_U" else 2
        R_cond = gt.pipes.conduction_thermal_resistance_circular_pipe(
            self.r_in, self.r_out, self.k_p)
        h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
            m_b / n_u, self.r_in, self.mu_f, self.density, self.k_f,
            self.heat_cap, self.epsilon)
        R_p = R_cond + 1.0 / (h_f * 2 * np.pi * self.r_in)
        borehole = gt.boreholes.Borehole(self.H, self.D, self.r_b, 0.0, 0.0)
        return self._make_pipe(borehole, R_p)

    def _build_eps_table(self):
        """
        eps(m_b) = (T_in - T_out) / (T_in - T_b) of one borehole, from
        pygfunction's U-tube model (multipole resistances + the in/out leg
        short-circuit along the depth), tabulated over the flow range once.
        calc_heat interpolates it in ln(m_b).
        """
        m_design = self.max_V * self.density / 3600.0 / self.N_b       # kg/s per borehole
        m_b = np.geomspace(0.005 * m_design, 1.5 * m_design, N_EPS_TABLE)
        eps = np.empty_like(m_b)
        R_b = np.empty_like(m_b)
        for i, m in enumerate(m_b):
            pipe = self._pipe_at_flow(m)
            T_out = pipe.get_outlet_temperature(1.0, 0.0, m, self.heat_cap)
            eps[i] = 1.0 - float(np.squeeze(T_out))
            R_b[i] = float(np.squeeze(pipe.effective_borehole_thermal_resistance(m, self.heat_cap)))
        self._eps_m = m_b
        self._eps_ln_m = np.log(m_b)
        self._eps_tab = np.clip(eps, 0.0, 1.0)
        self._Rb_tab = R_b
        self.m_design_per_borehole = m_design
        self.eps_design = float(np.interp(np.log(m_design), self._eps_ln_m, self._eps_tab))
        self.R_b_design = float(np.interp(np.log(m_design), self._eps_ln_m, self._Rb_tab))

    def _eps_eff(self, flow_m3, dt):
        """
        Effectiveness of the whole field for flow_m3 [m3] this step, INCLUDING
        the wall's response to this step's own load (see PHYSICS). Parallel
        boreholes with one wall temperature -> field eps = borehole eps.
        """
        m_b = flow_m3 * self.density / dt / self.N_b
        eps = np.interp(math.log(m_b), self._eps_ln_m, self._eps_tab)
        C_W = flow_m3 * self.density * self.heat_cap / dt            # W/K
        return eps / (1.0 + C_W * eps * self._dgk0 / self.H_tot)

    # ------------------------------------------------------------------ #
    #  Load aggregation (Claesson-Javed / Kallstrom, vectorised)          #
    # ------------------------------------------------------------------ #
    def _build_aggregation(self, dt, n_steps):
        """
        Cells of growing width: cells_per_level cells of 1 step, then of 2, 4, 8 ...
        (pygfunction's ClaesKallstrom layout). Cell 0 is the current step. Each
        new step every cell j >= 1 loses one step of its mean load and gains one
        step of cell j-1's mean load. Returns widths w [steps] and
        dgk_j = (g(a_{j+1} dt) - g(a_j dt)) / (2 pi k_s)  [K per W/m], with a_j the
        age boundaries, so that  dT_b = sum_j dgk_j * q_j.
        """
        p = self.cells_per_level
        widths, total, i = [], 0, 0
        while total < n_steps + 1:
            i += 1
            w = 2 ** (int(math.ceil(i / p)) - 1)
            widths.append(w)
            total += w
        w = np.asarray(widths, dtype=float)
        ages = np.concatenate([[0.0], np.cumsum(w)])
        g_edges = np.concatenate([[0.0], self.g_of_t(ages[1:] * dt)])
        dgk = np.diff(g_edges) / (2 * np.pi * self.k_s)
        return w, dgk

    # ------------------------------------------------------------------ #
    #  ATES_obj interface                                                 #
    # ------------------------------------------------------------------ #
    def initialize(self, volume, T_in, len_timestep):
        """
        Called by main2 after it has set self.flow_injected (per-timestep flow to
        storage, m3) with volume = sum(flow_injected) and T_in = the flow-weighted
        supply temperature. Stores both; the actual time-stepping happens in
        calc_heat(), which needs missing_energy.

        Standalone use: if flow_injected is not set (or has the wrong length),
        calc_heat() spreads `volume` over the summer with a half-cosine profile.
        """
        if volume < 1:
            print("Volume smaller than 1 m^3 per year, set to 0")
            volume = 0.0
        self.volume = float(volume)
        self.T_charge = float(T_in)
        self.len_timestep = len_timestep

    def init_cold_well(self, T_in, volume):
        """
        ATES parity only. A BTES has no cold well: the loop water that is cooled
        in the HX / HP evaporator goes straight back into the boreholes. main2
        calls this in the cold-well-loss loop for non-geothermal suppliers and
        reads cold_well_T_ave, so report 'no loss'.
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
            warnings.warn(f"BTES_obj: flow_injected has {len(fi)} entries, expected "
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
            warnings.warn(f"BTES_obj: synthetic charging profile capped at max_V = "
                          f"{self.max_V:g} m3/h; injected volume reduced to "
                          f"{prof.sum():.0f} m3/yr (asked {self.volume:.0f}).",
                          RuntimeWarning, stacklevel=3)
        self.flow_injected = prof
        return prof

    def _discharge(self, flow_m3, T_b0, T_demand_out, hp_running, dt):
        """
        One discharge step at field flow flow_m3 [m3] with wall temperature T_b0
        (history only). Same split as ATES_obj._energy_split, with the fluid
        leaving the field (T_out) as the source temperature.

        Returns (Q_dir, Q_evap, P_el, Q_tot, COP, Q_ground, T_out, T_back,
                 Q_evap_avail, capped), energies in kWh per step.
        Q_ground = Q_dir + Q_evap is the heat taken out of the ground.
        """
        C_A = flow_m3 * self._C_per_m3                 # kWh/K for this step's water
        eps = self._eps_eff(flow_m3, dt)
        T_ret = self.T_return

        # (a) HX only: the loop water is cooled to the DHN return.
        if not hp_running or self.HP is None:
            if T_b0 <= T_ret:
                return _NO_DISCHARGE
            Q_g = C_A * eps * (T_b0 - T_ret)
            return (Q_g, 0.0, 0.0, Q_g, np.nan, Q_g, T_ret + Q_g / C_A, T_ret, 0.0, False)

        # (b) HX + HP: nominally the loop water goes back at T_floor.
        T_fl = self.T_floor
        if T_b0 <= T_fl:
            return _NO_DISCHARGE
        Q_g = C_A * eps * (T_b0 - T_fl)
        T_out = T_fl + Q_g / C_A
        T_evap_in = min(T_out, T_ret)
        Q_dir = C_A * max(0.0, T_out - T_ret)
        Q_evap_avail = C_A * (T_evap_in - T_fl)

        # (c) COP from the target glide (varies in mode D), as ATES_obj / MTES_obj.
        COP = self.HP.Calculate_COP(T_demand_out, 0.5 * (T_evap_in + T_fl))
        if COP <= 1.0:                                  # HP useless -> HX only
            return self._discharge(flow_m3, T_b0, T_demand_out, False, dt)

        # (d) Fixed compressor power caps the source heat. The water then goes back
        #     WARMER than T_floor, so the ground gives less: solve that exactly.
        Q_evap_cap = self.HP.power_el * (COP - 1.0) * (dt / 3600.0)
        T_back = T_fl
        capped = Q_evap_avail > Q_evap_cap
        if not capped:
            Q_evap = Q_evap_avail
        else:
            Q_evap = Q_evap_cap
            # Mode D (no HX heat): evaporator takes everything the ground gives.
            T_back = T_b0 - Q_evap / (C_A * eps)
            T_out = T_back + Q_evap / C_A
            if T_out <= T_ret:
                Q_g, Q_dir = Q_evap, 0.0
            else:
                # Mode B: HX takes T_out -> T_return, evaporator T_return -> T_back.
                T_back = T_ret - Q_evap / C_A
                Q_g = C_A * eps * (T_b0 - T_back)
                T_out = T_back + Q_g / C_A
                Q_dir = Q_g - Q_evap
        P_el = Q_evap / (COP - 1.0)
        return (Q_dir, Q_evap, P_el, Q_dir + Q_evap + P_el, COP, Q_g,
                T_out, T_back, Q_evap_avail, capped)

    def calc_heat(self, T_cutoff, T_demand_out, storage_extraction, missing_energy,
                  hp_on=None, hp_override_below_cutoff=True,
                  HP=None, len_timestep=3600, firstyear=False, control=None):
        """
        Time-step the borefield through n_spinup_years identical years and return
        the heat delivered to the DHN per timestep in the last year [kWh]
        (= BTES + HP).

        Parameters
        ----------
        T_cutoff : float
            DHN return temperature. Below this the loop water cannot heat the
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
            Force the HP on whenever the wall is below T_cutoff (mode D), since
            the HX delivers nothing there.
        HP, firstyear : unused, kept for signature parity with ATES_obj.
        len_timestep : int
            Length of each timestep in seconds.
        control : str, optional
            "Peak shaving" is accepted but not implemented (warns).

        Per-timestep results of the last year are stored on the object:
        flow_extracted, output_dir, output_evap, output_HP, P_el, COP, T_extract,
        T_inject, mode, heat_charged, T_charge_return, T_b_hist, T_fluid_hist.
        Per-year totals in yearly_*.
        """
        if self.timing:
            t0 = time.time()
        self.len_timestep = len_timestep
        dt = float(len_timestep)

        missing = np.nan_to_num(np.asarray(missing_energy, dtype=float)
                                * np.asarray(storage_extraction, dtype=float))
        n = len(missing)
        if control == "Peak shaving":
            warnings.warn("BTES_obj: control='Peak shaving' is not implemented; "
                          "discharging whatever the demand deficit allows.",
                          RuntimeWarning, stacklevel=2)

        flow_inj = self._injection_profile(n, dt)
        max_flow_step = self.max_V * dt / 3600.0                       # m3 per step
        C_per_m3 = self._C_per_m3                                      # kWh/(m3 K)
        min_useful = self.min_useful_kW * (dt / 3600.0)                # kWh per step

        # NOTE: no "extracted <= injected" cap, unlike ATES_obj. The loop is CLOSED:
        # flow_injected and flow_extracted are THROUGHPUT through the boreholes,
        # not a stock of water. The limits are the pump rating (max_flow_step) and
        # the wall temperature (above T_return without HP, above T_floor with one).

        # --- Temperature levels (set once, never mutated) ---------------------------
        self.T_return = float(T_cutoff)                                # HX floor
        self.T_floor = cold_well_T(T_cutoff, self.T_g, self.HP)        # HP cold side

        # --- HP dispatch intent from main2 ------------------------------------------
        if hp_on is None or self.HP is None:
            hp_on = np.zeros(n, dtype=bool)
        else:
            if len(hp_on) != n:
                raise ValueError("hp_on must be one value per timestep")
            hp_on = np.asarray(hp_on).astype(bool)

        # --- Ground: load aggregation over the whole spin-up ------------------------
        n_years = max(1, self.n_spinup_years)
        w, dgk = self._build_aggregation(dt, n_years * n)
        self._dgk0 = float(dgk[0])          # wall response to the CURRENT step's load
        dgk_hist = dgk[1:].copy()
        inv_w = 1.0 / w[1:]
        q = np.zeros(len(w))                # W/m per cell, extraction > 0; q[0] = now
        kWh_to_q = 3.6e6 / dt / self.H_tot  # kWh per step -> W per m of borehole

        # --- Result arrays (last year) ----------------------------------------------
        output               = np.zeros(n)   # total heat to the DHN [kWh]
        self.flow_extracted  = np.zeros(n)
        self.output_dir      = np.zeros(n)   # via HX
        self.output_evap     = np.zeros(n)   # via HP source side
        self.output_HP       = np.zeros(n)   # HP condenser output = evap + P_el
        self.P_el            = np.zeros(n)   # compressor electricity
        self.COP             = np.full(n, np.nan)
        self.T_extract       = np.zeros(n)   # fluid out of the field when discharging
        self.T_inject        = np.full(n, float(T_cutoff))   # realised loop return T
        self.mode            = np.full(n, 'off', dtype=object)
        self.heat_charged    = np.zeros(n)   # heat absorbed by the ground [kWh]
        self.T_charge_return = np.full(n, np.nan)   # water back to the supplier [C]
        self.T_b_hist        = np.zeros(n)
        self.T_fluid_hist    = np.zeros(n)

        self.T_b_hist_all     = np.zeros(n_years * n + 1)
        self.T_fluid_hist_all = np.zeros(n_years * n + 1)
        self.q_hist_all       = np.zeros(n_years * n)
        self.yearly_delivered_kWh = np.zeros(n_years)
        self.yearly_extracted_kWh = np.zeros(n_years)
        self.yearly_charged_kWh   = np.zeros(n_years)
        self.yearly_offered_kWh   = np.zeros(n_years)
        self.yearly_T_b_start     = np.zeros(n_years)
        self.T_b_hist_all[0] = self.T_g
        self.T_fluid_hist_all[0] = self.T_g

        if self.verbose:
            print(f"{'year':>4} {'T_b Jan1':>9} {'offered':>12} {'absorbed':>12} "
                  f"{'extracted':>12} {'delivered':>12} {'Reff':>6}   [kWh]")

        # --- Spin-up: identical years back-to-back -----------------------------------
        T_b_end = self.T_g
        for year in range(n_years):
            last = (year == n_years - 1)
            self.yearly_T_b_start[year] = T_b_end
            Q_deliv = Q_extr = Q_ch = Q_off = 0.0

            for t in range(n):
                s = year * n + t
                # ---- 0. Ground history: shift the aggregation cells one step ---------
                q[1:] += (q[:-1] - q[1:]) * inv_w
                q[0] = 0.0
                T_hist = self.T_g - float(dgk_hist @ q[1:])   # wall T without this step
                T_b0 = T_hist
                q_now = 0.0
                T_fluid = T_hist                              # idle loop sits at T_b

                # ---- 1. Charging: supply water at T_charge through the field ---------
                v_in = flow_inj[t]
                if v_in > 0.0:
                    # What the supply side booked as sent to storage (main2 divides
                    # the surplus volume by Factor_due_HP = (T_sup - T_cold)/(T_sup - T_ret)).
                    Q_off += v_in * C_per_m3 * max(0.0, self.T_charge - self.T_floor)
                    if self.T_charge > T_b0:          # HX cannot heat a warmer field
                        C_A = v_in * C_per_m3
                        q_abs = C_A * self._eps_eff(v_in, dt) * (self.T_charge - T_b0)
                        Q_ch += q_abs
                        q_now -= q_abs * kWh_to_q     # injection = negative extraction
                        T_b0 = T_hist - self._dgk0 * q_now   # wall warmed by the charge
                        T_fluid = self.T_charge - q_abs / C_A
                        if last:
                            self.heat_charged[t] = q_abs
                            self.T_charge_return[t] = T_fluid

                # ---- 2. Discharging: cover the deficit from the field ----------------
                if missing[t] > 0.0:
                    hp_running = bool(hp_on[t])
                    # Mode-D override: wall below the DHN return -> the HX delivers
                    # nothing, the HP is the only way to get heat out. Overrules main2.
                    if (self.HP is not None and hp_override_below_cutoff
                            and T_b0 < self.T_return):
                        hp_running = True
                    T_inj = self.T_floor if hp_running else self.T_return

                    flow = max_flow_step
                    r = _NO_DISCHARGE
                    if T_b0 - T_inj > max(self.min_dT_extract, 0.0):
                        r = self._discharge(flow, T_b0, T_demand_out, hp_running, dt)
                        # Pump control: not worth running if even FULL flow delivers
                        # less than min_useful_kW (tested on capability, not demand).
                        if r[3] < min_useful:
                            r = _NO_DISCHARGE

                    # Don't over-deliver: shrink the flow to match missing_energy.
                    # Heat is sub-linear in flow (eps rises as the flow drops), so
                    # the factor loop converges from above.
                    if r[3] > missing[t]:
                        for _ in range(30):
                            factor = missing[t] / r[3]
                            if 0.995 <= factor <= 1.0:
                                break
                            flow *= factor
                            r = self._discharge(flow, T_b0, T_demand_out, hp_running, dt)
                            if r[3] <= 0.0:
                                break

                    # Mode D with the compressor capping Q_evap: no HX heat to gain
                    # from extra flow, so pump only what the evaporator can cool to
                    # T_floor (same heat, less pumping), as MTES_obj does.
                    if r[3] > 0.0 and r[9] and r[0] <= 0.0:
                        for _ in range(30):
                            factor = r[1] / r[8]              # cap / available
                            if factor >= 0.995:
                                break
                            flow *= factor
                            r = self._discharge(flow, T_b0, T_demand_out, hp_running, dt)
                            if not r[9] or r[3] <= 0.0:
                                break

                    Q_dir, Q_evap, P_el, Q_tot, COP, Q_ground, T_out, T_back = r[:8]
                    if Q_tot > 0.0:
                        q_now += Q_ground * kWh_to_q
                        T_fluid = T_out
                        Q_deliv += Q_tot
                        Q_extr += Q_ground
                        if last:
                            hp_active = (Q_evap > 0.0)
                            self.flow_extracted[t] = flow
                            self.output_dir[t]     = Q_dir
                            self.output_evap[t]    = Q_evap
                            self.output_HP[t]      = Q_evap + P_el
                            self.P_el[t]           = P_el
                            self.COP[t]            = COP
                            self.T_extract[t]      = T_out
                            self.T_inject[t]       = T_back
                            self.mode[t] = ('A' if not hp_active
                                            else ('B' if T_out >= self.T_return else 'D'))
                            output[t] = Q_tot

                # ---- 3. Book this step's net load into the ground --------------------
                q[0] = q_now
                self.q_hist_all[s] = q_now
                T_b_end = T_hist - self._dgk0 * q_now
                self.T_b_hist_all[s + 1] = T_b_end
                self.T_fluid_hist_all[s + 1] = T_fluid
                if last:
                    self.T_b_hist[t] = T_b_end
                    self.T_fluid_hist[t] = T_fluid

            self.yearly_delivered_kWh[year] = Q_deliv
            self.yearly_extracted_kWh[year] = Q_extr
            self.yearly_charged_kWh[year]   = Q_ch
            self.yearly_offered_kWh[year]   = Q_off
            if self.verbose:
                reff = Q_extr / Q_ch if Q_ch > 0 else np.nan
                print(f"{year + 1:>4} {self.yearly_T_b_start[year]:>9.2f} {Q_off:>12,.0f} "
                      f"{Q_ch:>12,.0f} {Q_extr:>12,.0f} {Q_deliv:>12,.0f} {reff:>6.3f}")

        # --- Aggregates main2 reads ---------------------------------------------------
        # 8-year ramp for LCOE_calc: years 1..7 as simulated, entry 8 = the mature
        # (last) year, which is also what the per-timestep arrays report.
        y = self.yearly_delivered_kWh
        if len(y) >= 8:
            ramp = np.concatenate([y[:7], [y[-1]]])
        else:
            ramp = np.concatenate([y, np.full(8 - len(y), y[-1])])
        self.total_heat_extracted_vs_T_ground_kWh_first_8_years = ramp

        # Heat that stayed in the ground each year: losses to the surroundings and
        # the surface, plus (early on) the build-up of the stored heat itself.
        self.yearly_net_to_ground_kWh = self.yearly_charged_kWh - self.yearly_extracted_kWh

        # Recovery efficiency of EVERY simulated year, so the spin-up can be
        # inspected/plotted. Rises as the field and its surroundings warm up.
        with np.errstate(divide="ignore", invalid="ignore"):
            self.yearly_Reff = np.where(self.yearly_charged_kWh > 0,
                                        self.yearly_extracted_kWh / self.yearly_charged_kWh,
                                        np.nan)

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
            print(f"BTES calc_heat: {n_years} x {n} steps took {time.time() - t0:.2f}s")
        return output

    def aggregation_error(self):
        """
        Self-test of the load aggregation: recompute the wall temperature of the
        whole run by EXACT temporal superposition of the recorded loads
        (T_b[s] = T_g - sum_k q_k (g((s-k+1) dt) - g((s-k) dt)) / (2 pi k_s), by FFT)
        and return the largest deviation from the simulated T_b [K].
        """
        from scipy.signal import fftconvolve
        N = len(self.q_hist_all)
        dt = float(self.len_timestep)
        g = np.concatenate([[0.0], self.g_of_t(np.arange(1, N + 1) * dt)])
        dgk = np.diff(g) / (2 * np.pi * self.k_s)
        T_exact = self.T_g - fftconvolve(self.q_hist_all, dgk)[:N]
        return float(np.max(np.abs(T_exact - self.T_b_hist_all[1:])))

    # ------------------------------------------------------------------ #
    #  Economics / emissions (called by main2.economic_analysis)          #
    # ------------------------------------------------------------------ #
    def calc_opex(self, kWh_generated):
        """Fixed OPEX + circulation-pump electricity for every m3 through the field."""
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
        Embodied CO2 of the recovered stored heat, same construction as ATES_obj /
        MTES_obj (extracted kWh inflated back to the supply heat it took, i.e.
        divided by Reff * utilisation), plus the HP compressor electricity.
        """
        try:
            sum_CO2 = 0
            for i in self.supplier:
                sum_CO2 = i.CO2_kg + sum_CO2
            ground_kWh = float(np.nansum(self.output_dir) + np.nansum(self.output_evap))
            return_value = ground_kWh * (sum_CO2 / (self.Reff * self.utilisation)
                                         / len(self.supplier))
        except Exception:
            return_value = 0
        if self.HP is not None:
            return_value = return_value + self.HP.calc_emissions(result)
        return return_value

    def summary(self):
        """Print the sizing and the mature-year balance."""
        print("-" * 72)
        print(f"BTES '{self.name}': {self.N_x} x {self.N_y} = {self.N_b} boreholes, "
              f"H = {self.H:g} m, B = {self.B:g} m, D = {self.D:g} m, r_b = {self.r_b:g} m")
        print(f"  drilled {self.H_tot:,.0f} m, ground volume {self.V_ground:,.0f} m3 "
              f"({self.C_ground_kWh_per_K / 1e3:.1f} MWh/K), "
              f"k_s = {self.k_s:g} W/mK, t_s = {self.t_s / 3.1536e7:.1f} yr")
        print(f"  g-function ({self.gfunc_method}, {self.gfunc_bc}, {self.gfunc_source}): "
              f"g(1 d) = {float(self.g_of_t(86400)):.2f}, "
              f"g(1 yr) = {float(self.g_of_t(3.1536e7)):.2f}, "
              f"g(10 yr) = {float(self.g_of_t(3.1536e8)):.2f}")
        print(f"  {self.pipe_type}: design flow {self.m_design_per_borehole:.3f} kg/s per "
              f"borehole, R_b,eff = {self.R_b_design:.3f} mK/W, eps = {self.eps_design:.3f}")
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
            print(f"  wall temperature mature year: {self.T_b_hist.min():.1f} - "
                  f"{self.T_b_hist.max():.1f} C")
            modes = pd.Series(self.mode).value_counts()
            print(f"  hours by mode: " + ", ".join(f"{k}: {v}" for k, v in modes.items()))
            print(f"  load aggregation vs exact superposition: max error "
                  f"{self.aggregation_error():.3f} K")
        print("-" * 72)


# ====================================================================== #
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # ------------------------------------------------------------------ #
    #  Standalone test, no main2 needed: default field, synthetic year     #
    #  (same demand, charging volume and HP as the MTES_obj_Peter demo)    #
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
        print("\n" + "#" * 72)
        print(f"#  RUN {k}/{len(cases)}: {label.upper()}   "
              f"({N_X} x {N_Y} boreholes, H = {H_BOREHOLE:g} m, max_V = {MAX_V:g} m3/h, "
              f"N_YEARS = {N_YEARS}"
              + (f", HP = {HP.power_el:g} kW_el, dT_cold = {HP.delta_T_coldside:g} K)"
                 if HP is not None else ")"))
        print("#" * 72)
        btes = BTES_obj([], HP=HP, verbose=True, timing=True)   # all defaults: top of file
        btes.initialize(Volume, T_charge, dt)
        hp_on = np.ones(n, dtype=bool) if HP is not None else None   # HP allowed all year
        out = btes.calc_heat(T_return, T_supply, storage_extraction, missing_energy,
                             hp_on=hp_on, len_timestep=dt)
        btes.summary()
        print(f"  covers {out.sum() / missing_energy.sum() * 100:.1f} % of the winter deficit; "
              f"8-year ramp [MWh]: "
              f"{np.round(btes.total_heat_extracted_vs_T_ground_kWh_first_8_years / 1e3, 1)}")
        if HP is not None:
            cop = btes.COP[np.isfinite(btes.COP)]
            if cop.size:
                print(f"  HP: {np.nansum(btes.P_el) / 1e3:,.0f} MWh_el, mean COP {cop.mean():.2f}")
        runs[label] = (btes, out)

    # --- Figure 1: spin-up of wall and fluid temperature, one panel per run -------
    # Same layout as the MTES spin-up figure: blue / orange at lw 1.8, grey dashed
    # year separators, integer year ticks, large fonts, legend below the axes. The
    # fluid temperature switches every hour between charging return, discharge
    # outlet and the idle wall temperature, so it is drawn thinner, underneath.
    fig, axes = plt.subplots(len(runs), 1, figsize=(12, 5.2 * len(runs)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, (label, (m, _)) in zip(axes, runs.items()):
        yrs = np.arange(len(m.T_b_hist_all)) / n

        ax.plot(yrs, m.T_fluid_hist_all, label="Fluid leaving the field [°C]",
                color="tab:orange", linewidth=0.6, alpha=0.7)
        ax.plot(yrs, m.T_b_hist_all, label="Borehole Wall Temperature [°C]",
                color="tab:blue", linewidth=1.8)

        ax.set_title(f"BTES Wall & Fluid Temperature Over Time  ({label})", fontsize=18)
        ax.set_ylabel("Temperature [°C]", fontsize=15)
        ax.grid(True, alpha=0.5)

        # Integer year ticks + grey year separators, as in the MTES figure.
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

    # --- Figure 2: mature year, per run --------------------------------------------
    for label, (m, out) in runs.items():
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        days = hours / 24
        a1.plot(days, m.T_fluid_hist, label="Fluid leaving the field", lw=0.6,
                color="tab:orange", alpha=0.7)
        a1.plot(days, m.T_b_hist, label="Borehole wall", lw=1.4, color="tab:blue")
        a1.axhline(m.T_charge, color="tab:red", lw=0.6, ls=":", label="Charging T")
        a1.axhline(m.T_return, color="k", lw=0.6, ls=":", label="DHN return")
        if m.HP is not None:
            a1.axhline(m.T_floor, color="grey", lw=0.6, ls=":", label="HP floor")
        a1.set_ylabel("Temperature [C]")
        a1.set_title(f"BTES mature year ({label})")
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

    # --- Figure 3: recovery efficiency per simulated year ---------------------------
    # Top panel is the answer, bottom panel is the reason: in the first years much
    # of the charge goes into warming the field and the ground around it, so Reff
    # is low. As that ground warms up the losses fall and Reff rises towards the
    # value set by the field's size and shape (heat keeps leaking to the
    # surroundings and the surface, so it stays below 1).
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

        # First year within 0.5 % of the final value -> "spun up" from here on.
        # Labels staggered so two runs converging in the same year stay readable.
        settled = np.where(np.abs(m.yearly_Reff - m.yearly_Reff[-1]) < 0.005)[0]
        if len(settled) and len(yrs) > 1:
            b1.axvline(yrs[settled[0]], color=c, lw=0.8, ls="--", alpha=0.5)
            b1.annotate(f"converged year {yrs[settled[0]]} ({label})",
                        (yrs[settled[0]], 0.08 + 0.09 * list(runs).index(label)),
                        textcoords="offset points", xytext=(5, 0),
                        fontsize=8, color=c, ha="left",
                        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))

        # Absorbed vs extracted: the gap is the heat that stayed in the ground
        # (losses + build-up), so their ratio is the number plotted above.
        b2.plot(yrs, m.yearly_charged_kWh / 1e3, "o--", ms=3, lw=1.0,
                color=c, alpha=0.55, label=f"absorbed ({label})")
        b2.plot(yrs, m.yearly_extracted_kWh / 1e3, "o-", ms=3, lw=1.5,
                color=c, label=f"extracted ({label})")

    _m0 = runs[list(runs)[0]][0]
    b1.axhline(1.0, color="k", lw=0.8, ls=":", alpha=0.5, label="lossless limit")
    b1.set_ylabel("Recovery efficiency\nextracted / absorbed  [-]")
    b1.set_title(f"BTES recovery efficiency per simulated year "
                 f"({_m0.N_b} boreholes x {_m0.H:g} m, B = {_m0.B:g} m, "
                 f"V_ground = {_m0.V_ground:,.0f} m3)")
    b1.set_ylim(0, 1.15)
    b1.grid(alpha=0.35)
    b1.legend(fontsize=8.5, loc="lower right")

    b2.set_xlabel("Simulated year")
    b2.set_ylabel("Heat [MWh/yr]")
    b2.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Right margin scales with the run length so the end-value label always fits.
    _ny = max(len(m.yearly_Reff) for m, _ in runs.values())
    b2.set_xlim(0.5, _ny + max(0.6, 0.06 * _ny))
    b2.grid(alpha=0.35)
    b2.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    # --- Figure 4: the g-function of this field (BTES only) --------------------------
    # The ground's step response. Early: each borehole on its own (~ln t); months to
    # years: the boreholes interact and g climbs; decades: surface losses level it off.
    fig, ax = plt.subplots(figsize=(9, 5))
    ln_ts = np.log(_m0.g_time / _m0.t_s)
    ax.plot(ln_ts, _m0.g_values, "o-", ms=3, lw=1.5, color="tab:blue",
            label=f"{_m0.N_x} x {_m0.N_y} field, B/H = {_m0.B / _m0.H:.3f}")
    for t_mark, lbl in [(86400, "1 day"), (30 * 86400, "1 month"),
                        (3.1536e7, "1 year"), (3.1536e8, "10 years")]:
        x = np.log(t_mark / _m0.t_s)
        ax.axvline(x, color="grey", lw=0.8, ls="--")
        ax.annotate(lbl, (x, ax.get_ylim()[1]), textcoords="offset points",
                    xytext=(3, -12), fontsize=8, color="grey")
    ax.set_xlabel("ln(t / t_s)   [-]")
    ax.set_ylabel("g-function [-]")
    ax.set_title(f"BTES g-function ({_m0.gfunc_method}, {_m0.gfunc_bc}; "
                 f"t_s = {_m0.t_s / 3.1536e7:.1f} yr)")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()

    plt.show()
