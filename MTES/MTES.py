"""
MTES (Mine Thermal Energy Storage) Simulation
Based on: "Co-Simulation of a District Heating and Cooling System in
           Combination with Mine Thermal Energy Storage" - Till Spengler

Reproduces Figure 4.15: 3 charge cycles + 3 discharge cycles, 30 days each.

ASSUMPTIONS MADE (beyond what the thesis states) are flagged with:
    [ASSUMED] - derived/estimated by implementer
    [THESIS]  - directly from the thesis

Run in PyCharm: pip install numpy matplotlib
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# 1. PARAMETERS (all from Table 4.2 and surrounding text unless flagged)
# =============================================================================

# --- Rock properties [THESIS Table 4.2] ---
lambda_rock = 3.5        # W/(m·K)  thermal conductivity of sandstone
rho_rock    = 2469.84    # kg/m³    density of sandstone
cp_rock     = 1400.0     # J/(kg·K) specific heat capacity of sandstone
                         #          (averaged dry/saturated, Table 4.2)

# --- Water properties [THESIS Table 4.2] ---
rho_water  = 997.0       # kg/m³
cp_water   = 4180.0      # J/(kg·K)

# --- Tank geometry [THESIS Table 4.2] ---
V_tank = 350000.0        # m³   total volume of the two mine chambers
L      = 178.5           # m    average length of the two chambers

# [ASSUMED] Cylinder radius derived from V = π·r²·L (thesis gives V and L,
# but never explicitly states r_tank)
r_tank = math.sqrt(V_tank / (math.pi * L))   # ≈ 3.533 m

# --- Thermal diffusivity [THESIS eq. 4.30] ---
alpha = lambda_rock / (rho_rock * cp_rock)   # m²/s ≈ 1.012e-6

# --- Maximum thermal radius [THESIS eq. 4.31 + Section 4.2.4.4] ---
# Thesis: max cycle assumed to be 100 days → fixed buffer radius
t_max_cycle = 100 * 24 * 3600               # 100 days in seconds
# r_th from eq. 4.31 = distance from EDGE of tank to thermal front edge
r_th_from_edge = 1.5 * math.sqrt(alpha * t_max_cycle)   # ≈ 4.436 m
# r_tr,max in eq. 4.32 = distance from CENTRE of tank to thermal front edge
# [ASSUMED] Adding r_tank because thesis distinguishes the two in Sec. 4.2.4.3
#r_tr_max = r_tank + r_th_from_edge           # ≈ 7.969 m

# --- Masses ---
m_tank = rho_water * V_tank                  # kg, mass of water in tank

# [ASSUMED] Rock buffer modeled as a cylindrical shell between r_tank and r_tr_max
# Thesis treats rock as a single lumped node; its mass follows from the
# shell geometry. Thesis never states m_rock explicitly.
#V_rock = math.pi * (r_tr_max**2 - r_tank**2) * L    # m³ shell volume
#m_rock = rho_rock * V_rock                            # kg ≈ 70.7e6 kg

# Pre-compute the log ratio used in eq. 4.32 (constant since r_tr,max is fixed)
#ln_ratio = math.log(r_tr_max / r_tank)      # = ln(7.969 / 3.533) ≈ 0.813

# --- Flow parameters [THESIS Sec. 4.2.3 / pump 3] ---
V_dot = 150.0 / 3600.0                      # m³/s (150 m³/h)
m_dot = rho_water * V_dot                   # kg/s ≈ 41.5 kg/s

# --- Test scenario temperatures [THESIS Sec. 4.2.4.4, Figure 4.15] ---
T_charge_in  = 80.0     # °C  inlet temperature during charging
T_discharge_in = 20.0   # °C  inlet temperature during discharging

# --- Initial conditions [THESIS Sec. 4.2.3: initial MTES temp = 11°C] ---
T_tank_init = 11.0      # °C
# [ASSUMED] Rock buffer starts at the same temperature as the tank.
# Thesis does not state a separate initial rock temperature for this test.
T_rock_init = 11.0      # °C

# =============================================================================
# 2. SIMULATION SETUP
# =============================================================================

# Time stepping
dt_hours = 1.0                        # hours  [ASSUMED] hourly time step
dt       = dt_hours * 3600.0          # seconds

# Cycle schedule: 3 charge + 3 discharge, each 30 days [THESIS Fig. 4.15]
# #cycle_days    = 30
# #n_cycles      = 6                     # 3 charge + 3 discharge alternating
cycle_days    = 180
n_cycles      = 10
total_hours   = n_cycles * cycle_days * 24
n_steps       = int(total_hours / dt_hours)

# Build a mode array: True = charging, False = discharging
cycle_hours   = cycle_days * 24
mode_charging = np.zeros(n_steps, dtype=bool)
for step in range(n_steps):
    t_h = step * dt_hours
    cycle_index = int(t_h // cycle_hours)
    if cycle_index % 2 == 0:          # even cycles = charging
        mode_charging[step] = True

# =============================================================================
# 3. SIMULATION LOOP  (explicit forward Euler)
# =============================================================================

T_tank = T_tank_init
T_rock = T_rock_init

history_time   = np.zeros(n_steps + 1)
history_T_tank = np.zeros(n_steps + 1)
history_T_rock = np.zeros(n_steps + 1)
history_Q_cond = np.zeros(n_steps + 1)   # conductive heat flux [W]

history_T_tank[0] = T_tank
history_T_rock[0] = T_rock
history_time[0]   = 0.0
history_Q_cond[0] = 0.0

for i in range(n_steps):

    charging = mode_charging[i]
    T_in = T_charge_in if charging else T_discharge_in

    t_s = (i + 1) * dt
    r_tr_now = r_tank + 1.5 * math.sqrt(alpha * t_s)
    m_rock_now = rho_rock * math.pi * (r_tr_now ** 2 - r_tank ** 2) * L
    ln_ratio_now = math.log(r_tr_now / r_tank)
    # ------------------------------------------------------------------
    # Step A: Mixing — incoming water blends instantly with tank volume
    # [ASSUMED] In the full co-simulation, Modelica handles the mixing
    # and passes the blended T_tank to Python (Sec. 4.2.3). Running
    # standalone, we implement it ourselves using the standard CSTR
    # (continuously stirred tank) energy balance, discretised with
    # forward Euler:
    #   ΔT_mix = (m_dot · dt / m_tank) · (T_in − T_tank)
    # ------------------------------------------------------------------
    delta_T_mix = (m_dot * dt / m_tank) * (T_in - T_tank)
    T_tank = T_tank + delta_T_mix

    # ------------------------------------------------------------------
    # Step B: Conduction between tank water and rock buffer
    # [THESIS] eq. 4.32 — 4.34
    # Q_cond > 0 → heat flows from tank into rock (typically during charging)
    # Q_cond < 0 → heat flows from rock into tank (typically during discharging)
    # ------------------------------------------------------------------
    Q_cond = (2 * math.pi * L * lambda_rock * (T_tank - T_rock) * dt) / ln_ratio_now

    T_tank = T_tank - Q_cond / (m_tank * cp_water)
    T_rock = T_rock + Q_cond / (m_rock_now  * cp_rock)

    # Store results — Q_cond converted from J per step → average power [W]
    history_time[i + 1]   = (i + 1) * dt_hours
    history_T_tank[i + 1] = T_tank
    history_T_rock[i + 1] = T_rock
    history_Q_cond[i + 1] = Q_cond / dt   # J → W

# Convert W → kW for readability in the plot
history_Q_cond_kW = history_Q_cond / 1000.0

# =============================================================================
# 4. PLOTS
# =============================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
fig.subplots_adjust(hspace=0.08)

# --- Shared background shading (charge=blue, discharge=orange) ---
for ax in (ax1, ax2):
    for c in range(n_cycles):
        t_start = c * cycle_hours
        t_end   = (c + 1) * cycle_hours
        color   = '#d0eaff' if c % 2 == 0 else '#ffe0d0'
        ax.axvspan(t_start, t_end, alpha=0.25, color=color, linewidth=0)

# --- Plot 1: Temperatures ---
ax1.plot(history_time, history_T_tank, color='steelblue',  linewidth=1.8,
         label='Tank Temperature')
ax1.plot(history_time, history_T_rock, color='darkorange', linewidth=1.8,
         label='Rock Temperature (Buffer)')
ax1.set_title('MTES Cyclic Charging and Discharging', fontsize=13)
ax1.set_ylabel('Temperature [°C]', fontsize=11)
ax1.set_xlim(0, total_hours)
ax1.set_ylim(0, 90)
ax1.grid(True, linestyle='--', alpha=0.5)

tank_patch      = mpatches.Patch(color='steelblue',  label='Tank Temperature')
rock_patch      = mpatches.Patch(color='darkorange', label='Rock Temperature (Buffer)')
charge_patch    = mpatches.Patch(color='#d0eaff', alpha=0.6, label='Charging (80 °C in)')
discharge_patch = mpatches.Patch(color='#ffe0d0', alpha=0.6, label='Discharging (20 °C in)')
ax1.legend(handles=[tank_patch, rock_patch, charge_patch, discharge_patch],
           loc='upper right', fontsize=9)

# --- Plot 2: Conductive heat flux ---
ax2.plot(history_time, history_Q_cond_kW, color='mediumseagreen', linewidth=1.8,
         label='Conductive heat flux (tank ↔ rock)')
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax2.set_xlabel('Time [hours]', fontsize=11)
ax2.set_ylabel('Heat flux [kW]', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper right', fontsize=9)

# Annotate sign convention
ax2.annotate('+ : tank losing heat to rock', xy=(0.01, 0.93),
             xycoords='axes fraction', fontsize=8, color='dimgray')
ax2.annotate('− : rock returning heat to tank', xy=(0.01, 0.04),
             xycoords='axes fraction', fontsize=8, color='dimgray')

plt.savefig('mtes_figure_4_15.png', dpi=150)
plt.show()
print("Done. Plot saved as mtes_figure_4_15.png")

# =============================================================================
# 5. SUMMARY OF ASSUMPTIONS MADE BY IMPLEMENTER
# =============================================================================
print("""
========================================================
ASSUMPTIONS MADE (not explicitly stated in the thesis)
========================================================
1. [r_tank]      Cylinder radius derived from V = π·r²·L using V=7000 m³,
                 L=178.5 m → r_tank ≈ 3.533 m. Thesis never states r_tank.

2. [r_tr_max]    r_tr,max (from CENTRE) = r_tank + r_th_from_edge.
                 Thesis distinguishes "from edge" vs "from centre" in Sec.
                 4.2.4.3 but does not give a final numerical value for r_tr,max.

3. [m_rock]      Rock buffer modelled as a cylindrical shell between r_tank
                 and r_tr_max. Thesis never gives m_rock explicitly; it is
                 derived from the shell geometry and rho_rock.

4. [T_rock_init] Initial rock temperature set equal to T_tank_init = 11°C.
                 Thesis gives the MTES initial temperature (11°C) but not
                 a separate initial rock temperature for the test scenario.

5. [Mixing eq.]  Mixing step uses standard CSTR balance:
                 ΔT_mix = (m_dot·dt / m_tank) · (T_in − T_tank).
                 In the full co-simulation, Modelica handles the mixing
                 (the MTES block is a mixing volume in Modelica, Sec 4.2.3)
                 and passes the already-blended T_tank to the Python model.
                 The Python model (eqs. 4.32-4.34) then only applies the
                 conduction correction. Since we run standalone here without
                 Modelica, we implement the mixing step ourselves before
                 calling the conduction equations.

6. [Step order]  Mixing applied first, conduction second within each time
                 step. Thesis does not specify the order of operations.

7. [dt = 1 hour] Time step chosen as 1 hour. Thesis does not state the
                 time step used for the Python model.
========================================================
""")