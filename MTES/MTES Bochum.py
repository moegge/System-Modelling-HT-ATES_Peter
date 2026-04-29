"""
Standalone MTES test driver
---------------------------
Drives Till Spengler's MTES model (mtes_class.MTES) with a PRESCRIBED
charge / discharge schedule instead of the Modelica FMU. The MTES working
principle (do_step, eqs. 4.32-4.34) is imported UNCHANGED from mtes_class.py.

Requires in the same folder: mtes_class.py
Requires installed:          numpy, matplotlib
No FMU needed.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mtes_class import MTES   # <-- Till's physics, imported untouched

# =============================================================================
# Physical parameters  (identical to execute_fmu_adaptable.py)
# =============================================================================
V_t = 7000  # m3

# --- Rock Properties ---
lambda_r = 3.5  # W/mK
rho_r = 2469.84  # kg/m3
cp_r = (800 + 2000) / 2  # J/kgK
alpha_r = lambda_r / (rho_r * cp_r)  # m2/s

# --- Tank Geometry ---
L = (207 + 150) / 2  # m
r_t = np.sqrt(V_t / (np.pi * L))  # m

# --- Water Properties ---
cp_water = 4180  # J/kgK
rho_water = 997  # kg/m3
m_tank = V_t * rho_water  # kg

# --- Initial Conditions ---
T_tank = 10.0  # °C
T_rock = 10.0  # °C

# --- Fixed thermal buffer radius (based on max charging time) ---
t_max = 100 * 24 * 3600  # s
r_th_max = 1.5 * np.sqrt(alpha_r * t_max) + r_t
V_rock = np.pi * L * (r_th_max ** 2 - r_t ** 2)
m_rock = V_rock * rho_r

# =============================================================================
# Simulation settings
# =============================================================================
step_size = 3600.0  # s  (1 hour; set to 60.0 to match the FMU driver exactly)

# =============================================================================
# PRESCRIBED CHARGE / DISCHARGE SCHEDULE   <-- edit this block to test
# Each phase: (label, duration [days], volume flow [m3/h], inlet temp [°C])
#   "charge"    -> hot water in  (T_in > T_tank): tank warms, heat into rock
#   "discharge" -> cold water in (T_in < T_tank): tank cools, heat from rock
# =============================================================================
# 1. Set the number of cycles you want to simulate
num_cycles = 5

# 2. Define a single cycle (e.g., 30 days charge, 30 days discharge)
base_cycle = [
    ("charge",    30, 150.0, 80.0),
    ("discharge", 30, 150.0, 20.0),
]

# 3. Multiply the list to build the full schedule automatically
SCHEDULE = base_cycle * num_cycles

# =============================================================================
# Build the MTES object (working principle untouched - Till's class)
# =============================================================================
mtes = MTES(T_tank, T_rock, step_size, L, lambda_r, r_t, r_th_max,
            m_tank, m_rock, cp_water, cp_r, rho_r)

# =============================================================================
# Run the prescribed schedule
# =============================================================================
t_s = []  # time [s]
T_tank_hist = []
T_rock_hist = []
Power_MW_hist = []  # <-- New list to track energy flow (Power in Megawatts)

current_time = 0.0
for label, dur_days, V_dot_m3h, T_in in SCHEDULE:
    m_dot = rho_water * (V_dot_m3h / 3600.0)  # m3/h -> kg/s
    n_steps = int(round(dur_days * 24 * 3600 / step_size))
    for _ in range(n_steps):
        # Calculate instantaneous power (MW) BEFORE the tank temp changes
        # Positive = heat going IN (Charge), Negative = heat going OUT (Discharge)
        power_MW = (m_dot * cp_water * (T_in - mtes.T_tank)) / 1_000_000.0
        Power_MW_hist.append(power_MW)

        mtes.do_step(T_in, m_dot)  # Till's physics, unchanged
        current_time += step_size
        t_s.append(current_time)
        T_tank_hist.append(mtes.T_tank)
        T_rock_hist.append(mtes.T_rock)

t_days = np.array(t_s) / (24 * 3600)
T_tank_hist = np.array(T_tank_hist)
T_rock_hist = np.array(T_rock_hist)
Power_MW_hist = np.array(Power_MW_hist)

print(f"Simulated {t_days[-1]:.0f} days in {len(t_days)} steps "
      f"(step = {step_size:.0f} s)")
print(f"Final tank temperature: {T_tank_hist[-1]:.2f} °C")
print(f"Final rock temperature: {T_rock_hist[-1]:.2f} °C")

# =============================================================================
# Plotting (Now with 2 subplots)
# =============================================================================
# Create a figure with 2 rows, sharing the X-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# --- Top Chart: Temperatures ---
t0 = 0.0
for label, dur_days, _, _ in SCHEDULE:
    t1 = t0 + dur_days
    ax1.axvspan(t0, t1, color=("tab:red" if label == "charge" else "tab:blue"), alpha=0.06)
    t0 = t1

ax1.plot(t_days, T_tank_hist, color="tab:blue", linewidth=1.8, label="Water (tank) temperature [°C]")
ax1.plot(t_days, T_rock_hist, color="tab:orange", linewidth=1.8, label="Rock temperature [°C]")

ax1.set_title("MTES test run - Temperatures & Energy Flow", fontsize=16)
ax1.set_ylabel("Temperature [°C]", fontsize=13)
ax1.grid(True, alpha=0.4)
ax1.legend(loc="upper right", fontsize=12)

# --- Bottom Chart: Energy Flow (Power) ---
# Plot a black zero-line for reference
ax2.axhline(0, color="black", linewidth=1, alpha=0.5)

# Plot the power curve
ax2.plot(t_days, Power_MW_hist, color="dimgrey", linewidth=1.2)

# Fill the area under the curve to clearly show IN vs OUT
ax2.fill_between(t_days, Power_MW_hist, 0, where=(Power_MW_hist >= 0),
                 color="tab:red", alpha=0.5, label="Energy In (Charge)")
ax2.fill_between(t_days, Power_MW_hist, 0, where=(Power_MW_hist < 0),
                 color="tab:blue", alpha=0.5, label="Energy Out (Discharge)")

ax2.set_xlabel("Time [days]", fontsize=13)
ax2.set_ylabel("Thermal Power [MW]", fontsize=13)
ax2.grid(True, alpha=0.4)
ax2.legend(loc="upper right", fontsize=12)

plt.tight_layout()

output_dir = os.path.dirname(os.path.abspath(__file__))
png_filename = os.path.join(output_dir, "MTES_test_prescribed.png")
plt.savefig(png_filename, bbox_inches="tight", dpi=150)
print(f"✅ Saved plot: {png_filename}")

plt.show()