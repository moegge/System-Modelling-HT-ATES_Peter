import os
import shutil
from fmpy import read_model_description, extract, instantiate_fmu
import numpy as np
import time
import matplotlib.pyplot as plt
#import pandas as pd
from mtes_class import MTES
import datetime as dt
import matplotlib.dates as mdates

# --- Variables to set ---
hp_mtes_Qmax = 5500000
V_t = 7000 #m3
max_mtes = (50 + 273.15) #K, 46.88 om precies te zijn maar dan kom je wel dicht in de buurt
min_mtes = (15 + 273.15) #K
min_CHP = 12.69
mtes_supply_temp = max_mtes 
deltaT_HP1 = 5
deltaT_HP2 = deltaT_HP1
dp_HD = 10e5

# --- User settings FMU ---
fmu_filename = 'Bochum_Final.fmu'   # Path to your FMU
start_time   = 0.0                  # Start time of simulation
stop_time    = 79837200.            # Stop time of simulation #647485200 for long term
step_size    = 60.                  # Step size for single-step simulation

# --- Rock Properties ---
lambda_r = 3.5  # W/mK
rho_r = 2469.84  # kg/m³
cp_r = (800 + 2000) / 2  # J/kgK
alpha_r = lambda_r / (rho_r * cp_r)  # m²/s

# --- Tank Geometry ---
#V_t = 7000  # m³
L = (207 + 150) / 2  # m
r_t = np.sqrt(V_t / (np.pi * L))  # m

# --- Water Properties ---
cp_water = 4180  # J/kgK
rho_water = 997  # kg/m³
m_tank = V_t * rho_water  # kg

# --- Initial Conditions ---
T_tank = 10.0  # °C
T_rock = 10.0  # initial rock temperature
T_values = [T_tank]
T_rock_values = [T_rock]
time_hours = [0]

# --- Compute fixed thermal buffer radius (based on max charging time) ---
t_max = 100 * 24 * 3600  # max cycle duration
r_th_max = 1.5 * np.sqrt(alpha_r * t_max) + r_t
V_rock = np.pi * L * (r_th_max**2 - r_t**2)
m_rock = V_rock * rho_r

mtes = MTES(T_tank, T_rock, step_size, L, lambda_r, r_t, r_th_max, m_tank, m_rock, cp_water, cp_r, rho_r)



# 1. Read model description
model_description = read_model_description(fmu_filename)

vrs = {}
for variable in model_description.modelVariables:
        vrs[variable.name] = variable.valueReference
# Print variable references for debugging
#print("Variable References:")
#for name, vr in vrs.items():
#    print(f"{name}: {vr}")

# 2. Extract the FMU to a temporary folder
unzipdir = extract(fmu_filename)

# 3. Instantiate the FMU (FMPy figures out if it’s Co-Sim or Model Exchange)
fmu = instantiate_fmu(unzipdir=unzipdir, model_description=model_description,visible=True, debug_logging=True,)

# 4. Instantiate and initialize the FMU

fmu.instantiate()
fmu.setupExperiment(startTime=start_time, tolerance=1e-5)
#----------- Set and get initial conditions if needed

fmu.setReal([vrs['inlet.m_flow'], ], [0.0])  # Set initial mass flow
fmu.setReal([vrs['inlet.forward.T'], ], [T_tank + 273.15])  # Set initial rock temperature
fmu.setReal([vrs['MTES_T'], ], [T_tank + 273.15])  # Set initial tank temperature

fmu.enterInitializationMode()

######################################################## ADAPT ############################################################
#Set parameters from cli
fmu.setReal([vrs['heat_pump.QCon_flow_max']], [hp_mtes_Qmax])
fmu.setReal([vrs['heat_pump_mtes.QCon_flow_max']], [hp_mtes_Qmax])
fmu.setReal([vrs['max_mtes'], ], [max_mtes])
fmu.setReal([vrs['min_mtes'], ], [min_mtes])
fmu.setReal([vrs['min_CHP'], ], [min_CHP])
fmu.setReal([vrs['mtes_supply_temp'], ], [mtes_supply_temp])
fmu.setReal([vrs['deltaT_HP1'], ], [deltaT_HP1])
fmu.setReal([vrs['deltaT_HP2'], ], [deltaT_HP2])
fmu.setReal([vrs['heating_demand.flowUnit.dp_nominal'], ], [dp_HD])
######################################################## ADAPT ############################################################


fmu.exitInitializationMode()


current_time = start_time

time_start = time.time()

temperature_rock = []

temperature_mtes = []
while (current_time < stop_time):

    # Step the FMU
    status = fmu.doStep(
        currentCommunicationPoint=current_time,
        communicationStepSize=step_size
    )

    # 5. Get results from the FMU
    pump_mtes_mflow = fmu.getReal([vrs['pump_mtes.m_flow_actual'],])[0]
    outlet_T_mtes = fmu.getReal([vrs['outlet.forward.T'],])[0] - 273.15 # Convert Kelvin to Celsius
    outlet_mflow_mtes = fmu.getReal([vrs['outlet.m_flow'],])[0]
    mtes_mode = fmu.getReal([vrs['mtes_mode'],])[0] # 1=charge, 2=storage, 3=discharge
    min_T =  fmu.getReal([vrs['min_mtes'],])[0]
    mtes.do_step(outlet_T_mtes, outlet_mflow_mtes)

    fmu.setReal([vrs['MTES_T'], ], [mtes.T_tank + 273.15])  # Update tank temperature
    fmu.setReal([vrs['inlet.m_flow'], ], [pump_mtes_mflow])  # Set mass flow
    fmu.setReal([vrs['inlet.forward.T'], ], [mtes.T_tank + 273.15])  # Set tank temperature


    temperature_mtes.append(mtes.T_tank)
    temperature_rock.append(mtes.T_rock)

    if current_time % (24*3600) == 0:  # Log every day
        time_intermediate = time.time()
        print(f"Current time: {current_time / 3600:.2f} hours, MTES Tank Temperature: {mtes.T_tank:.2f} °C")
        print(f"Time taken for this iteration: {time_intermediate - time_start:.2f} seconds")
        print(f"MTES Mode: {mtes_mode}")
        #print(f"min_mtes: {min_T}")

    # Update time
    current_time += step_size
    

# 6. Terminate and free the instance
fmu.terminate()
fmu.freeInstance()

time_end = time.time()
print(f"Simulation finished in {time_end - time_start:.2f} seconds.")
# Clean up temporary folder
shutil.rmtree(unzipdir, ignore_errors=True)

"""
# ----------------------------------------
# PLOT MTES & ROCK TEMPERATURES OVER TIME NEW
# ----------------------------------------

plt.figure(figsize=(12, 6))

# --- Build time axis in datetimes (same as before) ---
sim_start = dt.datetime(2021, 6, 23)
time_seconds = np.arange(len(temperature_mtes)) * step_size
time_axis = [sim_start + dt.timedelta(seconds=float(s)) for s in time_seconds]

# --- Plot the temperature lines (unchanged) ---
plt.plot(time_axis, temperature_mtes, label="Water Temperature [°C]", color="tab:blue", linewidth=1.8)
plt.plot(time_axis, temperature_rock, label="Rock Temperature [°C]", color="tab:orange", linewidth=1.8)

plt.title("MTES Water & Rock Temperature Over Time", fontsize=18)
plt.xlabel("Time [years]", fontsize=15)
plt.ylabel("Temperature [°C]", fontsize=15)
plt.grid(True, alpha=0.5)

# --- Format x-axis to show 1, 2, 3... for each January ---
ax = plt.gca()

# Find January 1st each year within range
years = sorted(set([t.year for t in time_axis]))
year_ticks = [dt.datetime(y, 1, 1) for y in years if y >= 2022]

# Create numeric labels (1, 2, 3, ...)
year_labels = [str(i + 1) for i in range(len(year_ticks))]

ax.set_xticks(year_ticks)
ax.set_xticklabels(year_labels, fontsize=12)
plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

# --- Limit axis to simulation period ---
ax.set_xlim([time_axis[0], time_axis[-1]])

# --- Extend y-limit downward for clean layout ---
ymin, ymax = ax.get_ylim()
extra_space = 0.1 * (ymax - ymin)
ax.set_ylim(ymin - extra_space, ymax)

# --- Grey vertical year separators (no labels) ---
for x in year_ticks:
    ax.axvline(x, color="gray", linestyle="--", linewidth=1)

# --- Legend below plot ---
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=13)

plt.tight_layout()

# --- Save figure ---
output_dir = "/ceph/home/til66021/Co-Simulation/pushit-cosimulation/CoSim_FMU/results execute/"
png_filename = os.path.join(output_dir, "MTES_Rock_Temperatures.png")
plt.savefig(png_filename, bbox_inches="tight")
plt.close()

print(f"✅ Saved temperature plot: {png_filename}")


"""
# ----------------------------------------
# PLOT MTES & ROCK TEMPERATURES OVER TIME ORIGINAL
# ----------------------------------------

plt.figure(figsize=(12, 6))

# --- Build time axis in datetimes (same as your good version) ---
sim_start = dt.datetime(2021, 6, 23)   # simulation starts June 23, 2021
time_seconds = np.arange(len(temperature_mtes)) * step_size
time_axis = [sim_start + dt.timedelta(seconds=float(s)) for s in time_seconds]

# --- Plot the temperature lines (keep as-is, your good version) ---
plt.plot(time_axis, temperature_mtes, label="Water Temperature [°C]", color="tab:blue", linewidth=1.8)
plt.plot(time_axis, temperature_rock, label="Rock Temperature [°C]", color="tab:orange", linewidth=1.8)

plt.title("MTES Water & Rock Temperature Over Time", fontsize=18)
plt.xlabel("Time [mo]", fontsize=15)
plt.ylabel("Temperature [°C]", fontsize=15)
plt.grid(True, alpha=0.5)

# --- Format x-axis like COP plot ---
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)

# --- Limit axis to simulation period (no Jan 2024) ---
ax.set_xlim([dt.datetime(2021, 6, 23), dt.datetime(2023, 12, 31)])

# --- Extend y-limit downward for year labels ---
ymin, ymax = ax.get_ylim()
extra_space = 0.1 * (ymax - ymin)
ax.set_ylim(ymin - extra_space, ymax)

# --- Year separator lines + labels (no 2024) ---
year_lines = [dt.datetime(2022, 1, 1), dt.datetime(2023, 1, 1)]
year_labels = ["2022 →", "2023 →"]

for x, label in zip(year_lines, year_labels):
    ax.axvline(x, color="gray", linestyle="--", linewidth=1)
    ax.text(x, ymin - extra_space * 0.1, label,
            rotation=0, va='top', ha='center', fontsize=14, color='gray',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))

# --- Legend below plot ---
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=13)

plt.tight_layout()

# --- Save figure ---
output_dir = "/ceph/home/til66021/Co-Simulation/pushit-cosimulation/CoSim_FMU/results execute/"
png_filename = os.path.join(output_dir, "MTES_Rock_Temperatures.png")
plt.savefig(png_filename, bbox_inches="tight")
plt.close()

print(f"✅ Saved temperature plot: {png_filename}")


plt.figure(figsize=(10, 5))
plt.plot(np.arange(0, len(temperature_mtes)) * step_size / 3600, temperature_mtes, label="MTES Tank Temperature")
plt.xlabel("Time [hours]")
plt.ylabel("Temperature [°C]")
plt.title("MTES Tank Temperature Over Time")
plt.legend()
plt.grid()
plt.show()
