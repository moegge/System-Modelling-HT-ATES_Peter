"""
Marginal electricity cost and ON/OFF dispatch signal for a heat pump on a
residential district-heating network (NL, 2026 tariffs, operator-side, excl. VAT).

Eq. 1:   c_marg,i = P_spot,i + M + eb_marg + tau        [EUR/kWh]
Signal:  HP_ON,i  = 1 if <basis>,i < threshold else 0

Scope
-----
- No annual cost total (Eq. 2) - that is deliberately out of scope here.
- No file output. The signal is printed to the console and left in the
  module-level variable `hp_signal` (list[int]) for downstream use.

Note on eb_marg
---------------
eb_marg is the energy-tax rate of the bracket that ANNUAL consumption reaches.
It is pinned to the top (zakelijk) bracket here, on the assumption that annual
consumption always exceeds 10 GWh. This keeps Eq. 1 a pure per-hour signal.
If the dispatch drops consumption below 10 GWh the pin is wrong - section 4
prints a warning when that happens.
"""

import pandas as pd

# ======================================================================
# 1. USER INPUTS  --  edit these
# ======================================================================

# --- Input file -------------------------------------------------------
CSV_PATH = r"C:\Users\0527831\PycharmProjects\System-Modelling-HT-ATES_Peter\Data_and_scripts\Electricity Data\Netherlands.csv"
COL_TIME  = "Datetime (Local)"
COL_PRICE = "Price (EUR/MWhe)"

YEAR = 2025                     # most recent complete year in the file

# --- Heat pump --------------------------------------------------------
HP_SIZE_KW  = 1200.0            # electrical rating [kW_el]
LOAD_FACTOR = 1.0               # power drawn while ON, as fraction of rating

# --- Eq. 1 terms ------------------------------------------------------
M_SUPPLIER        = 0.02        # M   - supplier sourcing markup      [EUR/kWh]
TAU_TRANSPORT_KWH = 0.0198      # tau - per-kWh grid transport        [EUR/kWh]
                                #       0.0198 for MS; set 0.0 if > 1500 kW

# --- Energy tax 2026, electricity, excl. VAT --------------------------
# Assumption: annual consumption always reaches the top (zakelijk) bracket,
# so eb_marg is fixed and the dispatch <-> bracket feedback disappears.
EB_MARG_EUR_KWH = 0.00310       # eb_marg, > 10.000.000 kWh zakelijk
# Reference table, not used while the top bracket is forced:
#        0 t/m      2.900   0.09161
#    2.901 t/m     10.000   0.09161
#   10.001 t/m     50.000   0.06671
#   50.001 t/m 10.000.000   0.03735
#      > 10.000.000 (zak.)  0.00310

# --- Dispatch rule ----------------------------------------------------
THRESHOLD_EUR_MWH = 60.0        # run when basis < this value        [EUR/MWh]
THRESHOLD_BASIS   = "c_marg"    # "c_marg" (Eq. 1, all-in marginal)
                                # "spot"   (raw wholesale only)
STRICT_LESS_THAN  = True        # True: basis <  threshold
                                # False: basis <= threshold

# --- Console output ---------------------------------------------------
PRINT_DAILY_GRID   = True       # 0/1 grid, one line per day (24 cols)
PRINT_PYTHON_LIST  = True       # copy-pasteable Python literal
PRINT_HEAD_N       = 0          # >0: print only first N hours of the grid



# ======================================================================
# 2. LOAD DATA
# ======================================================================

df = pd.read_csv(CSV_PATH)
df[COL_TIME] = pd.to_datetime(df[COL_TIME])
df = df[df[COL_TIME].dt.year == YEAR].copy()
df = df.dropna(subset=[COL_PRICE]).sort_values(COL_TIME).reset_index(drop=True)

if df.empty:
    raise ValueError(f"No data found for year {YEAR}. Check YEAR and the CSV.")

n_hours = len(df)
if n_hours not in (8760, 8784):
    print(f"  [warn] {n_hours} rows for {YEAR} (expected 8760 or 8784). "
          f"Check for DST duplicates, gaps, or sub-hourly resolution.")

df["spot_eur_kwh"] = df[COL_PRICE] / 1000.0     # P_spot,i  [EUR/kWh]

E_hourly = HP_SIZE_KW * LOAD_FACTOR             # kWh drawn in an ON hour

# ======================================================================
# 3. MARGINAL COST (Eq. 1) AND DISPATCH
# ======================================================================

if THRESHOLD_BASIS not in ("c_marg", "spot"):
    raise ValueError("THRESHOLD_BASIS must be 'c_marg' or 'spot'.")

eb_marg = EB_MARG_EUR_KWH
adder = M_SUPPLIER + eb_marg + TAU_TRANSPORT_KWH     # M + eb_marg + tau

df["c_marg_eur_kwh"] = df["spot_eur_kwh"] + adder    # Eq. 1

basis_mwh = (df["c_marg_eur_kwh"] if THRESHOLD_BASIS == "c_marg"
             else df["spot_eur_kwh"]) * 1000.0
on_mask = ((basis_mwh < THRESHOLD_EUR_MWH) if STRICT_LESS_THAN
           else (basis_mwh <= THRESHOLD_EUR_MWH)).to_numpy()

df["hp_on"] = on_mask.astype(int)

hp_signal = df["hp_on"].tolist()            # <-- the deliverable
n_on = int(on_mask.sum())
E_total = E_hourly * n_on


# ======================================================================
# 4. CONSOLE OUTPUT
# ======================================================================

print("=" * 68)
print(f"  HP dispatch signal  |  year {YEAR}")
print("=" * 68)
print(f"  HP size .................. {HP_SIZE_KW:,.0f} kW_el  (load factor {LOAD_FACTOR})")
print(f"  Hours in dataset ......... {n_hours:,}")

print("-" * 68)
print(f"  eb_marg .................. {eb_marg:.5f} EUR/kWh")
print(f"  Constant adder (M+eb+tau)  {adder:.5f} EUR/kWh  "
      f"(= {adder*1000:,.2f} EUR/MWh)")
print(f"  Mean spot ................ {df['spot_eur_kwh'].mean()*1000:,.2f} EUR/MWh")
print(f"  Mean c_marg .............. {df['c_marg_eur_kwh'].mean()*1000:,.2f} EUR/MWh")
print("-" * 68)
print(f"  Threshold ................ {THRESHOLD_EUR_MWH:,.2f} EUR/MWh "
      f"on {THRESHOLD_BASIS}")
if THRESHOLD_BASIS == "c_marg":
    equiv_spot = THRESHOLD_EUR_MWH - adder * 1000
    print(f"  -> equivalent spot price . {equiv_spot:,.2f} EUR/MWh")
print(f"  ON hours ................. {n_on:,} / {n_hours:,} "
      f"({n_on/n_hours*100:.1f} %)")
print(f"  Implied consumption ...... {E_total:,.0f} kWh/yr")
if E_total <= 10_000_000:
    print(f"  [warn] implied consumption is below the 10 GWh top-bracket "
          f"threshold. eb_marg is pinned at {eb_marg:.5f} but the true "
          f"marginal rate here would be 0.03735 EUR/kWh "
          f"(+{(0.03735 - eb_marg)*1000:,.2f} EUR/MWh on the adder).")
if n_on == 0:
    print("  [warn] signal is OFF for every hour - threshold is below the "
          "cheapest hour of the year.")
elif n_on == n_hours:
    print("  [warn] signal is ON for every hour - threshold is above the "
          "most expensive hour of the year.")
print("=" * 68)

if PRINT_DAILY_GRID:
    limit = PRINT_HEAD_N if PRINT_HEAD_N > 0 else n_hours
    print(f"  HP_ON by hour (0 = off, 1 = on), one line per day, "
          f"columns = hour 0..23")
    print("-" * 68)
    for start in range(0, limit, 24):
        chunk = hp_signal[start:start + 24]
        stamp = df[COL_TIME].iloc[start].strftime("%Y-%m-%d")
        print(f"  {stamp}  " + " ".join(str(v) for v in chunk))
    print("=" * 68)

if PRINT_PYTHON_LIST:
    print("  hp_signal = [")
    for start in range(0, n_hours, 48):
        chunk = hp_signal[start:start + 48]
        print("      " + ", ".join(str(v) for v in chunk) + ",")
    print("  ]")
    print("=" * 68)