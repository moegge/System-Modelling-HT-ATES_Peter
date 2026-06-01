"""
Hourly retail electricity price and annual cost for a heat pump on a
residential district-heating network (NL, 2026 tariffs, operator-side, excl. VAT).

What it does
------------
1. Reads an hourly wholesale (day-ahead) price series in EUR/MWh.
2. Builds the per-hour RETAIL price = wholesale + supplier markup
   + energy tax (effective rate) + per-kWh grid transport.
3. Computes the ANNUAL cost, split into every component (energy / power / fixed).
4. Plots: (a) retail vs wholesale per hour, (b) their % difference,
   (c) a bar chart of the annual-cost composition.

Tariff basis
------------
- Energy tax (energiebelasting) 2026, electricity, excl. VAT, degressive brackets.
- Grid: Stedin grootverbruik 2026, category MS (151-1500 kW) -> Table 3 + Table 2.
- VAT skipped (reclaimable for the operator).

"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================================
# 1. USER INPUTS  --  edit these
# ======================================================================

# --- Path to your wholesale price file (the only mandatory edit) -------
CSV_PATH = r"C:\Users\0527831\PycharmProjects\System-Modelling-HT-ATES_Peter\Data_and_scripts\Electricity Data\Netherlands.csv"

# --- Where to write the hourly-price output file -------------------------
#     None = same folder as the CSV; or set e.g. r"C:\...\outputs"
OUTPUT_DIR = r"C:\Users\0527831\PycharmProjects\System-Modelling-HT-ATES_Peter\Data_and_scripts\Outputs"

# --- Which year to analyse (your file spans 2015-2026; 2025 is the
#     most recent COMPLETE year) ------------------------------------------
YEAR = 2025

# --- Wholesale price threshold for the duration-curve chart -------------
PRICE_THRESHOLD_EUR_MWH = 20.0   # [EUR/MWh]

# --- Heat pump -----------------------------------------------------------
HP_SIZE_KW   = 1200.0   # electrical rating of the HP [kW_el]
LOAD_FACTOR  = 1.0      # 1.0 = runs at rated power every hour (constant load)
P_CONTRACT_KW = HP_SIZE_KW   # contracted transport capacity you book [kW]

# --- Commodity & supplier ------------------------------------------------
M_SUPPLIER = 0.02       # supplier sourcing markup [EUR/kWh]

# --- Grid: Stedin grootverbruik 2026, category MS (151-1500 kW), excl. VAT
#     (set TAU_TRANSPORT_KWH = 0.0 if your HP is > 1500 kW: no per-kWh term)
TAU_TRANSPORT_KWH      = 0.0198   # transport energy term [EUR/kWh]
VR_TRANSPORT_PER_MONTH = 36.75    # transport vastrecht [EUR/month]
C_CONTRACT_PER_KW_MONTH = 2.0228  # contracted-capacity charge [EUR/kW/month]
C_MAX_PER_KW_MONTH      = 3.0966  # monthly-peak charge      [EUR/kW/month]

# --- Other fixed annual costs -------------------------------------------
APV_PER_YEAR    = 1505.00   # periodieke aansluitvergoeding, Stedin Table 2, MS-distributie [EUR/yr]
C_METER_PER_YEAR = 300.00   # metering (meetbedrijf) - PLACEHOLDER, confirm with your meter co. [EUR/yr]
R_TAX_PER_YEAR  = 0.00      # vermindering energiebelasting [EUR/yr]; 0 for a plant connection
                            # (no 'verblijfsfunctie'); set to 519.80 only if it qualifies.

# --- One-time connection fee (Stedin Table 1) - CapEx, NOT in annual cost
#     For 1200 kW (~>1000 t/m 1750 kVA, MS met MS-meting): 59972.34
TABLE1_CONNECTION_FEE = 59972.34   # [EUR, one-time]

# --- Energy tax 2026, electricity, excl. VAT  (upper kWh limit, EUR/kWh) --
ENERGY_TAX_BRACKETS = [
    (2_900,        0.09161),
    (10_000,       0.09161),
    (50_000,       0.06671),
    (10_000_000,   0.03735),
    (np.inf,       0.00310),
]

# --- Column names in the CSV (change only if your file differs) ----------
COL_TIME  = "Datetime (Local)"
COL_PRICE = "Price (EUR/MWhe)"

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

# Wholesale price: EUR/MWh -> EUR/kWh
df["wholesale_eur_kwh"] = df[COL_PRICE] / 1000.0

# Constant hourly consumption [kWh] (1 hour steps): E_i = power * 1h
E_hourly = HP_SIZE_KW * LOAD_FACTOR
df["E_kwh"] = E_hourly
E_total = df["E_kwh"].sum()          # annual consumption [kWh]

# ======================================================================
# 3. ENERGY TAX (degressive)
# ======================================================================

def energy_tax_annual(total_kwh, brackets):
    """Total energy tax [EUR] across degressive brackets."""
    tax, lower = 0.0, 0.0
    for upper, rate in brackets:
        if total_kwh <= lower:
            break
        kwh_in_bracket = min(total_kwh, upper) - lower
        tax += kwh_in_bracket * rate
        lower = upper
    return tax

EB_annual = energy_tax_annual(E_total, ENERGY_TAX_BRACKETS)
eb_eff = EB_annual / E_total          # effective average tax rate [EUR/kWh]

# ======================================================================
# 4. HOURLY RETAIL PRICE  [EUR/kWh]
#    Only the energy (per-kWh) terms belong in a per-kWh price:
#    wholesale (varies) + markup + energy tax + per-kWh transport (constant adders)
# ======================================================================

constant_adder = M_SUPPLIER + eb_eff + TAU_TRANSPORT_KWH
df["retail_eur_kwh"] = df["wholesale_eur_kwh"] + constant_adder

# % difference vs wholesale  (= adder / wholesale * 100)
with np.errstate(divide="ignore", invalid="ignore"):
    pct = (df["retail_eur_kwh"] - df["wholesale_eur_kwh"]) / df["wholesale_eur_kwh"] * 100.0
df["pct_diff"] = pct.replace([np.inf, -np.inf], np.nan)

# ======================================================================
# 5. ANNUAL COST COMPONENTS  [EUR]
# ======================================================================

# Monthly peak power [kW] = max hourly power within each month
df["power_kw"] = df["E_kwh"] / 1.0     # 1-hour steps
monthly_peak = df.groupby(df[COL_TIME].dt.month)["power_kw"].max()

C_wholesale       = (df["wholesale_eur_kwh"] * df["E_kwh"]).sum()
C_markup          = M_SUPPLIER * E_total
C_energy_tax      = EB_annual
C_transport_kwh   = TAU_TRANSPORT_KWH * E_total
C_vastrecht       = 12 * VR_TRANSPORT_PER_MONTH
C_kw_contract     = 12 * C_CONTRACT_PER_KW_MONTH * P_CONTRACT_KW
C_kw_max          = (C_MAX_PER_KW_MONTH * monthly_peak).sum()
C_apv             = APV_PER_YEAR
C_meter           = C_METER_PER_YEAR
C_vermindering    = -R_TAX_PER_YEAR

components = {
    "Wholesale\ncommodity": C_wholesale,
    "Supplier\nmarkup":      C_markup,
    "Energy tax":            C_energy_tax,
    "Transport\n(per kWh)":  C_transport_kwh,
    "Transport\nvastrecht":  C_vastrecht,
    "kW-contract":           C_kw_contract,
    "kW-max":                C_kw_max,
    "Aansluitverg.\n(annual)": C_apv,
    "Metering":              C_meter,
    "Vermindering":          C_vermindering,
}
Cost_total = sum(components.values())

# ======================================================================
# 6. TEXT SUMMARY
# ======================================================================

print("=" * 64)
print(f"  DHN heat-pump electricity cost  |  year {YEAR}")
print("=" * 64)
print(f"  HP size .................. {HP_SIZE_KW:,.0f} kW_el  (load factor {LOAD_FACTOR})")
print(f"  Hours in dataset ......... {n_hours:,}")
print(f"  Annual consumption ....... {E_total:,.0f} kWh")
print(f"  Mean wholesale ........... {df['wholesale_eur_kwh'].mean()*1000:,.2f} EUR/MWh")
print(f"  Effective energy-tax rate  {eb_eff:.5f} EUR/kWh")
print(f"  Retail markup (constant) . {constant_adder*1000:,.2f} EUR/MWh  (= {constant_adder:.5f} EUR/kWh)")
print(f"  Mean retail price ........ {df['retail_eur_kwh'].mean()*1000:,.2f} EUR/MWh")
print("-" * 64)
print("  ANNUAL COST COMPOSITION [EUR]")
for name, val in components.items():
    print(f"    {name.replace(chr(10), ' '):<26} {val:>15,.2f}")
print("-" * 64)
print(f"    {'TOTAL ANNUAL COST':<26} {Cost_total:>15,.2f}")
print(f"    {'-> per kWh (all-in avg)':<26} {Cost_total / E_total:>15,.5f}")
print("-" * 64)
print(f"  One-time connection fee (Table 1, CapEx, NOT in total above):")
print(f"    {TABLE1_CONNECTION_FEE:,.2f} EUR")
print("=" * 64)

# ======================================================================
# 7. EXPORT HOURLY PRICES TO FILE  (one file per run)
# ======================================================================

export_df = pd.DataFrame({
    "datetime_local": df[COL_TIME],
    "wholesale_eur_mwh": df["wholesale_eur_kwh"] * 1000,
    "wholesale_eur_kwh": df["wholesale_eur_kwh"],
    "retail_eur_kwh": df["retail_eur_kwh"],
    "retail_eur_mwh": df["retail_eur_kwh"] * 1000,
    "premium_pct": df["pct_diff"],
})

# filename tagged with the scenario so different runs don't overwrite
out_name = f"hourly_retail_{YEAR}_{int(HP_SIZE_KW)}kW_LF{LOAD_FACTOR:g}.csv"
out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else Path(CSV_PATH).parent
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / out_name
export_df.to_csv(out_path, index=False)
print(f"  Wrote hourly prices to: {out_path}")
print("=" * 64)

# ======================================================================
# 8. CHARTS
# ======================================================================

# --- Chart 1: wholesale price per hour ----------------------------------
fig1, ax1 = plt.subplots(figsize=(13, 5))
ax1.plot(df[COL_TIME], df["wholesale_eur_kwh"] * 1000, lw=0.4, color="#378ADD")
ax1.set_title(f"Hourly wholesale electricity price - {YEAR}")
ax1.set_xlabel("Time")
ax1.set_ylabel("Wholesale price [EUR/MWh]")
ax1.grid(alpha=0.3)
fig1.tight_layout()

# --- Chart 2: annual cost composition -----------------------------------
fig3, ax3 = plt.subplots(figsize=(13, 6))
names = list(components.keys())
values = list(components.values())
# colour by bucket: energy (coral), power (teal), fixed (grey)
bucket_color = {
    "Wholesale\ncommodity": "#D85A30", "Supplier\nmarkup": "#D85A30",
    "Energy tax": "#D85A30", "Transport\n(per kWh)": "#D85A30",
    "kW-contract": "#1D9E75", "kW-max": "#1D9E75",
    "Transport\nvastrecht": "#888780", "Aansluitverg.\n(annual)": "#888780",
    "Metering": "#888780", "Vermindering": "#888780",
}
colors = [bucket_color[n] for n in names]
bars = ax3.bar(names, values, color=colors)
ax3.set_title(f"Annual cost composition - {YEAR}  |  Total = {Cost_total:,.0f} EUR")
ax3.set_ylabel("Annual cost [EUR]")
ax3.grid(alpha=0.3, axis="y")
for b, v in zip(bars, values):
    ax3.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}",
             ha="center", va="bottom", fontsize=8)
# legend for buckets
from matplotlib.patches import Patch
ax3.legend(handles=[
    Patch(color="#D85A30", label="Energy (EUR/kWh)"),
    Patch(color="#1D9E75", label="Power (EUR/kW)"),
    Patch(color="#888780", label="Fixed (EUR/yr)"),
])
fig3.tight_layout()

# --- Chart 3: wholesale price duration curve + threshold ----------------
fig4, ax4 = plt.subplots(figsize=(13, 5))
wholesale_mwh = df["wholesale_eur_kwh"].values * 1000
sorted_desc = np.sort(wholesale_mwh)[::-1]          # most -> least expensive
hours_axis = np.arange(1, len(sorted_desc) + 1)

n_below = int((wholesale_mwh < PRICE_THRESHOLD_EUR_MWH).sum())
pct_below = n_below / len(wholesale_mwh) * 100

ax4.plot(hours_axis, sorted_desc, lw=1.2, color="#378ADD", label="Wholesale (sorted)")
ax4.axhline(PRICE_THRESHOLD_EUR_MWH, color="#D85A30", lw=1.2, ls="--",
            label=f"Threshold = {PRICE_THRESHOLD_EUR_MWH:g} EUR/MWh")
ax4.axvspan(len(wholesale_mwh) - n_below, len(wholesale_mwh), color="#378ADD", alpha=0.12)
ax4.axvline(len(wholesale_mwh) - n_below, color="grey", lw=0.8, ls=":")
ax4.set_title(f"Wholesale price duration curve - {YEAR}  |  "
              f"{n_below:,} h ({pct_below:.1f}%) below {PRICE_THRESHOLD_EUR_MWH:g} EUR/MWh")
ax4.set_xlabel("Hours (sorted from most to least expensive)")
ax4.set_ylabel("Wholesale price [EUR/MWh]")
ax4.legend()
ax4.grid(alpha=0.3)
fig4.tight_layout()

# --- Chart 4: wholesale price distribution + threshold ------------------
fig4, ax4 = plt.subplots(figsize=(13, 5))
wholesale_mwh = df["wholesale_eur_kwh"].values * 1000

n_below = int((wholesale_mwh < PRICE_THRESHOLD_EUR_MWH).sum())
pct_below = n_below / len(wholesale_mwh) * 100

# split the data so the "below threshold" hours are shaded distinctly
bins = np.linspace(wholesale_mwh.min(), wholesale_mwh.max(), 80)
ax4.hist(wholesale_mwh[wholesale_mwh < PRICE_THRESHOLD_EUR_MWH], bins=bins,
         color="#378ADD", alpha=0.9, label=f"Below threshold ({n_below:,} h)")
ax4.hist(wholesale_mwh[wholesale_mwh >= PRICE_THRESHOLD_EUR_MWH], bins=bins,
         color="#B4B2A9", alpha=0.7, label="At/above threshold")
ax4.axvline(PRICE_THRESHOLD_EUR_MWH, color="#D85A30", lw=1.2, ls="--",
            label=f"Threshold = {PRICE_THRESHOLD_EUR_MWH:g} EUR/MWh")
ax4.set_title(f"Wholesale price distribution - {YEAR}  |  "
              f"{n_below:,} h ({pct_below:.1f}%) below {PRICE_THRESHOLD_EUR_MWH:g} EUR/MWh")
ax4.set_xlabel("Wholesale price [EUR/MWh]")
ax4.set_ylabel("Number of hours")
ax4.legend()
ax4.grid(alpha=0.3, axis="y")
fig4.tight_layout()

# --- Chart 5: when below-threshold hours occur, by hour of day ----------
fig5, ax5 = plt.subplots(figsize=(13, 5))
below = df[df["wholesale_eur_kwh"] * 1000 < PRICE_THRESHOLD_EUR_MWH]
counts = below[COL_TIME].dt.hour.value_counts().reindex(range(24), fill_value=0)
ax5.bar(counts.index, counts.values, color="#378ADD")
ax5.set_title(f"When wholesale is below {PRICE_THRESHOLD_EUR_MWH:g} EUR/MWh, by hour of day - {YEAR}  "
              f"|  {len(below):,} h total")
ax5.set_xlabel("Hour of day")
ax5.set_ylabel("Number of hours below threshold")
ax5.set_xticks(range(24))
ax5.grid(alpha=0.3, axis="y")
fig5.tight_layout()

# --- Chart 6: when below-threshold hours occur, by month ----------------
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
fig6, ax6 = plt.subplots(figsize=(13, 5))
below = df[df["wholesale_eur_kwh"] * 1000 < PRICE_THRESHOLD_EUR_MWH]
counts = below[COL_TIME].dt.month.value_counts().reindex(range(1, 13), fill_value=0)
ax6.bar(range(1, 13), counts.values, color="#378ADD")
ax6.set_title(f"When wholesale is below {PRICE_THRESHOLD_EUR_MWH:g} EUR/MWh, by month - {YEAR}  "
              f"|  {len(below):,} h total")
ax6.set_xlabel("Month")
ax6.set_ylabel("Number of hours below threshold")
ax6.set_xticks(range(1, 13))
ax6.set_xticklabels(month_labels)
ax6.grid(alpha=0.3, axis="y")
fig6.tight_layout()

# --- Chart 7: when below-threshold hours occur, by week -----------------
fig7, ax7 = plt.subplots(figsize=(13, 5))
below = df[df["wholesale_eur_kwh"] * 1000 < PRICE_THRESHOLD_EUR_MWH]
week = below[COL_TIME].dt.isocalendar().week
counts = week.value_counts().reindex(range(1, 54), fill_value=0)
ax7.bar(counts.index, counts.values, color="#378ADD")
ax7.set_title(f"When wholesale is below {PRICE_THRESHOLD_EUR_MWH:g} EUR/MWh, by week - {YEAR}  "
              f"|  {len(below):,} h total")
ax7.set_xlabel("Week of year")
ax7.set_ylabel("Number of hours below threshold")
ax7.set_xticks(range(1, 54, 4))
ax7.set_xlim(0.5, 53.5)
ax7.grid(alpha=0.3, axis="y")
fig7.tight_layout()

plt.show()