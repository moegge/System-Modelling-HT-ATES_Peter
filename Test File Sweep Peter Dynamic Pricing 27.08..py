# -*- coding: utf-8 -*-
"""
Test File Sweep Peter Dynamic Pricing 27.08..py
==================================================================
Dynamic-pricing sweep for the discharge-side heat pump.

Run 1  : baseline, dispatch OFF (HP on every discharge hour).
Runs 2+: dispatch ON, sweeping the price threshold.

The threshold acts on c_marg (Eq. 1). The economically correct level is
COP * gas_marginal_cost = COP * gas_price/eff; at COP 3-5 and 0.10 EUR/kWh
gas that is ~320-540 EUR/MWh. Values below that deliberately switch the HP
off in hours it would still have won - the sweep maps that cost.

CAVEAT: heat_pump_ATES.elec_price is a scalar, so shifting the HP into cheap
hours does NOT reduce opex yet. Until an hourly price array is wired in, every
threshold below always-on can only look worse. Read the runtime/mode columns,
not the LCOH, for now.

Output -> ./results sweep/dynamic pricing/
==================================================================
"""
import os
import numpy as np
import pandas as pd

from Test_File_Peter_17_08 import run_case
from main2_Peter import build_hp_dispatch

# ================================================================== #
#  SWEEP SETTINGS                                                    #
# ================================================================== #

THRESHOLDS_EUR_MWH = [50, 100, 150, 200, 300, 430, 600, 900, 1e6]
# 1e6 = always on -> should reproduce the baseline exactly (validation row).

CONFIG = "GGAH"              # must include the HP for the sweep to do anything
WRITE_PER_RUN_EXCEL = True  # True -> full timeseries workbook per run (heavy)

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT_DIR = os.path.join(_HERE, "results sweep", "dynamic pricing")

# ================================================================== #


def _mode_counts(ts_mode):
    """A/B/D discharge-hour counts from the per-timestep mode array."""
    m = np.asarray(ts_mode, dtype=object)
    return (int((m == 'A').sum()), int((m == 'B').sum()), int((m == 'D').sum()))


def _row(res, label, signal_on_hours):
    a, b, d = _mode_counts(res["ts_mode"])
    return {
        "Case": label,
        "Threshold [EUR/MWh]": res["threshold_eur_mwh"],
        "Signal ON hours": signal_on_hours,
        "Mode A (HX only)": a,
        "Mode B (HP, price-driven)": b,
        "Mode D (HP, override)": d,
        "HP heat [GWh]": res["hp_GWh"],
        "HP electricity [GWh]": res["hp_elec_GWh"],
        "Pricing": "hourly" if res["hp_hourly_pricing"] else "flat",
        "Mean price paid [EUR/MWh]": res["hp_price_paid_eur_kwh"] * 1000,
        "Elec cost [kEUR]": res["hp_elec_cost_eur"] / 1e3,
        "Elec cost flat [kEUR]": res["hp_elec_cost_flat_eur"] / 1e3,
        "HP mean COP": res["hp_mean_COP"],
        "ATES direct [GWh]": res["ates_direct_GWh"],
        "Gas boiler [GWh]": res["gas_GWh"],
        "Unmet [GWh]": res["unmet_GWh"],
        "System LCOH (Yang) [EUR/kWh]": res["system_lcoh_yang"],
        "ATES LCOH [EUR/kWh]": res["ates_lcoh"],
        "Gas LCOH [EUR/kWh]": res["gas_lcoh"],
        "CO2 [t/yr]": res["total_CO2_t"],
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    # --- Run 1: baseline, dispatch off ------------------------------------
    print("\n" + "#" * 68)
    print("#  BASELINE - dynamic dispatch OFF (HP on every discharge hour)")
    print("#" * 68)
    res = run_case(CONFIG=CONFIG,
                   HP_DYNAMIC_DISPATCH=False,
                   OUTFILE=os.path.join(OUT_DIR, "baseline_static.xlsx"),
                   tag="baseline",
                   make_plots=False,
                   write_excel=WRITE_PER_RUN_EXCEL)
    rows.append(_row(res, "Baseline (always on)", np.nan))

    # --- Runs 2+: threshold sweep -----------------------------------------
    for thr in THRESHOLDS_EUR_MWH:
        # Rebuild the signal here purely to report its ON count; run_case
        # rebuilds it internally from the same inputs.
        n_on = int(build_hp_dispatch(n_timesteps=8760,
                                     threshold_eur_mwh=thr,
                                     verbose=False).sum())

        print("\n" + "#" * 68)
        print(f"#  DYNAMIC - threshold {thr:g} EUR/MWh  ({n_on}/8760 h ON)")
        print("#" * 68)
        res = run_case(CONFIG=CONFIG,
                       HP_DYNAMIC_DISPATCH=True,
                       HP_THRESHOLD_EUR_MWH=thr,
                       OUTFILE=os.path.join(OUT_DIR, f"dynamic_thr{thr:g}.xlsx"),
                       tag=f"thr{thr:g}",
                       make_plots=False,
                       write_excel=WRITE_PER_RUN_EXCEL)
        rows.append(_row(res, f"Dynamic thr={thr:g}", n_on))

    # --- Summary ----------------------------------------------------------
    summary = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "summary_dynamic_pricing.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Sweep", index=False)

    print("\n" + "=" * 68)
    print("  SWEEP SUMMARY")
    print("=" * 68)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()