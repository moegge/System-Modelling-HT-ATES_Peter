# -*- coding: utf-8 -*-
"""
Result Sweep Delft Dynamic Pricing 27.08..py
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
import sys

# model_driver.py / main2_Peter.py / ATES_obj_Peter.py live one level up, in
# System-Modelling-HT-ATES_Peter; this file sits in the "Delft Case" subfolder.
_MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODEL_ROOT not in sys.path:
    sys.path.insert(0, _MODEL_ROOT)
    
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

from model_driver import run_case, TIMESTEP, DEMAND_EXAMPLE, DEMAND_T_IN, DEMAND_T_OUT
from main2_Peter import build_hp_dispatch, demand_class

# ================================================================== #
#  SWEEP SETTINGS                                                    #
# ================================================================== #

ALWAYS_ON_THRESHOLD = 1e6    # gate never binds -> HP available every discharge hour
THRESHOLDS_EUR_MWH = [100, 130, 160, 190, 220, 250, 300, 430, ALWAYS_ON_THRESHOLD]
# 1e6 = always on

CONFIG = "GGAH"              # must include the HP for the sweep to do anything
GAS_PRICE = 0.030            # [EUR/kWh_gas] = 80 EUR/MWh
CO2_PRICE = 75.0            # [EUR/tonne]
WRITE_PER_RUN_EXCEL = True  # True -> full timeseries workbook per run (heavy)

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT_DIR = os.path.join(_HERE, "results sweep",
                       f"dynamic pricing gas{GAS_PRICE*1000:.0f} co2{CO2_PRICE:.0f}")

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

def _plots(summary, out_dir):
    """
    Six panels on one figure. x-axis is the dispatch threshold on a log scale,
    because the sweep spans 50 to 1e6. The always-on run and the flat-priced
    baseline are drawn as horizontal reference lines, not as curve points.
    """
    thr = summary["Threshold [EUR/MWh]"]
    curve = summary[thr.notna() & (thr < ALWAYS_ON_THRESHOLD)].sort_values(
        "Threshold [EUR/MWh]")
    always = summary[thr == ALWAYS_ON_THRESHOLD]
    base = summary[thr.isna()]
    if curve.empty:
        print("  (no threshold rows -- skipping figures)")
        return

    x = curve["Threshold [EUR/MWh]"].values
    ref = (lambda col: float(always[col].iloc[0]) if not always.empty else np.nan)
    bas = (lambda col: float(base[col].iloc[0]) if not base.empty else np.nan)

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0))
    (a, b, c), (d, e, f) = axes

    # (a) the headline: does gating the HP on price lower the system LCOH?
    a.plot(x, curve["System LCOH (Yang) [EUR/kWh]"].values * 1000.0,
           "o-", ms=4, color="tab:blue")
    for val, col, lbl in ((ref("System LCOH (Yang) [EUR/kWh]"), "tab:green",
                           "always on (hourly)"),
                          (bas("System LCOH (Yang) [EUR/kWh]"), "0.5",
                           "baseline (flat price)")):
        if np.isfinite(val):
            a.axhline(val * 1000.0, ls="--", lw=1.2, color=col, label=lbl)
    a.set_ylabel("System LCOH (\u20ac/MWh)")
    a.set_title("(a) System LCOH", fontsize=9)
    a.legend(fontsize=7)

    # (b) how much the HP actually runs
    b.plot(x, curve["HP heat [GWh]"].values, "o-", ms=4,
           color="tab:green", label="HP heat delivered")
    b.plot(x, curve["HP electricity [GWh]"].values, "s--", ms=4,
           color="tab:red", label="HP electricity")
    if not always.empty:
        b.axhline(ref("HP heat [GWh]"), ls=":", lw=1.2, color="tab:green")
    b.set_ylabel("GWh/yr")
    b.set_title("(b) Heat-pump throughput", fontsize=9)
    b.legend(fontsize=7)

    # (c) the fixed-charge trap: Eq. 2's vastrecht / kW-contract / APV spread
    #     over ever less energy as the gate tightens.
    c.plot(x, curve["Mean price paid [EUR/MWh]"].values, "o-", ms=4,
           color="tab:purple")
    if np.isfinite(bas("Mean price paid [EUR/MWh]")):
        c.axhline(bas("Mean price paid [EUR/MWh]"), ls="--", lw=1.2,
                  color="0.5", label="flat price")
        c.legend(fontsize=7)
    c.set_ylabel("Mean price paid (\u20ac/MWh)")
    c.set_title("(c) All-in electricity price", fontsize=9)

    # (d) where the displaced heat goes
    d.plot(x, curve["Gas boiler [GWh]"].values, "o-", ms=4,
           color="tab:red", label="Gas boiler")
    d.plot(x, curve["ATES direct [GWh]"].values, "s--", ms=4,
           color="tab:orange", label="ATES direct")
    d.set_ylabel("GWh/yr")
    d.set_title("(d) Backup and direct-HX heat", fontsize=9)
    d.legend(fontsize=7)

    # (e) emissions -- the HP trades gas CO2 for grid CO2
    e.plot(x, curve["CO2 [t/yr]"].values, "o-", ms=4, color="tab:brown")
    if np.isfinite(ref("CO2 [t/yr]")):
        e.axhline(ref("CO2 [t/yr]"), ls="--", lw=1.2, color="tab:green",
                  label="always on")
        e.legend(fontsize=7)
    e.set_ylabel("System CO\u2082 (t/yr)")
    e.set_title("(e) Emissions", fontsize=9)

    for ax in (a, b, c, d, e):
        ax.set_xscale("log")
        ax.set_xlabel("Dispatch threshold on c_marg (\u20ac/MWh)")
        ax.grid(alpha=0.3, which="both")

    # (f) mode split -- categorical, so the always-on run sits alongside.
    #     Mode D is the override: it fires regardless of price, so it is the
    #     floor the threshold cannot cut into.
    bars = pd.concat([curve, always])
    labels = [f"{t:g}" if t < ALWAYS_ON_THRESHOLD else "always\non"
              for t in bars["Threshold [EUR/MWh]"]]
    xi = np.arange(len(bars))
    bot = np.zeros(len(bars))
    for col, colr, lbl in (("Mode A (HX only)", "tab:blue", "A (HX only)"),
                           ("Mode B (HP, price-driven)", "tab:green", "B (HP)"),
                           ("Mode D (HP, override)", "tab:red", "D (override)")):
        v = bars[col].values.astype(float)
        f.bar(xi, v, bottom=bot, color=colr, label=lbl)
        bot += v
    f.set_xticks(xi)
    f.set_xticklabels(labels, fontsize=7)
    f.set_xlabel("Dispatch threshold (\u20ac/MWh)")
    f.set_ylabel("Discharge hours")
    f.set_title("(f) Dispatch mode split", fontsize=9)
    f.legend(fontsize=7)
    f.grid(axis="y", alpha=0.3)

    fig.suptitle("Heat-pump dispatch threshold sweep  --  reference is the "
                 "always-on run, priced hourly (Eq. 2)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "dynamic_pricing_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure -> {path}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    # --- Run 1: baseline, dispatch off ------------------------------------
    print("\n" + "#" * 68)
    print("#  BASELINE - dynamic dispatch OFF (HP on every discharge hour)")
    print("#" * 68)
    res = run_case(CONFIG=CONFIG,
                   GAS_PRICE=GAS_PRICE,
                   CO2_PRICE=CO2_PRICE,
                   HP_DYNAMIC_DISPATCH=False,
                   OUTFILE=os.path.join(OUT_DIR, "baseline_static.xlsx"),
                   tag="baseline",
                   make_plots=False,
                   write_excel=WRITE_PER_RUN_EXCEL)
    rows.append(_row(res, "Baseline (always on)", np.nan))

    # --- Runs 2+: threshold sweep -----------------------------------------
    # Timestep count as run_case sees it: demand_class.adjust_for_timesetting
    # resamples the profile, so this is 8760 * 3600/TIMESTEP.
    _dem = demand_class(T_in=DEMAND_T_IN, T_out=DEMAND_T_OUT,
                        example_demand=DEMAND_EXAMPLE)
    _dem.adjust_for_timesetting(len_timestep=TIMESTEP)
    n_steps = len(_dem.data)

    for thr in THRESHOLDS_EUR_MWH:
        # Rebuild the signal here purely to report its ON count; run_case
        # rebuilds it internally from the same inputs.
        n_on = int(build_hp_dispatch(n_timesteps=n_steps,
                                     len_timestep=TIMESTEP,
                                     threshold_eur_mwh=thr,
                                     verbose=False).sum())

        print("\n" + "#" * 68)
        print(f"#  DYNAMIC - threshold {thr:g} EUR/MWh  ({n_on}/{n_steps} steps ON)")
        print("#" * 68)
        res = run_case(CONFIG=CONFIG,
                       GAS_PRICE=GAS_PRICE,
                       CO2_PRICE=CO2_PRICE,
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
    _plots(summary, OUT_DIR)


if __name__ == "__main__":
    main()