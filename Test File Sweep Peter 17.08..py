# -*- coding: utf-8 -*-
"""
sweep.py
==================================================================
Drives Test_File_Peter_17_08.run_case() across a grid of configurations and
collects the headline results into one summary table + workbook.

Put this file in the SAME folder as Test_File_Peter_17_08.py, main2_Peter.py,
ATES_obj_Peter.py and the data files, then:

    python sweep.py

run_case() returns a dict of results per run. This script only sets the knobs
it wants to vary; everything else falls back to the module defaults.

Notes
-----
* make_plots=False  -> no matplotlib windows during the sweep (essential; a big
  grid would otherwise open dozens of figures and can block on plt.show()).
* write_excel=False -> skip the per-run 5-sheet workbook and just collect the
  returned numbers. Flip a single run to True if you want its full workbook.
* Watch 'Reff' and 'injected_volume_m3' NEXT TO the LCOH columns: toggling the
  HP or changing ATES_MAX_V moves the injected volume (via the cold-well
  temperature assumption / Factor_due_HP), which shifts Reff and therefore the
  ATES LCOH. Keeping them side by side makes that coupling visible.
==================================================================
"""

import matplotlib
matplotlib.use("Agg")          # headless: no windows even if something calls show()

import pandas as pd
from Test_File_Peter_17_08 import run_case

# ================================================================== #
#  DEFINE THE SWEEP GRID HERE                                        #
# ================================================================== #

# Example: HP on/off across a range of storage sizes.
MAX_V_GRID  = [100, 200, 300, 400]
USE_HP_GRID = [False, True]

# Example (uncomment to sweep the HP compressor size instead):
# HP_POWER_GRID = [500, 1000, 1500, 2000, 3000]

# ================================================================== #


def main():
    results = []

    for max_v in MAX_V_GRID:
        for use_hp in USE_HP_GRID:
            tag = f"maxV{max_v}_{'HPon' if use_hp else 'HPoff'}"
            print("\n" + "#" * 70)
            print(f"# RUN: {tag}")
            print("#" * 70)
            res = run_case(
                USE_HP=use_hp,
                ATES_MAX_V=max_v,
                tag=tag,
                make_plots=False,     # no popups during the sweep
                write_excel=False,    # set True for a run you want the full workbook of
            )
            results.append(res)

    # --- Collect into a summary table -----------------------------------------
    summary_cols = [
        "tag", "USE_HP", "ATES_MAX_V",
        "Reff", "injected_volume_m3", "extracted_volume_m3",
        "demand_GWh", "geo_GWh", "ates_direct_GWh", "hp_GWh", "gas_GWh", "unmet_GWh",
        "system_lcoh", "geo_lcoh", "ates_lcoh", "gas_lcoh",
        "hp_elec_GWh", "hp_elec_cost_eur", "hp_mean_COP", "hp_capex_Meur",
        "total_CO2_t", "total_CO2_cost_eur",
    ]
    df = pd.DataFrame([{k: r.get(k) for k in summary_cols} for r in results])

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))

    out = "sweep_summary.xlsx"
    df.to_excel(out, index=False)
    print(f"\nSaved sweep summary -> {out}  ({len(df)} runs)")

    return df, results


if __name__ == "__main__":
    main()