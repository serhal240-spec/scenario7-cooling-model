"""
Scenario 7: AI data centre cooling in Phoenix, Arizona
Hourly model over the 8,760 hours of the TMYx weather year

    Baseline  evaporative cooling tower behind a plate heat exchanger (HX-2), trim chiller as backup
    Option A  dry cooler + trim chiller, no water on site
    Option B  hybrid: the same dry cooler, with adiabatic spray on the inlet air. Two settings:
              the spray switches on when the outdoor air is hotter than T_sw, and then cools the
              coil inlet air down to T_set (or as far as the spray can). Both are picked together
              as the pair with the lowest electricity that keeps site water under the annual limit V_lim.

Equal service in all three: same IT load, same 40 °C limit on the water going to the chips,
same CDU and rack pumps, same non cooling overhead (UPS losses, lighting, room air system).
The overhead is built bottom up, so PUE is a result of the model, not an input.

Put Phoenix_TMYx_2011-2025_clean_hourly.csv in the same folder as these files.
Libraries: numpy, pandas, matplotlib.
Run this file (python main.py): results print step by step (about 40 seconds), then all figures
open on screen. Nothing is saved.

Files:
    inputs.py    every input value with its source
    physics.py   air properties, tower and coil formulas, chiller COP, overhead
    options.py   baseline, Option A, Option B, indicators
    analysis.py  runs the model and prints STEP 0 to 10
    figures.py   report Figures 3 to 13 and four extra figures
"""

import matplotlib.pyplot as plt

import analysis   # runs the model and prints STEP 0 to 10
import figures    # draws the figures

plt.show()
