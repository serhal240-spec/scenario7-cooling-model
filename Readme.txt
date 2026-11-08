Scenario 7: AI data centre cooling in Phoenix, Arizona

Hourly model of a 1 MW data centre over one year. It compares three cooling options:
  Baseline: evaporative cooling tower
  Option A: dry cooler with a trim chiller (no water)
  Option B: hybrid dry cooler with water spray on hot hours

Libraries needed:
  Python 3.9 or newer
  numpy
  pandas
  matplotlib

Install them with:
  pip install numpy pandas matplotlib

How to run:
  1. Keep all the files and Phoenix_TMYx_2011-2025_clean_hourly.csv in the same folder.
  2. Run: python main.py
  3. The results print step by step (about 40 seconds), then the figures open.

Files:
  main.py      the file you run
  inputs.py    all input values and their sources
  physics.py   formulas (air properties, tower, dry cooler, chiller)
  options.py   the three cooling options and the indicators
  analysis.py  runs the model and prints the results
  figures.py   draws the figures

To change an input, edit its value in inputs.py and run main.py again.