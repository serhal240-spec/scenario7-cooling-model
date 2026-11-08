"""
Scenario 7 model: results.
Runs the base case and prints STEP 0 to 10: input register, common system, weather, each option,
energy balance, comparison table, trade offs, second law checks and the sensitivity study.
Imported by figures.py and main.py; it can also be run on its own (prints only, no figures).
"""

from pathlib import Path

import numpy as np
import pandas as pd

import inputs
import physics
import options
from inputs import *
from physics import *
from options import *

# Folder these files sit in (falls back to the current folder in Jupyter)
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()


# =====================================================================
# BASE CASE
# =====================================================================

print("Running the base case (a few seconds)...")
raw = pd.read_csv(HERE / "Phoenix_TMYx_2011-2025_clean_hourly.csv")
run = run_all(raw)
wx, lp, (base, optA, optB), sweep, ov = run["wx"], run["lp"], run["res"], run["sweep"], run["ov"]
grid = run["grid"]
ind = [indicators(r, wx, ov) for r in (base, optA, optB)]
names = ["Baseline", "Option A", "Option B"]

# Input register: value, unit, status, source. Same status codes as the report's Table 1:
# B = brief, S = sourced, D = derived from sourced data, A = assumption checked against sources, G = group choice
REGISTER = [
    ("P_IT", "kW", "B", "IT load, brief p3"),
    ("H_op", "h/yr", "A", "every weather hour at full load; brief 8,000"),
    ("f_liq", "-", "A", "liquid share; Lenovo Neptune up to 80%; tested 0.70 to 0.80"),
    ("T_supply", "°C", "B", "water to chips, brief p3"),
    ("T_return", "°C", "B", "water from chips, brief p3"),
    ("cp_w", "kJ/kgK", "B", "brief p3"),
    ("h_fg", "kJ/kg", "S", "Cengel and Boles"),
    ("T_wb_d", "°C", "S", "ASHRAE Fundamentals 2025 ch.14 (TMYx header), 0.4% evaporation"),
    ("A_cdu", "K", "S", "facility loop 35/55; Vertiv XDU450, Delta SuperCDU 4 to 5 K"),
    ("A_hx", "K", "A", "Deppmann (2021); Alfa Laval"),
    ("SFP_pump", "kW/kW", "A", "4.0 kW, about 270 kPa at 9.57 kg/s"),
    ("eta_p", "-", "A", "pump + motor efficiency, typical 0.6 to 0.75"),
    ("dp_hx2", "kPa", "S", "Alfa Laval HVAC PHE practice"),
    ("dp_coil", "kPa", "D", "Kelvion dry cooler datasheet (24 kPa) + piping"),
    ("dp_P3", "kPa", "S", "BAC pumping and piping guide"),
    ("f_min", "-", "S", "DesignBuilder"),
    ("A_tw", "K", "S", "Brentwood (sets tower size)"),
    ("LG_d", "-", "A", "tower L/G at full fan, typical 0.75 to 1.5"),
    ("n_tw", "-", "S", "Merkel exponent, Kloppers and Kroger (2005)"),
    ("SFP_tw", "kW/kW", "S", "Brentwood"),
    ("phi_evap", "-", "D", "ASHRAE HVAC S&E ch.40"),
    ("COC", "-", "S", "US DOE FEMP"),
    ("drift_frac", "-", "A", "typical of modern drift eliminators"),
    ("A_dc", "K", "A", "Brentwood"),
    ("dTa_d", "K", "D", "Trane DRY500 / Airedale"),
    ("n_dc", "-", "S", "coil UA exponent, Zukauskas (Incropera 2011)"),
    ("SFP_dc", "kW/kW", "S", "ASHRAE HVAC S&E ch.41"),
    ("COP_rated", "-", "A", "AHRI rating point; below the 2.96 code minimum, tested 2.2 to 3.4"),
    ("dT_ev", "K", "S", "evaporator approach, Stoecker and Jones (1982)"),
    ("dT_cd", "K", "S", "condenser approach, Stoecker and Jones (1982)"),
    ("eps_ad", "-", "S", "IIR"),
    ("eta_sp", "-", "A", "share of spray evaporated; SPX/CTI 67 to 75%"),
    ("P_sp", "kW", "S", "Evans and Corral (SPX/CTI 2025) wetting rate on pad top area"),
    ("WUE_lim", "L/kWh", "G", "Microsoft 2025 fleet WUE"),
    ("V_lim", "m3/yr", "D", "WUE_lim x IT energy, on consumption"),
    ("r_B", "-", "A", "drained water returned (USGS consumptive use)"),
    ("f_ups", "-", "A", "UPS + distribution losses; new UPSs 92 to 95% efficient"),
    ("P_light", "kW", "D", "lighting and controls, WSEC 10.1 W/m2 x about 1,000 m2"),
    ("T_chw_room", "°C", "A", "room chiller water temperature"),
    ("COP_max", "-", "A", "chiller COP cap (room and trim)"),
    ("SFP_crah", "kW/kW", "A", "room air handler fans"),
    ("CI_grid", "kg/kWh", "S", "EPA eGRID2023 AZNM"),
    ("EWIF", "L/kWh", "D", "Macknick (2012) x AZNM mix"),
    ("PUE_ref", "-", "B", "brief p3, used only as a check"),
]
print("=" * 70)
print("STEP 0: INPUT REGISTER (B brief, S sourced, D derived, A assumption, G group choice)")
print("=" * 70)
for name, unit, status, src in REGISTER:
    print(f"  {name:<13} {globals()[name]:>10g} {unit:<7} {status}  {src}")

print("\n" + "=" * 70)
print("STEP 1: COMMON SYSTEM (same in all three options)")
print("=" * 70)
print(f"  Q_IT_liquid = {Q_liq:.0f} kW  (f_liq = {f_liq:.2f} of IT, assumption, tested 0.70 to 0.80 in STEP 10)")
print(f"  Q_IT_air = {Q_res:.0f} kW  (to room air, removed by the room air system, STEP 6)")
print(f"  m_w = {lp['m_w']:.2f} kg/s  (rack loop and facility loop)")
print(f"  V_w = {lp['V_w'] * 1000:.2f} L/s")
print(f"  P_loop (rack + facility pumps) = SFP_pump x Q_IT_liquid = {SFP_pump} x {Q_liq:.0f} = {lp['P_loop']:.2f} kW")
print(f"  Rack water: {T_supply:.0f} °C to the chips, {T_supply + lp['Q_rack'] / lp['C_w']:.2f} °C back")
print(f"  Facility water supply needed T_fs = {lp['T_fs']:.1f} °C  (CDU approach {A_cdu:.0f} K)")
print("  Trim chiller COP formula (same chiller model in all three options):")
print(f"    COP = eta_II x T_ev / (T_cd - T_ev),  T_ev = chilled water out - {dT_ev:.0f} K,  T_cd = outdoor dry bulb + {dT_cd:.0f} K")
print(f"    Rating point (AHRI 550/590): {T_chw_rated} °C chilled water, {T_db_rated:.0f} °C air -> "
      f"T_ev = {T_ev_rated - KELVIN:.1f} °C, T_cd = {T_cd_rated - KELVIN:.1f} °C")
print(f"    COP_carnot_rated = {COP_carnot_rated:.2f},  COP_rated = {COP_rated}  ->  eta_II = {eta_II:.3f}")
for T_air in (35, 40, 46):
    _, cop_x, _ = trim_chiller(np.array([1.0]), lp["T_fs"], np.array([float(T_air)]))
    print(f"    Our duty: {lp['T_fs']:.0f} °C water, {T_air} °C air -> COP = {cop_x[0]:.2f}")

print("\n" + "=" * 70)
print("STEP 2: WEATHER (TMYx 2011-2025, Phoenix Sky Harbor)")
print("=" * 70)
print(f"  Weather hours = {len(raw)}")
print(f"  h_weight = {wx['h_weight']:.4f} operating hours per weather hour (all h/yr below are operating hours)")
print(f"  Max dry bulb = {wx['T_db'].max():.1f} °C,  max wet bulb = {wx['T_wb'].max():.1f} °C")
print(f"  Wet bulb check vs CSV: largest difference = "
      f"{np.abs(wx['T_wb'] - raw['Wet bulb (°C), calculated'].to_numpy()).max():.2f} K")

print("\n" + "=" * 70)
print("STEP 3: BASELINE, EVAPORATIVE COOLING TOWER")
print("=" * 70)
print(f"  P_P2 (through HX-2) = {base['P_P2']:.2f} kW")
print(f"  P_P3 (tower water) = {base['P_P3']:.2f} kW")
print(f"  Q_tower at full duty = {base['Q_unit_full']:.1f} kW")
print(f"  Tower water: {base['T_in']:.2f} °C in, {base['T_out_set']:.1f} °C out needed")
print(f"  Tower sized for {T_wb_d + A_tw:.1f} °C cold water at {T_wb_d} °C wet bulb "
      f"(approach A_tw = {A_tw:.1f} K): Merkel number at full fan = {base['Me_d']:.3f}")
print(f"  Cold water actually needed = {base['T_out_set']:.1f} °C, so the tower has {base['T_out_set'] - T_wb_d - A_tw:.1f} K "
      f"of spare approach on the design day")
print(f"  Fan power at full speed = {SFP_tw * base['Q_unit_full']:.2f} kW")
print(f"  Mean fan speed = {base['f'].mean():.2f},  hours at full fan = {annual(base['f'] > 0.999, wx):,.0f} h/yr")
print(f"  Hours the tower cannot hold the set point (trim chiller) = {ind[0]['chiller_h']:,.0f} h/yr")
print(f"  E_fan = {annual(base['P_fan'], wx):,.0f} kWh/yr")
print(f"  E_pumps = {annual(base['P_pumps'], wx):,.0f} kWh/yr")
print(f"  E_chiller = {annual(base['W_ch'], wx):,.0f} kWh/yr")
print(f"  E_cool = {ind[0]['E_cool']:,.0f} kWh/yr")
print(f"  Evaporation = {to_m3(base['m_evap'], wx):,.0f} m3/yr")
print(f"  Drift = {to_m3(base['m_drift'], wx):,.0f} m3/yr")
print(f"  Blowdown = {to_m3(base['m_blow'], wx):,.0f} m3/yr")
print(f"  V_consumed (evaporation + drift + (1 - r_B) x blowdown, r_B = {r_B:g}) = {ind[0]['V_cons']:,.0f} m3/yr")
print(f"  V_withdrawn (make up) = {ind[0]['V_with']:,.0f} m3/yr")

print("\n" + "=" * 70)
print("STEP 4: OPTION A, DRY COOLER + TRIM CHILLER")
print("=" * 70)
print(f"  P_P2 (through coil) = {optA['P_P2']:.2f} kW")
print(f"  Q_dry_cooler at full duty = {optA['Q_fac']:.1f} kW,  water {optA['T_in']:.2f} °C in, {lp['T_fs']:.1f} °C out needed")
print(f"  Coil sized for air at {optA['T_ai_d']:.1f} °C: effectiveness = {optA['eps_d']:.3f}, NTU = {optA['NTU_d']:.2f}, "
      f"UA = {optA['UA_d']:.1f} kW/K")
print(f"  Air flow at full fan = {optA['m_air_d']:.1f} kg/s")
print(f"  Fan power at full speed = {SFP_dc * optA['Q_fac']:.2f} kW")
print(f"  Hours the trim chiller runs = {ind[1]['chiller_h']:,.0f} h/yr  (air above about {optA['T_ai_d']:.0f} °C)")
print(f"  Trim chiller peak duty = {ind[1]['chiller_peak']:.0f} kW  (size of the chiller)")
print(f"  Trim chiller peak electricity = {optA['W_ch'].max():.0f} kW")
run_h = optA["Q_ch"] > 0.01
print(f"  Mean COP when running (energy weighted) = {optA['Q_ch'][run_h].sum() / optA['W_ch'][run_h].sum():.2f}")
print(f"  E_fan = {annual(optA['P_fan'], wx):,.0f} kWh/yr")
print(f"  E_pumps = {annual(optA['P_pumps'], wx):,.0f} kWh/yr")
print(f"  E_chiller = {annual(optA['W_ch'], wx):,.0f} kWh/yr")
print(f"  E_cool = {ind[1]['E_cool']:,.0f} kWh/yr")
print(f"  Site water = 0 m3/yr")

print("\n" + "=" * 70)
print("STEP 5: OPTION B, HYBRID (DRY COOLER + ADIABATIC SPRAY)")
print("=" * 70)
print(f"  Water limit V_lim = WUE_lim x E_IT = {WUE_lim} x {E_IT:,.0f} kWh / 1000 = {V_lim:,.0f} m3/yr (site water consumed)")
print(f"  Chosen settings: spray on above T_sw = {optB['T_sw']:.1f} °C, cooling the coil air to T_set = "
      f"{optB['T_set']:.1f} °C  (lowest electricity within the limit)")
_one = sweep[sweep["V_cons"] <= V_lim].sort_values("E_cool").iloc[0]
print(f"  For comparison, the one setting rule (spray down to T_sw only) would pick T_sw = {_one['T_sw']:.1f} °C: "
      f"{_one['E_cool'] / 1000:,.1f} MWh/yr, {_one['V_cons']:,.0f} m3/yr")
_r = optB["Q_ch"] > 0.01
print(f"  Trim chiller hours = {annual(_r, wx):,.0f} h/yr, of which below 40 kW = "
      f"{annual(_r & (optB['Q_ch'] < 40), wx):,.0f} h/yr")
print(f"  Spray hours = {annual(optB['spraying'], wx):,.0f} h/yr")
print(f"  Hours the trim chiller runs = {ind[2]['chiller_h']:,.0f} h/yr")
print(f"  Trim chiller peak duty = {ind[2]['chiller_peak']:.0f} kW")
print(f"  E_fan = {annual(optB['P_fan'], wx):,.0f} kWh/yr")
print(f"  E_pumps (incl. spray pump) = {annual(optB['P_pumps'] + optB['P_spray'], wx):,.0f} kWh/yr")
print(f"  E_chiller = {annual(optB['W_ch'], wx):,.0f} kWh/yr")
print(f"  E_cool = {ind[2]['E_cool']:,.0f} kWh/yr")
print(f"  V_consumed (evaporated + (1 - r_B) x drained, r_B = {r_B:g}) = {ind[2]['V_cons']:,.0f} m3/yr")
print(f"  V_withdrawn (incl. drained spray) = {ind[2]['V_with']:,.0f} m3/yr")
optB_ref = option_b(wx, lp, T_spray_ref)
ind_ref = indicators(optB_ref, wx, ov)
print(f"  Reference: T_sw = {T_spray_ref} °C (Microsoft Arizona practice): water = {ind_ref['V_cons']:,.0f} m3/yr, "
      f"cooling electricity = {ind_ref['E_cool'] / 1000:,.1f} MWh/yr, trim chiller {ind_ref['chiller_h']:,.0f} h/yr")
print("  What each switchover setting costs and saves (full sweep in Figure 11):")
for T_lab in (45, 40, 35, 30, 25, 20, 15):
    row = sweep[np.isclose(sweep["T_sw"], T_lab)].iloc[0]
    print(f"    T_sw = {T_lab} °C: water = {row['V_cons']:,.0f} m3/yr, cooling electricity = {row['E_cool'] / 1000:,.1f} MWh/yr")

print("\n" + "=" * 70)
print("STEP 6: OVERHEAD AND WHOLE SYSTEM ENERGY BALANCE (first law, annual)")
print("=" * 70)
print("  Non cooling overhead, built bottom up (same equipment in all three options):")
print(f"    P_UPS = f_ups x P_IT = {f_ups} x {P_IT} = {ov['P_ups']:.1f} kW")
print(f"    P_light = {ov['P_light']:.1f} kW")
print(f"    Room heat Q_room = Q_IT_air + P_UPS + P_light = {Q_res:.0f} + {ov['P_ups']:.0f} + {ov['P_light']:.0f} "
      f"= {ov['Q_room']:.0f} kW")
print(f"    Room chiller: {T_chw_room:.0f} °C water, COP {ov['COP_room'].min():.2f} to {ov['COP_room'].max():.2f} "
      f"(capped at {COP_max}), hours at the cap = {annual(ov['COP_room'] >= COP_max - 1e-9, wx):,.0f} h/yr")
print(f"    Room air handler fans = SFP_crah x Q_room = {ov['P_crah']:.1f} kW")
print(f"    Room air system: mean {ov['P_room'].mean():.1f} kW, peak {ov['P_room'].max():.1f} kW, "
      f"{annual(ov['P_room'], wx) / 1000:,.1f} MWh/yr")
print(f"    E_other (UPS + lighting + room air) = {ind[0]['E_other'] / 1000:,.1f} MWh/yr")
print(f"  PUE is now a result: baseline {ind[0]['PUE']:.3f} vs brief reference {PUE_ref:.2f} (check only)")
print(f"  {'MWh/yr':<38}" + "".join(f"{n:>11}" for n in names))
rows = {
    "Electricity in: IT": [P_IT * H_op for _ in names],
    "Electricity in: other overhead": [i["E_other"] for i in ind],
    "Electricity in: cooling": [i["E_cool"] for i in ind],
    "Heat out: outdoor unit + fans": [annual(r["Q_unit"] + r["P_fan"] + r["P_spray"], wx) for r in (base, optA, optB)],
    "Heat out: trim chiller condenser": [annual(r["Q_ch"] + r["W_ch"], wx) for r in (base, optA, optB)],
    "Heat out: room air system": [Q_res * H_op + i["E_other"] for i in ind],
}
for k, v in rows.items():
    print(f"  {k:<38}" + "".join(f"{x / 1000:>11,.1f}" for x in v))
for r, i in zip((base, optA, optB), ind):
    heat_out = sum(rows[k][names.index(r["name"])] for k in rows if k.startswith("Heat"))
    gap = heat_out - i["E_DC"]
    print(f"  {r['name']}: electricity in = {i['E_DC'] / 1000:,.1f} MWh, heat out = {heat_out / 1000:,.1f} MWh, "
          f"gap = {gap:.2e} kWh  -> {'PASS' if abs(gap) < 1 else 'FAIL'}")

print("\n" + "=" * 70)
print("STEP 7: BASELINE vs OPTIONS (annual, CI = %.2f kg/kWh, EWIF = %.1f L/kWh)" % (CI_grid, EWIF))
print("=" * 70)
table_rows = [
    ("Cooling electricity (MWh/yr)", [i["E_cool"] / 1000 for i in ind], "{:,.1f}"),
    ("Facility electricity E_DC (MWh/yr)", [i["E_DC"] / 1000 for i in ind], "{:,.0f}"),
    ("Peak cooling power (kW)", [i["P_cool_peak"] for i in ind], "{:,.0f}"),
    ("Trim chiller hours (h/yr)", [i["chiller_h"] for i in ind], "{:,.0f}"),
    ("Heat rejected to air (MWh/yr)", [i["E_DC"] / 1000 for i in ind], "{:,.0f}"),
    ("Heat reused (MWh/yr)", [0, 0, 0], "{:,.0f}"),
    ("Emissions (t CO2e/yr)", [i["G_DC"] for i in ind], "{:,.0f}"),
    ("Site water consumed (m3/yr)", [i["V_cons"] for i in ind], "{:,.0f}"),
    ("Site water withdrawn (m3/yr)", [i["V_with"] for i in ind], "{:,.0f}"),
    ("Power station water (m3/yr)", [i["V_source"] for i in ind], "{:,.0f}"),
    ("PUE (-)", [i["PUE"] for i in ind], "{:.3f}"),
    ("EUE_IT (-)", [i["EUE_IT"] for i in ind], "{:.3f}"),
    ("CUE (kg CO2e/kWh IT)", [i["CUE"] for i in ind], "{:.3f}"),
    ("WUE_site (L/kWh IT)", [i["WUE_site"] for i in ind], "{:.3f}"),
    ("WUE_source (L/kWh IT)", [i["WUE_source"] for i in ind], "{:.3f}"),
]
col_names = ["Baseline (tower)", "Option A (dry)", f"Option B (hybrid, {optB['T_sw']:.1f} / {optB['T_set']:.0f} °C)"]
print(f"  {'':<38}" + "".join(f"{n:>34}" for n in col_names))
for label, vals, fmt in table_rows:
    print(f"  {label:<38}" + "".join(f"{fmt.format(v):>34}" for v in vals))

print("\n" + "=" * 70)
print("STEP 8: TRADE OFFS AND THRESHOLDS (vs baseline)")
print("=" * 70)
for k in (1, 2):
    dE = ind[k]["E_DC"] - ind[0]["E_DC"]
    dV = ind[0]["V_cons"] - ind[k]["V_cons"]
    print(f"  {names[k]}: extra electricity = {dE / 1000:,.1f} MWh/yr, site water saved = {dV:,.0f} m3/yr")
    print(f"      electricity per m3 saved = {dE / dV:.1f} kWh/m3")
    print(f"      extra emissions = {CI_grid * dE / 1000:,.1f} t CO2e/yr  ({CI_grid * dE / dV:.1f} kg CO2e per m3 saved)")
    print(f"      power station water added = {EWIF * dE / 1000:,.0f} m3/yr; "
          f"total water only rises if EWIF > {dV * 1000 / dE:,.0f} L/kWh")
cheapest_free = sweep[sweep["V_cons"] < 1]["E_cool"].min()
print(f"  Option B with no spray at all: E_cool = {cheapest_free / 1000:,.1f} MWh/yr (the same as Option A)")
print(f"  Option B lowest electricity at any water use: {sweep['E_cool'].min() / 1000:,.1f} MWh/yr "
      f"with {sweep.loc[sweep['E_cool'].idxmin(), 'V_cons']:,.0f} m3/yr (T_sw = {sweep.loc[sweep['E_cool'].idxmin(), 'T_sw']:.1f} °C)")

print("\n" + "=" * 70)
print("STEP 9: SECOND LAW AND TEMPERATURE CHECKS")
print("=" * 70)
K = KELVIN
def T_entropic(T1, T2):
    """Entropic mean temperature (K) of a stream going from T1 to T2 (°C)."""
    a, b = T1 + K, T2 + K
    return np.where(np.abs(a - b) > 1e-6, (a - b) / np.log(a / b), a)

T0 = wx["T_db"] + K                                   # dead state = outdoor air, each hour
# 9a. Trim chiller vs Carnot (Option A, the option that relies on it)
run_h = optA["Q_ch"] > 0.01
ratio = optA["COP"][run_h] / optA["COP_carnot"][run_h]
print(f"  9a Trim chiller, Option A: COP {optA['COP'][run_h].min():.2f} to {optA['COP'][run_h].max():.2f}, "
      f"Carnot COP {optA['COP_carnot'][run_h].min():.2f} to {optA['COP_carnot'][run_h].max():.2f}")
print(f"     COP / Carnot COP = {ratio.max():.2f} at most  -> {'PASS (below 1)' if ratio.max() < 1 else 'FAIL'}")
Q_ev = annual(optA["Q_ch"], wx); W_tot = annual(optA["W_ch"], wx)
print(f"     Chiller energy balance: Q_evap {Q_ev / 1000:,.1f} + W {W_tot / 1000:,.1f} = Q_cond {(Q_ev + W_tot) / 1000:,.1f} MWh/yr")
# 9b. Entropy generated by heat transfer across finite temperature differences
Tr_h, Tr_c = T_entropic(lp["T_s"] + lp["Q_rack"] / lp["C_w"], lp["T_s"]), T_entropic(lp["T_fs"], lp["T_fs"] + lp["Q_rack"] / lp["C_w"])
S_cdu = lp["Q_rack"] * (1 / Tr_c - 1 / Tr_h)          # kW/K
print(f"  9b CDU: rack water mean {Tr_h:.2f} K, facility water mean {Tr_c:.2f} K, S_gen = {S_cdu * 1000:.1f} W/K, "
      f"exergy destroyed = {annual(T0 * S_cdu, wx) / 1000:,.1f} MWh/yr (all options)")
S_dc = optA["Q_unit"] * (1 / T_entropic(optA["T_ai"], optA["T_ao"]) - 1 / T_entropic(optA["T_in"], optA["T_out"]))
print(f"     Dry cooler, Option A: S_gen always >= 0 -> {'PASS' if S_dc.min() >= 0 else 'FAIL'}, "
      f"exergy destroyed = {annual(T0 * S_dc, wx) / 1000:,.1f} MWh/yr")
S_dcB = optB["Q_unit"] * (1 / T_entropic(optB["T_ai"], optB["T_ao"]) - 1 / T_entropic(optB["T_in"], optB["T_out"]))
print(f"     Dry cooler, Option B: S_gen always >= 0 -> {'PASS' if S_dcB.min() >= 0 else 'FAIL'}, "
      f"exergy destroyed = {annual(T0 * S_dcB, wx) / 1000:,.1f} MWh/yr")
# 9c. Available thermal exergy of the loop heat that every option throws away
T_loop = T_entropic(T_return, T_supply)
X_loop = Q_liq * np.maximum(1 - T0 / T_loop, 0)
print(f"  9c Exergy of the {Q_liq:.0f} kW loop heat vs outdoor air = {annual(X_loop, wx) / 1000:,.0f} MWh/yr "
      f"({annual(X_loop, wx) / (Q_liq * H_op) * 100:.1f}% of the heat). Rejected, not reused, in all options.")
# 9d. Heat exchanger temperature compatibility, every hour
checks = [
    ("Tower cold water stays above the wet bulb", (base["T_out"] - wx["T_wb"]).min()),
    ("Dry cooler water out stays above air in (A)", (optA["T_out"] - optA["T_ai"]).min()),
    ("Dry cooler air out stays below water in (A)", (optA["T_in"] - optA["T_ao"]).min()),
    ("Dry cooler water out stays above air in (B)", (optB["T_out"] - optB["T_ai"]).min()),
    ("Sprayed air stays above the wet bulb (B)", (optB["T_ai"] - wx["T_wb"])[optB["spraying"]].min()),
    ("Evaporator below the chilled water (all)", dT_ev),
    ("Condenser above the outdoor air (all)", dT_cd),
]
print("  9d Temperature compatibility (smallest margin over the year):")
for label, margin in checks:
    print(f"     {label:<46} {margin:6.2f} K  -> {'PASS' if margin > 0 else 'FAIL'}")
balance = max(np.abs(r["Q_unit"] + r["Q_ch"] - r["Q_unit_full"]).max() for r in (base, optA, optB))
print(f"     Hourly loop balance, outdoor unit + chiller = duty: worst gap {balance:.2e} kW -> "
      f"{'PASS' if balance < 1e-6 else 'FAIL'}")

# 9e. Design peak check: can each option still hold the facility supply, and how big must the chillers be?
p_des = float(wx["p"].mean())
W_des = ((2501 - 2.326 * T_wb_des) * W_sat(T_wb_des, p_des) - cp_a * (T_db_des - T_wb_des)) / \
        (2501 + 1.86 * T_db_des - 4.186 * T_wb_des)
wx_des = {"T_db": np.array([T_db_des]), "T_wb": np.array([T_wb_des]), "p": np.array([p_des]),
          "W": np.array([W_des]), "h_a": h_air(np.array([T_db_des]), W_des), "h_weight": 1.0}
des = [baseline_tower(wx_des, lp), option_a(wx_des, lp), option_b(wx_des, lp, optB["T_sw"], optB["T_set"])]
ov_des = overhead(wx_des, lp)
print(f"  9e Design peak check: {T_db_des} °C dry bulb with {T_wb_des} °C wet bulb at the same time "
      f"(harsher than any real hour)")
print(f"     {'':<40}" + "".join(f"{n:>12}" for n in names))
rows_des = [
    ("Outdoor unit duty (kW)", [d["Q_unit"][0] for d in des]),
    ("Trim chiller duty at design (kW)", [d["Q_ch"][0] for d in des]),
    ("Trim chiller duty, worst TMYx hour (kW)", [r["Q_ch"].max() for r in (base, optA, optB)]),
    ("Cooling electricity at design (kW)", [d["P_cool"][0] for d in des]),
    ("Room air system at design (kW)", [ov_des["P_room"][0]] * 3),
    ("Facility power at design (kW)", [P_IT + ov_des["P_other"][0] + d["P_cool"][0] for d in des]),
    ("Site water at design (m3/h)", [d["m_cons"][0] * 3.6 for d in des]),
]
for label, vals in rows_des:
    print(f"     {label:<40}" + "".join(f"{v:>12,.1f}" for v in vals))
gap_des = max(abs(d["Q_unit"][0] + d["Q_ch"][0] - d["Q_unit_full"]) for d in des)
print(f"     {lp['T_fs']:.0f} °C facility supply held at design in all options (duty gap {gap_des:.1e} kW) -> "
      f"{'PASS' if gap_des < 1e-6 else 'FAIL'}; size the trim chillers to at least the larger duty above")

print("\n" + "=" * 70)
print("STEP 10: SENSITIVITY (one input at a time; overhead recalculated and Option B settings re-picked in each case)")
print("=" * 70)
print("  Running about 40 extra cases (about half a minute)...")


MODEL_MODULES = (inputs, physics, options)     # every module that reads the inputs


def run_with(changes, **kw):
    """Run the whole model with some inputs changed, then put the base values back.
    Each module keeps its own copy of the inputs, so the change is made in all of them."""
    saved = [(m, k, getattr(m, k)) for m in MODEL_MODULES for k in changes]
    for m in MODEL_MODULES:
        for k, v in changes.items():
            setattr(m, k, v)
    try:
        return run_all(raw, **kw)
    finally:
        for m, k, v in saved:
            setattr(m, k, v)


coil_catalogue = {"A_dc": A_dc_cases[1], "dTa_d": dTa_catalogue}
sens_cases = [("Main case", "-", {}, {})]
sens_cases += [("Chip water limit (C)", f"{T:g}", {}, {"T_s": T}) for T in T_s_cases]
sens_cases += [("Water allowance (L/kWh)", f"{w:.2f}", {"V_lim": w * E_IT / 1000}, {}) for w in WUE_lim_cases]
sens_cases += [("Dry cooler approach (K)", "3", {"A_dc": A_dc_cases[0]}, {}),
               ("Dry cooler approach (K)", "10", coil_catalogue, {})]
sens_cases += [("Chiller rated COP", f"{c:g}", {"eta_II": eta_II * c / COP_rated}, {}) for c in COP_rated_cases]
sens_cases += [("Chiller COP cap", "none" if c is None else f"{c:g}", {"COP_max": c}, {}) for c in COP_max_cases]
sens_cases += [("Pad effectiveness", f"{e:.2f}", {"eps_ad": e}, {}) for e in eps_ad_cases]
sens_cases += [("Tower evaporation share", f"{p:.2f}", {"phi_evap": p}, {}) for p in phi_cases]
sens_cases += [("Hotter year (K)", f"+{d:g}", {}, {"dT_clim": d}) for d in dT_clim_cases]
sens_cases += [("Drained water consumed", "r_B = 0", {"r_B": 0.0}, {})]
sens_cases += [("Combined bad case for B", "+3 K, 0.15 L/kWh, r_B = 0, 10 K coil",
                {"V_lim": WUE_lim_cases[0] * E_IT / 1000, "r_B": 0.0, **coil_catalogue}, {"dT_clim": dT_clim_cases[1]})]
sens_cases += [("Liquid capture share", f"{f:.2f}", {}, {"f": f}) for f in f_liq_cases]
sens_cases += [("CDU approach (K)", f"{a:g}", {"A_cdu": a}, {}) for a in A_cdu_cases]
# Thresholds: where Option B meets the baseline, and how Option B responds to the allowance
sens_cases += [("Threshold chip limit", f"{T:.1f}", {}, {"T_s": float(T)}) for T in np.arange(40.5, 45.0, 0.5)]
sens_cases += [("Threshold coil approach", f"{a:g}", {"A_dc": a}, {}) for a in (4.0, 5.0, 6.0)]
sens_cases += [("Threshold allowance", f"{w:.2f}", {"V_lim": w * E_IT / 1000}, {})
               for w in (0.05, 0.10, 0.20, 0.30, 0.50, 0.60, 0.80, 1.00)]

sens = []
def add_case(key, value, results, wx_case, ov_case, CI=CI_grid):
    for n, r in zip(names, results):
        i = indicators(r, wx_case, ov_case, CI)
        sens.append((key, value, n, i["E_cool"] / 1000, i["PUE"], i["CUE"], i["G_DC"], i["V_cons"], i["V_with"],
                     i["WUE_site"], i["chiller_h"], r.get("T_sw", np.nan), r.get("T_set", np.nan)))
for key, value, changes, kw in sens_cases:
    if key == "Main case":
        add_case(key, value, (base, optA, optB), wx, ov)
        for CI in CI_cases:                 # grid carbon intensity only changes emissions: no rerun needed
            add_case("Grid carbon intensity", f"{CI:.2f}", (base, optA, optB), wx, ov, CI)
        continue
    rr = run_with(changes, **kw)
    add_case(key, value, rr["res"], rr["wx"], rr["ov"])
sens = pd.DataFrame(sens, columns=["Input", "Value", "Option", "E_cool_MWh", "PUE", "CUE", "G_tCO2", "V_cons",
                                   "V_with", "WUE_site", "chiller_h", "T_sw", "T_set"])
# extra emissions compared with the baseline under the same case
sens["dG_tCO2"] = sens["G_tCO2"] - sens.groupby(["Input", "Value"], sort=False)["G_tCO2"].transform("first")
main_cases = sens[~sens["Input"].str.startswith("Threshold")]
with pd.option_context("display.width", 200, "display.max_rows", 200):
    print(main_cases.drop(columns=["V_with", "chiller_h"]).to_string(index=False, formatters={
        "E_cool_MWh": "{:,.1f}".format, "PUE": "{:.3f}".format, "CUE": "{:.3f}".format,
        "G_tCO2": "{:,.0f}".format, "V_cons": "{:,.0f}".format, "WUE_site": "{:.3f}".format,
        "T_sw": "{:.1f}".format, "T_set": "{:.1f}".format, "dG_tCO2": "{:+,.1f}".format}))

# 10b Thresholds at which the recommendation changes
E_s = sens.pivot_table(index=["Input", "Value"], columns="Option", values="E_cool_MWh", sort=False)
V_s = sens.pivot_table(index=["Input", "Value"], columns="Option", values="V_cons", sort=False)
B_minus_base = E_s["Option B"] - E_s["Baseline"]


def crossing(xs, ys):
    """x where a rising line through (xs, ys) first crosses zero (linear between the tested points)."""
    for k in range(len(xs) - 1):
        if (ys[k] - 0) * (ys[k + 1] - 0) <= 0:
            return xs[k] + (0 - ys[k]) * (xs[k + 1] - xs[k]) / (ys[k + 1] - ys[k])
    return np.nan


main_gap = ind[2]["E_cool"] / 1000 - ind[0]["E_cool"] / 1000       # Option B minus baseline, main case
chip_T = np.arange(40.5, 45.0, 0.5)
chip_x = [T_supply] + list(chip_T) + [T_s_cases[1]]
chip_y = ([main_gap] + [B_minus_base.loc[("Threshold chip limit", f"{T:.1f}")] for T in chip_T]
          + [B_minus_base.loc[("Chip water limit (C)", f"{T_s_cases[1]:g}")]])
coil_x = [A_dc_cases[0], 4.0, 5.0, 6.0, A_dc]
coil_y = ([B_minus_base.loc[("Dry cooler approach (K)", "3")]]
          + [B_minus_base.loc[("Threshold coil approach", f"{a:g}")] for a in (4.0, 5.0, 6.0)] + [main_gap])
allow_order = ["0.05", "0.10", "0.20", "0.30", "0.50", "0.60", "0.80", "1.00"]
allow = E_s.loc["Threshold allowance"].reindex(allow_order)
print("\n  10b Thresholds")
print(f"     Allowance at which the baseline becomes feasible:         {ind[0]['V_cons'] * 1000 / E_IT:.2f} L/kWh "
      f"({V_s.loc[('Drained water consumed', 'r_B = 0'), 'Baseline'] * 1000 / E_IT:.2f} with r_B = 0)")
print(f"     Chip water limit at which Option B matches the baseline: {crossing(chip_x, chip_y):.1f} °C")
print(f"     Dry cooler approach at which Option B matches baseline:  {crossing(coil_x, coil_y):.1f} K")
print(f"     Smallest saving of Option B over Option A (0.05 L/kWh):   "
      f"{allow.loc['0.05', 'Option A'] - allow.loc['0.05', 'Option B']:.1f} MWh/yr")
print(f"     Option B with plenty of water (1.00 L/kWh):               {allow.loc['1.00', 'Option B']:.1f} MWh/yr "
      f"(baseline {ind[0]['E_cool'] / 1000:.1f})")
with pd.option_context("display.width", 200):
    print("     Option B against the allowance (MWh/yr, m3/yr):")
    print(pd.DataFrame({"E_cool_B": allow["Option B"],
                        "V_cons_B": V_s.loc["Threshold allowance", "Option B"].reindex(allow_order)})
          .round(1).to_string())
