"""
Scenario 7 model: figures.
Draws report Figures 3 to 13 and four extra figures from the results in analysis.py.
Run main.py to see them; running this file directly also works.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis import *


# =====================================================================
# FIGURES (shown on screen, not saved)
# Figures 3 to 13 are the report figures, in report order (window titles give the figure numbers).
# Figures 1 and 2 are diagrams with no model results, so they are not drawn here.
# After them come extra figures that are not in the report (hourly detail, a comparison table).
# =====================================================================

OPT = {"Baseline": "#2a78d6", "Option A": "#eb6834", "Option B": "#1baf7a"}
COL = OPT
PART = {"Pumps": "#008300", "Fans": "#6250d6", "Spray pump": "#e87ba4", "Trim chiller": "#eda100"}
plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 200,
})
hours = wx["hour"]
month = wx["month"]
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS = month_names
month_starts = [hours[month == m][0] for m in range(1, 13)]
x12 = np.arange(12)
res = [base, optA, optB]
labels = ["Baseline\n(evaporative tower)", "Option A\n(dry cooler)", "Option B\n(hybrid)"]
REPORT_FIGS = {}                      # report figures, by file name


def month_axis(ax):
    """Label an hourly x axis with month names."""
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_names)
    ax.set_xlim(hours[0] - 0.5, hours[-1] + 0.5)
    ax.grid(axis="x", visible=False)


def monthly(x, kind="kwh"):
    """Monthly totals: MWh from hourly kW, or m3 from hourly kg/s."""
    s = pd.Series(x).groupby(month).sum().to_numpy()
    return s * wx["h_weight"] / 1000 if kind == "kwh" else s * 3600 * wx["h_weight"] / 1000


def monthly_mwh(p_kw):
    return monthly(p_kw, "kwh")


def monthly_m3(m_kgs):
    return monthly(m_kgs, "m3")


def hourly_columns(ax, parts, title):
    """One column per hour, stacked by component."""
    bottom = np.zeros(len(hours))
    edges = np.append(hours - 0.5, hours[-1] + 0.5)
    for label, values in parts:
        if np.max(values) > 0:
            # filled steps, one flat top per hour: looks the same as 8,760 touching bars but draws much faster
            ax.stairs(bottom + values, edges, baseline=bottom, fill=True, color=PART[label], label=label)
            bottom = bottom + values
    ax.set_ylim(0, bottom.max() * 1.15)
    ax.set_ylabel("Cooling electricity (kW)")
    ax.set_title(title)
    ax.legend(ncol=4, loc="upper left")
    month_axis(ax)


# ---------------------------------------------------------------------
# Report Figure 3: weather
# ---------------------------------------------------------------------
db = pd.Series(wx["T_db"]).groupby(month)
wb = pd.Series(wx["T_wb"]).groupby(month)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), num="Figure 3 Weather")
a1.plot(x12, db.mean(), "-o", color="#d35400", label="Mean dry bulb")
a1.plot(x12, db.max(), "--o", color="#d35400", alpha=0.55, ms=4, label="Maximum dry bulb")
a1.plot(x12, wb.mean(), "-s", color="#2471a3", label="Mean wet bulb")
a1.plot(x12, wb.max(), "--s", color="#2471a3", alpha=0.55, ms=4, label="Maximum wet bulb")
a1.set_xticks(x12, MONTHS)
a1.set_ylabel("Temperature (°C)")
a1.set_title("(a) Monthly air temperatures")
a1.legend(fontsize=8, ncol=2, loc="upper left")
a1.set_ylim(0, 58)
h28 = pd.Series(wx["T_db"] > 28.0).groupby(month).sum()
hspray = pd.Series(optB["spraying"]).groupby(month).sum()
a2.bar(x12 - 0.2, h28, 0.4, color="#eb6834", label="Dry bulb above 28 °C (coil needs help)")
a2.bar(x12 + 0.2, hspray, 0.4, color="#1baf7a", label="Option B spray on (above %.1f °C)" % optB["T_sw"])
a2.set_xticks(x12, MONTHS)
a2.set_ylabel("Hours per month")
a2.set_title("(b) Hours that drive the dry and hybrid results")
a2.legend(fontsize=8, loc="upper left")
a2.set_ylim(0, 820)
fig.suptitle("Phoenix Sky Harbor TMYx 2011 to 2025 weather year (8,760 h)", weight="bold", fontsize=11)
fig.tight_layout()
REPORT_FIGS["fig03_weather"] = fig

# ---------------------------------------------------------------------
# Report Figure 4: annual cooling electricity by component
# ---------------------------------------------------------------------
shared = lp["P_loop"] * 8760 / 1000          # MWh/yr, rack + facility loop pumps, same in every option
parts = {
    "Shared loop pumps (P1, P2)": [shared] * 3,
    "Option pumps (HX-2, P3, coil)": [annual(r["P_pumps"], wx) / 1000 - shared for r in res],
    "Fans": [annual(r["P_fan"], wx) / 1000 for r in res],
    "Spray pump": [annual(r["P_spray"], wx) / 1000 for r in res],
    "Trim chiller": [annual(r["W_ch"], wx) / 1000 for r in res],
}
pcol = ["#b3b6b7", "#27ae60", "#6250d6", "#e87ba4", "#eda100"]
fig, ax = plt.subplots(figsize=(8, 4.6), num="Figure 4 Cooling electricity by component")
bottom = np.zeros(3)
for (k, v), c in zip(parts.items(), pcol):
    ax.bar(labels, v, 0.55, bottom=bottom, color=c, label=k, edgecolor="white", lw=0.6)
    bottom += np.array(v)
for i, t in enumerate(bottom):
    ax.text(i, t + 4, "%.1f MWh" % t, ha="center", fontsize=9, weight="bold")
ax.set_ylabel("Cooling electricity (MWh/yr)")
ax.set_ylim(0, bottom.max() * 1.18)
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(axis="x", visible=False)
ax.set_title("Annual cooling electricity by component (shared facility overhead of %.0f MWh not shown)"
             % (annual(ov["P_other"], wx) / 1000), fontsize=10)
REPORT_FIGS["fig04_cooling_components"] = fig

# ---------------------------------------------------------------------
# Report Figure 5: monthly cooling electricity
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.8), num="Figure 5 Monthly cooling electricity")
for i, (n, r) in enumerate(zip(names, res)):
    ax.bar(x12 + (i - 1) * 0.27, monthly_mwh(r["P_cool"]), 0.27, color=COL[n], label=n)
ax.set_xticks(x12, MONTHS)
ax.set_ylabel("Cooling electricity (MWh/month)")
ax.legend(fontsize=9, loc="upper left")
ax.grid(axis="x", visible=False)
ax.set_title("Monthly cooling electricity: the options separate only in the hot months")
REPORT_FIGS["fig05_monthly_electricity"] = fig

# ---------------------------------------------------------------------
# Report Figure 6: trim chiller duty duration curve
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 3.9), num="Figure 6 Chiller duration")
hrs = np.arange(1, len(hours) + 1)
for n, r in zip(names, res):
    ax.plot(hrs, np.sort(r["Q_ch"])[::-1], color=COL[n], lw=2, label="%s (%s h, peak %.0f kW)"
            % (n, f"{int(np.sum(r['Q_ch'] > 0.01)):,}", r["Q_ch"].max()))
ax.set_xlim(0, 4000)
ax.set_xlabel("Hours of the year, sorted from largest to smallest chiller duty")
ax.set_ylabel("Trim chiller duty (kW)")
ax.legend(fontsize=9)
ax.set_title("Trim chiller duty duration curves (hours beyond 4,000 have zero duty)")
REPORT_FIGS["fig06_chiller_duration"] = fig

# ---------------------------------------------------------------------
# Report Figure 7: trim chiller duty at critical hours
# ---------------------------------------------------------------------
i_mild = int(np.argmin(np.abs(wx["T_db"] - 20.0) + np.abs(wx["hour"] - 2200) / 1e4))
i_dry = int(np.argmax(wx["T_db"]))
i_wet = int(np.argmax(wx["T_wb"]))
conds = [("Mild\n%.1f / %.1f °C" % (wx["T_db"][i_mild], wx["T_wb"][i_mild]), [r["Q_ch"][i_mild] for r in res]),
         ("Hottest hour\n%.1f / %.1f °C" % (wx["T_db"][i_dry], wx["T_wb"][i_dry]), [r["Q_ch"][i_dry] for r in res]),
         ("Most humid hour\n%.1f / %.1f °C" % (wx["T_db"][i_wet], wx["T_wb"][i_wet]), [r["Q_ch"][i_wet] for r in res]),
         ("ASHRAE design\n%.1f / %.1f °C" % (T_db_des, T_wb_des), [d["Q_ch"][0] for d in des])]
fig, ax = plt.subplots(figsize=(8.5, 4), num="Figure 7 Critical hours")
xc = np.arange(len(conds))
for j, n in enumerate(names):
    vals = [c[1][j] for c in conds]
    ax.bar(xc + (j - 1) * 0.26, vals, 0.26, color=COL[n], label=n)
    for k, v in enumerate(vals):
        ax.text(xc[k] + (j - 1) * 0.26, v + 8, "%.0f" % v, ha="center", fontsize=7.5)
ax.set_xticks(xc, [c[0] for c in conds], fontsize=8.5)
ax.set_ylabel("Trim chiller duty (kW)")
ax.set_ylim(0, 620)
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="x", visible=False)
ax.set_title("Trim chiller duty at critical hours (dry bulb / wet bulb)")
REPORT_FIGS["fig11_critical_hours"] = fig   # report Figure 7

# ---------------------------------------------------------------------
# Report Figure 8: monthly site water
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4), num="Figure 8 Monthly site water")
bw = 0.38
ax.bar(x12 - bw / 2, monthly_m3(base["m_evap"]), bw, color="#2471a3", label="Baseline: evaporation")
ax.bar(x12 - bw / 2, monthly_m3(base["m_blow"]), bw, bottom=monthly_m3(base["m_evap"]), color="#a9cce3",
       label="Baseline: blowdown (returned)")
ax.bar(x12 + bw / 2, monthly_m3(optB["m_evap"]), bw, color="#1baf7a", label="Option B: evaporated spray")
ax.bar(x12 + bw / 2, monthly_m3(optB["m_blow"]), bw, bottom=monthly_m3(optB["m_evap"]), color="#a3e4d7",
       label="Option B: spray drain (returned)")
ax.set_xticks(x12, MONTHS)
ax.set_ylabel("Site water withdrawn (m³/month)")
ax.legend(fontsize=8.5, ncol=2, loc="upper left")
ax.set_ylim(0, monthly_m3(base["m_with"]).max() * 1.35)
ax.grid(axis="x", visible=False)
ax.set_title("Monthly site water (drift, %.0f m³/yr, too small to show; Option A uses none)"
             % to_m3(base["m_drift"], wx), fontsize=10.5)
REPORT_FIGS["fig07_monthly_water"] = fig   # report Figure 8

# ---------------------------------------------------------------------
# Report Figure 9: annual water, site and power station
# ---------------------------------------------------------------------
cons = [i["V_cons"] for i in ind]
ret = [i["V_with"] - i["V_cons"] for i in ind]
src = [i["V_source"] for i in ind]
fig, ax = plt.subplots(figsize=(8, 4.4), num="Figure 9 Annual water")
ax.bar(labels, cons, 0.55, color="#2471a3", label="Site water consumed (evaporation, drift)")
ax.bar(labels, ret, 0.55, bottom=cons, color="#a9cce3", label="Site water withdrawn and returned (drain)")
ax.bar(labels, src, 0.55, bottom=np.array(cons) + np.array(ret), color="#d5d8dc",
       label=f"Power station water ({EWIF} L/kWh)")
for i in range(3):
    ax.text(i, cons[i] + ret[i] + src[i] + 300, "%s m³" % f"{cons[i] + src[i]:,.0f}", ha="center", fontsize=9,
            weight="bold")
ax.axhline(V_lim, color="#555555", ls="--", lw=1)
ax.text(0.5, V_lim + 300, "site limit\n%s m³" % f"{V_lim:,.0f}", fontsize=8, color="#555555", ha="center")
ax.set_ylabel("Water (m³/yr)")
ax.set_ylim(0, max(np.array(cons) + np.array(ret) + np.array(src)) * 1.15)
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(axis="x", visible=False)
ax.set_title("Annual water: site and power station (labels: total consumed)", fontsize=10.5)
REPORT_FIGS["fig08_annual_water"] = fig   # report Figure 9

# ---------------------------------------------------------------------
# Report Figure 10: heat flows
# ---------------------------------------------------------------------
lat = to_m3(base["m_evap"], wx) * h_fg / 3600            # MWh: m3 x 1000 kg x kJ/kg / 3.6e6 kJ per MWh
outdoor = [annual(r["Q_unit"] + r["P_fan"] + r["P_spray"], wx) / 1000 for r in res]
cond = [annual(r["Q_ch"] + r["W_ch"], wx) / 1000 for r in res]
room = annual(ov["Q_room"] + ov["P_room"], wx) / 1000
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.25]},
                             num="Figure 10 Heat flows")
lat_l = [lat, 0, 0]
sens_l = [outdoor[0] - lat, outdoor[1], outdoor[2]]
a1.bar(labels, lat_l, 0.55, color="#2471a3", label="Tower, latent (evaporation)")
a1.bar(labels, sens_l, 0.55, bottom=lat_l, color="#85c1e9", label="Outdoor unit, sensible (incl. fan heat)")
a1.bar(labels, cond, 0.55, bottom=np.array(outdoor), color="#eda100", label="Trim chiller condenser")
a1.bar(labels, [room] * 3, 0.55, bottom=np.array(outdoor) + np.array(cond), color="#b3b6b7",
       label="Room air system condenser")
for i in range(3):
    a1.text(i, outdoor[i] + cond[i] + room + 150, "%s" % f"{outdoor[i] + cond[i] + room:,.0f}", ha="center",
            fontsize=8.5, weight="bold")
a1.set_ylabel("Heat rejected to atmosphere (MWh/yr)")
a1.set_ylim(0, 16500)
a1.set_yticks(np.arange(0, 12001, 2000))      # no tick labels behind the legend
a1.legend(fontsize=7.5, loc="upper left")
a1.tick_params(axis="x", labelsize=8)
a1.grid(axis="x", visible=False)
a1.set_title("(a) Annual heat rejected, by route")
for n, r in zip(names, res):
    a2.plot(x12, monthly_mwh(r["Q_ch"]), "-o", color=COL[n], label=n)
a2.set_xticks(x12, MONTHS)
a2.set_ylabel("Liquid heat taken by trim chiller (MWh/month)")
a2.legend(fontsize=9)
a2.set_title("(b) Seasonal heat moved by refrigeration")
fig.suptitle("Annual and seasonal heat flows (no heat is reused in any option)", weight="bold", fontsize=11)
fig.tight_layout()
REPORT_FIGS["fig09_heat_flows"] = fig   # report Figure 10

# ---------------------------------------------------------------------
# Report Figure 11: energy vs water trade off
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.8), num="Figure 11 Trade off")
front = pareto(grid)
ax.plot(sweep["V_cons"], sweep["E_cool"] / 1000, color="#aaaaaa", lw=1.2, ls="--", zorder=2,
        label="One setting: spray down to T$_{sw}$ only")
sc = ax.scatter(front["V_cons"], front["E_cool"] / 1000, c=front["T_sw"], cmap="viridis", s=22, zorder=3)
ax.plot(front["V_cons"], front["E_cool"] / 1000, color="#666666", lw=0.8, zorder=2,
        label="Two settings: best (T$_{sw}$, T$_{set}$) for each water use")
ax.legend(fontsize=8, loc="upper right")
cb = fig.colorbar(sc, ax=ax, pad=0.01)
cb.set_label("Spray switch on temperature T$_{sw}$ (°C)")
ax.axvline(V_lim, color="#555555", ls="--", lw=1)
ax.text(V_lim + 80, 75, "site limit\n%s m³/yr" % f"{V_lim:,.0f}", fontsize=8, color="#555555", va="bottom")
ax.plot(ind[2]["V_cons"], ind[2]["E_cool"] / 1000, "o", ms=13, mfc="none", mec=COL["Option B"], mew=2.2, zorder=4)
ax.annotate("Option B selected\non above %.1f °C, to %.0f °C" % (optB["T_sw"], optB["T_set"]),
            (ind[2]["V_cons"], ind[2]["E_cool"] / 1000),
            (ind[2]["V_cons"] + 900, ind[2]["E_cool"] / 1000 + 35), fontsize=8.5, color=COL["Option B"],
            arrowprops=dict(arrowstyle="-", color=COL["Option B"]))
ax.plot(0, ind[1]["E_cool"] / 1000, "s", ms=9, color=COL["Option A"], zorder=4)
ax.text(150, ind[1]["E_cool"] / 1000 + 6, "Option A (= no spray)", fontsize=8.5, color=COL["Option A"])
ax.plot(ind[0]["V_cons"], ind[0]["E_cool"] / 1000, "D", ms=9, color=COL["Baseline"], zorder=4)
ax.text(ind[0]["V_cons"] - 150, ind[0]["E_cool"] / 1000 + 9, "Baseline tower", fontsize=8.5, color=COL["Baseline"],
        ha="right")
ax.set_xlabel("Site water consumed (m³/yr)")
ax.set_ylabel("Cooling electricity (MWh/yr)")
ax.set_xlim(-300, 9800)
ax.set_ylim(50, 260)
ax.set_title("Electricity against site water for the Option B spray settings")
REPORT_FIGS["fig10_tradeoff"] = fig   # report Figure 11

# ---------------------------------------------------------------------
# Report Figure 12: sensitivity of the recommendation (tornado charts from STEP 10)
# ---------------------------------------------------------------------
gapA = E_s["Option A"] - E_s["Option B"]     # Option B's saving over Option A
gapB = E_s["Option B"] - E_s["Baseline"]     # Option B's extra over the baseline
LOW, HIGH, COMB = "#2a78d6", "#eb6834", "#7a7a7a"
# (row label, input, low value, low label, high value, high label); None = tested one way only
tornado_rows = [
    ("Chip water limit", "Chip water limit (C)", "35", "35 °C", "45", "45 °C"),
    ("Dry cooler approach", "Dry cooler approach (K)", "3", "3 K", "10", "10 K"),
    ("Water allowance", "Water allowance (L/kWh)", "0.15", "0.15", "0.40", "0.40 L/kWh"),
    ("Hotter year", "Hotter year (K)", None, None, "+3", "+3 K"),
    ("Drained water consumed", "Drained water consumed", None, None, "r_B = 0", "r$_B$ = 0"),
    ("Chiller COP cap", "Chiller COP cap", "5", "5", "none", "none"),
    ("Chiller rated COP", "Chiller rated COP", "2.2", "2.2", "3.4", "3.4"),
    ("Pad effectiveness", "Pad effectiveness", "0.65", "0.65", "0.85", "0.85"),
]
comb_key = ("Combined bad case for B", "+3 K, 0.15 L/kWh, r_B = 0, 10 K coil")
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True, num="Figure 12 Sensitivity")
t_labels = [r[0] for r in tornado_rows] + ["Combined bad case for B"]
yy = list(range(len(t_labels)))[::-1]
for ax, gap, title, xlabel in [
        (axes[0], gapA, "(a) Option B's saving over Option A",
         "Option A minus Option B (MWh/yr)\nleft of the dashed line, Option A would use less"),
        (axes[1], gapB, "(b) Option B's extra over the baseline",
         "Option B minus baseline (MWh/yr)\nleft of the dashed line, Option B uses less than the baseline")]:
    mid = gap[("Main case", "-")]
    for yi, (lab, key, lo, lo_lab, hi, hi_lab) in zip(yy, tornado_rows):
        for val, txt, c in ((lo, lo_lab, LOW), (hi, hi_lab, HIGH)):
            if val is None:
                continue
            v = gap[(key, val)]
            ax.barh(yi, v - mid, left=mid, color=c, height=0.6, alpha=0.9)
            ax.text(v + (3 if v >= mid else -3), yi, txt, va="center", fontsize=8,
                    ha="left" if v >= mid else "right")
    v = gap[comb_key]
    ax.barh(yy[-1], v - mid, left=mid, color=COMB, height=0.6, hatch="//", edgecolor="white")
    ax.text(v + (3 if v >= mid else -3), yy[-1], f"{v:.0f}", va="center", fontsize=8,
            ha="left" if v >= mid else "right")
    ax.axvline(mid, color="black", lw=1)
    ax.text(mid, len(t_labels) - 0.35, f"main case {mid:.0f}", ha="center", fontsize=8)
    ax.axvline(0, color="#b00020", lw=1, ls="--")
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(yy)
axes[0].set_yticklabels(t_labels)
axes[0].set_xlim(-15, 200)
axes[1].set_xlim(-60, 290)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=LOW, label="lower input value"), Patch(color=HIGH, label="higher input value"),
                    Patch(facecolor=COMB, hatch="//", edgecolor="white",
                          label="hot year + 0.15 L/kWh + r$_B$ = 0 + 10 K coil")],
           loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.09))
fig.suptitle("Sensitivity of the recommendation, one input at a time (Option B re-selects its spray settings in every case)",
             fontsize=11, weight="bold")
fig.tight_layout()
REPORT_FIGS["fig12_sensitivity"] = fig

# ---------------------------------------------------------------------
# Report Figure 13: sensitivity to grid carbon intensity
# ---------------------------------------------------------------------
CI_range = np.linspace(0, 0.6, 61)                  # kg CO2e/kWh, zero carbon grid to a coal heavy grid
fig, ax = plt.subplots(figsize=(8, 4.4), num="Figure 13 Carbon intensity")
for n, i in zip(names, ind):
    ax.plot(CI_range, CI_range * (i["E_DC"] - ind[0]["E_DC"]) / 1000, color=COL[n], label=n)
ax.axvline(CI_grid, color="grey", ls="--", lw=1)
ax.text(CI_grid + 0.008, 85, "Arizona grid\n(AZNM %.2f)" % CI_grid, fontsize=8.5)
ax.set_xlabel("Grid carbon intensity (kg CO$_2$e/kWh)")
ax.set_ylabel("Extra emissions vs baseline (t CO$_2$e/yr)")
ax.legend(fontsize=9, loc="upper left")
ax.set_title("Extra emissions against grid carbon intensity\n(negative of the avoided emissions; the ranking does not change)",
             fontsize=10)
fig.tight_layout()
REPORT_FIGS["fig13_carbon_intensity"] = fig

# =====================================================================
# EXTRA FIGURES (not in the report)
# =====================================================================

# Extra 1: baseline hourly cooling electricity
fig, ax = plt.subplots(figsize=(11, 4.5), num="Extra 1 Baseline hourly")
hourly_columns(ax, [("Pumps", base["P_pumps"]), ("Fans", base["P_fan"]), ("Trim chiller", base["W_ch"])],
               "Extra 1. Baseline (cooling tower): cooling electricity every hour")
fig.tight_layout()

# Extra 2: Option A hourly cooling electricity
fig, ax = plt.subplots(figsize=(11, 4.5), num="Extra 2 Option A hourly")
hourly_columns(ax, [("Pumps", optA["P_pumps"]), ("Fans", optA["P_fan"]), ("Trim chiller", optA["W_ch"])],
               "Extra 2. Option A (dry cooler + trim chiller): cooling electricity every hour")
fig.tight_layout()

# Extra 3: hot day vs mild day, hour by hour
day_max = pd.Series(wx["T_db"]).groupby(wx["day"]).max()
day_mean = pd.Series(wx["T_db"]).groupby(wx["day"]).mean()
hot_day = int(day_max.idxmax())
mild_day = int((day_mean - wx["T_db"].mean()).abs().idxmin())
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, num="Extra 3 Hot and mild day")
for ax, d, label in zip(axes, (hot_day, mild_day), ("Hottest day", "Mild day")):
    sel = wx["day"] == d
    for k, (n, r) in enumerate(zip(names, (base, optA, optB))):
        ax.bar(np.arange(24) + (k - 1) * 0.28, r["P_cool"][sel], width=0.28, color=OPT[n], label=n)
    m_, d_ = wx["month"][sel][0], raw["Day"].to_numpy()[sel][0]
    ax.set_title(f"{label}: {d_} {month_names[m_ - 1]}, {wx['T_db'][sel].min():.0f} to {wx['T_db'][sel].max():.0f} °C")
    ax.set_xlabel("Hour of the day")
    ax.set_xticks(range(0, 24, 3))
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Cooling electricity (kW)")
axes[0].legend(loc="upper left")
fig.suptitle("Extra 3. Cooling electricity hour by hour on a hot and a mild day")
fig.tight_layout()

# Extra 4: comparison table
fig, ax = plt.subplots(figsize=(11, 6), num="Extra 4 Comparison table")
ax.axis("off")
cell_text = [[fmt.format(v) for v in vals] for _, vals, fmt in table_rows]
row_labels = [r[0].replace("m3", "m³") for r in table_rows]
tbl = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_names,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 1.45)
for (r_, c_), cell in tbl.get_celld().items():
    cell.set_edgecolor("0.85")
    if r_ == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#eef2f7")
ax.set_title(f"Extra 4. Baseline vs options, annual (1 MW IT, {H_op:,} h, CI {CI_grid} kg/kWh, EWIF {EWIF} L/kWh)",
             pad=12)
fig.tight_layout()


if __name__ == "__main__":
    plt.show()
