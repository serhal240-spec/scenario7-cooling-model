"""
Scenario 7 model: the three cooling options.
One function per option (baseline tower, Option A dry cooler, Option B hybrid), the Option B
settings search, the indicators (PUE, CUE, WUE) and run_all, which runs the whole model once.
"""

import numpy as np
import pandas as pd

from inputs import *
from physics import *


# =====================================================================
# MODEL FUNCTIONS: one per option, each returns hourly arrays and annual totals
# =====================================================================

def baseline_tower(wx, lp):
    C_w, V_w = lp["C_w"], lp["V_w"]
    P_P2 = V_w * dp_hx2 / eta_p                      # kW, extra facility pumping through HX-2 (m3/s x kPa = kW)
    P_P3 = V_w * dp_P3 / eta_p                       # kW, tower water pump (same flow both sides of HX-2)
    Q_fac = lp["Q_rack"] + P_P2                      # kW leaving the facility loop through HX-2
    Q_tw = Q_fac + P_P3                              # kW the tower must reject at full duty
    R = Q_tw / C_w                                   # K, tower range
    T_out_set = lp["T_fs"] - A_hx                    # °C, cold tower water needed
    T_in = T_out_set + R                             # °C, hot tower water (return is fixed while supply is held)

    # Size the tower: Merkel number at full fan that gives cold water at design wet bulb + approach.
    # Fixed size, so the sensitivity runs test the same tower, not a new one.
    p_d = wx["p"].mean()
    T_out_d = T_wb_d + A_tw
    Me_d = float(merkel(T_out_d + R, T_out_d, h_sat(T_wb_d, p_d), LG_d, p_d))

    # Fan speed each hour: smallest f where the tower's Merkel number covers the duty
    h_a, p = wx["h_a"], wx["p"]
    def surplus(f):
        return Me_d * f**n_tw - merkel(T_in, T_out_set, h_a, LG_d / f, p)
    f = bisect(surplus, np.full_like(h_a, f_min), np.ones_like(h_a))
    limited = surplus(np.ones_like(h_a)) < 0          # cannot hold the set point even at full fan

    # Hours at the limit: full fan, cold water floats up, trim chiller covers the rest
    T_out = np.full_like(h_a, T_out_set)
    if limited.any():
        hl, pl = h_a[limited], p[limited]
        T_out[limited] = bisect(lambda T: Me_d - merkel(T_in, T, hl, LG_d, pl),
                                np.full_like(hl, T_out_set), np.full_like(hl, T_in - 0.01))
    Q_tower = C_w * (T_in - T_out)                   # kW actually rejected by the tower
    Q_ch = Q_tw - Q_tower                            # kW left for the trim chiller
    W_ch, COP, COP_c = trim_chiller(Q_ch, lp["T_fs"], wx["T_db"], COP_max=COP_max)

    P_fan = SFP_tw * Q_tw * f**3                     # kW, fan law: power goes with speed cubed
    P_pumps = np.full_like(h_a, lp["P_loop"] + P_P2 + P_P3)

    # Water: evaporation and drift are consumed, blowdown is drained (withdrawn, not consumed)
    m_evap = phi_evap * Q_tower / h_fg               # kg/s
    m_drift = np.full_like(h_a, drift_frac * lp["m_w"])
    m_blow = np.maximum(m_evap / (COC - 1) - m_drift, 0)

    return {
        "name": "Baseline", "P_P2": P_P2, "P_P3": P_P3, "Q_fac": Q_fac, "Q_unit_full": Q_tw,
        "T_in": T_in, "T_out_set": T_out_set, "Me_d": Me_d,
        "f": f, "T_out": T_out, "Q_unit": Q_tower, "Q_ch": Q_ch, "W_ch": W_ch, "COP": COP, "COP_carnot": COP_c,
        "P_fan": P_fan, "P_pumps": P_pumps, "P_spray": np.zeros_like(h_a),
        "P_cool": P_pumps + P_fan + W_ch,
        "m_evap": m_evap, "m_drift": m_drift, "m_blow": m_blow,
        "m_cons": m_evap + m_drift + (1 - r_B) * m_blow, "m_with": m_evap + m_drift + m_blow,
    }


def dry_cooler(wx, lp, T_ai, name):
    """Dry cooler + trim chiller for a given coil inlet air temperature (Option A: dry bulb, Option B: after spray)."""
    C_w, V_w = lp["C_w"], lp["V_w"]
    P_P2 = V_w * dp_coil / eta_p                     # kW, extra facility pumping through the coil
    Q_fac = lp["Q_rack"] + P_P2                      # kW to reject
    T_in = lp["T_fs"] + Q_fac / C_w                  # °C, hot water into the coil (fixed while supply is held)

    # Size the coil: full fan, full load, air at T_fs - A_dc gives water out at T_fs
    C_a_d = Q_fac / dTa_d                            # kW/K, air capacity rate at full fan
    T_ai_d = lp["T_fs"] - A_dc
    Cmin_d, Cmax_d = min(C_w, C_a_d), max(C_w, C_a_d)
    eps_d = Q_fac / (Cmin_d * (T_in - T_ai_d))
    NTU_d = float(bisect(lambda N: eps_crossflow(N, Cmin_d / Cmax_d) - eps_d, 0.01, 20.0))
    UA_d = NTU_d * Cmin_d                            # kW/K

    def Q_coil(f):
        C_a = f * C_a_d
        UA = UA_d * f**n_dc
        Cmin, Cmax = np.minimum(C_w, C_a), np.maximum(C_w, C_a)
        eps = eps_crossflow(UA / Cmin, Cmin / Cmax)
        return np.maximum(eps * Cmin * (T_in - T_ai), 0)

    f = bisect(lambda f: Q_coil(f) - Q_fac, np.full_like(T_ai, f_min), np.ones_like(T_ai))
    Q_dc = np.minimum(Q_coil(f), Q_fac)              # above the load at f_min the bypass holds the supply
    Q_ch = Q_fac - Q_dc
    W_ch, COP, COP_c = trim_chiller(Q_ch, lp["T_fs"], wx["T_db"], COP_max=COP_max)

    m_air = f * C_a_d / cp_a                         # kg/s of air through the coil
    T_ao = T_ai + Q_dc / (f * C_a_d)                 # °C, air leaving the coil
    T_w_out = T_in - Q_dc / C_w                      # °C, water leaving the coil
    P_fan = SFP_dc * Q_fac * f**3
    P_pumps = np.full_like(T_ai, lp["P_loop"] + P_P2)

    return {
        "name": name, "P_P2": P_P2, "Q_fac": Q_fac, "Q_unit_full": Q_fac, "T_in": T_in,
        "C_a_d": C_a_d, "m_air_d": C_a_d / cp_a, "T_ai_d": T_ai_d, "eps_d": eps_d, "NTU_d": NTU_d, "UA_d": UA_d,
        "f": f, "m_air": m_air, "T_ai": T_ai, "T_ao": T_ao, "T_out": T_w_out,
        "Q_unit": Q_dc, "Q_ch": Q_ch, "W_ch": W_ch, "COP": COP, "COP_carnot": COP_c,
        "P_fan": P_fan, "P_pumps": P_pumps,
    }


def option_a(wx, lp):
    r = dry_cooler(wx, lp, wx["T_db"], "Option A")
    zero = np.zeros_like(wx["T_db"])
    r.update({"P_spray": zero, "P_cool": r["P_pumps"] + r["P_fan"] + r["W_ch"],
              "m_evap": zero, "m_drift": zero, "m_blow": zero, "m_cons": zero, "m_with": zero})
    return r


def option_b(wx, lp, T_sw, T_set=None):
    """Spray on when the dry bulb is above T_sw, cooling the coil air down to T_set (T_set = T_sw if not given)."""
    if T_set is None:
        T_set = T_sw
    T_db, T_wb = wx["T_db"], wx["T_wb"]
    T_lowest = T_db - eps_ad * (T_db - T_wb)         # coolest air the spray can give
    T_ai = np.where(T_db > T_sw, np.maximum(T_set, T_lowest), T_db)  # spray only as much as needed to reach T_set
    spraying = T_ai < T_db - 1e-9
    r = dry_cooler(wx, lp, T_ai, "Option B")
    m_evap = r["m_air"] * cp_a * (T_db - T_ai) / h_fg    # kg/s, adiabatic: sensible drop = latent gain
    P_spray = np.where(spraying, P_sp, 0.0)
    r.update({"T_sw": T_sw, "T_set": T_set, "spraying": spraying, "P_spray": P_spray,
              "P_cool": r["P_pumps"] + r["P_fan"] + r["W_ch"] + P_spray,
              "m_evap": m_evap, "m_drift": np.zeros_like(T_db),
              "m_blow": m_evap / eta_sp - m_evap,      # drained spray: withdrawn, not consumed (plays the blowdown role)
              "m_cons": m_evap + (1 - r_B) * (m_evap / eta_sp - m_evap), "m_with": m_evap / eta_sp})
    return r


def pareto(df):
    """Lower envelope of electricity against water: the settings no other setting beats on both."""
    d = df.sort_values(["V_cons", "E_cool"])
    return d[d["E_cool"] < d["E_cool"].cummin().shift(fill_value=np.inf)]


def option_b_best(wx, lp):
    """Sweep every (T_sw, T_set) pair, keep the lowest electricity that meets the water limit.
    Also returns the one setting sweep (T_set = T_sw, spray only down to the switch on temperature) for comparison.
    Shortcut: in any pair, an hour is either dry (T_db <= T_sw, the coil sees the outdoor air) or sprayed
    down to max(T_set, coolest the pads allow). So each hour is solved once dry and once for each target
    temperature, and every pair is assembled from those hourly results. This gives exactly the same numbers
    as solving option_b for all 520 pairs, about eight times faster."""
    T_db = wx["T_db"]
    dry = option_b(wx, lp, np.inf)                                  # never sprays
    wet = {}
    for T in np.union1d(T_sw_options, T_set_options):              # every target temperature used below
        r = option_b(wx, lp, -np.inf, T)                            # sprays in every hour (only hot hours are used)
        wet[T] = (r["P_cool"], r["m_cons"])

    def pair(T_sw, T_set):
        on = T_db > T_sw                                            # hours with the pads wet
        P = np.where(on, wet[T_set][0], dry["P_cool"])
        m = np.where(on, wet[T_set][1], dry["m_cons"])
        return annual(P, wx), to_m3(m, wx)

    sweep, grid = [], []
    for T_sw in T_sw_options:
        sweep.append((T_sw, *pair(T_sw, T_sw)))
        for T_set in T_set_options[T_set_options < T_sw]:
            grid.append((T_sw, T_set, *pair(T_sw, T_set)))
    sweep = pd.DataFrame(sweep, columns=["T_sw", "E_cool", "V_cons"])
    grid = pd.DataFrame(grid + [(t, t, e, v) for t, e, v in sweep.itertuples(index=False)],
                        columns=["T_sw", "T_set", "E_cool", "V_cons"])
    ok = grid[grid["V_cons"] <= V_lim]
    best = ok.loc[ok["E_cool"].idxmin()]
    return option_b(wx, lp, best["T_sw"], best["T_set"]), sweep, grid


def indicators(r, wx, ov, CI=CI_grid):
    E_cool = annual(r["P_cool"], wx)
    E_other = annual(ov["P_other"], wx)              # UPS + lighting + room air system, built bottom up
    E_DC = E_IT + E_other + E_cool
    V_cons = to_m3(r["m_cons"], wx)
    V_with = to_m3(r["m_with"], wx)
    return {
        "E_cool": E_cool, "E_other": E_other, "E_DC": E_DC,
        "PUE": E_DC / E_IT,
        "EUE_IT": E_IT / E_DC,                       # Q_reuse = 0 (no heat user in Scenario 7)
        "CUE": CI * E_DC / E_IT,                     # kg CO2e per kWh IT, purchased electricity only
        "G_DC": CI * E_DC / 1000,                    # t CO2e/yr
        "V_cons": V_cons, "V_with": V_with,
        "WUE_site": V_cons * 1000 / E_IT,            # L/kWh IT (consumption boundary)
        "WUE_source": (V_cons * 1000 + EWIF * E_DC) / E_IT,
        "V_source": EWIF * E_DC / 1000,              # m3/yr at power stations
        "chiller_h": float(np.sum(r["Q_ch"] > 0.01) * wx["h_weight"]),
        "chiller_peak": float(r["Q_ch"].max()),
        "P_cool_peak": float(r["P_cool"].max()),
    }


def run_all(raw, T_s=T_supply, dT_clim=0.0, f=f_liq):
    """Whole model for one set of inputs (chip water limit, climate shift, liquid share)."""
    wx = make_weather(raw, dT_clim)
    lp = loop_design(T_s, f)
    base = baseline_tower(wx, lp)
    a = option_a(wx, lp)
    b, sweep, grid = option_b_best(wx, lp)
    ov = overhead(wx, lp)
    return {"wx": wx, "lp": lp, "res": [base, a, b], "sweep": sweep, "grid": grid, "ov": ov}
