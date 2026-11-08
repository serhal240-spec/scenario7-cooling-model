"""
Scenario 7 model: helper functions.
Moist air properties, tower (Merkel) and crossflow coil formulas, the chiller COP model,
weather arrays, the common loop design and the non cooling overhead.
"""

import numpy as np

from inputs import *


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def p_ws(T):
    """Saturation pressure of water vapour (Pa), T in °C.
    Hyland and Wexler (1983), ASHRAE Handbook Fundamentals ch.1, same as the weather converter."""
    Tk = np.asarray(T, dtype=float) + KELVIN
    ln_p = (-5.8002206e3 / Tk + 1.3914993 - 4.8640239e-2 * Tk + 4.1764768e-5 * Tk**2
            - 1.4452093e-8 * Tk**3 + 6.5459673 * np.log(Tk))
    return np.exp(ln_p)


def W_sat(T, p):
    """Humidity ratio of saturated air (kg water / kg dry air)."""
    pw = p_ws(T)
    return 0.621945 * pw / (p - pw)


def h_air(T, W):
    """Enthalpy of moist air, kJ per kg dry air (ASHRAE Fundamentals ch.1)."""
    return cp_a * T + W * (2501 + 1.86 * T)


def h_sat(T, p):
    """Enthalpy of saturated air at water temperature T (the air film touching the water)."""
    return h_air(T, W_sat(T, p))


def wet_bulb(T_db, W, p):
    """Thermodynamic wet bulb (°C) from dry bulb, humidity ratio and pressure (ASHRAE psychrometric equation)."""
    lo = np.full_like(T_db, -30.0)
    hi = np.array(T_db, dtype=float)
    for _ in range(50):
        mid = (lo + hi) / 2
        W_star = ((2501 - 2.326 * mid) * W_sat(mid, p) - cp_a * (T_db - mid)) / (2501 + 1.86 * T_db - 4.186 * mid)
        too_wet = W_star > W
        hi = np.where(too_wet, mid, hi)
        lo = np.where(too_wet, lo, mid)
    return (lo + hi) / 2


def bisect(fun, lo, hi, n=32):
    """Vectorised bisection for a function that rises with x. Returns x where fun(x) = 0,
    or the end of the range when there is no crossing inside it."""
    lo = np.array(lo, dtype=float)
    hi = np.array(hi, dtype=float)
    for _ in range(n):
        mid = (lo + hi) / 2
        above = fun(mid) >= 0
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return hi


def merkel(T_in, T_out, h_a_in, LG, p):
    """Merkel number needed to cool water from T_in to T_out (4 point Chebyshev integration).
    Air enthalpy rises along the tower by (L/G) cp_w per K of water cooling.
    Returns infinity where the air would reach saturation (impossible duty)."""
    dT = T_in - T_out
    total = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        for c in (0.1, 0.4, 0.6, 0.9):
            T = T_out + c * dT
            dh = h_sat(T, p) - (h_a_in + LG * cp_w * (T - T_out))
            total = total + np.where(dh > 0, 1 / dh, np.inf)
    return cp_w * dT / 4 * total


def eps_crossflow(NTU, Cr):
    """Effectiveness of a crossflow exchanger, both streams unmixed (Incropera, Table 11.3)."""
    Cr = np.maximum(Cr, 1e-9)
    return 1 - np.exp((NTU**0.22 / Cr) * (np.exp(-Cr * NTU**0.78) - 1))


def trim_chiller(Q_ch, T_fs, T_db, COP_max=None):
    """Air cooled chiller. COP = eta_II x Carnot COP between evaporating and condensing temperatures.
    T_fs is the chilled water it delivers. COP_max caps the COP at low lift (used for the room chiller).
    Returns electricity (kW), COP and Carnot COP for every hour."""
    T_ev = T_fs - dT_ev + KELVIN
    T_cd = T_db + dT_cd + KELVIN
    COP_carnot = T_ev / np.maximum(T_cd - T_ev, 0.1)
    COP = eta_II * COP_carnot
    if COP_max is not None:
        COP = np.minimum(COP, COP_max)
    W_ch = np.where(Q_ch > 0, Q_ch / COP, 0.0)
    return W_ch, COP, COP_carnot


def make_weather(raw, dT_clim=0.0):
    """Hourly weather arrays. A climate shift moves the dry bulb and keeps the humidity ratio."""
    wx = {
        "hour": raw["Hour of year"].to_numpy(),
        "month": raw["Month"].to_numpy(),
        "day": (raw["Hour of year"].to_numpy() - 1) // 24,
        "T_db": raw["Dry bulb (°C)"].to_numpy(dtype=float) + dT_clim,
        "W": raw["Humidity ratio (g/kg dry air), calculated"].to_numpy(dtype=float) / 1000,
        "p": raw["Station pressure (Pa)"].to_numpy(dtype=float),
    }
    wx["T_wb"] = wet_bulb(wx["T_db"], wx["W"], wx["p"])
    wx["h_a"] = h_air(wx["T_db"], wx["W"])
    wx["h_weight"] = H_op / len(raw)          # operating hours represented by one weather hour
    return wx


def loop_design(T_s, f=f_liq):
    """Common system: heat split, rack loop, CDU and facility supply temperature. Same in every option."""
    Q_liq = f * P_IT                         # kW, IT heat into the water
    Q_res = (1 - f) * P_IT                  # kW, IT heat into the room air
    m_w = Q_liq / (cp_w * dT_loop)             # kg/s, from the enthalpy change of the water (brief p3)
    V_w = m_w / rho_w                          # m3/s
    P_loop = SFP_pump * Q_liq                 # kW, rack + facility loop pumps (specific pump power)
    return {
        "T_s": T_s,
        "Q_liq": Q_liq,
        "Q_res": Q_res,
        "m_w": m_w,
        "C_w": m_w * cp_w,                  # kW/K, same flow on both sides of the CDU
        "V_w": V_w,
        "P_loop": P_loop,
        "Q_rack": Q_liq + P_loop,             # kW into the water (chips + pump work ends up as heat)
        "T_fs": T_s - A_cdu,                   # °C, facility water the outdoor side must deliver
    }


def overhead(wx, lp):
    """Non cooling overhead, built bottom up: UPS losses, lighting and the room air system.
    The room air system removes the air side IT heat plus the UPS and lighting heat with an air cooled
    chiller (same COP formula, capped) and indoor fans. Same equipment in every option."""
    P_ups = f_ups * P_IT                                   # kW
    Q_room = lp["Q_res"] + P_ups + P_light                   # kW of heat in the room air
    Q_room_h = np.full_like(wx["T_db"], Q_room)
    W_room, COP_room, _ = trim_chiller(Q_room_h, T_chw_room, wx["T_db"], COP_max=COP_max)
    P_crah = SFP_crah * Q_room                             # kW, indoor fans
    P_room = W_room + P_crah                             # kW, room air system electricity each hour
    return {"P_ups": P_ups, "P_light": P_light, "Q_room": Q_room, "COP_room": COP_room,
            "W_room": W_room, "P_crah": P_crah, "P_room": P_room,
            "P_other": P_ups + P_light + P_room}              # kW each hour, everything that is not IT or liquid cooling


def annual(x, wx):
    """kWh/yr from an hourly kW array (or kg/yr from kg/h)."""
    return float(np.sum(x) * wx["h_weight"])


def to_m3(m_dot, wx):
    """m3/yr from an hourly kg/s array (1 m3 of make up water = 1000 kg)."""
    return float(np.sum(m_dot) * 3600 * wx["h_weight"] / 1000)
