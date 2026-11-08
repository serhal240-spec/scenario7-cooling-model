"""
Scenario 7 model: inputs.
Every input value with its unit and source, plus the values derived from them.
"""

import numpy as np


# =====================================================================
# INPUTS
# ASSUMPTION = typical value checked against published ranges (status A in the report, Table 1).
# The brief's values are used as a starting point. STEP 0 prints the full input register.
# =====================================================================

# From the brief (p3, common starting point)
P_IT = 1000            # kW, IT load, brief p3
H_op = 8760            # h/yr at full IT load: every hour of the weather year, so no idle hours and no standby
                       # row. Brief p3 suggests 8,000 h; the module leader agreed the full year could be used
PUE_ref = 1.20         # PUE of the terrestrial reference, brief p3. Used ONLY as a check on our result
f_liq = 0.80           # share of IT heat captured by the cold plate (liquid) loop; the rest heats the room air.
                       # ASSUMPTION for cold plate servers: Lenovo Neptune removes "up to 80%" through water.
                       # Same number as the brief's recoverable heat reference, but not the same thing.
T_supply = 40          # °C, water going to the chips (rack side of the CDU), brief p3
T_return = 60          # °C, water coming back from the chips. brief p3
cp_w = 4.18            # kJ/(kg K), water. Cengel and Cimbala, Fluid Mechanics

# Physical constants
cp_a = 1.006           # kJ/(kg K), dry air. Cengel and Boles, Thermodynamics, Table A-2
h_fg = 2430            # kJ/kg, latent heat of water at about 30 °C. Cengel and Boles, Table A-4
rho_w = 990            # kg/m3, water at about 50 °C (pump volume flow). Steam tables
KELVIN = 273.15        # K

# Site design condition
T_wb_d = 24.1          # °C, design wet bulb, ASHRAE 0.4% evaporation, Phoenix. ASHRAE Handbook Fundamentals (2025)
                       # ch.14, read from the TMYx file header. (NOAA 1990s data, gave 24.4; changed so all
                       # design values come from one source)

# Heat exchangers (approach = temperature difference between the two streams)
A_cdu = 5.0            # K, CDU (HX-1), rack water to facility water, so facility loop 35/55 °C.
                       # Catalogue CDUs: Vertiv XDU450 4 K nominal (8 K max), Delta SuperCDU 4 to 5 K
A_hx = 3.0             # K, HX-2, facility water to open tower water (baseline only). Deppmann (2021);
                       # Alfa Laval HVAC PHE practice uses about 1 to 2 K, 3 K kept as a margin

# Pumps
SFP_pump = 0.005       # kW of pumping per kW of loop heat, rack + facility loop (P1 + common part of P2).
                       # ASHRAE HVAC Applications Handbook ch.43
eta_p = 0.65           # pump + motor efficiency, for the option specific extras below. ASSUMPTION, typical 0.6 to 0.75
dp_hx2 = 69            # kPa, P2 extra through HX-2 (baseline). Alfa Laval, HVAC PHE best design practices (~10 psi)
dp_coil = 50           # kPa, P2 extra through the dry cooler coil (Options A and B). Kelvion LF-PD203 datasheet
                       # (516 kW, 24 kPa) plus about 26 kPa for headers and roof piping (our allowance)
dp_P3 = 200            # kPa, P3 tower water pump: static lift, spray nozzles, HX-2 tower side, piping.
                       # BAC Pumping and Piping guide, examples 65 to 70 ft of head (about 200 kPa)

# Fans
f_min = 0.20           # lowest fan speed as a fraction of full speed. DesignBuilder / MEP Academy

# Baseline: cooling tower (Merkel method)
A_tw = 4.0             # K, tower approach (cold water minus wet bulb) at full fan on the design day. Brentwood
                       # This sets the tower size. It can make 28.1 °C water on the design day; we only need 32 °C,
                       # so the fans run slowly most of the year.
LG_d = 1.0             # water to air mass flow ratio L/G at full fan. ASSUMPTION, typical 0.75 to 1.5 in tower selection data
n_tw = 0.6             # Merkel number scales with (L/G)^-n, so with fan speed f^n. Kloppers and Kroger (2005),
                       # fitted exponents 0.5 to 0.8 for splash and film fills; middle value used
SFP_tw = 0.012         # kW of fan per kW of heat, at full speed. Brentwood (ASHRAE 90.1 minimum is 0.0127)
phi_evap = 0.85        # share of tower heat removed by evaporation (fixed). ASHRAE HVAC S&E (2016) ch.40
COC = 4                # cycles of concentration. US DOE FEMP
drift_frac = 0.00005   # drift as a fraction of tower water flow (0.005%). ASSUMPTION, typical of modern drift eliminators

# Options A and B: dry cooler (effectiveness NTU method, crossflow)
A_dc = 7.0             # K, approach (water out minus air in) at full fan and full load. Brentwood
dTa_d = 10.0           # K, air temperature rise across the coil at full fan and full load. Trane DRY500 / Airedale DR95
n_dc = 0.6             # UA scales with airflow^n (air side controls). Zukauskas tube bank correlation, Nu ~ Re^0.63
                       # for Re 1e3 to 2e5 (Incropera et al. 2011, Table 7.5)
SFP_dc = 0.025         # kW of fan per kW of heat, at full speed. ASHRAE HVAC S&E Handbook ch.41

# Trim chiller (air cooled, in series after the outdoor unit, same model in every option)
# COP formula: COP = eta_II x Carnot COP, with T_ev = chilled water out - dT_ev and T_cd = outdoor dry bulb + dT_cd.
# eta_II is not guessed: it is fitted so the formula gives the rated COP at the standard rating point.
COP_rated = 2.8        # full load COP of an air cooled chiller at the rating point. Below the
                       # ASHRAE 90.1-2019 minimum of 2.96 (Table 6.8.1-3, Path A, < 150 t) so it is conservative;
                       # 2.2 to 3.4 tested in STEP 10, which covers 2.96
T_db_rated = 35.0      # °C, outdoor air at the AHRI 550/590 rating point
T_chw_rated = 6.7      # °C, chilled water out at the AHRI 550/590 rating point
dT_ev = 5.0            # K, chilled water out minus evaporating temperature. Typical selection (Stoecker and Jones
                       # 1982, condensers and evaporators). Only shapes the COP curve: eta_II is refitted so the
                       # rated COP is always met
dT_cd = 15.0           # K, condensing temperature minus outdoor dry bulb. Typical air cooled condenser selection
                       # (Stoecker and Jones 1982); kept at the high side for a hot climate

# Option B: adiabatic spray
eps_ad = 0.75          # saturation effectiveness of the spray / pads. IIR (adiabatic condensers and fluid coolers)
eta_sp = 0.80          # share of sprayed water that evaporates (rest drains). ASSUMPTION; Evans and Corral (SPX/CTI 2025): 67 to 75%
P_sp = 0.5             # kW, spray / pad pump when spraying. Evans and Corral (SPX/CTI TP25-25, 2025): 0.9 to 1.5
                       # gpm per ft2 of pad TOP area; pad about 23 m long x 0.15 m thick (46 m2 face, Hegg et al. 2019)
                       # = 2.1 to 3.5 L/s at 10 m head, pump 0.6 -> 0.35 to 0.57 kW.
                       # Pad face 46 m2 gives 1.6 m/s at the 74 m3/s design air flow (Hegg et al. 2019 tested 2 to 3.5 m/s)
WUE_lim = 0.27         # L/kWh IT, site water allowance. Group choice: Microsoft 2025 fleet WUE
                       # V_lim (below) = WUE_lim x IT energy, applied to site water CONSUMED
T_spray_ref = 29.4     # °C, spray switch on used by Microsoft in Arizona (Microsoft blog, 2026). Shown for comparison
T_sw_options = np.arange(15.0, 47.5, 0.5)   # °C, spray switch on temperatures tested for Option B
T_set_options = np.arange(22.0, 28.5, 1.0)  # °C, coil inlet air targets tested (28 °C = coil alone limit)

# Drained water (tower blowdown, Option B spray run off)
r_B = 1.0              # share of drained water returned to the water system (sewer -> treatment -> reuse), so it
                       # is withdrawn but not consumed. ASSUMPTION: USGS definition of consumptive use (evaporated
                       # or removed); Phoenix treats and reuses its wastewater. r_B = 0 is run in STEP 10

# Non cooling overhead, built bottom up (same equipment in every option)
f_ups = 0.04           # UPS + power distribution losses as a share of IT power. ASSUMPTION: ENERGY STAR gives
                       # 92 to 95% for new UPSs; 4% assumes a high efficiency unit
P_light = 10.0         # kW, lighting, controls, security. WSEC allowance 0.94 W/ft2 (10.1 W/m2) over about 1,000 m2
T_chw_room = 15.0      # °C, chilled water to the room air handlers (warm data hall, ASHRAE supply air 18 to 27 °C).
                       # ASSUMPTION. Room chiller uses the same COP formula as the trim chiller (same eta_II)
COP_max = 8.0          # cap on chiller COP at low lift (compressor minimum pressure ratio). Applied to BOTH the
                       # room chiller and the trim chillers. ASSUMPTION, typical of scroll chiller data
SFP_crah = 0.03        # kW of indoor air handler fan per kW of room heat. ASSUMPTION, typical CRAH fan power

# Design peak check (STEP 9e): both at once, harsher than any real hour
T_db_des = 44.0        # °C, ASHRAE 0.4% cooling dry bulb, Phoenix (EPW header, ASHRAE 2025)
T_wb_des = T_wb_d      # °C, ASHRAE 0.4% evaporation wet bulb (2025), same source as the dry bulb

# Grid
CI_grid = 0.3203       # kg CO2e/kWh, location based. EPA eGRID2023 rev2, AZNM (706.2 lb/MWh)
EWIF = 1.1             # L/kWh, water consumed by power stations per kWh. Macknick et al. (2012) x AZNM mix

# Sensitivity ranges (STEP 10). Each range comes from a source or from the spread in our input data.
CI_cases = [0.20, 0.45]              # kg CO2e/kWh, cleaner grid / more gas. Group choice (base: eGRID2023 AZNM 0.3203)
T_s_cases = [35, 45]                 # °C, chip water limit: ASHRAE liquid cooling classes W32 to W45; GB200 accepts 45
WUE_lim_cases = [0.15, 0.40]         # L/kWh IT, site water allowance, around Chandler's rule of 0.18 to 0.36 L/kWh
A_dc_cases = [3.0, 10.0]             # K, dry cooler approach: low value / catalogue selections (Kelvion, Trane)
dTa_catalogue = 12.0                 # K, air rise used with the 10 K catalogue approach (catalogues show 11 to 13 K)
COP_rated_cases = [2.2, 3.4]         # rated chiller COP about -20% / +20%, covers the 2.96 ASHRAE 90.1 minimum.
                                     # Applied to trim AND room chillers (same eta_II)
COP_max_cases = [5.0, None]          # COP cap: stricter / no cap (fixed eta_II at every lift)
eps_ad_cases = [0.65, 0.85]          # pad effectiveness around the 0.72 to 0.78 measured by Hegg et al. (2019)
phi_cases = [0.80, 0.88]             # tower evaporation share, winter / summer from the air side check
dT_clim_cases = [1.5, 3.0]           # K added to every dry bulb (humidity ratio kept): warmer years than the TMYx
# Checks quoted in the report text but not in Figure 12 (they change every option together or repeat the chip limit)
f_liq_cases = [0.70, 0.75]           # liquid share, range of published cold plate capture shares
A_cdu_cases = [3.0, 4.0]             # K, CDU approach: facility loop 37/57, 36/56 °C (catalogue 4 to 5 K)

# Derived
dT_loop = T_return - T_supply        # K, 20
Q_liq = f_liq * P_IT                 # kW, IT heat into the liquid loop (base case)
Q_res = (1 - f_liq) * P_IT           # kW, IT heat into room air (base case), removed by the room air system
E_IT = P_IT * H_op                   # kWh/yr, IT energy
V_lim = WUE_lim * E_IT / 1000        # m3/yr, annual site water limit (consumption)

# Chiller COP formula fitted to the rating point
T_ev_rated = T_chw_rated - dT_ev + KELVIN                              # K, evaporating temperature at rating
T_cd_rated = T_db_rated + dT_cd + KELVIN                               # K, condensing temperature at rating
COP_carnot_rated = T_ev_rated / (T_cd_rated - T_ev_rated)              # Carnot COP at the rating point
eta_II = COP_rated / COP_carnot_rated                                  # second law efficiency of the chiller
