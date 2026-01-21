# -*- coding: utf-8 -*-
"""D-Pile (single pile) rekenkern op basis van OpenSeesPy.

Dit bestand is bedoeld om als module te worden geïmporteerd door de Streamlit app.
- Bevat: CONFIG (default), build_and_run(config)
- Geen interactieve UI / geen __main__ runner (deploy-ready)

Auteur: W. Elgersma (origineel)
Opschoning/packaging: 2026-01-20
"""

import openseespylinux as ops
import numpy as np
import csv
import os
import math
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG – pas hier je parameters aan
# (met opties/keuzes per instelling)
# =============================================================================
CONFIG = {
    # =============================
    # SOIL (Bodem)
    # =============================
    'soil': {
        'use_layers': False,  # opties: True \ False
        # - False: gebruik 'single'
        # - True : gebruik layer_1..layer_3 (enabled=True)
        'gwt_depth_m': -1.0,  # Grondwaterstand in m onder maaiveld (>=0). None = geen GWT.
        'top_py_eval_depth_m': None,  # evaluatiediepte [m] voor bovenste p-y veer
        # None = automatische keuze (afhankelijk van D en GWT)
        'print_layers': True,  # opties: True \ False
        # print lagenoverzicht als use_layers=True

        # ---------------------------
        # Single soil (alleen als use_layers=False)
        # ---------------------------
        'single': {
            'type': 'clay',  # opties: 'sand' \ 'clay'
            'dry_unit_weight': 16.0,  # kN/m3 (gamma_droog)
            'wet_unit_weight': 16.0,  # kN/m3 (gamma_nat)
            'lateral_rule': 'api_static',  # opties: 'api_static' \ 'api_cyclic'

            # ---- klei-parameters (alleen relevant bij type='clay')
            'su': 30.0,        # kN/m2 (ongedraineerde schuifsterkte)
            'epsilon50': 0.010,# [-] rek bij 50% mobilisatie (Matlock/API)
# -------------------------------------------------------------------------
# ε50 (epsilon50) keuzehulp – API/Matlock (klei)
#
# ε50 = strain at 50% failure load (dimensionloos) voor API/Matlock p-y curves.
# API context geldt o.a.: y50 = 2.5 * ε50 * D
#
# Richtwaarden:
#   su [kN/m²]     ->  ε50 [-]
#   5   – 25       ->  0.020
#   25  – 50       ->  0.010
#   50  – 100      ->  0.007
#   100 – 200      ->  0.005
#   200 – 400      ->  0.004
#
# Tips:
# - Als je geen lab/field test hebt voor ε50, kies een waarde uit bovenstaande band
#   op basis van su (ongedraineerde schuifsterkte).
# - Conservatief (stijver/soepeler) hangt af van interpretatie en projectcontext
# Bron: D-Pile Group User Manual, hoofdstuk 15, Table 15.1 (API/Matlock). 
# -------------------------------------------------------------------------
            'J': 0.25,         # [-] empirische factor (API/Matlock, default 0.25)

            # ---- zand-parameters (alleen relevant bij type='sand')
            'phi_deg': 30.0,        # graden (wrijvingshoek)

        },
        # ---------------------------
        # Soil layers (alleen als use_layers=True)
        # Elke layer_x: enabled True/False
        # type: 'sand'/'clay'
        # ---------------------------
        'layer_1': {
            'enabled': True,  # opties: True \ False
            'top_m': 0.0,     # m vanaf maaiveld (laag top)
            'type': 'clay',   # opties: 'sand' \ 'clay'
            'dry_unit_weight': 16.0,  # kN/m3
            'wet_unit_weight': 16.0,  # kN/m3\
            'lateral_rule': 'api_static',  # opties: 'api_static' \ 'api_cyclic'
            # klei
            'su': 30.0,       # kN/m2 (verplicht voor clay)
            'epsilon50': 0.010,# [-] (verplicht voor clay. Zie tabel boven)
            'J': 0.25,        # [-]
            # zand
            'phi_deg': 30.0,  # graden (verplicht voor sand, anders None)
        },
        'layer_2': {
            'enabled': True,  # True \ False
            'top_m': 3.0,     # m
            'type': 'sand',   # 'sand' \ 'clay'
            'dry_unit_weight': 18.0,  # kN/m3
            'wet_unit_weight': 18.0,  # kN/m3
            'lateral_rule': 'api_static',  # opties: 'api_static' \ 'api_cyclic'
            'su': 30.0,        # kN/m2 (alleen clay)
            'epsilon50': 0.010,# [-] (alleen clay. Zie tabel boven)
            'J': 0.25,         # [-]
            'phi_deg': 30.0,   # graden (alleen sand)
        },
        'layer_3': {
            'enabled': False,  # True \ False
            'top_m': 8.0,      # m
            'type': 'clay',    # 'sand' \ 'clay'
            'dry_unit_weight': 20.0,  # kN/m3
            'wet_unit_weight': 20.0,  # kN/m3
            'lateral_rule': 'api_static',  # opties: 'api_static' \ 'api_cyclic'
            'su': 60.0,        # kN/m2 (clay)
            'epsilon50': 0.007,# [-] (clay. Zie tabel boven)
            'J': 0.25,         # [-]
            'phi_deg': None,   # graden (sand)
        },
    },
    # =============================
    # PILE (Paal)
    # =============================
    'pile': {
        'type': 'concrete_round',  # opties: 'steel' \ 'concrete_round' \ 'concrete_square'
        # - steel: outer_diameter + wall_thickness
        # - concrete_round: outer_diameter
        # - concrete_square: width
        # - user_defined: EA + EI (override_EA_EI hoeft dan niet)
        'length': 12.8,  # m totale paallengte
        'head_above_ground': 0.0,  # m paalkop boven maaiveld (0 = op maaiveld)
        'outer_diameter': 0.3,  # m (steel / concrete_round)
        'wall_thickness': 0.01,  # m (alleen steel; eis: 0 < t < D/2)
        'width': 0.3,  # m (alleen concrete_square)
        'E': 2.2e7,  # kN/m2 of consistente eenheidset (E moet matchen met krachten/afmetingen) 
        # (2.2e7 voor concrete_round en concrete_square. 2.1e8 voor steel)
        # (optioneel) override van EA/EI:
        'override_EA_EI': False,  # True \ False
        'EA': None,  # (alleen nodig als override_EA_EI=True of type='user_defined')
        'EI': None,  # idem
    },
    # =============================
    # LOAD (Belasting)
    # =============================
    'load': {
        'lateral_head_load': 80.0,  # kN horizontale kracht op paalkop
        'head_moment_z': 0.0,  # kNm moment op paalkop (z-as)
    },
    # =============================
    # BOUNDARY (Randvoorwaarden)
    # =============================
    'boundary': {
        'top_condition': 'free',  # opties: 'free' \ 'fixed'
        # - 'free' : rotatie aan paalkop vrij
        # - 'fixed' : rotatie aan paalkop vast
        'fix_vertical_dof': True,  # True \ False
        # True = verticale verplaatsing (uy) vastgezet
    },
    # =============================
    # ANALYSIS
    # =============================
    'analysis': {
        'n_steps': 100,  # int aantal load steps
        'tolerance': 1e-6,  # float convergentietolerantie
        'max_iters': 25,  # int max iteraties per stap
    },

    # =============================
    # OUTPUT
    # =============================
    'output': {
        'folder': 'Resultaten',  # mapnaam voor output
        'csv': 'results.csv',  # bestandsnaam resultaten
        'png': 'results.png',  # plot output
        'plot': True,  # True \ False: maak de 3 standaard plots
        'py_plots': {
            'enabled': True,  # True \ False: extra P-Y plots maken
            'folder': 'PY_Plots',  # submap in outputfolder
            'levels_mode': 'manual',  # opties: 'auto' \ 'manual'
            # - 'auto': automatisch levels kiezen (incl top en tip)
            # - anders: handmatig via levels_m
            'levels_m': [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5],  # list[float] dieptes [m] (alleen bij manual)
            'spacing_m': None,  # float \ None: bij auto kun je vaste spacing afdwingen
            'max_plots': 8,  # int max aantal P-Y plots (min 2)
            'per_unit_length': True,  # True => P in kN/m \ False => P in kN
            'multilinear': True,  # True => 5-takken knikpunten \ False => glad (meer sample punten)
            'dpi': 200,  # int resolutie van PNG
            'show': False,  # True \ False: plots ook tonen (handig bij lokaal draaien)
        }
    },
}
# =============================================================================
# -------------------- Helpers: gamma_eff(z) & top z eval ---------------------
def normalize_gwt_depth(gwt_depth):
    if gwt_depth is None:
        return None
    try:
        val = float(gwt_depth)
    except Exception:
        return None
    return abs(val)
def gamma_effective(z, gamma_dry, gamma_wet, gwt_depth, gamma_water=9.81):
    if gwt_depth is None or gwt_depth < 0:
        return gamma_dry
    if z <= gwt_depth:
        return gamma_dry
    return max(gamma_wet - gamma_water, 0.0)


def effective_overburden_stress(z, layers, gwt_depth, gamma_water=9.81):
    """Bereken verticale effectieve overburden-spanning σ'v0 op diepte z (kN/m2).

    - z: diepte onder maaiveld [m] (positief naar beneden)
    - layers: lijst met dicts met 'top_m', 'bottom_m', 'dry_unit_weight', 'wet_unit_weight'
              (zoals opgebouwd in build_and_run). Indien None: retourneer None.
    - gwt_depth: grondwaterstand [m onder maaiveld] (None = geen GWT)

    Opmerking:
    In API-formules komt vaak γ' * H voor. Bij lagen met wisselende volumieke gewichten
    is dat gelijk aan de integraal van γ'(z) over 0..H, oftewel σ'v0(H).
    """
    try:
        H = max(float(z), 0.0)
    except Exception:
        return 0.0
    if H <= 0.0:
        return 0.0
    if layers is None:
        return None

    sigma = 0.0
    gwt = gwt_depth if (gwt_depth is not None and gwt_depth >= 0.0) else None

    for lay in layers:
        top = float(lay.get('top_m', 0.0))
        bot = float(lay.get('bottom_m', 1e12))
        if H <= top:
            break
        seg_top = max(0.0, top)
        seg_bot = min(H, bot)
        if seg_bot <= seg_top:
            continue

        gamma_dry = float(lay.get('dry_unit_weight', 20.0))
        gamma_wet = float(lay.get('wet_unit_weight', 20.0))

        # Split eventueel op GWT binnen dit segment
        if gwt is None:
            sigma += gamma_dry * (seg_bot - seg_top)
        else:
            if seg_bot <= gwt:
                # geheel boven GWT
                sigma += gamma_dry * (seg_bot - seg_top)
            elif seg_top >= gwt:
                # geheel onder GWT
                sigma += max(gamma_wet - gamma_water, 0.0) * (seg_bot - seg_top)
            else:
                # segment kruist GWT
                sigma += gamma_dry * (gwt - seg_top)
                sigma += max(gamma_wet - gamma_water, 0.0) * (seg_bot - gwt)

    return sigma

def compute_top_z_eval(D, gwt_depth, override=None):
    """Bepaal evaluatiediepte voor de bovenste p-y veer (H in API-formules).

    D-Pile Group toont/werkt standaard met een topniveau van ca. 0.02 m onder maaiveld
    voor de bovenste P-Y veer. Een vaste kleine diepte voorkomt spronggedrag bij
    variërende grondwaterstand.
    """
    if override is not None:
        return max(float(override), 1e-3)
    return 0.02

# -------------------- Sectie/stijfheid ---------------------------------------
def _compute_section(pile_cfg):
    ptype = str(pile_cfg.get('type', 'steel')).lower()
    E = float(pile_cfg.get('E', 2.1e8))
    override = bool(pile_cfg.get('override_EA_EI', False)) if 'override_EA_EI' in pile_cfg else False
    EA_override = pile_cfg.get('EA', None)
    EI_override = pile_cfg.get('EI', None)
    if override or ptype == 'user_defined':
        if EA_override is None or EI_override is None:
            raise ValueError("override_EA_EI/user_defined vereist 'EA' en 'EI' in CONFIG['pile'].")
        # OpenSees 'elasticBeamColumn' vraagt A, E en I afzonderlijk en gebruikt E*A en E*I.
        # Bij override_EA_EI of type='user_defined' zijn EA en EI gegeven; leid daarom af:
        #   A = EA / E  en  I = EI / E
        EA = float(EA_override)
        EI = float(EI_override)
        if EA <= 0.0 or EI <= 0.0:
            raise ValueError("EA en EI moeten > 0 zijn voor user_defined/override.")
        if abs(E) < 1e-18:
            # Als E niet zinvol is opgegeven, val terug op E=1.0 (dan geldt A=EA en I=EI).
            E = 1.0
        A = EA / E
        I = EI / E
        if A <= 0.0 or I <= 0.0:
            raise ValueError("Afgeleide A/I ongeldig (A<=0 of I<=0). Controleer EA/EI en E.")
        return A, I, EA, EI, E
    if ptype == 'steel':
        D = float(pile_cfg['outer_diameter'])
        t = float(pile_cfg.get('wall_thickness', 0.0))
        if t <= 0.0 or t >= 0.5 * D:
            raise ValueError("Ongeldige wall_thickness voor steel: geef 0 < t < D/2.")
        Di = D - 2.0 * t
        A = math.pi * 0.25 * (D**2 - Di**2)
        I = math.pi * 0.015625 * (D**4 - Di**4)  # pi/64
    elif ptype == 'concrete_round':
        D = float(pile_cfg['outer_diameter'])
        if D <= 0:
            raise ValueError("outer_diameter moet > 0 zijn voor concrete_round.")
        A = math.pi * 0.25 * D**2
        I = math.pi * 0.015625 * D**4  # pi/64
    elif ptype == 'concrete_square':
        b = float(pile_cfg['width'])
        if b <= 0:
            raise ValueError("width moet > 0 zijn voor concrete_square-paal.")
        A = b**2
        I = b**4 / 12.0
    else:
        raise ValueError(f"Onbekend pile.type '{ptype}'. Kies uit steel, concrete_round, concrete_square, user_defined.")
    EA = E * A
    EI = E * I
    return A, I, EA, EI, E
# -------------------- CLAY (API/Matlock) -------------------------------------
def calculate_py_clay_api(z, D, su, eps50, J, gamma_dry, gamma_wet, gwt_depth, lateral_rule='api_static', sigma_v_eff=None):
    """
    API/Matlock (klei, statisch/cyclisch) p-y parameters met GWT en (dry/wet) unit weights.
    In: su [kN/m^2], z [m], D [m], eps50 [-], J [-], gamma_dry/wet [kN/m^3], gwt_depth [m]
    Uit: y50 [m], p_u_per_m [kN/m] (lijnlast)
    y50 = 2.5 * eps50 * D
    p_u,shallow = (3*su + gamma_eff*z + J*su*(z/D)) * D
    p_u,deep = 9*su*D
    p_u = min(p_u,shallow, p_u,deep)
    (Voor cyclic wordt de vorm later gezet in de multispring-opbouw; p_u/y50 zijn dezelfde.)
    """
    z_eff = max(z, 0.01)
    y50 = 2.5 * eps50 * D
    gamma_eff = gamma_effective(z_eff, gamma_dry, gamma_wet, gwt_depth)
    sigma_use = float(sigma_v_eff) if (sigma_v_eff is not None) else (gamma_eff * z_eff)
    pu_shallow = (3.0 * su + sigma_use + J * su * (z_eff / D)) * D  # kN/m
    pu_deep = 9.0 * su * D  # kN/m
    pu_per_m = min(pu_shallow, pu_deep)
    return y50, pu_per_m
# -------------------- SAND (API, helpers) ------------------------------------
def _A_factor_linear_static(z_over_D: float) -> float:
    """A-factor (statisch) volgens eenvoudige API-benadering: A = 3 - 0.8*(z/D) met minimum 0.9."""
    try:
        val = 3.0 - 0.8 * float(z_over_D)
    except Exception:
        val = 0.9
    return max(0.9, val)
def _k_from_phi_table(phi_deg: float, gwt_side: str = 'below') -> float:
    """Getabelleerde relatie k(φ) in kN/m³ met lineaire interpolatie."""
    phi_pts = [29.0, 29.5, 30.0, 33.0, 36.0, 38.0, 40.0]
    k_dry = [2715.0, 6109.0, 11199.0, 25453.0, 42761.0, 59051.0, 75341.0]
    k_wet = [2715.0, 5090.0, 8145.0, 16303.0, 25453.0, 32580.0, 41743.0]
    use_dry = str(gwt_side).lower() == 'above'
    k_pts = k_dry if use_dry else k_wet
    if phi_deg <= phi_pts[0]:
        return k_pts[0]
    if phi_deg >= phi_pts[-1]:
        return k_pts[-1]
    for i in range(len(phi_pts) - 1):
        if phi_pts[i] <= phi_deg <= phi_pts[i + 1]:
            return (k_pts[i + 1] - k_pts[i]) / (phi_pts[i + 1] - phi_pts[i]) * (phi_deg - phi_pts[i]) + k_pts[i]
    return k_pts[0]
def _C123_from_phi_table(phi_deg):
    """API tabel (Table 15.3): C1, C2, C3 als functie van φ [deg]."""
    phi_pts = [20.0, 25.0, 30.0, 35.0, 40.0]
    C1_pts = [0.77, 1.22, 1.90, 3.00, 4.67]
    C2_pts = [1.58, 2.03, 2.67, 3.45, 4.35]
    C3_pts = [9.00, 15.50, 28.50, 54.25, 100.00]
    if phi_deg <= phi_pts[0]:
        return C1_pts[0], C2_pts[0], C3_pts[0]
    if phi_deg >= phi_pts[-1]:
        return C1_pts[-1], C2_pts[-1], C3_pts[-1]
    for i in range(len(phi_pts) - 1):
        if phi_pts[i] <= phi_deg <= phi_pts[i + 1]:
            t = (phi_deg - phi_pts[i]) / (phi_pts[i + 1] - phi_pts[i])
            C1 = C1_pts[i] + t * (C1_pts[i + 1] - C1_pts[i])
            C2 = C2_pts[i] + t * (C2_pts[i + 1] - C2_pts[i])
            C3 = C3_pts[i] + t * (C3_pts[i + 1] - C3_pts[i])
            return C1, C2, C3
    return C1_pts[0], C2_pts[0], C3_pts[0]
# -------------------- SAND (API, static/cyclic) ------------------------------
def calculate_py_sand_api(z, D, phi_deg, gamma_dry, gamma_wet, gwt_depth, lateral_rule='api_static', sigma_v_eff=None):
    """
    API zand p-y (tanh) met GWT en (dry/wet) unit weights.
    matching: pu via Table 15.3 (C1,C2,C3) en Eq. (15.7)-(15.8).
    y50 volgt uit y_max = (A*pu)/(k*z) en y50 = atanh(0.5)*y_max.
    Retour: y50 [m], pu_per_m [kN/m].
    - 'api_static': A = max(0.9, 3 - 0.8*(z/D)) [origineel gedrag intact]
    - 'api_cyclic': A = 0.9 (API Cyclic voor zand).
    """
    z_eval = max(float(z), 0.01)
    # --- A-factor ---
    lr = str(lateral_rule).lower()
    if lr == 'api_cyclic':
        A = 0.9
    else:
        A = _A_factor_linear_static(z_eval / D)
    # --- gamma' ---
    gamma_eff = gamma_effective(z_eval, gamma_dry, gamma_wet, gwt_depth)
    # --- pu (API Table 15.3 via C1,C2,C3) ---
    C1, C2, C3 = _C123_from_phi_table(phi_deg)
    H = z_eval
    sigma_use = float(sigma_v_eff) if (sigma_v_eff is not None) else (gamma_eff * H)
    pu_shallow = (C1 * H + C2 * D) * sigma_use
    pu_deep = (C3 * D) * sigma_use
    # --- k(phi) uit tabel ---
    k_side_depth = 'above' if (gwt_depth is not None and z_eval <= gwt_depth) else 'below'
    k_si = _k_from_phi_table(phi_deg, gwt_side=k_side_depth)
    # --- eind p_u en y50 ---
    pu_per_m = A * min(pu_shallow, pu_deep)
    # y_max = (A * p_u) / (k * H)
    y_max = pu_per_m / max(k_si * H, 1e-12)
    atanh_half = 0.5 * math.log((1.0 + 0.5) / (1.0 - 0.5))
    y50 = atanh_half * y_max
    return y50, pu_per_m
def build_and_run(config):
    # BODEM
    soil_cfg = config['soil']
    use_layers = bool(soil_cfg.get('use_layers', False) or isinstance(soil_cfg.get('layers', None), list))
    gwt_depth = normalize_gwt_depth(soil_cfg.get('gwt_depth_m', None))
    top_py_override = soil_cfg.get('top_py_eval_depth_m', None)
    if top_py_override is not None:
        top_py_override = float(top_py_override)
    # PAAL/BELASTING
    L = float(config['pile']['length'])
    n_el = 500  # Hardcoded: aantal beam-elementen (mesh resolution)
    H_load = float(config['load']['lateral_head_load'])
    M_head = float(config['load'].get('head_moment_z', 0.0))
    fix_vertical_dof = bool(config['boundary'].get('fix_vertical_dof', True))
    top_condition_global = str(config['boundary'].get('top_condition', 'free')).lower()
    # Sectie/stijfheid
    A_sec, I_sec, EA, EI, E = _compute_section(config['pile'])
    D = float(config['pile']['outer_diameter'])
    # BODEM single
    if not use_layers:
        single = soil_cfg.get('single', {})
        soil_type = str(single.get('type', 'clay')).lower()
        gamma_dry = float(single.get('dry_unit_weight', 20.0))
        gamma_wet = float(single.get('wet_unit_weight', 20.0))
        lateral_rule_single = str(single.get('lateral_rule', 'api_static'))
        if soil_type == 'clay':
            su = float(single.get('su'))
            eps50 = float(single.get('epsilon50'))
            J = float(single.get('J', 0.25))
        elif soil_type == 'sand':
            float(single.get('phi_deg'))
        else:
            raise ValueError("soil.single.type moet 'clay' of 'sand' zijn")
    # BODEM layers
    layers = None
    if use_layers:
        # Nieuw: voorkeur via soil['layers'] (onbeperkt aantal lagen)
        layers_cfg = soil_cfg.get('layers', None)

        # Legacy fallback: layer_1..layer_3 (backwards compatible)
        if not isinstance(layers_cfg, list):
            layers_cfg = []
            for key in ['layer_1', 'layer_2', 'layer_3']:
                lay = soil_cfg.get(key, None)
                if isinstance(lay, dict) and bool(lay.get('enabled', False)):
                    layers_cfg.append(lay)

        # Accepteer dicts; 'enabled' is optioneel (default True)
        layers_cfg = [lay for lay in layers_cfg if isinstance(lay, dict) and bool(lay.get('enabled', True))]
        if len(layers_cfg) == 0:
            raise ValueError("use_layers=True maar geen lagen opgegeven")

        layers = []
        for idx, lay in enumerate(layers_cfg, start=1):
            top_m = float(lay.get('top_m', 0.0))
            ltype = str(lay.get('type', '')).lower()
            if ltype not in ['clay', 'sand']:
                raise ValueError(f"Laag {idx}: type moet 'clay' of 'sand' zijn")
            if ltype == 'clay' and (lay.get('su') is None or lay.get('epsilon50') is None):
                raise ValueError(f"Laag {idx} clay: vul su en epsilon50")
            if ltype == 'sand' and (lay.get('phi_deg') is None):
                raise ValueError(f"Laag {idx} sand: vul phi_deg")

            norm = {**lay}
            norm['type'] = ltype
            norm['top_m'] = top_m
            if ltype == 'clay':
                norm['phi_deg'] = None
            layers.append(norm)

        layers.sort(key=lambda d: d['top_m'])
        for i in range(len(layers)):
            layers[i]['bottom_m'] = layers[i+1]['top_m'] if i < len(layers)-1 else 1e12

        if bool(soil_cfg.get('print_layers', False)):
            print('Gebruikte lagen (top->bottom):')
            for j, LAY in enumerate(layers, start=1):
                print(f" Laag {j}: top={LAY['top_m']} m, bottom={LAY['bottom_m']} m, type={LAY['type']}")

    def _soil_at_depth(z_m):
        if not use_layers:
            return None
        z = float(z_m)
        for lay in layers:
            if lay['top_m'] <= z < lay['bottom_m']:
                return lay
        return layers[-1]

    # Uitvoerbestanden
    out_dir = config.get('output', {}).get('folder', 'Resultaten')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, config.get('output', {}).get('csv', 'results.csv'))
    png_path = os.path.join(out_dir, config.get('output', {}).get('png', 'results.png'))
    do_plot = bool(config.get('output', {}).get('plot', True))
    # OpenSees (modelopbouw)
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)
    ops.geomTransf('Linear', 1)
    n_steps = int(config['analysis']['n_steps'])
    tol = float(config['analysis']['tolerance'])
    iters = int(config['analysis']['max_iters'])
    dz = L / n_el
    h_above = float(config['pile'].get('head_above_ground', 0.0))
    z_eval_top = compute_top_z_eval(D, gwt_depth, override=top_py_override)
    def _tributary_length(i_node, embedded_local, depths_list):
        pos = embedded_local.index(i_node)
        z_i = depths_list[i_node]
        z_top = 0.0
        z_bot = depths_list[embedded_local[-1]]
        if pos == 0:
            if len(embedded_local) == 1:
                return max(z_bot - z_top, 0.0)
            z_next = depths_list[embedded_local[pos + 1]]
            return 0.5 * ((z_i - z_top) + (z_next - z_top))
        elif pos == len(embedded_local) - 1:
            z_prev = depths_list[embedded_local[pos - 1]]
            return 0.5 * (z_bot - z_prev)
        else:
            z_prev = depths_list[embedded_local[pos - 1]]
            z_next = depths_list[embedded_local[pos + 1]]
            return 0.5 * (z_next - z_prev)
    def build_single_pile(pile_id, x0, top_condition_local):
        base = int(pile_id) * 100000
        node_tags = [base + i for i in range(n_el + 1)]
        ele_tags = [base + 50000 + i for i in range(n_el)]
        for i, nd in enumerate(node_tags):
            ops.node(nd, float(x0), h_above - i * dz)
        uy_fix = 1 if fix_vertical_dof else 0
        top_is_fixed = (str(top_condition_local).lower() == 'fixed')
        for i, nd in enumerate(node_tags):
            rz_fix = 1 if (i == 0 and top_is_fixed) else 0
            ops.fix(nd, 0, uy_fix, rz_fix)
        for i, et in enumerate(ele_tags):
            ops.element('elasticBeamColumn', et, node_tags[i], node_tags[i + 1], A_sec, E, I_sec, 1)
        first_mat_tag = base + 20000
        first_spring_tag = base + 30000
        first_soil_node = base + 40000
        # --- p-y metadata opslaan (voor P-Y plots) ---
        py_meta = []
        depths_below = [max(0.0, -(h_above - i * dz)) for i in range(n_el + 1)]
        embedded_local = [i for i in range(n_el + 1) if (h_above - i * dz) <= 1e-12]
        if len(embedded_local) == 0:
            raise ValueError('Geen paal-embedment')
        for i_local in embedded_local:
            nd = node_tags[i_local]
            z_node = depths_below[i_local]
            z_eval = z_eval_top if (i_local == embedded_local[0]) else max(z_node, 0.01)
            if use_layers:
                lay = _soil_at_depth(z_eval)
                soil_type_local = lay['type']
                gamma_dry_l = float(lay.get('dry_unit_weight', 20.0))
                gamma_wet_l = float(lay.get('wet_unit_weight', 20.0))
            else:
                soil_type_local = soil_type
                gamma_dry_l = gamma_dry
                gamma_wet_l = gamma_wet

            soil_label_nl = 'klei' if soil_type_local == 'clay' else ('zand' if soil_type_local == 'sand' else str(soil_type_local))

            # --- overburden spanning σ'v0 op evaluatiediepte (voor juiste laag-overgang) ---
            if use_layers:
                sigma_v_eff = effective_overburden_stress(z_eval, layers, gwt_depth)
            else:
                # homogeen: σ'v0(z) = ∫0..z γ'(ζ)dζ (met knik bij de grondwaterstand)
                pseudo_layers = [
                    {'top_m': 0.0, 'bottom_m': 1e12, 'dry_unit_weight': gamma_dry_l, 'wet_unit_weight': gamma_wet_l}
                ]
                sigma_v_eff = effective_overburden_stress(z_eval, pseudo_layers, gwt_depth)

            if soil_type_local == 'clay':
                if use_layers:
                    su_l = float(lay.get('su'))
                    eps50_l = float(lay.get('epsilon50'))
                    J_l = float(lay.get('J', 0.25))
                    lateral_rule_l = str(lay.get('lateral_rule', 'api_static'))
                else:
                    su_l, eps50_l, J_l = su, eps50, J
                    lateral_rule_l = lateral_rule_single
                y50, pu_per_m = calculate_py_clay_api(
                    z_eval, D, su_l, eps50_l, J_l, gamma_dry_l, gamma_wet_l, gwt_depth,
                    lateral_rule=lateral_rule_l, sigma_v_eff=sigma_v_eff
                )
                soilType_id = 1
            else:
                if use_layers:
                    phi_l = float(lay.get('phi_deg'))
                    lateral_rule_l = str(lay.get('lateral_rule', 'api_static'))
                else:
                    phi_l = float(single.get('phi_deg'))
                    lateral_rule_l = lateral_rule_single
                y50, pu_per_m = calculate_py_sand_api(
                    z_eval, D, phi_l, gamma_dry_l, gamma_wet_l, gwt_depth,
                    lateral_rule=lateral_rule_l, sigma_v_eff=sigma_v_eff
                )
                soilType_id = 2
            Ltrib = _tributary_length(i_local, embedded_local, depths_below)
            pult_element = pu_per_m * Ltrib
            mat_tag = first_mat_tag + i_local
            atanh_half = 0.5 * math.log((1.0 + 0.5) / (1.0 - 0.5))
            pu_per_m_local = pult_element / max(Ltrib, 1e-12)
            if soilType_id == 1:  # CLAY
                if str(lateral_rule_l).lower() == 'api_cyclic':
                    # Cyclic (H>HR): plateau op 0.72*pu vanaf 3*y50 tot 8*y50
                    y_break = [0.1*y50, 0.3*y50, 1.0*y50, 3.0*y50, 8.0*y50]
                    def p_per_m(y):
                        if y < 3.0*y50:
                            return pu_per_m_local * (0.5 * (y / y50) ** (1.0/3.0))
                        else:
                            return 0.72 * pu_per_m_local
                else:
                    # STATIC (klassieke Matlock)
                    y_break = [0.1*y50, 0.3*y50, 1.0*y50, 3.0*y50, 8.0*y50]
                    def p_per_m(y):
                        return pu_per_m_local * (0.5 * (y / y50) ** (1.0/3.0)) if y < 8.0*y50 else pu_per_m_local
            else:  # SAND
                y_max = y50 / atanh_half if atanh_half > 0 else (8.0*y50)
                y_break = [0.25*y_max, 0.5*y_max, 1.0*y_max, 1.5*y_max, 2.5*y_max]
                def p_per_m(y):
                    return pu_per_m_local * math.tanh(atanh_half * y / max(y50, 1e-12))
            y_pts = [0.0] + y_break
            p_pts = [0.0] + [p_per_m(y) for y in y_break]
            p_pts_e = [pp * Ltrib for pp in p_pts]
            Kt = []
            for ii in range(1, len(y_pts)):
                dy = max(y_pts[ii] - y_pts[ii-1], 1e-12)
                Kt.append((p_pts_e[ii] - p_pts_e[ii-1]) / dy)
            Kt.append(0.0)
            sub_tags = []
            for m in range(1, len(y_pts)):
                k_m = Kt[m-1] - Kt[m]
                if k_m <= 1e-9:
                    continue
                Fy_m = k_m * y_pts[m]
                sub_tag = mat_tag*10 + m
                ops.uniaxialMaterial('Steel01', sub_tag, Fy_m, k_m, 0.0)
                sub_tags.append(sub_tag)
            if len(sub_tags) == 0:
                k_lin = Kt[0] if len(Kt) else 1.0
                sub_tag = mat_tag*10 + 1
                ops.uniaxialMaterial('Elastic', sub_tag, k_lin)
                sub_tags = [sub_tag]
            if len(sub_tags) == 1:
                mat_use = sub_tags[0]
            else:
                ops.uniaxialMaterial('Parallel', mat_tag, *sub_tags)
                mat_use = mat_tag
            soil_node = first_soil_node + i_local
            x_i, y_i = ops.nodeCoord(nd)
            ops.node(soil_node, x_i, y_i)
            ops.fix(soil_node, 1, 1, 1)
            spring_tag = first_spring_tag + i_local
            ops.element('zeroLength', spring_tag, soil_node, nd, '-mat', mat_use, '-dir', 1)
            # --- p-y metadata opslaan (voor P-Y plots) ---
            py_meta.append({
                'i_local': i_local,
                'depth_m': z_node,
                'z_eval_m': z_eval,
                'Ltrib_m': Ltrib,
                'mat_tag': mat_use,
                'soilType_id': soilType_id,
                'soil_label': soil_label_nl,
                'pult_kN': pult_element,
                'y50_m': y50,
                'rule': str(lateral_rule_l).lower()
            })
        return {'node_tags': node_tags, 'ele_tags': ele_tags, 'top_node': node_tags[0], 'x0': float(x0), 'py_meta': py_meta}
    # Systeem opbouwen
    piles_built = []

    piles_built.append(build_single_pile(0, 0.0, top_condition_global))
    load_node = piles_built[0]['top_node']

    # Belastingen
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    ops.load(load_node, H_load, 0.0, M_head)
    # Analyse
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormDispIncr', tol, iters)
    ops.algorithm('Newton')
    ops.integrator('LoadControl', 1.0 / n_steps)
    ops.analysis('Static')
    # --- Force-Displacement history (D-Pile Group: Pile Force-Displacement chart) ---
    fd_steps = []
    fd_ux_top = []
    fd_Fx_top = []

    for _step in range(1, n_steps + 1):
        ok = ops.analyze(1)
        if ok != 0:
            print(f'Analyse faalde bij stap {_step} (code={ok}).')
            break
        # representatieve paal 0 (of eerste paal in groep)
        _rep_top = piles_built[0]['top_node']
        _rep_ele0 = piles_built[0]['ele_tags'][0]
        try:
            _ux = float(ops.nodeDisp(_rep_top, 1))
        except Exception:
            _ux = float('nan')
        try:
            _f0 = ops.eleForce(_rep_ele0)
            _Fx1 = float(_f0[0])
            _Fx2 = float(_f0[1])
            # Neem de component met de grootste absolute waarde als horizontale paalkopkracht
            _Fx = _Fx1 if abs(_Fx1) >= abs(_Fx2) else _Fx2
        except Exception:
            _Fx = float('nan')
        fd_steps.append(_step)
        fd_ux_top.append(_ux)
        fd_Fx_top.append(_Fx)

    # Resultaten voor representatieve paal 0
    rep = piles_built[0]
    node_tags = rep['node_tags']
    ele_tags = rep['ele_tags']
    depths = np.array([i * dz for i in range(n_el + 1)], dtype=float)
    ux = np.array([ops.nodeDisp(node_tags[i], 1) for i in range(n_el + 1)], dtype=float)
    Mz = np.zeros(n_el + 1, dtype=float)
    for i in range(n_el):
        f = ops.eleForce(ele_tags[i])
        Mz[i] = -f[2]
    if n_el >= 1:
        f_last = ops.eleForce(ele_tags[-1])
        Mz[n_el] = -f_last[5]
    Vy = np.zeros(n_el + 1, dtype=float)
    if n_el >= 1:
        Vy[0] = -(Mz[1] - Mz[0]) / dz
    for k in range(1, n_el):
        Vy[k] = -(Mz[k + 1] - Mz[k - 1]) / (2.0 * dz)
    Vy[n_el] = -(Mz[n_el] - Mz[n_el - 1]) / dz
    # Plotten
    Mz_plot = Mz.copy(); Vy_plot = Vy.copy()
    if True:
        Mz_plot *= -1.0
        Vy_plot *= -1.0
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['depth_m', 'ux_m', 'V_kN', 'Mz_kNm'])
        for i in range(n_el + 1):
            w.writerow([depths[i], ux[i], Vy_plot[i], Mz_plot[i]])

    # --- Extra output: Pile Force-Displacement (paal 0) ---
    fd_csv_path = os.path.join(out_dir, 'pile_force_displacement_pile0.csv')
    with open(fd_csv_path, 'w', newline='') as ffd:
        wfd = csv.writer(ffd)
        wfd.writerow(['step', 'ux_top_m', 'Fx_top_kN'])
        for s, uxi, Fi in zip(fd_steps, fd_ux_top, fd_Fx_top):
            wfd.writerow([s, uxi, Fi])

    if do_plot:
        # Zorg voor positieve richting (zoals in D-Pile Group charts)
        _sign = 1.0
        if len(fd_ux_top) > 0 and len(fd_Fx_top) > 0:
            if (fd_ux_top[-1] * fd_Fx_top[-1]) < 0:
                _sign = -1.0
        fig_fd, ax_fd = plt.subplots(1, 1, figsize=(6.5, 4.5))
        ax_fd.plot([u*1000.0 for u in fd_ux_top], [ _sign*f for f in fd_Fx_top], 'k-', lw=2)
        ax_fd.set_xlabel('Verplaatsing paalkop X [mm]')
        ax_fd.set_ylabel('Kracht paalkop X [kN]')
        ax_fd.set_title('Pile Force-Displacement (paal 0)')
        ax_fd.grid(True, alpha=0.3)
        plt.tight_layout()
        fd_png_path = os.path.join(out_dir, 'pile_force_displacement_pile0.png')
        plt.savefig(fd_png_path, dpi=200)
        plt.show()
        plt.close(fig_fd)
    if do_plot:
        z_plot = -depths
        fig, ax = plt.subplots(1, 3, figsize=(14, 5))
        ax[0].plot(ux * 1000.0, z_plot, 'b-', lw=2)
        ax[0].set_xlabel('Verplaatsing X [mm]')
        ax[0].set_ylabel('Diepte [m]')
        ax[0].set_title('Laterale verplaatsing')
        ax[0].grid(True, alpha=0.3)
        ax[1].plot(Mz_plot, z_plot, 'r-', lw=2)
        ax[1].set_xlabel('Moment Z [kNm]')
        ax[1].set_title('Buigend moment')
        ax[1].grid(True, alpha=0.3)
        ax[2].plot(Vy_plot, z_plot, 'g-', lw=2)
        ax[2].set_xlabel('Dwarskracht X [kN]')
        ax[2].set_title('Dwarskracht')
        ax[2].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.show()
        plt.close(fig)
    # ============================================================
    # P-Y plots
    # ============================================================
    py_cfg = config.get('output', {}).get('py_plots', {})
    if py_cfg.get('enabled', False):
        py_list = piles_built[0].get('py_meta', [])
        if isinstance(py_list, list) and len(py_list) > 0:
            py_dir = os.path.join(out_dir, str(py_cfg.get('folder', 'PY_Plots')))
            os.makedirs(py_dir, exist_ok=True)
            z_avail = sorted({float(it.get('z_eval_m', it.get('depth_m', 0.0))) for it in py_list})
            if len(z_avail) == 0:
                z_avail = [0.0]
            z_top_avail = min(z_avail)
            L_total = float(config.get('pile', {}).get('length', 0.0))
            h_above = float(config.get('pile', {}).get('head_above_ground', 0.0))
            z_tip_target = max(L_total - h_above, 0.0)
            levels_mode = str(py_cfg.get('levels_mode', 'auto')).lower()
            max_plots = int(py_cfg.get('max_plots', 8))
            if max_plots < 2:
                max_plots = 2
            levels_req = []
            spacing = py_cfg.get('spacing_m', None)
            if levels_mode in ['auto', 'multispring', 'default']:
                spacing_val = None
                if spacing is not None:
                    try:
                        spacing_val = float(spacing)
                    except Exception:
                        spacing_val = None
                if spacing_val is not None and spacing_val > 0:
                    targets = []
                    t = 0.0
                    while t < z_tip_target - 1e-9:
                        targets.append(t)
                        t += spacing_val
                    targets.append(z_tip_target)
                    if len(targets) > max_plots:
                        idxs = np.linspace(0, len(targets)-1, max_plots).round().astype(int)
                        targets = [targets[i] for i in idxs]
                else:
                    targets = list(np.linspace(0.0, z_tip_target, max_plots))
                if len(targets) >= 1:
                    targets[0] = z_top_avail
                if len(targets) >= 2:
                    targets[-1] = z_tip_target
                for zt in targets:
                    z_near = min(z_avail, key=lambda z: abs(z - zt))
                    if z_near not in levels_req:
                        levels_req.append(z_near)
                z_tip_near = min(z_avail, key=lambda z: abs(z - z_tip_target))
                if z_tip_near not in levels_req:
                    levels_req.append(z_tip_near)
                levels_req = sorted(levels_req)
                if len(levels_req) > max_plots:
                    idxs = np.linspace(0, len(levels_req)-1, max_plots).round().astype(int)
                    levels_req = [levels_req[i] for i in idxs]
            else:
                levels_in = py_cfg.get('levels_m', [0.02])
                if isinstance(levels_in, (int, float)):
                    levels_in = [float(levels_in)]
                for zt in [float(z) for z in levels_in]:
                    z_near = min(z_avail, key=lambda z: abs(z - zt))
                    if z_near not in levels_req:
                        levels_req.append(z_near)
                z_tip_near = min(z_avail, key=lambda z: abs(z - z_tip_target))
                if z_tip_near not in levels_req:
                    levels_req.append(z_tip_near)
                levels_req = sorted(levels_req)
            def _item_near_level(z_level):
                return min(py_list, key=lambda it: abs(float(it.get('z_eval_m', it.get('depth_m', 0.0))) - float(z_level)))
            def _sample_p(mat_tag, y):
                ops.testUniaxialMaterial(int(mat_tag))
                ops.setStrain(0.0)
                _ = ops.getStress()
                ops.setStrain(float(y))
                return float(ops.getStress())
            atanh_half = 0.5 * math.log((1.0 + 0.5) / (1.0 - 0.5))
            for z_sel in levels_req:
                it = _item_near_level(z_sel)
                mat_tag = int(it['mat_tag'])
                soil_id = int(it.get('soilType_id', 0))
                soil_label = str(it.get('soil_label', 'klei'))
                y50 = float(it.get('y50_m', 0.0))
                Ltrib = float(it.get('Ltrib_m', 1.0))
                z_plot = float(it.get('z_eval_m', it.get('depth_m', 0.0)))
                rule = str(it.get('rule', 'api_static'))
                y_pts = [0.0]
                if bool(py_cfg.get('multilinear', True)):
                    if soil_id == 1:
                        # clay
                        if 'cyclic' in rule:
                            y_pts += [0.1*y50, 0.3*y50, 1.0*y50, 3.0*y50]
                        else:
                            y_pts += [0.1*y50, 0.3*y50, 1.0*y50, 3.0*y50, 8.0*y50]
                    else:
                        # sand
                        y_max = y50 / atanh_half if atanh_half > 0 else (8.0*y50)
                        y_pts += [0.25*y_max, 0.5*y_max, 1.0*y_max, 1.5*y_max, 2.5*y_max]
                else:
                    # gladde sampling
                    max_y_plot = 3.0*y50 if (soil_id == 1 and 'cyclic' in rule) else max(8.0*y50, 1e-4)
                    y_pts = list(np.linspace(0.0, max_y_plot, 120))
                p_pts = [_sample_p(mat_tag, y) for y in y_pts]
                if bool(py_cfg.get('per_unit_length', True)) and Ltrib > 0:
                    p_pts = [p / Ltrib for p in p_pts]
                p_label = 'P [kN/m]' if bool(py_cfg.get('per_unit_length', True)) else 'P [kN]'
                if len(p_pts) > 1 and p_pts[-1] < 0:
                    p_pts = [-p for p in p_pts]

                # CSV export P-Y
                title_rule = 'cyclic' if 'cyclic' in rule else 'static'
                csv_fname = f"PY_pile0_level_{z_plot:06.2f}m_{soil_label}.csv".replace(' ', '_')
                csv_full = os.path.join(py_dir, csv_fname)
                with open(csv_full, 'w', newline='') as fcsv:
                    wcsv = csv.writer(fcsv, delimiter=';')
                    wcsv.writerow(['Y [m]', p_label])
                    for yy, pp in zip(y_pts, p_pts):
                        yy_s = f"{yy:.4f}".replace('.', ',')
                        pp_s = f"{pp:.1f}".replace('.', ',')
                        wcsv.writerow([yy_s, pp_s])

                fig_py, ax_py = plt.subplots(1, 1, figsize=(6.5, 4.5))
                ax_py.plot([yy*1000.0 for yy in y_pts], p_pts,
                           color='#1f77b4', lw=2,
                           marker='o' if bool(py_cfg.get('multilinear', True)) else None, ms=4)
                ax_py.set_xlabel("X'' [mm]")
                ax_py.set_ylabel(p_label)
                ax_py.set_title(f"P-Y curve ({soil_label}, {title_rule}, niveau={z_plot:.2f} m)")
                ax_py.grid(True, alpha=0.3)
                plt.tight_layout()
                dpi = int(py_cfg.get('dpi', 200))
                fname = f"PY_pile0_level_{z_plot:06.2f}m_{soil_label}.png".replace(' ', '_')
                plt.savefig(os.path.join(py_dir, fname), dpi=dpi)
                if bool(py_cfg.get('show', False)):
                    plt.show()
                plt.close(fig_py)

    return {
        'depths_m': depths,
        'ux_m': ux,
        'Vy_kN': Vy,
        'Mz_kNm': Mz,
        'csv_path': csv_path,
        'png_path': png_path,
    }
