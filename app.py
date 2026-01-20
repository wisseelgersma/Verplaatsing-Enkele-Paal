# -*- coding: utf-8 -*-
"""D-Pile – Single pile (Streamlit)

Wijzigingen:
- Verticale verplaatsing is standaard vastgezet (geen UI-optie) voor stabiliteit.
- Validatie bodemlagen: waarschuwing bij dubbele/niet-oplopende laagtoppen.
- Slimme defaults: nieuwe laagtop = laatste + 1.0 m.
- Export/Import: JSON via sidebar.
- Nettere helpteksten bij parameters.

Start:
  streamlit run app.py
"""

import os
import json
import copy
import tempfile
from pathlib import Path

import streamlit as st

os.environ.setdefault("MPLBACKEND", "Agg")

import dpile_model


# ---------------------------
# Helpers
# ---------------------------

def _float_list_from_text(txt: str):
    if txt is None:
        return []
    txt = str(txt)
    parts = [p.strip() for p in txt.replace('\n', ',').replace(';', ',').split(',')]
    out = []
    for p in parts:
        if not p:
            continue
        try:
            out.append(float(p))
        except Exception:
            pass
    return out


def _safe_get(dct, keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _validate_layers(layers):
    """Return (warnings:list[str], errors:list[str])."""
    warnings = []
    errors = []
    if not isinstance(layers, list) or len(layers) == 0:
        errors.append("Er is geen enkele laag aanwezig.")
        return warnings, errors

    tops = []
    for i, lay in enumerate(layers, start=1):
        try:
            tops.append(float(lay.get('top_m', 0.0)))
        except Exception:
            errors.append(f"Laag {i}: laagtop is geen geldig getal.")

    if any(t < 0 for t in tops):
        warnings.append("Er zijn laagtoppen < 0 m. Gebruik bij voorkeur diepte onder maaiveld (>= 0).")

    if len(set([round(t, 6) for t in tops])) != len(tops):
        warnings.append("Er zijn dubbele laagtoppen. Dit kan tot onduidelijke laagindeling leiden.")

    if tops != sorted(tops):
        warnings.append("Laagtoppen zijn niet oplopend. Klik op 'Sorteer op laagtop' om dit te corrigeren.")

    return warnings, errors


def _default_layer(top_m=0.0):
    return {
        'top_m': float(top_m),
        'type': 'clay',
        'dry_unit_weight': 16.0,
        'wet_unit_weight': 16.0,
        'lateral_rule': 'api_static',
        'su': 30.0,
        'epsilon50': 0.010,
        'J': 0.25,
        'phi_deg': None,
        'enabled': True,
    }


SOIL_TYPE_LABELS = {"clay": "Klei", "sand": "Zand"}
SOIL_TYPE_TO_KEY = {v: k for k, v in SOIL_TYPE_LABELS.items()}

API_RULE_LABELS = {"api_static": "API Statisch", "api_cyclic": "API Cyclisch"}
API_RULE_TO_KEY = {v: k for k, v in API_RULE_LABELS.items()}

PILE_TYPE_LABELS = {
    "steel": "Staal",
    "concrete_round": "Beton Rond",
    "concrete_square": "Beton Vierkant",
    "user_defined": "Eigen (EA/EI)",
}
PILE_TYPE_TO_KEY = {v: k for k, v in PILE_TYPE_LABELS.items()}

RECOMMENDED_E = {
    "steel": 2.1e8,
    "concrete_round": 2.2e7,
    "concrete_square": 2.2e7,
    "user_defined": 2.2e7,
}


def _build_cfg_from_state(base_cfg):
    soil = {
        'use_layers': True,
        'gwt_depth_m': float(st.session_state.get('gwt_depth', -1.0)),
        'top_py_eval_depth_m': None if str(st.session_state.get('top_py_eval', '')).strip() == '' else float(st.session_state.get('top_py_eval')),
        'print_layers': False,
        'layers': st.session_state.get('layers_list', [_default_layer(0.0)]),
    }

    ptype_label = st.session_state.get('ptype_label', PILE_TYPE_LABELS.get(str(_safe_get(base_cfg,['pile','type'],'concrete_round')).lower(), 'Beton Rond'))
    ptype = PILE_TYPE_TO_KEY.get(ptype_label, 'concrete_round')

    pile = {
        'type': ptype,
        'length': float(st.session_state.get('pile_length', float(_safe_get(base_cfg,['pile','length'],12.8)))),
        'head_above_ground': float(st.session_state.get('head_above', float(_safe_get(base_cfg,['pile','head_above_ground'],0.0)))),
        'outer_diameter': float(st.session_state.get('outer_d', float(_safe_get(base_cfg,['pile','outer_diameter'],0.3)))),
        'wall_thickness': float(st.session_state.get('wall_t', float(_safe_get(base_cfg,['pile','wall_thickness'],0.01)))),
        'width': float(st.session_state.get('width_b', float(_safe_get(base_cfg,['pile','width'],0.3)))),
        'E': float(st.session_state.get('E_input', float(_safe_get(base_cfg,['pile','E'], RECOMMENDED_E.get(ptype, 2.2e7))))),
        'override_EA_EI': bool(st.session_state.get('override_EA_EI', False) or ptype == 'user_defined'),
        'EA': None,
        'EI': None,
    }

    if pile['override_EA_EI']:
        pile['EA'] = float(st.session_state.get('EA_in', float(_safe_get(base_cfg,['pile','EA'],0.0) or 0.0)))
        pile['EI'] = float(st.session_state.get('EI_in', float(_safe_get(base_cfg,['pile','EI'],0.0) or 0.0)))

    mesh = {'n_elements': int(st.session_state.get('n_el', int(_safe_get(base_cfg,['mesh','n_elements'],500))))}

    load = {
        'lateral_head_load': float(st.session_state.get('H', float(_safe_get(base_cfg,['load','lateral_head_load'],80.0)))),
        'head_moment_z': float(st.session_state.get('M', float(_safe_get(base_cfg,['load','head_moment_z'],0.0)))),
    }

    top_label = st.session_state.get('top_cond', 'Vrij')
    boundary = {
        'top_condition': 'free' if top_label == 'Vrij' else 'fixed',
        'fix_vertical_dof': True,
    }

    analysis = {
        'n_steps': int(st.session_state.get('n_steps', int(_safe_get(base_cfg,['analysis','n_steps'],100)))),
        'tolerance': float(st.session_state.get('tol', float(_safe_get(base_cfg,['analysis','tolerance'],1e-6)))),
        'max_iters': int(st.session_state.get('max_iters', int(_safe_get(base_cfg,['analysis','max_iters'],25)))),
    }

    py_mode_label = st.session_state.get('py_mode', 'Handmatig')
    output = {
        'folder': 'Resultaten',
        'csv': 'results.csv',
        'png': 'results.png',
        'plot': bool(st.session_state.get('do_plot', True)),
        'show_plots': False,
        'py_plots': {
            'enabled': bool(st.session_state.get('py_enabled', True)),
            'folder': 'PY_Plots',
            'levels_mode': 'auto' if py_mode_label == 'Automatisch' else 'manual',
            'levels_m': _float_list_from_text(st.session_state.get('levels_txt', '0.5, 1.5')),
            'spacing_m': None,
            'max_plots': int(st.session_state.get('max_plots', 8)),
            'per_unit_length': bool(st.session_state.get('per_unit', True)),
            'multilinear': True,
            'dpi': int(st.session_state.get('dpi', 200)),
            'show': False,
        }
    }

    return {
        'soil': soil,
        'pile': pile,
        'mesh': mesh,
        'load': load,
        'boundary': boundary,
        'analysis': analysis,
        'output': output,
    }


def _apply_cfg_to_state(cfg, base_cfg):
    layers = _safe_get(cfg, ['soil','layers'], None)
    st.session_state.layers_list = layers if isinstance(layers, list) and len(layers)>0 else [_default_layer(0.0)]

    st.session_state.gwt_depth = float(_safe_get(cfg, ['soil','gwt_depth_m'], -1.0) or -1.0)
    tpy = _safe_get(cfg, ['soil','top_py_eval_depth_m'], None)
    st.session_state.top_py_eval = '' if tpy is None else str(tpy)

    ptype = str(_safe_get(cfg, ['pile','type'], _safe_get(base_cfg,['pile','type'],'concrete_round'))).lower()
    st.session_state.ptype_label = PILE_TYPE_LABELS.get(ptype, 'Beton Rond')
    st.session_state.pile_length = float(_safe_get(cfg, ['pile','length'], _safe_get(base_cfg,['pile','length'],12.8)))
    st.session_state.head_above = float(_safe_get(cfg, ['pile','head_above_ground'], _safe_get(base_cfg,['pile','head_above_ground'],0.0)))
    st.session_state.outer_d = float(_safe_get(cfg, ['pile','outer_diameter'], _safe_get(base_cfg,['pile','outer_diameter'],0.3)))
    st.session_state.wall_t = float(_safe_get(cfg, ['pile','wall_thickness'], _safe_get(base_cfg,['pile','wall_thickness'],0.01)))
    st.session_state.width_b = float(_safe_get(cfg, ['pile','width'], _safe_get(base_cfg,['pile','width'],0.3)))
    st.session_state.E_input = float(_safe_get(cfg, ['pile','E'], RECOMMENDED_E.get(ptype, 2.2e7)))

    st.session_state.override_EA_EI = bool(_safe_get(cfg, ['pile','override_EA_EI'], False) or ptype=='user_defined')
    st.session_state.EA_in = float(_safe_get(cfg, ['pile','EA'], _safe_get(base_cfg,['pile','EA'],0.0) or 0.0))
    st.session_state.EI_in = float(_safe_get(cfg, ['pile','EI'], _safe_get(base_cfg,['pile','EI'],0.0) or 0.0))

    st.session_state.n_el = int(_safe_get(cfg, ['mesh','n_elements'], _safe_get(base_cfg,['mesh','n_elements'],500)))

    st.session_state.H = float(_safe_get(cfg, ['load','lateral_head_load'], _safe_get(base_cfg,['load','lateral_head_load'],80.0)))
    st.session_state.M = float(_safe_get(cfg, ['load','head_moment_z'], _safe_get(base_cfg,['load','head_moment_z'],0.0)))

    top_cond = str(_safe_get(cfg, ['boundary','top_condition'], _safe_get(base_cfg,['boundary','top_condition'],'free'))).lower()
    st.session_state.top_cond = 'Vrij' if top_cond == 'free' else 'Ingeklemd'

    st.session_state.n_steps = int(_safe_get(cfg, ['analysis','n_steps'], _safe_get(base_cfg,['analysis','n_steps'],100)))
    st.session_state.tol = float(_safe_get(cfg, ['analysis','tolerance'], _safe_get(base_cfg,['analysis','tolerance'],1e-6)))
    st.session_state.max_iters = int(_safe_get(cfg, ['analysis','max_iters'], _safe_get(base_cfg,['analysis','max_iters'],25)))

    out = _safe_get(cfg, ['output'], {})
    st.session_state.do_plot = bool(out.get('plot', True))
    py = out.get('py_plots', {}) if isinstance(out, dict) else {}
    st.session_state.py_enabled = bool(py.get('enabled', True))
    st.session_state.max_plots = int(py.get('max_plots', 8))
    st.session_state.py_mode = 'Automatisch' if str(py.get('levels_mode', 'manual')).lower() == 'auto' else 'Handmatig'
    st.session_state.levels_txt = ', '.join(str(x) for x in (py.get('levels_m', [0.5,1.5]) or [0.5,1.5]))
    st.session_state.per_unit = bool(py.get('per_unit_length', True))
    st.session_state.dpi = int(py.get('dpi', 200))


# ---------------------------
# Page setup
# ---------------------------

st.set_page_config(page_title="Verplaatsing Enkele paal", layout="wide")
st.title("Verplaatsing Enkele paal")

# Patch plt.show() zodat headless/web niet blokkeert
try:
    if hasattr(dpile_model, 'plt'):
        dpile_model.plt.show = lambda *args, **kwargs: None
except Exception:
    pass

base_cfg = copy.deepcopy(dpile_model.CONFIG)

# ---------------------------
# Sidebar: Import/Export
# ---------------------------
# ---------------------------
# Configuratie (rechtsboven)
# ---------------------------

# Kleine knop rechtsboven met Import/Export
_top_left, _top_right = st.columns([0.82, 0.18])
with _top_right:
    if hasattr(st, 'popover'):
        with st.popover("⚙️ Configuratie"):
            upl = st.file_uploader("Importeer JSON", type=['json'], key='cfg_uploader')
            if upl is not None:
                try:
                    cfg_in = json.loads(upl.read().decode('utf-8'))
                    if not isinstance(cfg_in, dict):
                        raise ValueError('JSON moet een object/dict zijn.')
                    _apply_cfg_to_state(cfg_in, base_cfg)
                    st.success('Config geïmporteerd. Pagina wordt vernieuwd…')
                    st.rerun()
                except Exception as e:
                    st.error(f"Import mislukt: {e}")

            cfg_now = _build_cfg_from_state(base_cfg)
            st.download_button(
                "Download JSON",
                data=json.dumps(cfg_now, indent=2, ensure_ascii=False),
                file_name="dpile_config.json",
                mime="application/json",
            )
            st.caption("Bewaar scenario's door je invoer als JSON te downloaden.")
    else:
        # Fallback voor oudere Streamlit versies
        with st.expander("⚙️ Configuratie"):
            upl = st.file_uploader("Importeer JSON", type=['json'], key='cfg_uploader')
            if upl is not None:
                try:
                    cfg_in = json.loads(upl.read().decode('utf-8'))
                    if not isinstance(cfg_in, dict):
                        raise ValueError('JSON moet een object/dict zijn.')
                    _apply_cfg_to_state(cfg_in, base_cfg)
                    st.success('Config geïmporteerd. Pagina wordt vernieuwd…')
                    st.rerun()
                except Exception as e:
                    st.error(f"Import mislukt: {e}")

            cfg_now = _build_cfg_from_state(base_cfg)
            st.download_button(
                "Download JSON",
                data=json.dumps(cfg_now, indent=2, ensure_ascii=False),
                file_name="dpile_config.json",
                mime="application/json",
            )
            st.caption("Bewaar scenario's door je invoer als JSON te downloaden.")

soil_tab, pile_tab, load_tab, boundary_tab, analysis_tab, output_tab = st.tabs(
    ["Bodem", "Paal", "Belasting", "Randvoorwaarden", "Analyse", "Output"]
)

# ---------------------
# Bodem (altijd lagen)
# ---------------------
with soil_tab:
    st.subheader("Bodemopbouw")
    c_soil = base_cfg['soil']

    # Init defaults
    if 'layers_list' not in st.session_state:
        layers_init = c_soil.get('layers', None)
        st.session_state.layers_list = [dict(x) for x in layers_init] if isinstance(layers_init, list) and len(layers_init)>0 else [_default_layer(0.0)]
    if 'gwt_depth' not in st.session_state:
        st.session_state.gwt_depth = float(c_soil.get('gwt_depth_m', -1.0) or -1.0)
    if 'top_py_eval' not in st.session_state:
        st.session_state.top_py_eval = '' if c_soil.get('top_py_eval_depth_m', None) is None else str(c_soil.get('top_py_eval_depth_m'))

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown("**Lagen**")
        st.caption("Voeg lagen toe met '+ Laag toevoegen'. De volgorde wordt bepaald door de laagtop.")
    with col2:
        st.number_input(
            "Grondwaterstand (m onder maaiveld)",
            value=float(st.session_state.gwt_depth),
            step=0.1,
            help="Gebruik -1 voor: geen grondwaterstand.",
            key='gwt_depth'
        )
    with col3:
        st.text_input(
            "Evaluatiediepte bovenste p-y veer (m) – leeg = automatisch",
            value=str(st.session_state.top_py_eval),
            help="Laat leeg voor standaardwaarde uit de rekenkern.",
            key='top_py_eval'
        )

    c_add, c_sort = st.columns([1, 1])
    with c_add:
        if st.button("+ Laag toevoegen"):
            last_top = float(st.session_state.layers_list[-1].get('top_m', 0.0)) if st.session_state.layers_list else 0.0
            st.session_state.layers_list.append(_default_layer(round(last_top + 1.0, 2)))
            st.rerun()
    with c_sort:
        if st.button("Sorteer op laagtop"):
            st.session_state.layers_list = sorted(st.session_state.layers_list, key=lambda d: float(d.get('top_m', 0.0)))
            st.rerun()
    warnings, errors = _validate_layers(st.session_state.layers_list)
    for w in warnings:
        st.warning(w)
    for e in errors:
        st.error(e)

    new_layers = []
    for i, lay in enumerate(st.session_state.layers_list, start=1):
        with st.expander(f"Laag {i}", expanded=(i <= 2)):
            r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns([1, 1, 1, 1, 0.25])

            with r1c1:
                top_m = st.number_input(
                    "Laagtop (m)",
                    value=float(lay.get('top_m', 0.0)),
                    step=0.1,
                    help="Diepte van de bovenzijde van de laag onder maaiveld.",
                    key=f"lay_top_{i}",
                )

            with r1c2:
                soil_label = SOIL_TYPE_LABELS.get(str(lay.get('type', 'clay')).lower(), 'Klei')
                soil_label = st.selectbox(
                    "Grondsoort",
                    options=list(SOIL_TYPE_TO_KEY.keys()),
                    index=list(SOIL_TYPE_TO_KEY.keys()).index(soil_label),
                    help="Kies Klei (ongedraineerd) of Zand (gedraineerd).",
                    key=f"lay_type_{i}",
                )
                ltype = SOIL_TYPE_TO_KEY[soil_label]

                api_label = API_RULE_LABELS.get(str(lay.get('lateral_rule', 'api_static')).lower(), 'API Statisch')
                api_label = st.selectbox(
                    "P-y methode",
                    options=list(API_RULE_TO_KEY.keys()),
                    index=list(API_RULE_TO_KEY.keys()).index(api_label),
                    help="API Statisch/Cyclisch bepaalt de vorm/sterkte van de p-y curve.",
                    key=f"lay_api_{i}",
                )
                lateral = API_RULE_TO_KEY[api_label]

            with r1c3:
                gd = st.number_input(
                    "Volumiek gewicht droog (kN/m³)",
                    value=float(lay.get('dry_unit_weight', 16.0)),
                    step=0.1,
                    help="Gebruikt voor effectieve spanning boven grondwater.",
                    key=f"lay_gd_{i}",
                )
                gw = st.number_input(
                    "Volumiek gewicht nat (kN/m³)",
                    value=float(lay.get('wet_unit_weight', 16.0)),
                    step=0.1,
                    help="Gebruikt onder grondwater.",
                    key=f"lay_gw_{i}",
                )

            with r1c4:
                if ltype == 'clay':
                    su = st.number_input(
                        "Su (kN/m²)",
                        value=float(lay.get('su', 30.0) or 30.0),
                        step=1.0,
                        help="Ongedraineerde schuifsterkte (typisch 5–400 kN/m² afhankelijk van klei).",
                        key=f"lay_su_{i}",
                    )
                    # ε50 (-)
                    eps = st.number_input(
                        "ε50 (-)",
                        value=float(lay.get('epsilon50', 0.010) or 0.010),
                        step=0.001,
                        format="%.3f",
                        help="""Vervormingsparameter voor klei (afhankelijk van Su).

```text
Richtwaarden:
su [kN/m²]  ->  ε50 [-]
5   – 25    ->  0.020
25  – 50    ->  0.010
50  – 100   ->  0.007
100 – 200   ->  0.005
200 – 400   ->  0.004
```
""",

                        key=f"lay_eps_{i}",
                    )

                    Jv = st.number_input(
                        "J (-)",
                        value=float(lay.get('J', 0.25) or 0.25),
                        step=0.01,
                        help="Dimensionloze parameter voor klei in API-formulering (vaak 0.25 als default).",
                        key=f"lay_J_{i}",
                    )
                    phi = None
                else:
                    phi = st.number_input(
                        "Wrijvingshoek φ (°)",
                        value=float(lay.get('phi_deg', 30.0) or 30.0),
                        step=0.5,
                        help="Interne wrijvingshoek van zand (typisch ~28–40°).",
                        key=f"lay_phi_{i}",
                    )
                    su = None
                    eps = None
                    Jv = float(lay.get('J', 0.25) or 0.25)

            with r1c5:
                if st.button("🗑️", key=f"lay_rm_{i}"):
                    st.session_state.layers_list.pop(i-1)
                    if len(st.session_state.layers_list) == 0:
                        st.session_state.layers_list = [_default_layer(0.0)]
                    st.rerun()

            new_layers.append({
                'enabled': True,
                'top_m': float(top_m),
                'type': str(ltype),
                'dry_unit_weight': float(gd),
                'wet_unit_weight': float(gw),
                'lateral_rule': str(lateral),
                'su': None if su is None else float(su),
                'epsilon50': None if eps is None else float(eps),
                'J': float(Jv),
                'phi_deg': None if phi is None else float(phi),
            })

    st.session_state.layers_list = new_layers


# --------
# Paal
# --------
with pile_tab:
    st.subheader("Paal")
    c_pile = base_cfg['pile']

    if 'ptype_label' not in st.session_state:
        st.session_state.ptype_label = PILE_TYPE_LABELS.get(str(c_pile.get('type','concrete_round')).lower(), 'Beton Rond')
    if 'pile_length' not in st.session_state:
        st.session_state.pile_length = float(c_pile.get('length', 12.8))
    if 'head_above' not in st.session_state:
        st.session_state.head_above = float(c_pile.get('head_above_ground', 0.0))
    if 'E_input' not in st.session_state:
        st.session_state.E_input = float(c_pile.get('E', 2.2e7))
    if 'override_EA_EI' not in st.session_state:
        st.session_state.override_EA_EI = bool(c_pile.get('override_EA_EI', False))
    if 'EA_in' not in st.session_state:
        st.session_state.EA_in = float(c_pile.get('EA', 0.0) or 0.0)
    if 'EI_in' not in st.session_state:
        st.session_state.EI_in = float(c_pile.get('EI', 0.0) or 0.0)
    if 'outer_d' not in st.session_state:
        st.session_state.outer_d = float(c_pile.get('outer_diameter', 0.3))
    if 'wall_t' not in st.session_state:
        st.session_state.wall_t = float(c_pile.get('wall_thickness', 0.01))
    if 'width_b' not in st.session_state:
        st.session_state.width_b = float(c_pile.get('width', 0.3))

    def _on_pile_type_change():
        ptype = PILE_TYPE_TO_KEY.get(st.session_state.ptype_label, 'concrete_round')
        st.session_state.E_input = float(RECOMMENDED_E.get(ptype, 2.2e7))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.selectbox(
            "Paaltype",
            options=list(PILE_TYPE_TO_KEY.keys()),
            index=list(PILE_TYPE_TO_KEY.keys()).index(st.session_state.ptype_label),
            key='ptype_label',
            on_change=_on_pile_type_change,
            help="Kies het type paal. Bij wijzigen wordt E teruggezet naar een aanbevolen waarde.",
        )
        ptype = PILE_TYPE_TO_KEY[st.session_state.ptype_label]

        st.number_input("Paallengte (m)", value=float(st.session_state.pile_length), step=0.1, key='pile_length')
        st.number_input("Paalkop boven maaiveld (m)", value=float(st.session_state.head_above), step=0.1, key='head_above')

    with col2:
        st.number_input(
            "Elasticiteitsmodulus E (kN/m²)",
            value=float(st.session_state.E_input),
            step=1e6,
            format="%.3e",
            help=f"Aanbevolen voor {st.session_state.ptype_label}: {RECOMMENDED_E.get(ptype, 2.2e7):.3e}",
            key='E_input'
        )

        if ptype == 'user_defined':
            st.session_state.override_EA_EI = True
            st.caption("Eigen (EA/EI): EA en EI zijn verplicht.")
        else:
            st.checkbox("EA/EI overschrijven", value=bool(st.session_state.override_EA_EI), key='override_EA_EI')

    with col3:
        if ptype == 'steel':
            st.number_input("Buitendiameter D (m)", value=float(st.session_state.outer_d), step=0.01, key='outer_d')
            st.number_input("Wanddikte t (m)", value=float(st.session_state.wall_t), step=0.001, key='wall_t')
        elif ptype == 'concrete_round':
            st.number_input("Diameter D (m)", value=float(st.session_state.outer_d), step=0.01, key='outer_d')
        elif ptype == 'concrete_square':
            st.number_input("Breedte b (m)", value=float(st.session_state.width_b), step=0.01, key='width_b')
            st.number_input("Equivalent diameter D voor p-y (m)", value=float(st.session_state.outer_d), step=0.01, key='outer_d')
        else:
            st.number_input("Equivalent diameter D voor p-y (m)", value=float(st.session_state.outer_d), step=0.01, key='outer_d')

    if bool(st.session_state.override_EA_EI):
        c4, c5 = st.columns(2)
        with c4:
            st.number_input("Axiale stijfheid EA (kN)", value=float(st.session_state.EA_in), step=1e4, format="%.3e", key='EA_in')
        with c5:
            st.number_input("Buigstijfheid EI (kN·m²)", value=float(st.session_state.EI_in), step=1e4, format="%.3e", key='EI_in')


# ---------
# Belasting
# ---------
with load_tab:
    st.subheader("Belasting")
    c_load = base_cfg['load']
    if 'H' not in st.session_state:
        st.session_state.H = float(c_load.get('lateral_head_load', 80.0))
    if 'M' not in st.session_state:
        st.session_state.M = float(c_load.get('head_moment_z', 0.0))

    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Horizontale koplast H (kN)", value=float(st.session_state.H), step=1.0, key='H')
    with c2:
        st.number_input("Kopmoment Mz (kNm)", value=float(st.session_state.M), step=1.0, key='M')


# ----------------
# Randvoorwaarden
# ----------------
with boundary_tab:
    st.subheader("Randvoorwaarden")
    c_bnd = base_cfg['boundary']
    if 'top_cond' not in st.session_state:
        st.session_state.top_cond = 'Vrij' if str(c_bnd.get('top_condition','free')).lower() == 'free' else 'Ingeklemd'

    st.selectbox(
        "Kopconditie",
        options=['Vrij', 'Ingeklemd'],
        index=['Vrij','Ingeklemd'].index(st.session_state.top_cond),
        key='top_cond'
    )
# ------
# Analyse
# ------
with analysis_tab:
    st.subheader("Analyse")
    c_an = base_cfg['analysis']
    if 'n_steps' not in st.session_state:
        st.session_state.n_steps = int(c_an.get('n_steps', 100))
    if 'tol' not in st.session_state:
        st.session_state.tol = float(c_an.get('tolerance', 1e-6))
    if 'max_iters' not in st.session_state:
        st.session_state.max_iters = int(c_an.get('max_iters', 25))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Aantal stappen", value=int(st.session_state.n_steps), step=10, min_value=1, key='n_steps')
    with c2:
        st.number_input("Tolerantie", value=float(st.session_state.tol), step=1e-6, format="%.2e", key='tol')
    with c3:
        st.number_input("Max iteraties per stap", value=int(st.session_state.max_iters), step=1, min_value=1, key='max_iters')


# -----
# Output
# -----
with output_tab:
    st.subheader("Output")
    c_out = base_cfg.get('output', {})

    if 'do_plot' not in st.session_state:
        st.session_state.do_plot = bool(c_out.get('plot', True))
    if 'py_enabled' not in st.session_state:
        st.session_state.py_enabled = bool(_safe_get(c_out,['py_plots','enabled'], True))
    if 'max_plots' not in st.session_state:
        st.session_state.max_plots = int(_safe_get(c_out,['py_plots','max_plots'], 8))
    if 'py_mode' not in st.session_state:
        st.session_state.py_mode = 'Handmatig'
    if 'levels_txt' not in st.session_state:
        st.session_state.levels_txt = ', '.join(str(x) for x in (_safe_get(c_out,['py_plots','levels_m'], [0.5,1.5]) or [0.5,1.5]))
    if 'per_unit' not in st.session_state:
        st.session_state.per_unit = bool(_safe_get(c_out,['py_plots','per_unit_length'], True))
    if 'dpi' not in st.session_state:
        st.session_state.dpi = int(_safe_get(c_out,['py_plots','dpi'], 200))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.checkbox("Maak standaard plots", value=bool(st.session_state.do_plot), key='do_plot')
    with c2:
        st.checkbox("Maak P-Y plots", value=bool(st.session_state.py_enabled), key='py_enabled')
    with c3:
        st.number_input("Maximaal aantal P-Y plots", value=int(st.session_state.max_plots), step=1, min_value=2, key='max_plots')

    st.markdown("##### P-Y instellingen")
    st.selectbox("Selectie niveaus", options=['Automatisch', 'Handmatig'], index=['Automatisch','Handmatig'].index(st.session_state.py_mode), key='py_mode')
    st.text_area("Niveaus (m) – alleen bij Handmatig", value=str(st.session_state.levels_txt), key='levels_txt')

    st.checkbox("P per strekkende meter (kN/m)", value=bool(st.session_state.per_unit), key='per_unit')
    st.number_input("Resolutie (DPI)", value=int(st.session_state.dpi), step=10, min_value=50, key='dpi')


# ----------------
# Run
# ----------------
st.divider()

cfg = _build_cfg_from_state(base_cfg)
warnings, errors = _validate_layers(cfg['soil']['layers'])

run_disabled = len(errors) > 0
if run_disabled:
    st.error("Run geblokkeerd: los eerst de fouten in de lagen op.")

if st.button("Run berekening", type="primary", disabled=run_disabled):
    run_dir = Path(tempfile.mkdtemp(prefix="dpile_run_"))
    cfg['output']['folder'] = str(run_dir)

    try:
        with st.spinner("Rekenen... (OpenSeesPy)"):
            res = dpile_model.build_and_run(cfg)
    except Exception as e:
        st.error(f"Fout tijdens berekening: {e}")
        st.stop()

    # Bewaar resultaten in session_state zodat ze blijven staan na (her)render / downloads
    st.session_state.last_run_dir = str(run_dir)
    st.session_state.last_res = res
    st.session_state.last_cfg = cfg

    st.success("Klaar! Resultaten staan hieronder en blijven zichtbaar.")
    st.rerun()


# ----------------
# Resultaten (blijven zichtbaar)
# ----------------
if 'last_run_dir' in st.session_state and st.session_state.get('last_res', None) is not None:
    run_dir = Path(st.session_state.last_run_dir)
    res = st.session_state.last_res

    st.divider()
    st.subheader("Resultaten")

    csv_path = Path(res.get('csv_path', run_dir / 'results.csv'))
    png_path = Path(res.get('png_path', run_dir / 'results.png'))
    fd_png = run_dir / 'pile_force_displacement_pile0.png'
    fd_csv = run_dir / 'pile_force_displacement_pile0.csv'

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### Plots")
        if png_path.exists():
            st.image(str(png_path), caption=png_path.name, use_container_width=True)
        if fd_png.exists():
            st.image(str(fd_png), caption=fd_png.name, use_container_width=True)

    with colB:
        st.markdown("### Downloads")
        # Download buttons triggeren een rerun; omdat resultaten in session_state staan blijven ze zichtbaar
        if csv_path.exists():
            st.download_button("Download results.csv", data=csv_path.read_bytes(), file_name=csv_path.name, mime="text/csv", key="dl_results_csv")
        if png_path.exists():
            st.download_button("Download results.png", data=png_path.read_bytes(), file_name=png_path.name, mime="image/png", key="dl_results_png")
        if fd_csv.exists():
            st.download_button("Download pile_force_displacement.csv", data=fd_csv.read_bytes(), file_name=fd_csv.name, mime="text/csv", key="dl_fd_csv")
        if fd_png.exists():
            st.download_button("Download pile_force_displacement.png", data=fd_png.read_bytes(), file_name=fd_png.name, mime="image/png", key="dl_fd_png")

        py_dir = run_dir / 'PY_Plots'
        if py_dir.exists() and any(py_dir.iterdir()):
            import io, zipfile
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for p in py_dir.rglob('*'):
                    if p.is_file():
                        zf.write(p, arcname=str(p.relative_to(run_dir)))
            st.download_button("Download PY_Plots.zip", data=buffer.getvalue(), file_name="PY_Plots.zip", mime="application/zip", key="dl_py_zip")

    st.caption(f"Outputmap: {run_dir}")

