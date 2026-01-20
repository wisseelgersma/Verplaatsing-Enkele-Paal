# D-Pile – Single pile (Streamlit)

## Start (lokaal)
```bash
pip install -r requirements.txt
python start_dpile_app.py
```

## Wijzigingen
- Verticale verplaatsing is altijd vastgezet (geen UI-optie).
- Validatie lagen (dubbele/niet-oplopende laagtoppen) met waarschuwingen.
- Slimme default: nieuwe laagtop = laatste + 1.0 m.
- Config Export/Import via JSON in de sidebar.
