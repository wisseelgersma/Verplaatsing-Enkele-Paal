# -*- coding: utf-8 -*-
import sys, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
subprocess.call([sys.executable, '-m', 'streamlit', 'run', str(HERE/'app.py')], cwd=str(HERE))
