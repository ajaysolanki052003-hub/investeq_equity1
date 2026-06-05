@echo off
REM Launch the Daily Swing Scanner on http://localhost:8532
cd /d "%~dp0"
python -m streamlit run scan_app.py --server.port 8532
