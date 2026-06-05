@echo off
REM Launch the EMA Cross + Swing-Low SL visualizer on http://localhost:8531
cd /d "%~dp0"
python -m streamlit run app.py --server.port 8531
