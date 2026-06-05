@echo off
REM Launch the EMA21/50 Zone Scanner on http://localhost:8530
cd /d "%~dp0"
python -m streamlit run app_ema_cross.py --server.port 8530
