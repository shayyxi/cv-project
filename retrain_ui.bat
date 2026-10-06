@echo off
rem Double-click launcher for the retraining page (Streamlit).
rem Opens http://localhost:8501 in the browser.
cd /d "%~dp0"
python -m streamlit run scripts\retrain_ui.py
if errorlevel 1 (
    echo.
    echo Streamlit failed to start. Is it installed?  pip install streamlit
    pause
)
