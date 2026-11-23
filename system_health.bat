@echo off
rem Double-click launcher for the system health page (Streamlit).
rem Opens http://localhost:8503 in the browser.
cd /d "%~dp0"
python -m streamlit run frontend\system-health\app.py --server.port 8503
if errorlevel 1 (
    echo.
    echo Streamlit failed to start. Is it installed?  pip install streamlit
    pause
)
