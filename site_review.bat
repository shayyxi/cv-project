@echo off
rem Double-click launcher for the site safety review page (Streamlit).
rem Opens http://localhost:8502 in the browser.
cd /d "%~dp0"
python -m streamlit run frontend\site-review\app.py --server.port 8502
if errorlevel 1 (
    echo.
    echo Streamlit failed to start. Is it installed?  pip install streamlit
    pause
)
