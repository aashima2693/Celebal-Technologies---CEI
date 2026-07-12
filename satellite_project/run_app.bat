@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Starting Streamlit dashboard...
echo Open browser at: http://localhost:8501
streamlit run app.py
pause
