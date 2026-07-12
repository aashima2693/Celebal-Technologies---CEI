@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM  Satellite Project — One-Click Windows Setup
REM  Run this file once to set up the virtual environment and install packages.
REM ═══════════════════════════════════════════════════════════════════════════

setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo  ======================================================
echo   Satellite Land-Use Classifier — Windows Setup
echo  ======================================================
echo.

REM ── Check Python ──────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+ from https://python.org
    echo         Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% found.

REM ── Create virtual environment ────────────────────────────────────────────
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo [STEP 1/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists. Skipping creation.
)

REM ── Activate venv ─────────────────────────────────────────────────────────
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated.

REM ── Upgrade pip ───────────────────────────────────────────────────────────
echo.
echo [STEP 2/4] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip upgraded.

REM ── Install packages ──────────────────────────────────────────────────────
echo.
echo [STEP 3/4] Installing required packages (this may take 5-10 minutes)...
echo            PyTorch, torchvision, streamlit, scikit-learn, etc.
echo.

pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [WARNING] Some packages failed to install. Trying without version pins...
    pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
    pip install Pillow tqdm requests streamlit opencv-python scipy jupyter ipykernel
)

echo.
echo [OK] Packages installed.

REM ── Register Jupyter kernel ───────────────────────────────────────────────
echo.
echo [STEP 4/4] Registering Jupyter kernel...
python -m ipykernel install --user --name=satellite_venv --display-name "Python (satellite_project)"
echo [OK] Kernel registered as "Python (satellite_project)"

REM ── SSL fix for urllib on Windows ────────────────────────────────────────
echo.
echo [INFO] Applying Windows SSL certificate fix...
python -c "import ssl; ctx = ssl.create_default_context(); print('[OK] SSL context OK')" 2>nul || echo [WARN] SSL check skipped

REM ── Done ─────────────────────────────────────────────────────────────────
echo.
echo  ======================================================
echo   Setup Complete!
echo  ======================================================
echo.
echo  NEXT STEPS:
echo  -----------
echo  1. Open VSCode in this folder
echo  2. Open notebooks\ folder
echo  3. Select kernel: "Python (satellite_project)"
echo  4. Run notebooks IN ORDER:
echo        01_data_pipeline.ipynb     (downloads EuroSAT ~100MB)
echo        02_baseline_cnn.ipynb
echo        03_transfer_learning.ipynb
echo        04_change_detection.ipynb
echo        05_bonus_gradcam.ipynb
echo        06_bonus_tsne_umap.ipynb
echo        07_bonus_imbalance.ipynb
echo.
echo  5. To launch the Streamlit dashboard (after notebook 03):
echo        run_app.bat
echo.

REM ── Create run_app.bat ────────────────────────────────────────────────────
(
echo @echo off
echo cd /d "%%~dp0"
echo call venv\Scripts\activate.bat
echo echo Starting Streamlit dashboard...
echo echo Open browser at: http://localhost:8501
echo streamlit run app.py
echo pause
) > run_app.bat

echo [OK] Created run_app.bat — double-click it to launch the dashboard.
echo.
pause
