@echo off
setlocal
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe

echo ============================================================
echo CL-MEE - Windows reproduction, tests and package verification
echo ============================================================

if not exist "%PY%" (
  echo [1/6] Creating Python virtual environment...
  py -3.13 -m venv .venv
  if errorlevel 1 goto :error
) else (
  echo [1/6] Reusing existing .venv
)

echo [2/6] Installing pinned dependencies...
%PY% -m pip install --upgrade pip
if errorlevel 1 goto :error
%PY% -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo [3/6] Capturing environment...
%PY% capture_environment.py --output-dir environment_capture
if errorlevel 1 goto :error

echo [4/6] Running unit tests...
%PY% -m unittest discover -s tests -v
if errorlevel 1 goto :error

echo [5/6] Running complete simulation and Foursquare experiments...
%PY% run_revision_experiments.py --dataset dataset_TSMC2014_NYC.txt --output-dir verification_results
if errorlevel 1 goto :error

echo [6/6] Verifying outputs and manuscript metrics...
%PY% verify_package.py --results-dir verification_results --dataset dataset_TSMC2014_NYC.txt --paper ..\main.tex --reference-dir revision_outputs --pdf ..\main.pdf --max-pages 9
if errorlevel 1 goto :error

echo.
echo SUCCESS: all tests and verification checks passed.
echo Read reproducibility\verification_results\verification_report.txt
pause
exit /b 0

:error
echo.
echo ERROR: a step failed. Read the message above and check Python 3.13 is installed.
pause
exit /b 1
