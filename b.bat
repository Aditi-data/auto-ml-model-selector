@echo off
setlocal enabledelayedexpansion
title ML Algorithm Accuracy Testing

echo ====================================
echo   ML Algorithm Accuracy Testing
echo ====================================
echo.
echo Installing required packages...
pip install tabulate >nul 2>&1
echo Opening file dialog...

for /f "delims=" %%i in ('powershell -WindowStyle Hidden -Command "Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.OpenFileDialog; $f.Filter = 'CSV files (*.csv)|*.csv'; $f.Title = 'Select CSV file for ML testing'; if($f.ShowDialog() -eq 'OK') { $f.FileName }"') do set "filepath=%%i"

if "!filepath!"=="" (
    echo No file selected. Running with sample data...
    python accuracy_testing.py
    goto end
)

echo Selected: !filepath!
echo.
set /p "target=Enter target column name (or press Enter for last column): "

if "!target!"=="" (
    python accuracy_testing.py --file "!filepath!"
) else (
    python accuracy_testing.py --file "!filepath!" --target "!target!"
)

:end
echo.
echo Press any key to exit...
pause >nul