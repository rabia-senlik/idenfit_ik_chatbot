@echo off
title Idenfit IK Asistani (Ollama)
cd /d "%~dp0"
cls
echo ========================================================
echo       IDENFIT IK VE KURAL ASISTANI BASLATILIYOR
echo ========================================================
echo.
echo Asistan hazirlaniyor, lutfen 3-4 saniye bekleyin...
echo Tarayiciniz sistem hazir oldugunda otomatik acilacaktir.
echo.
echo [ONEMLI] Bu siyah komut penceresini asistan acikken
echo          KAPATMAYIN! (Simge durumuna kucultebilirsiniz)
echo ========================================================
echo.
python -m streamlit run app.py
pause
