@echo off
title Idenfit Docker Baslatici
cd /d "%~dp0"
cls
echo ===================================================
echo   IDENFIT RAG DOCKER ILE BASLATILIYOR...
echo ===================================================
echo.
echo Docker Desktop kontrol ediliyor...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [HATA] Docker Desktop su anda kapali!
    echo Docker ile calistirmak icin once Windows Baslat menusunden
    echo 'Docker Desktop' uygulamasini acmaniz gerekir.
    echo.
    echo Eger Docker ile ugrasmak istemiyorsaniz dogrudan
    echo 'baslat.bat' dosyasina cift tiklayarak calistirabilirsiniz!
    echo.
    pause
    exit /b
)

echo Docker Desktop aktif. Konteyner derleniyor ve baslatiliyor...
docker compose up --build -d
echo.
echo ===================================================
echo [OK] Konteyner baslatildi!
echo Tarayicidan http://localhost:8501 aciliyor...
echo ===================================================
start "" "http://localhost:8501"
pause
