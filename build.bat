@echo off
setlocal
cd /d "%~dp0"
echo [*] INICIANDO COMPILACION NIGHTWATCH-VISION (CUDA + OpenCV + Blob Tracking)

:: ── Visual Studio ─────────────────────────────────────────────────────────────
call "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if %errorlevel% neq 0 (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)

:: ── OpenCV (Unreal Engine 5.7) ────────────────────────────────────────────────
set "UE_OPENCV_INC=C:\Program Files\Epic Games\UE_5.7\Engine\Plugins\Runtime\OpenCV\Source\ThirdParty\OpenCV\include"
set "OPENCV_LIB=C:\Program Files\Epic Games\UE_5.7\Engine\Plugins\Runtime\OpenCV\Source\ThirdParty\OpenCV\lib\Win64\opencv_world455.lib"

:: ── Kinect for Windows SDK 2.0 (optional) ────────────────────────────────────
set "KINECT_EXTRA="
if not "%KINECTSDK20_DIR%"=="" (
    set KINECT_EXTRA=-I"%KINECTSDK20_DIR%inc" -DNIGHTWATCH_USE_KINECT "%KINECTSDK20_DIR%lib\x64\Kinect20.lib"
    echo [OK] Kinect SDK: %KINECTSDK20_DIR%
) else (
    echo [WARN] KINECTSDK20_DIR no definida - solo modo sintetico
)

echo [*] OpenCV lib: %OPENCV_LIB%
echo [*] Compilando: main.cpp vision_kernel.cu synth_kinect.cu

:: ── NVCC ──────────────────────────────────────────────────────────────────────
nvcc -allow-unsupported-compiler -std=c++17 -o nightwatch_vision.exe ^
     main.cpp vision_kernel.cu synth_kinect.cu trackformer_trt.cpp ^
     -I"%UE_OPENCV_INC%" ^
     %KINECT_EXTRA% ^
     "%OPENCV_LIB%" ^
     -lcudart

if %errorlevel% neq 0 goto :compile_error

:: ── Compilacion exitosa ───────────────────────────────────────────────────────
echo [OK] COMPILACION EXITOSA.
if not exist opencv_world455.dll (
    copy "C:\Program Files\Epic Games\UE_5.7\Engine\Plugins\Runtime\OpenCV\Binaries\ThirdParty\Win64\opencv_world455.dll" . >nul 2>&1
)

echo.
echo Uso: .\nightwatch_vision.exe --synthetic
echo      .\nightwatch_vision.exe              (requiere Kinect v2)
echo.

nightwatch_vision.exe %*
goto :end

:: ── Error de compilacion ──────────────────────────────────────────────────────
:compile_error
echo.
echo [!] ERROR EN COMPILACION. Checklist:
echo     1. nvcc en PATH? Ejecuta: nvcc --version
echo     2. VS C++ workload instalado?
echo     3. Existe: %OPENCV_LIB%
echo.
pause

:end
endlocal
