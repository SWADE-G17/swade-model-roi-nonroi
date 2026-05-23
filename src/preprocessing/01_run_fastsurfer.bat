@echo off
REM ==============================================================================
REM 01_run_fastsurfer.bat
REM
REM DESCRIPCION:
REM   Este script recorre las carpetas AD, MCI y CN dentro de RAW_DIR,
REM   y por cada archivo .nii o .nii.gz llama a FastSurfer via Docker
REM   para hacer skull stripping y segmentacion automatica.
REM
REM   El resultado se guarda en OUTPUT_DIR con la estructura:
REM   OUTPUT_DIR\AD\subj001\mri\orig.mgz
REM   OUTPUT_DIR\AD\subj001\mri\aparc.DKTatlas+aseg.deep.mgz
REM
REM USO:
REM   1. Edita las variables de configuracion debajo
REM   2. Abre PowerShell como Administrador
REM   3. Ejecuta: scripts\01_run_fastsurfer.bat
REM ==============================================================================

REM ==============================================================================
REM CONFIGURACION - EDITA ESTAS 3 LINEAS ANTES DE EJECUTAR
REM ==============================================================================

REM Carpeta donde estan tus .nii.gz organizados en AD/, MCI/, CN/
set RAW_DIR=C:\alzheimer_data\raw

REM Carpeta donde se guardara la salida de FastSurfer
set OUTPUT_DIR=C:\alzheimer_data\processed

REM Ruta a tu archivo license.txt de FreeSurfer
set FS_LICENSE=C:\fastsurfer_license\license.txt

REM ==============================================================================
REM NO MODIFICAR LO DE ABAJO
REM ==============================================================================

echo.
echo ============================================================
echo  PIPELINE FASTSURFER - Alzheimer Project
echo ============================================================
echo  Datos crudos:     %RAW_DIR%
echo  Datos procesados: %OUTPUT_DIR%
echo  Licencia:         %FS_LICENSE%
echo ============================================================
echo.

REM Verificar que Docker esta corriendo
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker no esta corriendo. Abre Docker Desktop y espera que diga "Engine running".
    pause
    exit /b 1
)

REM Verificar que existe la licencia
if not exist "%FS_LICENSE%" (
    echo ERROR: No se encontro la licencia en: %FS_LICENSE%
    echo Descargala gratis en: https://surfer.nmr.mgh.harvard.edu/registration.html
    pause
    exit /b 1
)

REM Crear carpetas de salida si no existen
if not exist "%OUTPUT_DIR%\AD" mkdir "%OUTPUT_DIR%\AD"
if not exist "%OUTPUT_DIR%\MCI" mkdir "%OUTPUT_DIR%\MCI"
if not exist "%OUTPUT_DIR%\CN" mkdir "%OUTPUT_DIR%\CN"

REM Contador de sujetos procesados
set TOTAL=0
set ERRORS=0

REM Procesar cada clase
for %%C in (AD) do (
    echo.
    echo --- Procesando clase: %%C ---
    echo.

    if not exist "%RAW_DIR%\%%C" (
        echo ADVERTENCIA: No existe la carpeta %RAW_DIR%\%%C - saltando...
    ) else (
        REM Procesar .nii.gz
        for %%F in ("%RAW_DIR%\%%C\*.nii.gz") do (
            call :process_subject "%%~nF" "%%~nxF" "%%C" "%%~dpF"
        )
        REM Procesar .nii (sin gz)
        for %%F in ("%RAW_DIR%\%%C\*.nii") do (
            REM Solo procesar si no existe ya la version .gz
            if not exist "%RAW_DIR%\%%C\%%~nF.nii.gz" (
                call :process_subject "%%~nF" "%%~nxF" "%%C" "%%~dpF"
            )
        )
    )
)

echo.
echo ============================================================
echo  RESUMEN FINAL
echo ============================================================
echo  Total procesados: %TOTAL%
echo  Errores:          %ERRORS%
echo ============================================================
echo.
echo Ahora corre: python scripts\02_verify_data.py
echo para verificar que todo salio bien.
pause
exit /b 0


REM ==============================================================================
REM Subrutina: procesar un sujeto
REM Parametros: %1=nombre_sin_ext, %2=nombre_archivo, %3=clase, %4=carpeta
REM ==============================================================================
:process_subject
set SUBJ_ID=%~1
set SUBJ_FILE=%~2
set CLASS=%~3
set SUBJ_INPUT_DIR=%~4

REM Quitar la barra final si existe
if "%SUBJ_INPUT_DIR:~-1%"=="\" set SUBJ_INPUT_DIR=%SUBJ_INPUT_DIR:~0,-1%

REM Carpeta de salida para este sujeto
set SUBJ_OUT=%OUTPUT_DIR%\%CLASS%

REM Verificar si ya fue procesado (para no repetir)
if exist "%SUBJ_OUT%\%SUBJ_ID%\mri\aparc.DKTatlas+aseg.deep.mgz" (
    echo [SKIP] %CLASS%\%SUBJ_ID% ya fue procesado anteriormente
    goto :eof
)

echo [INICIO] Procesando: %CLASS%\%SUBJ_ID%

REM Llamar a FastSurfer via Docker
docker run --rm --user root --gpus all ^
  -v "%SUBJ_INPUT_DIR%":/data_in ^
  -v "%SUBJ_OUT%":/data_out ^
  -v "%FS_LICENSE%":/fs_license.txt ^
  deepmi/fastsurfer:latest ^
  --t1 /data_in/%SUBJ_FILE% ^
  --sid %SUBJ_ID% ^
  --sd /data_out ^
  --fs_license /fs_license.txt ^
  --seg_only ^
  --no_cereb ^
  --allow_root

if errorlevel 1 (
    echo [ERROR] Fallo el procesamiento de: %CLASS%\%SUBJ_ID%
    set /a ERRORS+=1
) else (
    echo [OK] Completado: %CLASS%\%SUBJ_ID%
    set /a TOTAL+=1
)

goto :eof
