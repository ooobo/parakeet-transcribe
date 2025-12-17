@echo off
REM Transcribe audio file with timestamps
REM Drag and drop an audio file onto this .bat file to transcribe it

if "%~1"=="" (
    echo Error: No audio file provided.
    echo.
    echo Usage: Drag and drop an audio file onto this .bat file
    echo.
    pause
    exit /b 1
)

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"

REM Run parakeet-transcribe with timestamps
"%SCRIPT_DIR%parakeet-transcribe.exe" "%~1" --timestamps

REM Batch file will pause automatically due to wait_for_keypress in the Rust code
