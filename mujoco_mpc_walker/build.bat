@echo off
REM Configure and build walker_mpc.exe (Phase 2 MPC teacher).
REM
REM This file used to hardcode two absolute paths - the VS "18" BuildTools folder and
REM C:\Program Files\CMake - and both are gone on the machine this repository lives on, which
REM is why the MPC phase could not be rebuilt after June. vswhere ships with every Visual
Studio installer since 2017, so ask it instead of guessing the edition.

setlocal enabledelayedexpansion

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo [BUILD] vswhere not found at "%VSWHERE%".
  echo [BUILD] Install Visual Studio Build Tools with the "Desktop development with C++"
  echo [BUILD] workload ^(vs_buildtools.exe --add Microsoft.VisualStudio.Workload.VCTools^).
  exit /b 1
)

for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDEV=%%i"
if not defined VSDEV (
  echo [BUILD] no Visual Studio installation with the C++ toolchain was found by vswhere.
  exit /b 1
)
echo [BUILD] using "%VSDEV%"
call "!VSDEV!\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 exit /b 1

REM Prefer the VS-bundled CMake, then whatever is on PATH.
set "CMAKE=!VSDEV!\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if not exist "%CMAKE%" where cmake >nul 2>&1 && set "CMAKE=cmake"
if not defined CMAKE (
  echo [BUILD] cmake not found. Install CMake, or the C++ CMake tools for Windows component.
  exit /b 1
)
echo [BUILD] using cmake "%CMAKE%"

cd "%~dp0"
if not exist build mkdir build
cd build
"%CMAKE%" .. -DCMAKE_BUILD_TYPE=Release || exit /b 1
"%CMAKE%" --build . --config Release -j || exit /b 1
echo [BUILD] done: build\walker_mpc.exe
