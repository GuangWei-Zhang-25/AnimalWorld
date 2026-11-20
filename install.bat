@echo off
rem Animal World: one-command install for Windows. UNTESTED on Windows (written on Linux; see INSTALL.md).
rem   install.bat          create %USERPROFILE%\animalworld-env and install Animal World with the run extra (CPU engine)
rem Before the PyPI release: set ANIMALWORLD_PACKAGE=animalworld@git+https://github.com/GuangWei-Zhang-25/AnimalWorld.git
rem The full GPU engine relies on symbolic links and Linux-style tools and is not supported on Windows yet.
setlocal
if "%ANIMALWORLD_ENV%"=="" set "ANIMALWORLD_ENV=%USERPROFILE%\animalworld-env"
if "%PYTHON%"=="" set "PYTHON=python"
set "PKG=%ANIMALWORLD_PACKAGE%"
if "%PKG%"=="" for %%f in ("%~dp0dist\animalworld-*.whl") do set "PKG=%%f"
if "%PKG%"=="" set "PKG=animalworld"
"%PYTHON%" -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" || (echo Animal World needs Python 3.9 or newer. & exit /b 1)
echo == 1/2  Creating the environment in %ANIMALWORLD_ENV%
"%PYTHON%" -m venv "%ANIMALWORLD_ENV%" || exit /b 1
"%ANIMALWORLD_ENV%\Scripts\python.exe" -m pip install --quiet --upgrade pip
echo == 2/2  Installing Animal World from: %PKG%
"%ANIMALWORLD_ENV%\Scripts\python.exe" -m pip install --quiet "%PKG%[run]" || exit /b 1
echo.
echo Animal World is installed. Start it with:
echo     "%ANIMALWORLD_ENV%\Scripts\animalworld.exe" studio
echo     "%ANIMALWORLD_ENV%\Scripts\animalworld.exe" run
endlocal
