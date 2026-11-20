@echo off
cd /d "%~dp0"
python -m pip install -q mujoco
python -m mujoco.viewer --mjcf=scene.xml
pause
