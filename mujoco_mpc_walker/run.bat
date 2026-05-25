@echo off
set "PATH=%PATH%;D:\mujoco-walker\mujoco_mpc_walker\build\bin"
set "MJPC_TASKS_DIR=D:\mujoco-walker\mujoco_mpc_walker\build"
cd build
walker_mpc.exe
