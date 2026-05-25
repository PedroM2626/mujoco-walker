@echo off
set "PATH=%PATH%;D:\mujoco-walker\mujoco_mpc_walker\build\bin"
set "MJPC_TASKS_DIR=D:\mujoco-walker\mujoco_mpc_walker\build\_deps\mujoco_mpc-build\mjpc\tasks"
cd build
.\generate_dataset.exe
echo EXIT CODE: %errorlevel%
