cmake_minimum_required(VERSION 3.16)
project(mujoco_mpc_walker)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch mujoco_mpc
include(FetchContent)
FetchContent_Declare(
  mujoco_mpc
  GIT_REPOSITORY https://github.com/google-deepmind/mujoco_mpc.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(mujoco_mpc)

# Create our executable
add_executable(walker_mpc
  main.cc
  walker_task.cc
)

# Need Threads
find_package(Threads REQUIRED)

# Link against MJPC
target_link_libraries(walker_mpc
  PRIVATE
    libmjpc
    mujoco::mujoco
    absl::flags
    absl::flags_parse
    absl::random_random
    absl::strings
    threadpool
    Threads::Threads
)

# Copy the XML files to the output directory so the executable can find them
add_custom_command(TARGET walker_mpc POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/task_walker.xml
    ${CMAKE_CURRENT_BINARY_DIR}/task_walker.xml
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/../walker_ragdoll.xml
    ${CMAKE_CURRENT_BINARY_DIR}/walker_ragdoll.xml
)

# This file used to end with a second target:
#
#   add_executable(mjpc_dataset_tool generate_data.cc walker_task.cc)   + link libraries
#
# `generate_data.cc` is not in the repository and never was committed, so CMake aborted at
# *configure* time with "Cannot find source file: generate_data.cc". That made the whole MPC
# phase unbuildable - with or without a compiler - which is the real reason the Phase-2
# dataset cannot be regenerated today, and why build/ holds a June walker_mpc.exe but no
# dataset. Recording now lives in main.cc behind --dataset_path, so this target has nothing
# left to build.
