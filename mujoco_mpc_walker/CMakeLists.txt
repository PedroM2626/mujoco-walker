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
