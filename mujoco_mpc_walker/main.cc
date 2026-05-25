#include <memory>
#include <vector>

#include <absl/flags/parse.h>
#include <mujoco/mujoco.h>
#include "mjpc/app.h"
#include "mjpc/task.h"
#include "walker_task.h"
#include "mjpc/tasks/tasks.h"
#include <absl/flags/parse.h>
#include <iostream>
#include <exception>

int main(int argc, char** argv) {
  std::cout << "Starting parsing..." << std::endl;
  absl::ParseCommandLine(argc, argv);
  std::cout << "Parsing done." << std::endl;

  try {
    std::cout << "Initializing tasks..." << std::endl;
    std::vector<std::shared_ptr<mjpc::Task>> tasks = mjpc::GetTasks();
    tasks.insert(tasks.begin(), std::make_shared<mjpc::WalkerTask>());
    std::cout << "Tasks initialized." << std::endl;

    std::cout << "Starting MJPC GUI..." << std::endl;
    mjpc::StartApp(tasks, 0);
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
