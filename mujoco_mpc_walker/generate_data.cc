#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <vector>
#include <thread>
#include <algorithm>

#include <mujoco/mujoco.h>
#include "mjpc/agent.h"
#include "mjpc/threadpool.h"
#include "mjpc/tasks/tasks.h"
#include "walker_task.h"

#include <absl/flags/parse.h>

using namespace mjpc;

namespace {
  Task* task;
  void residual_callback(const mjModel* model, mjData* data, int stage) {
    if (stage == mjSTAGE_ACC) {
      task->Residual(model, data, data->sensordata);
    }
  }
}

int main(int argc, char** argv) {
  std::cout << "[DEBUG] Starting process..." << std::endl;
  try {
    absl::ParseCommandLine(argc, argv);
    std::cout << "[DEBUG] ParseCommandLine OK." << std::endl;
    // Bypass GetTasks() which crashes on Windows MSVC due to library bugs
    std::vector<std::shared_ptr<Task>> tasks;
    tasks.push_back(std::make_shared<WalkerTask>());
    std::cout << "[DEBUG] Task inserted without GetTasks()." << std::endl;

  auto agent = std::make_unique<Agent>();
  std::cout << "[DEBUG] Agent created." << std::endl;
  agent->SetTaskList(tasks);
  std::cout << "[DEBUG] TaskList set." << std::endl;
  agent->gui_task_id = agent->GetTaskIdByName("Walker Ragdoll");
  std::cout << "[DEBUG] gui_task_id: " << agent->gui_task_id << std::endl;
  if (agent->gui_task_id == -1) {
    std::cerr << "Task 'Walker Ragdoll' not found!\n";
    return 1;
  }
  
  std::cout << "[DEBUG] About to LoadModel..." << std::endl;
  Agent::LoadModelResult load_model = agent->LoadModel();
  std::cout << "[DEBUG] LoadModel finished." << std::endl;
  mjModel* model = load_model.model.get();
  if (!model) {
    std::cerr << "Failed to load model: " << load_model.error << "\n";
    return 1;
  }
  std::cout << "[DEBUG] Model loaded." << std::endl;
  
  mjData* data = mj_makeData(model);
  std::cout << "[DEBUG] Data created." << std::endl;
  
  agent->estimator_enabled = false;
  std::cout << "[DEBUG] Agent initializing..." << std::endl;
  agent->Initialize(model);
  std::cout << "[DEBUG] Agent initialized." << std::endl;
  agent->Allocate();
  std::cout << "[DEBUG] Agent allocated." << std::endl;
  agent->Reset(data->ctrl);
  std::cout << "[DEBUG] Agent reset." << std::endl;
  agent->plan_enabled = true;
  
  task = agent->ActiveTask();
  task->mode = 2; // Mode 2: Target
  task->Transition(model, data); // Apply mode
  std::cout << "[DEBUG] Task transition applied." << std::endl;
  
  mjcb_sensor = &residual_callback;
  
  int num_threads = 4;
  unsigned int hardware_threads = std::thread::hardware_concurrency();
  if (hardware_threads > 0) {
    num_threads = std::max(1, (int)hardware_threads - 4);
  }
  if (const char* env_threads = std::getenv("NUM_THREADS")) {
    try { num_threads = std::stoi(env_threads); } catch (...) {}
  }
  std::cout << "[GEN] Using " << num_threads << " threads for ThreadPool" << std::endl;
  ThreadPool pool(num_threads);
  
  std::ofstream out("dataset.csv");
  if (!out.is_open()) {
    std::cerr << "Failed to open dataset.csv\n";
    return 1;
  }
  
  // Header: target_x, target_y, qpos_0...qpos_n, qvel_0...qvel_n, ctrl_0...ctrl_n
  out << "target_x,target_y";
  for (int i=0; i < model->nq; i++) out << ",qpos_" << i;
  for (int i=0; i < model->nv; i++) out << ",qvel_" << i;
  for (int i=0; i < model->nu; i++) out << ",ctrl_" << i;
  out << ",reward,done\n";
  
  int target_mocap_id = mj_name2id(model, mjOBJ_BODY, "target_marker");
  int mocapid = model->body_mocapid[target_mocap_id];
  
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);
  
  int steps_per_plan = 8;
  int good_steps_saved = 0;
  // Permite configurar o numero de passos via variavel de ambiente
  int target_steps = 15000;
  if (const char* env_steps = std::getenv("TARGET_STEPS")) {
    try { target_steps = std::stoi(env_steps); } catch (...) {}
  }
  std::cout << "Target steps: " << target_steps << "\n" << std::flush;
  
  while (good_steps_saved < target_steps) {
    agent->Reset(data->ctrl);
    mj_resetData(model, data);
    
    // Removed 50% chance of starting fallen because iLQG hangs on contact derivatives
    
    // Set random target (closer, to avoid falling immediately)
    double tx = dist(rng) * 0.4; // -2.0 to 2.0
    double ty = dist(rng) * 0.4;
    if (mocapid >= 0) {
      data->mocap_pos[3 * mocapid + 0] = tx;
      data->mocap_pos[3 * mocapid + 1] = ty;
    }
    
    task->Transition(model, data);
    agent->state.Set(model, data);
    
    // Warmup planner
    for (int i = 0; i < 50; i++) {
        agent->PlanIteration(&pool);
    }
    
    // Let it stabilize and walk towards the target
    for (int i = 0; i < 400; i++) {
      task->Transition(model, data);
      agent->state.Set(model, data);
      
      agent->ActivePlanner().ActionFromPolicy(
          data->ctrl, agent->state.state().data(),
          agent->state.time(), /*use_previous=*/false);
          
      // Write to CSV only if standing properly (Z height > 0.5)
      // Calculate positions
      int torso_id = mj_name2id(model, mjOBJ_BODY, "torso");
      double torso_x = data->xpos[3 * torso_id + 0];
      double torso_y = data->xpos[3 * torso_id + 1];
      double torso_z = data->xpos[3 * torso_id + 2];
      
      double target_dist = std::sqrt((torso_x - tx)*(torso_x - tx) + (torso_y - ty)*(torso_y - ty));
      
      // Target reached or dynamic timeout? Spawn new target!
      if (target_dist < 0.5 || (i > 0 && i % 100 == 0)) {
          tx = torso_x + dist(rng) * 0.5; // New random target nearby
          ty = torso_y + dist(rng) * 0.5;
          if (mocapid >= 0) {
              data->mocap_pos[3 * mocapid + 0] = tx;
              data->mocap_pos[3 * mocapid + 1] = ty;
          }
          // Recalculate target_dist for reward
          target_dist = std::sqrt((torso_x - tx)*(torso_x - tx) + (torso_y - ty)*(torso_y - ty));
      }
      
      bool done = (torso_z < 0.5); // Fallen
      
      // Reward logic
      double reward = -target_dist * 2.0;
      if (done) reward -= 100.0;
      else reward += 1.0;
          
      // Write to CSV including falling states
      out << tx << "," << ty;
      for(int j=0; j < model->nq; j++) out << "," << data->qpos[j];
      for(int j=0; j < model->nv; j++) out << "," << data->qvel[j];
      for(int j=0; j < model->nu; j++) out << "," << data->ctrl[j];
      out << "," << reward << "," << (done ? 1 : 0) << "\n";
      good_steps_saved++;
      
      // Removed reset on done to allow the network to learn recovery/self-rising behavior.
      // The episode will run for the full 400 steps unless good_steps_saved is reached.
      
      if (good_steps_saved >= target_steps) break;
      
      mj_step(model, data);
      
      // Verificacao de saude antes de planejar
      bool healthy = true;
      for (int j = 0; j < model->nq && healthy; j++) {
        if (!std::isfinite(data->qpos[j])) healthy = false;
      }
      
      if (i % steps_per_plan == 0 && healthy) {
        agent->PlanIteration(&pool);
      } else if (!healthy) {
        break;
      }
    }
    // Flush periodico para nao perder dados em caso de crash posterior
    out.flush();
    std::cout << "Good Steps Collected: " << good_steps_saved << " / " << target_steps << "\n" << std::flush;
  }
  std::cout << "\n";
  
  out.close();
  mj_deleteData(data);
  mjcb_sensor = nullptr;
  
  std::cout << "Dataset generated successfully! " << good_steps_saved << " steps.\n";
  std::cout.flush();
  // Usa _exit para evitar crash no destrutor do MuJoCo no Windows
  _exit(0);
  } catch (const std::exception& e) {
    std::cerr << "Fatal Exception: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown Fatal Exception!" << std::endl;
    return 1;
  }
}
