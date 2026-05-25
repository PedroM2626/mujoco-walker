#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/agent.h"
#include "mjpc/threadpool.h"
#include "mjpc/tasks/tasks.h"
#include "walker_task.h"

using namespace mjpc;

namespace {
  Task* task;
  void residual_callback(const mjModel* model, mjData* data, int stage) {
    if (stage == mjSTAGE_ACC) {
      task->Residual(model, data, data->sensordata);
    }
  }
}

#include <absl/flags/parse.h>

int main(int argc, char** argv) {
  try {
    absl::ParseCommandLine(argc, argv);
    std::cout << "Starting Headless MPC Dataset Generation...\n";
    auto tasks = GetTasks();
    tasks.insert(tasks.begin(), std::make_shared<WalkerTask>());

  auto agent = std::make_unique<Agent>();
  agent->SetTaskList(tasks);
  agent->gui_task_id = agent->GetTaskIdByName("Walker Ragdoll");
  if (agent->gui_task_id == -1) {
    std::cerr << "Task 'Walker Ragdoll' not found!\n";
    return 1;
  }
  
  Agent::LoadModelResult load_model = agent->LoadModel();
  mjModel* model = load_model.model.get();
  if (!model) {
    std::cerr << "Failed to load model: " << load_model.error << "\n";
    return 1;
  }
  
  mjData* data = mj_makeData(model);
  
  agent->estimator_enabled = false;
  agent->Initialize(model);
  agent->Allocate();
  agent->Reset(data->ctrl);
  agent->plan_enabled = true;
  
  task = agent->ActiveTask();
  task->mode = 2; // Mode 2: Target
  task->Transition(model, data); // Apply mode
  
  mjcb_sensor = &residual_callback;
  
  int num_threads = 4;
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
  out << "\n";
  
  int target_mocap_id = mj_name2id(model, mjOBJ_BODY, "target_marker");
  int mocapid = model->body_mocapid[target_mocap_id];
  
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);
  
  int steps_per_plan = 2;
  int total_episodes = 20;
  int steps_per_episode = 800; // 800 steps * 0.015s = 12 seconds per episode
  
  for (int ep = 0; ep < total_episodes; ep++) {
    mj_resetData(model, data);
    
    // Set random target for this episode
    double tx = dist(rng);
    double ty = dist(rng);
    if (mocapid >= 0) {
      data->mocap_pos[3 * mocapid + 0] = tx;
      data->mocap_pos[3 * mocapid + 1] = ty;
    }
    
    agent->state.Set(model, data);
    
    // Let it stabilize and walk towards the target
    for (int i = 0; i < steps_per_episode; i++) {
      task->Transition(model, data);
      agent->state.Set(model, data);
      
      agent->ActivePlanner().ActionFromPolicy(
          data->ctrl, agent->state.state().data(),
          agent->state.time(), /*use_previous=*/false);
          
      // Write to CSV (Imitation Learning target features: target_x, target_y, qpos, qvel)
      out << tx << "," << ty;
      for(int j=0; j < model->nq; j++) out << "," << data->qpos[j];
      for(int j=0; j < model->nv; j++) out << "," << data->qvel[j];
      for(int j=0; j < model->nu; j++) out << "," << data->ctrl[j];
      out << "\n";
      
      mj_step(model, data);
      
      if (i % steps_per_plan == 0) {
        agent->PlanIteration(&pool);
      }
    }
    std::cout << "Episode " << ep + 1 << " / " << total_episodes << " completed.\n";
  }
  
  out.close();
  mj_deleteData(data);
  mjcb_sensor = nullptr;
  
  std::cout << "Dataset generated successfully! (" << (total_episodes * steps_per_episode) << " total steps)\n";
  return 0;
  } catch (const std::exception& e) {
    std::cerr << "Fatal Exception: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown Fatal Exception!" << std::endl;
    return 1;
  }
}
