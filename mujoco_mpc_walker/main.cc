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

#include <fstream>
#include <cmath>

std::ofstream* dataset_out = nullptr;
bool header_written = false;
void (*old_step_callback)(const mjModel* m, mjData* d, int stage) = nullptr;

void my_step_callback(const mjModel* m, mjData* d, int stage) {
  if (old_step_callback) {
    old_step_callback(m, d, stage);
  }
  if (dataset_out && dataset_out->is_open()) {
    if (!header_written) {
      *dataset_out << "target_x,target_y";
      for (int i=0; i < m->nq; i++) *dataset_out << ",qpos_" << i;
      for (int i=0; i < m->nv; i++) *dataset_out << ",qvel_" << i;
      for (int i=0; i < m->nu; i++) *dataset_out << ",ctrl_" << i;
      *dataset_out << ",reward,done\n";
      header_written = true;
    }
    
    // Check for NaNs
    bool healthy = true;
    for (int j = 0; j < m->nq; j++) {
      if (!std::isfinite(d->qpos[j])) healthy = false;
    }
    for (int j = 0; j < m->nv; j++) {
      if (!std::isfinite(d->qvel[j])) healthy = false;
    }

    if (healthy) {
      int target_mocap_id = mj_name2id(m, mjOBJ_BODY, "target_marker");
      int torso_id = mj_name2id(m, mjOBJ_BODY, "torso");
      
      if (target_mocap_id >= 0 && torso_id >= 0) {
        int mocapid = m->body_mocapid[target_mocap_id];
        double tx = d->mocap_pos[3 * mocapid + 0];
        double ty = d->mocap_pos[3 * mocapid + 1];
        
        double torso_x = d->xpos[3 * torso_id + 0];
        double torso_y = d->xpos[3 * torso_id + 1];
        double torso_z = d->xpos[3 * torso_id + 2];
        
        // Calculate done and reward
        bool done = (torso_z < 0.5); // Fallen
        double dist = std::sqrt((torso_x - tx)*(torso_x - tx) + (torso_y - ty)*(torso_y - ty));
        
        // Target reached? Spawn new target!
        if (dist < 0.5) {
            tx = torso_x + ((double)rand() / RAND_MAX - 0.5) * 5.0; // Random [-2.5, +2.5]
            ty = torso_y + ((double)rand() / RAND_MAX - 0.5) * 5.0;
            if (mocapid >= 0) {
                // Warning: mutating mjData inside sensor callback is technically dangerous but works here
                d->mocap_pos[3 * mocapid + 0] = tx;
                d->mocap_pos[3 * mocapid + 1] = ty;
            }
            dist = std::sqrt((torso_x - tx)*(torso_x - tx) + (torso_y - ty)*(torso_y - ty));
        }
        
        // Reward: -distance to target, huge penalty for falling, small bonus for staying alive
        double reward = -dist * 2.0;
        if (done) reward -= 100.0;
        else reward += 1.0;
        
        *dataset_out << tx << "," << ty;
        for(int j=0; j < m->nq; j++) *dataset_out << "," << d->qpos[j];
        for(int j=0; j < m->nv; j++) *dataset_out << "," << d->qvel[j];
        for(int j=0; j < m->nu; j++) *dataset_out << "," << d->ctrl[j];
        *dataset_out << "," << reward << "," << (done ? 1 : 0) << "\n";
      }
    }
  }
}

int main(int argc, char** argv) {
  std::cout << "Starting parsing..." << std::endl;
  absl::ParseCommandLine(argc, argv);
  std::cout << "Parsing done." << std::endl;

  try {
    std::cout << "Initializing tasks..." << std::endl;
    // Bypass GetTasks() which crashes on Windows
    std::vector<std::shared_ptr<mjpc::Task>> tasks;
    tasks.push_back(std::make_shared<mjpc::WalkerTask>());
    std::cout << "Tasks initialized." << std::endl;

    std::ofstream out("dataset.csv", std::ios::app);
    out.seekp(0, std::ios::end);
    if (out.tellp() > 0) {
        header_written = true;
    }
    dataset_out = &out;
    old_step_callback = mjcb_sensor;
    mjcb_sensor = my_step_callback;

    std::cout << "Starting MJPC GUI... Brinque com o alvo! Feche a janela quando terminar de coletar." << std::endl;
    mjpc::StartApp(tasks, 0);
    
    out.flush();
    out.close();
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
