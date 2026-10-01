#include <memory>
#include <string>
#include <vector>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <mujoco/mujoco.h>
#include "mjpc/app.h"
#include "mjpc/task.h"
#include "walker_task.h"
#include "mjpc/tasks/tasks.h"
#include <iostream>
#include <exception>

#include <fstream>
#include <cmath>

// Where to write the imitation dataset, and how many transitions to write. An empty path
// leaves this binary behaving exactly as before: interactive viewer, nothing recorded.
ABSL_FLAG(std::string, dataset_path, "", "CSV file for (obs, ctrl, reward, done) transitions")
ABSL_FLAG(int, transitions, 15000, "Stop recording after this many transitions (0 = record until the window closes)")

std::ofstream* dataset_out = nullptr;
bool header_written = false;
bool recording_done = false;
int transitions_written = 0;
int transitions_target = 0;
void (*old_step_callback)(const mjModel* m, mjData* d, int stage) = nullptr;

void my_step_callback(const mjModel* m, mjData* d, int stage) {
  if (old_step_callback) {
    old_step_callback(m, d, stage);
  }
  if (dataset_out && dataset_out->is_open() && !recording_done) {
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

        transitions_written++;
        if (transitions_target > 0 && transitions_written >= transitions_target) {
          // The sensor callback cannot close the app, so this stops the recording and tells the
          // operator to close the window; the file is flushed below when main() returns.
          recording_done = true;
          dataset_out->flush();
          std::cout << "[DATASET] " << transitions_written << " transitions written to "
                    << "the output file - close the viewer window to finish." << std::endl;
        }
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

    // Recording is opt-in: with no --dataset_path this binary does what it always did.
    const std::string dataset_path = absl::GetFlag(FLAGS_dataset_path);
    std::ofstream out;
    if (!dataset_path.empty()) {
      transitions_target = absl::GetFlag(FLAGS_transitions);
      out.open(dataset_path, std::ios::app);
      if (!out.is_open()) {
        std::cerr << "[DATASET] cannot open " << dataset_path << " for writing" << std::endl;
        return 1;
      }
      out.seekp(0, std::ios::end);
      if (out.tellp() > 0) {
        header_written = true;  // resuming a file that already has a header
      }
      dataset_out = &out;
      old_step_callback = mjcb_sensor;
      mjcb_sensor = my_step_callback;
      std::cout << "[DATASET] recording to " << dataset_path
                << (transitions_target > 0 ? " (limit " + std::to_string(transitions_target) + ")" : "")
                << " - drive the walker, then close the window." << std::endl;
    }

    std::cout << "Starting MJPC GUI... Brinque com o alvo! Feche a janela quando terminar de coletar." << std::endl;
    mjpc::StartApp(tasks, 0);

    if (dataset_out) {
      dataset_out = nullptr;
      mjcb_sensor = old_step_callback;
      out.flush();
      out.close();
      std::cout << "[DATASET] " << transitions_written << " transitions in " << dataset_path
                << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
