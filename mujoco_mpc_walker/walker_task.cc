#include "walker_task.h"

#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string WalkerTask::Name() const { return "Walker Ragdoll"; }
std::string WalkerTask::XmlPath() const { return "task_walker.xml"; }

void WalkerTask::ResetLocked(const mjModel* model) {
  mode = 0; // Seated mode initially (or Walk)
  residual_.current_mode_ = 0;
}

WalkerTask::ResidualFn::ResidualFn(const WalkerTask* task, int current_mode)
    : mjpc::BaseResidualFn(task), current_mode_(current_mode) {}

void WalkerTask::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                      double* residual) const {
  int counter = 0;

  int torso_id = mj_name2id(model, mjOBJ_BODY, "torso");
  if (torso_id < 0) {
    mju_error("Body 'torso' not found.");
  }

  // 1. Upright posture (Z-axis alignment of torso)
  double* torso_mat = data->xmat + 9 * torso_id;
  double torso_up[3] = {torso_mat[2], torso_mat[5], torso_mat[8]};
  residual[counter++] = torso_up[2] - 1.0; // We want Z to be 1

  // 2. Height
  double torso_z = data->xpos[3 * torso_id + 2];
  residual[counter++] = torso_z - 1.25; // Target height ~1.25m

  double* pelvis_pos = data->xpos + 3 * torso_id;
  double* pelvis_vel = data->cvel + 6 * torso_id; // CoM velocity

  if (current_mode_ == 0) {
    // Mode 0: Stand
    // No position target
    residual[counter++] = 0.0;
    residual[counter++] = 0.0;
    residual[counter++] = 0.0;

    // Velocity target: 0 (Stay still)
    residual[counter++] = pelvis_vel[3] - 0.0;
    residual[counter++] = pelvis_vel[4] - 0.0;
  } else if (current_mode_ == 1) {
    // Mode 1: Walk
    // No position target
    residual[counter++] = 0.0;
    residual[counter++] = 0.0;
    residual[counter++] = 0.0;

    // Velocity target: Walk forward at 1.2 m/s
    residual[counter++] = pelvis_vel[3] - 1.2;
    residual[counter++] = pelvis_vel[4] - 0.0;
  } else if (current_mode_ == 2) {
    // Mode 2: Target
    int target_mocap_id = mj_name2id(model, mjOBJ_BODY, "target_marker");
    double target_x = 0.0;
    double target_y = 0.0;
    if (target_mocap_id >= 0) {
      int mocapid = model->body_mocapid[target_mocap_id];
      if (mocapid >= 0) {
        target_x = data->mocap_pos[3 * mocapid + 0];
        target_y = data->mocap_pos[3 * mocapid + 1];
      }
    }
    
    // Position target: Walk to target X, Y
    residual[counter++] = pelvis_pos[0] - target_x;
    residual[counter++] = pelvis_pos[1] - target_y;
    residual[counter++] = 0.0;

    // Velocity target: Unconstrained (let position cost drive the movement)
    residual[counter++] = 0.0;
    residual[counter++] = 0.0;
  } else {
    // Fallback
    residual[counter++] = 0.0;
    residual[counter++] = 0.0;
    residual[counter++] = 0.0;
    residual[counter++] = pelvis_vel[3];
    residual[counter++] = pelvis_vel[4];
  }

  // 5. Control Effort
  for (int i = 0; i < model->nu; i++) {
    residual[counter++] = data->ctrl[i];
  }

  // sensor dim sanity check
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error_i("mismatch between total user-sensor dimension "
                "and actual length of residual %d", counter);
  }
}

void WalkerTask::TransitionLocked(mjModel* model, mjData* data) {
  residual_.current_mode_ = mode;
}

}  // namespace mjpc
