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
}

WalkerTask::ResidualFn::ResidualFn(const WalkerTask* task)
    : mjpc::BaseResidualFn(task) {}

void WalkerTask::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                      double* residual) const {
  int counter = 0;

  // 1. Upright posture (Z-axis alignment of torso)
  int torso_id = mj_name2id(model, mjOBJ_BODY, "torso");
  if (torso_id < 0) {
    mju_error("Body 'torso' not found.");
  }
  double* torso_mat = data->xmat + 9 * torso_id;
  double torso_up[3] = {torso_mat[2], torso_mat[5], torso_mat[8]};
  residual[counter++] = torso_up[2] - 1.0; // We want Z to be 1

  // 2. Height
  double torso_z = data->xpos[3 * torso_id + 2];
  residual[counter++] = torso_z - 1.25; // Target height ~1.25m

  // 3. Position (Walk towards target)
  // We can use a target mocap or dummy. For simplicity, we just incentivize moving forward in X.
  // We will assume the target is at X=10.0, Y=0.0
  double* pelvis_pos = data->xpos + 3 * torso_id;
  residual[counter++] = pelvis_pos[0] - 10.0;
  residual[counter++] = pelvis_pos[1] - 0.0;
  residual[counter++] = 0.0; // Z position handled by height

  // 4. Velocity (Incentivize forward velocity)
  double* pelvis_vel = data->cvel + 6 * torso_id; // CoM velocity
  // Depending on the mode (parameters), we could target different velocities.
  // Here we hardcode a target forward velocity of 1.0 m/s
  residual[counter++] = pelvis_vel[3] - 1.0; // v_x
  residual[counter++] = pelvis_vel[4] - 0.0; // v_y

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
  // Can be used to move targets or update logic per step
}

}  // namespace mjpc
