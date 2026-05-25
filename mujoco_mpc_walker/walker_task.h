#ifndef MJPC_TASKS_WALKER_TASK_H_
#define MJPC_TASKS_WALKER_TASK_H_

#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {

class WalkerTask : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;

  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const WalkerTask* task, int current_mode = 0);
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
    int current_mode_;
  };

  WalkerTask() : residual_(this, 0) {}
  void TransitionLocked(mjModel* model, mjData* data) override;
  void ResetLocked(const mjModel* model) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this, residual_.current_mode_);
  }
  mjpc::BaseResidualFn* InternalResidual() override { return &residual_; }
  ResidualFn residual_;
};

}  // namespace mjpc

#endif  // MJPC_TASKS_WALKER_TASK_H_
