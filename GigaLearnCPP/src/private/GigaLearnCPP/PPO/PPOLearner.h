#pragma once
#include "ExperienceBuffer.h"
#include <GigaLearnCPP/PPO/PPOLearnerConfig.h>
#include <GigaLearnCPP/PPO/TransferLearnConfig.h>
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/Util/Timer.h>

#include "../Util/Models.h"

#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/loss.h>
#include <torch/optim/adam.h>

#include <future>
#include <vector>

namespace GGL {

// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/ppo_learner.py
class PPOLearner {
public:
  ModelSet models = {};
  ModelSet guidingPolicyModels = {};

  PPOLearnerConfig config;
  torch::Device device;

  PPOLearner(int obsSize, int numActions, PPOLearnerConfig config,
             torch::Device device,
             std::vector<torch::Device> dataParallelDevices = {});
  ~PPOLearner();

  static void MakeModels(bool makeCritic, int obsSize, int numActions,
                         PartialModelConfig sharedHeadConfig,
                         PartialModelConfig policyConfig,
                         PartialModelConfig criticConfig, torch::Device device,
                         ModelSet &outModels);

  // If models is null, this->models will be used
  void InferActions(torch::Tensor obs, torch::Tensor actionMasks,
                    torch::Tensor *outActions, torch::Tensor *outLogProbs,
                    ModelSet *models = NULL);
  torch::Tensor InferCritic(torch::Tensor obs);

  // Perhaps they should be somewhere else? Should probably make an inference
  // interface...
  static torch::Tensor InferPolicyProbsFromModels(ModelSet &models,
                                                  torch::Tensor obs,
                                                  torch::Tensor actionMasks,
                                                  float temperature,
                                                  bool halfPrec);
  static void InferActionsFromModels(ModelSet &models, torch::Tensor obs,
                                     torch::Tensor actionMasks,
                                     bool deterministic, float temperature,
                                     bool halfPrec, torch::Tensor *outActions,
                                     torch::Tensor *outLogProbs);

  void Learn(ExperienceBuffer &experience, Report &report,
             bool isFirstIteration);

  void TransferLearn(ModelSet &oldModels, torch::Tensor newObs,
                     torch::Tensor oldObs, torch::Tensor newActionMasks,
                     torch::Tensor oldActionMasks, torch::Tensor actionMaps,
                     Report &report,
                     const TransferLearnConfig &transferLearnConfig);

  void SaveTo(std::filesystem::path folderPath, bool saveOptims = true);
  void LoadFrom(std::filesystem::path folderPath);
  void SetLearningRates(float policyLR, float criticLR);

  ModelSet GetPolicyModels();
  ModelSet CloneModelsWithOptims() const;

private:
  struct Replica {
    torch::Device device = torch::Device(torch::kCPU);
    ModelSet models;
    ModelSet guidingPolicyModels;
  };

  std::vector<torch::Device> dataParallelDevices;
  std::vector<Replica> replicas;
  bool useDataParallel = false;

  void InitDataParallel(const std::vector<torch::Device> &devices);
  void AccumulateReplicaGradsToPrimary();
  void ZeroReplicaGrads();
  void SyncReplicasFromPrimary();

  std::future<void> _lastSaveFuture;
};
} // namespace GGL
