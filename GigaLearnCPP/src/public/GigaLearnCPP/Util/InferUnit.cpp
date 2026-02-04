#include "InferUnit.h"

#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <GigaLearnCPP/Util/Models.h>
#include <RLGymCPP/ThreadPool.h>

#include <algorithm>
#include <utility>

#ifdef RG_CUDA_SUPPORT
#if __has_include(<c10/cuda/CUDAGuard.h>)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
namespace ggl_cuda = c10::cuda;
#elif __has_include(<ATen/cuda/CUDAGuard.h>)
#include <ATen/cuda/CUDAGuard.h>
namespace ggl_cuda = at::cuda;
#else
#error "RG_CUDA_SUPPORT is enabled but no CUDA guard header was found."
#endif
#endif

GGL::InferUnit::InferUnit(RLGC::ObsBuilder *obsBuilder, int obsSize,
                          RLGC::ActionParser *actionParser,
                          PartialModelConfig sharedHeadConfig,
                          PartialModelConfig policyConfig,
                          std::filesystem::path modelsFolder, bool useGPU)
    : obsBuilder(obsBuilder), obsSize(obsSize), actionParser(actionParser),
      useGPU(useGPU) {

  this->models = std::make_unique<ModelSet>();

  try {
    PPOLearner::MakeModels(false, obsSize, actionParser->GetActionAmount(),
                           sharedHeadConfig, policyConfig, {},
                           useGPU ? torch::kCUDA : torch::kCPU, *this->models);
  } catch (std::exception &e) {
    RG_ERR_CLOSE(
        "InferUnit: Exception when trying to construct models: " << e.what());
  }

  try {
    this->models->Load(modelsFolder, false, false);
  } catch (std::exception &e) {
    RG_ERR_CLOSE(
        "InferUnit: Exception when trying to load models: " << e.what());
  }
}

RLGC::Action GGL::InferUnit::InferAction(const RLGC::Player &player,
                                         const RLGC::GameState &state,
                                         bool deterministic,
                                         float temperature) {
  return BatchInferActions({player}, {state}, deterministic, temperature)[0];
}

std::vector<RLGC::Action>
GGL::InferUnit::BatchInferActions(const std::vector<RLGC::Player> &players,
                                  const std::vector<RLGC::GameState> &states,
                                  bool deterministic, float temperature) {
  RG_ASSERT(players.size() > 0 && states.size() > 0);
  RG_ASSERT(players.size() == states.size());

  int batchSize = players.size();
  int actionAmount = actionParser->GetActionAmount();
  int combinedWidth = obsSize + actionAmount;

  bool usePinned = useGPU;
  auto combinedOptions = torch::TensorOptions().dtype(torch::kFloat32);
  if (usePinned) {
    combinedOptions = combinedOptions.pinned_memory(true);
  }

  thread_local TensorPool hostPool;
  thread_local TensorPool devicePool;

  auto tCombinedHost =
      hostPool.Acquire({batchSize, combinedWidth}, combinedOptions, usePinned);
  float *combinedPtr = tCombinedHost.data_ptr<float>();

  auto fnPrepareBatch = [&](int i) {
    FList curObs = obsBuilder->BuildObs(players[i], states[i]);
    if (curObs.size() != obsSize) {
      RG_ERR_CLOSE("InferUnit: Obs builder produced an obs that differs from "
                   "the provided size (expected: "
                   << obsSize << ", got: " << curObs.size() << ")\n"
                   << "Make sure you provided the correct obs size to the "
                      "InferUnit constructor.\n"
                   << "Also, make sure there aren't an incorrect number of "
                      "players (there are "
                   << states[i].players.size() << " in this state)");
    }
    float *rowPtr = combinedPtr + (static_cast<int64_t>(i) * combinedWidth);
    std::copy(curObs.begin(), curObs.end(), rowPtr);

    auto curMask = actionParser->GetActionMask(players[i], states[i]);
    if (curMask.size() != actionAmount) {
      RG_ERR_CLOSE("InferUnit: Action parser produced a mask that differs from "
                   "the provided action amount (expected: "
                   << actionAmount << ", got: " << curMask.size() << ")");
    }
    float *maskPtr = rowPtr + obsSize;
    for (int j = 0; j < actionAmount; j++)
      maskPtr[j] = static_cast<float>(curMask[j]);
  };

  RLGC::g_ThreadPool.StartBatchedJobs(fnPrepareBatch, batchSize, false);

  std::vector<RLGC::Action> results = {};

  try {
    RG_NO_GRAD;

    auto device = useGPU ? torch::kCUDA : torch::kCPU;
    torch::Tensor tActions, tLogProbs;

#ifdef RG_CUDA_SUPPORT
    if (useGPU) {
      auto stream = ggl_cuda::getStreamFromPool();
      ggl_cuda::CUDAStreamGuard streamGuard(stream);

      auto deviceOptions =
          torch::TensorOptions().dtype(torch::kFloat32).device(device);
      auto tCombinedDevice =
          devicePool.Acquire({batchSize, combinedWidth}, deviceOptions, false);
      tCombinedDevice.copy_(tCombinedHost, /*non_blocking=*/true);

      auto tObs = tCombinedDevice.narrow(1, 0, obsSize);
      auto tActionMasks = tCombinedDevice.narrow(1, obsSize, actionAmount);
      PPOLearner::InferActionsFromModels(*models, tObs, tActionMasks,
                                         deterministic, temperature, false,
                                         &tActions, &tLogProbs);

      stream.synchronize();
      devicePool.Release(std::move(tCombinedDevice));
    } else {
      auto tObs = tCombinedHost.narrow(1, 0, obsSize);
      auto tActionMasks = tCombinedHost.narrow(1, obsSize, actionAmount);
      PPOLearner::InferActionsFromModels(*models, tObs, tActionMasks,
                                         deterministic, temperature, false,
                                         &tActions, &tLogProbs);
    }
#else
    if (useGPU)
      RG_ERR_CLOSE("InferUnit: CUDA support is not enabled in this build.");
    {
      auto tObs = tCombinedHost.narrow(1, 0, obsSize);
      auto tActionMasks = tCombinedHost.narrow(1, obsSize, actionAmount);
      PPOLearner::InferActionsFromModels(*models, tObs, tActionMasks,
                                         deterministic, temperature, false,
                                         &tActions, &tLogProbs);
    }
#endif

    auto actionIndices = TENSOR_TO_VEC<int>(tActions);

    for (int i = 0; i < batchSize; i++)
      results.push_back(
          actionParser->ParseAction(actionIndices[i], players[i], states[i]));

  } catch (std::exception &e) {
    RG_ERR_CLOSE("InferUnit: Exception when inferring model: " << e.what());
  }

  hostPool.Release(std::move(tCombinedHost));

  return results;
}

GGL::InferUnit::~InferUnit() {
  if (models) {
    models->Free();
    models.reset();
  }
}
