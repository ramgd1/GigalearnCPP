#include "Learner.h"

#include <GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <GigaLearnCPP/PPO/PPOLearner.h>

#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <pybind11/embed.h>
#include <queue>
#include <random>
#include <thread>
#include <torch/cuda.h>

#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>
#endif
#include <RLGymCPP/ThreadPool.h>
#include <private/GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <private/GigaLearnCPP/PPO/GAE.h>
#include <private/GigaLearnCPP/PolicyVersionManager.h>
#include <private/GigaLearnCPP/Util/Models.h>

#include "Util/AvgTracker.h"
#include "Util/KeyPressDetector.h"
#include <private/GigaLearnCPP/Util/WelfordStat.h>

using namespace RLGC;

namespace GGL {
struct SaveJob {
  std::filesystem::path folder;
  ModelSet models;
  bool saveOptims = true;
};

class SaveWorker {
public:
  SaveWorker() {
    worker = std::thread([this]() { Run(); });
  }

  ~SaveWorker() { Stop(); }

  void Enqueue(SaveJob job) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      queue.push(std::move(job));
    }
    cv.notify_one();
  }

  void Flush() {
    std::unique_lock<std::mutex> lock(mutex);
    flushCv.wait(lock, [&]() { return queue.empty() && !busy; });
  }

private:
  void Stop() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stop = true;
    }
    cv.notify_all();
    if (worker.joinable())
      worker.join();
  }

  void Run() {
    while (true) {
      SaveJob job;
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&]() { return stop || !queue.empty(); });
        if (stop && queue.empty())
          return;
        job = std::move(queue.front());
        queue.pop();
        busy = true;
      }

      try {
        job.models.Save(job.folder, job.saveOptims);
      } catch (const std::exception &e) {
        RG_LOG("SaveWorker: Failed to save checkpoint at " << job.folder
                                                           << ". Exception: "
                                                           << e.what());
      } catch (...) {
        RG_LOG("SaveWorker: Failed to save checkpoint at "
               << job.folder << ". Unknown exception.");
      }

      {
        std::lock_guard<std::mutex> lock(mutex);
        busy = false;
        if (queue.empty())
          flushCv.notify_all();
      }
    }
  }

  std::mutex mutex;
  std::condition_variable cv;
  std::condition_variable flushCv;
  std::queue<SaveJob> queue;
  std::thread worker;
  bool stop = false;
  bool busy = false;
};
} // namespace GGL

static void WriteCrashReport(const char *context, const std::string &runID,
                             uint64_t totalTimesteps,
                             uint64_t totalIterations,
                             const GGL::LearnerConfig &config,
                             const std::exception *e) {
  try {
    std::filesystem::path reportDir = "crash_reports";
    std::filesystem::create_directories(reportDir);

    uint64_t now = RS_CUR_MS();
    std::filesystem::path reportPath =
        reportDir / (RS_STR("crash_" << now << ".txt"));

    std::ofstream out(reportPath);
    if (!out.good()) {
      RG_LOG("WriteCrashReport: Failed to open crash report at " << reportPath);
      return;
    }

    out << "Context: " << context << "\n";
    out << "Time (ms): " << now << "\n";
    out << "Thread: " << std::this_thread::get_id() << "\n";
    out << "Run ID: " << runID << "\n";
    out << "Total Timesteps: " << totalTimesteps << "\n";
    out << "Total Iterations: " << totalIterations << "\n";
    out << "Config:\n";
    out << "  numGames: " << config.numGames << "\n";
    out << "  tickSkip: " << config.tickSkip << "\n";
    out << "  actionDelay: " << config.actionDelay << "\n";
    out << "  renderMode: " << (config.renderMode ? "true" : "false") << "\n";
    out << "  tsPerSave: " << config.tsPerSave << "\n";
    out << "  checkpointsToKeep: " << config.checkpointsToKeep << "\n";
    out << "  deviceType: " << (int)config.deviceType << "\n";
    out << "  ppo.tsPerItr: " << config.ppo.tsPerItr << "\n";
    out << "  ppo.batchSize: " << config.ppo.batchSize << "\n";
    out << "  ppo.miniBatchSize: " << config.ppo.miniBatchSize << "\n";
    out << "  ppo.epochs: " << config.ppo.epochs << "\n";
    out << "  ppo.useHalfPrecision: "
        << (config.ppo.useHalfPrecision ? "true" : "false") << "\n";
    out << "  ppo.policyLR: " << config.ppo.policyLR << "\n";
    out << "  ppo.criticLR: " << config.ppo.criticLR << "\n";

    if (e) {
      out << "Exception: " << e->what() << "\n";
    } else {
      out << "Exception: <unknown>\n";
    }
  } catch (std::exception &ex) {
    RG_LOG("WriteCrashReport: Failed with exception: " << ex.what());
  } catch (...) {
    RG_LOG("WriteCrashReport: Failed with unknown exception.");
  }
}

// Tensor utilities now in FrameworkTorch.h

GGL::Learner::Learner(EnvCreateFn envCreateFn, LearnerConfig config,
                      StepCallbackFn stepCallback)
    : envCreateFn(envCreateFn), config(config), stepCallback(stepCallback) {
  saveWorker = std::make_unique<SaveWorker>();

  pybind11::initialize_interpreter();

#ifndef NDEBUG
  RG_LOG("===========================");
  RG_LOG("WARNING: GigaLearn runs extremely slowly in debug, and there are "
         "often bizzare issues with debug-mode torch.");
  RG_LOG("It is recommended that you compile in release mode without "
         "optimization for debugging.");
  RG_SLEEP(1000);
#endif

  if (config.tsPerSave == 0)
    config.tsPerSave = config.ppo.tsPerItr;

  RG_LOG("Learner::Learner():");

  if (config.randomSeed == -1)
    config.randomSeed = RS_CUR_MS();

  RG_LOG("\tCheckpoint Save/Load Dir: " << config.checkpointFolder);

  torch::manual_seed(config.randomSeed);

  at::Device device = at::Device(at::kCPU);
  std::vector<torch::Device> cudaDevices = {};
  if (config.deviceType == LearnerDeviceType::GPU_CUDA ||
      (config.deviceType == LearnerDeviceType::AUTO &&
       torch::cuda::is_available())) {
    RG_LOG("\tUsing CUDA GPU device(s)...");

    int deviceCount = torch::cuda::device_count();
    if (deviceCount <= 0)
      RG_ERR_CLOSE("Learner::Learner(): CUDA was reported available but "
                   "no CUDA devices were found.");

    std::vector<int> deviceIds = config.cudaDeviceIds;
    if (deviceIds.empty())
      deviceIds.push_back(0);

    // Validate and build device list
    for (int id : deviceIds) {
      if (id < 0 || id >= deviceCount)
        RG_ERR_CLOSE("Learner::Learner(): Invalid CUDA device id " << id
                                                                   << " (found "
                                                                   << deviceCount
                                                                   << " devices)");
      cudaDevices.emplace_back(torch::kCUDA, id);
    }

    // Test out moving a tensor to GPU and back to make sure the device is
    // working
    torch::Tensor t;
    bool deviceTestFailed = false;
    try {
      t = torch::tensor(0);
      t = t.to(cudaDevices[0]);
      t = t.cpu();
    } catch (...) {
      deviceTestFailed = true;
    }

    if (!torch::cuda::is_available() || deviceTestFailed)
      RG_ERR_CLOSE("Learner::Learner(): Can't use CUDA GPU because "
                   << (torch::cuda::is_available()
                           ? "libtorch cannot access the GPU"
                           : "CUDA is not available to libtorch")
                   << ".\n"
                   << "Make sure your libtorch comes with CUDA support, and "
                      "that CUDA is installed properly.")
    device = cudaDevices[0];
  } else {
    RG_LOG("\tUsing CPU device...");
    device = at::Device(at::kCPU);
  }

  if (RocketSim::GetStage() != RocketSimStage::INITIALIZED) {
    RG_LOG("\tInitializing RocketSim...");
    RocketSim::Init("collision_meshes", true);
  }

  {
    RG_LOG("\tCreating envs...");
    EnvSetConfig envSetConfig = {};
    envSetConfig.envCreateFn = envCreateFn;
    envSetConfig.numArenas = config.renderMode ? 1 : config.numGames;
    envSetConfig.tickSkip = config.tickSkip;
    envSetConfig.actionDelay = config.actionDelay;
    envSetConfig.saveRewards = config.addRewardsToMetrics;
    envSet = new RLGC::EnvSet(envSetConfig);
    obsSize = envSet->state.obs.size[1];
    numActions = envSet->actionParsers[0]->GetActionAmount();
  }

  {
    if (config.standardizeReturns) {
      this->returnStat = new WelfordStat();
    } else {
      this->returnStat = NULL;
    }

    if (config.standardizeObs) {
      this->obsStat = new BatchedWelfordStat(obsSize);
    } else {
      this->obsStat = NULL;
    }
  }

  try {
    RG_LOG("\tMaking PPO learner...");
    ppo = new PPOLearner(obsSize, numActions, config.ppo, device, cudaDevices);
  } catch (std::exception &e) {
    RG_ERR_CLOSE("Failed to create PPO learner: " << e.what());
  }

  if (config.renderMode) {
    renderSender = new RenderSender(config.renderTimeScale);
  } else {
    renderSender = NULL;
  }

  if (config.skillTracker.enabled || config.trainAgainstOldVersions)
    config.savePolicyVersions = true;

  if (config.savePolicyVersions && !config.renderMode) {
    if (config.checkpointFolder.empty())
      RG_ERR_CLOSE("Cannot save/load old policy versions with no checkpoint "
                   "save folder");
    versionMgr = new PolicyVersionManager(
        config.checkpointFolder / "policy_versions", config.maxOldVersions,
        config.tsPerVersion, config.skillTracker, envSet->config);
  } else {
    versionMgr = NULL;
  }

  if (!config.checkpointFolder.empty())
    Load();

  if (config.savePolicyVersions && !config.renderMode) {
    if (config.checkpointFolder.empty())
      RG_ERR_CLOSE("Cannot save/load old policy versions with no checkpoint "
                   "save folder");
    auto models = ppo->GetPolicyModels();
    versionMgr->LoadVersions(models, static_cast<int64_t>(totalTimesteps));
  }

  if (config.metricsType != LearnerConfig::MetricsType::PYTHON_WANDB) {
    // If not using python metrics, we can skip initializing the python
    // interpreter if it's not needed by other things
  }

  if (!config.renderMode) {
    if (!runID.empty())
      RG_LOG("\tRun ID: " << runID);
    metricSender = MetricSender::Make(config, runID);
  } else {
    metricSender = NULL;
  }

  RG_LOG(RG_DIVIDER);
}

void GGL::Learner::SaveStats(std::filesystem::path path) {
  using namespace nlohmann;

  constexpr const char *ERROR_PREFIX = "Learner::SaveStats(): ";

  std::ofstream fOut(path);
  if (!fOut.good())
    RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

  json j = {};
  j["total_timesteps"] = totalTimesteps;
  j["total_iterations"] = totalIterations;

  if (config.metricsType != LearnerConfig::MetricsType::DISABLED)
    j["run_id"] = metricSender->GetRunID();

  if (returnStat)
    j["return_stat"] = returnStat->ToJSON();
  if (obsStat)
    j["obs_stat"] = obsStat->ToJSON();

  if (versionMgr)
    versionMgr->AddRunningStatsToJSON(j);

  std::string jStr = j.dump();
  fOut << jStr;
}

void GGL::Learner::LoadStats(std::filesystem::path path) {
  // TODO: Repetitive code, merge repeated code into one function called from
  // both SaveStats() and LoadStats()

  using namespace nlohmann;
  constexpr const char *ERROR_PREFIX = "Learner::LoadStats(): ";

  std::ifstream fIn(path);
  if (!fIn.good())
    RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

  json j = json::parse(fIn);
  totalTimesteps = j["total_timesteps"];
  totalIterations = j["total_iterations"];

  if (j.contains("run_id"))
    runID = j["run_id"];

  if (returnStat)
    returnStat->ReadFromJSON(j["return_stat"]);
  if (obsStat)
    obsStat->ReadFromJSON(j["obs_stat"]);

  if (versionMgr)
    versionMgr->LoadRunningStatsFromJSON(j);
}

// Different than RLGym-PPO to show that they are not compatible
constexpr const char *STATS_FILE_NAME = "RUNNING_STATS.json";

void GGL::Learner::Save() {
  if (config.checkpointFolder.empty())
    RG_ERR_CLOSE("Learner::Save(): Cannot save because "
                 "config.checkpointSaveFolder is not set");

  std::filesystem::path saveFolder =
      config.checkpointFolder / std::to_string(totalTimesteps);
  try {
    std::filesystem::create_directories(saveFolder);
  } catch (std::exception &e) {
    RG_LOG("Learner::Save(): Failed to create save directory " << saveFolder
                                                               << ". "
                                                               << e.what());
    return;
  }

  RG_LOG("Saving to folder " << saveFolder << "...");
  try {
    SaveStats(saveFolder / STATS_FILE_NAME);
  } catch (std::exception &e) {
    RG_LOG("Learner::Save(): Failed to save stats file. " << e.what());
  }
  {
    SaveJob job = {};
    job.folder = saveFolder;
    job.models = ppo->CloneModelsWithOptims();
    job.saveOptims = (totalIterations % ppo->config.optimSaveInterval == 0);
    saveWorker->Enqueue(std::move(job));
  }

  // Remove old checkpoints
  if (config.checkpointsToKeep != -1) {
    std::set<int64_t> allSavedTimesteps =
        Utils::FindNumberedDirs(config.checkpointFolder);
    while (allSavedTimesteps.size() > config.checkpointsToKeep) {
      int64_t lowestCheckpointTS = INT64_MAX;
      for (int64_t savedTimesteps : allSavedTimesteps)
        lowestCheckpointTS = RS_MIN(lowestCheckpointTS, savedTimesteps);

      std::filesystem::path removePath =
          config.checkpointFolder / std::to_string(lowestCheckpointTS);
      try {
        std::filesystem::remove_all(removePath);
      } catch (std::exception &e) {
        RG_LOG("Failed to remove old checkpoint from "
               << removePath << ", exception: " << e.what());
      }
      allSavedTimesteps.erase(lowestCheckpointTS);
    }
  }

  if (versionMgr)
    versionMgr->SaveVersions();

  RG_LOG(" > Done.");
}

void GGL::Learner::Load() {
  if (config.checkpointFolder.empty())
    RG_ERR_CLOSE("Learner::Load(): Cannot load because "
                 "config.checkpointLoadFolder is not set");

  RG_LOG("Loading most recent checkpoint in " << config.checkpointFolder
                                              << "...");

  int64_t highest = -1;
  std::set<int64_t> allSavedTimesteps =
      Utils::FindNumberedDirs(config.checkpointFolder);
  for (int64_t timesteps : allSavedTimesteps)
    highest = RS_MAX(timesteps, highest);

  if (highest != -1) {
    std::filesystem::path loadFolder =
        config.checkpointFolder / std::to_string(highest);
    RG_LOG(" > Loading checkpoint " << loadFolder << "...");
    if (!config.expectedRunID.empty()) {
      std::filesystem::path statsPath = loadFolder / STATS_FILE_NAME;
      if (std::filesystem::exists(statsPath)) {
        try {
          nlohmann::json j = nlohmann::json::parse(std::ifstream(statsPath));
          if (j.contains("run_id")) {
            std::string savedRunID = j["run_id"];
            if (savedRunID != config.expectedRunID) {
              while (true) {
                std::cout << "Checkpoint run_id mismatch.\n"
                          << "Saved run_id: " << savedRunID << "\n"
                          << "Expected run_id: " << config.expectedRunID << "\n"
                          << "Continue and use checkpoint run_id? [Y/N]: ";
                std::string input;
                if (!std::getline(std::cin, input)) {
                  RG_ERR_CLOSE("Learner::Load(): Failed to read user input for "
                               "run_id mismatch prompt");
                }

                if (input.empty())
                  continue;
                char c = static_cast<char>(toupper(input[0]));
                if (c == 'Y') {
                  break;
                }
                if (c == 'N') {
                  RG_LOG(" > Deleting checkpoints and starting a new run.");
                  std::filesystem::remove_all(config.checkpointFolder);
                  runID.clear();
                  totalTimesteps = 0;
                  totalIterations = 0;
                  return;
                }
              }
            }
          }
        } catch (std::exception &e) {
          RG_ERR_CLOSE(
              "Learner::Load(): Failed to read run_id for mismatch check: "
              << e.what());
        }
      }
    }
    try {
      LoadStats(loadFolder / STATS_FILE_NAME);
    } catch (std::exception &e) {
      RG_LOG("Learner::Load(): Failed to load stats. " << e.what());
    }
    try {
      ppo->LoadFrom(loadFolder);
    } catch (std::exception &e) {
      RG_LOG("Learner::Load(): Failed to load PPO models. " << e.what());
    }
    if (config.autoTransferLearnOnMismatch && ConsumeTransferLearnRequest()) {
      pendingTransferLearn = true;
      pendingTransferLearnPath = loadFolder;
    }
    RG_LOG(" > Done.");
  } else {
    RG_LOG(" > No checkpoints found, starting new model.")
    if (config.expectedRunID.empty()) {
      uint64_t now = RS_CUR_MS();
      uint32_t randPart = std::random_device{}();
      runID = RS_STR(now << "_" << randPart);
      config.expectedRunID = runID;
    }
  }
}

void GGL::Learner::StartQuitKeyThread(std::atomic<bool> &quitPressed,
                                      std::thread &outThread) {
  quitPressed.store(false);

  RG_LOG("Press 'Q' to save and quit!");
  outThread = std::thread([&] {
    quitThreadStop = false;
    while (!quitThreadStop.load()) {
      char c = toupper(KeyPressDetector::GetPressedChar());
      if (c == 'Q') {
        RG_LOG("Save queued, will save and exit next iteration.");
        quitPressed.store(true);
      }
      THREAD_WAIT();
    }
  });
}
void GGL::Learner::StartTransferLearn(const TransferLearnConfig &tlConfig) {

  RG_LOG("Starting transfer learning...");

  // TODO: Lots of manual obs builder stuff going on which is quite volatile
  //	Although I can't really think another way to do this

  std::vector<ObsBuilder *> oldObsBuilders = {};
  for (int i = 0; i < envSet->arenas.size(); i++)
    oldObsBuilders.push_back(tlConfig.makeOldObsFn());

  // Reset all obs builders initially
  for (int i = 0; i < envSet->arenas.size(); i++)
    oldObsBuilders[i]->Reset(envSet->state.gameStates[0]);

  std::vector<ActionParser *> oldActionParsers = {};
  for (int i = 0; i < envSet->arenas.size(); i++)
    oldActionParsers.push_back(tlConfig.makeOldActFn());

  int oldNumActions = oldActionParsers[0]->GetActionAmount();

  if (oldNumActions != numActions) {
    if (!tlConfig.mapActsFn) {
      RG_ERR_CLOSE(
          "StartTransferLearn: Old and new action parsers have a different "
          "number of actions, but tlConfig.mapActsFn is NULL.\n"
          << "You must implement this function to translate the action "
             "indices.");
    };
  }

  // Determine old obs size
  int oldObsSize;
  {
    GameState testState = envSet->state.gameStates[0];
    oldObsSize =
        oldObsBuilders[0]->BuildObs(testState.players[0], testState).size();
  }

  ModelSet oldModels = {};
  {
    RG_NO_GRAD;
    PPOLearner::MakeModels(
        false, oldObsSize, oldNumActions, tlConfig.oldSharedHeadConfig,
        tlConfig.oldPolicyConfig, {}, ppo->device, oldModels);

    oldModels.Load(tlConfig.oldModelsPath, false, false);
  }

  auto cleanupTransferLearn = [&]() {
    for (auto *builder : oldObsBuilders)
      delete builder;
    oldObsBuilders.clear();
    for (auto *parser : oldActionParsers)
      delete parser;
    oldActionParsers.clear();
    oldModels.Free();
  };

  std::atomic<bool> saveQueued(false);
  std::thread keyPressThread;
  try {
    StartQuitKeyThread(saveQueued, keyPressThread);
    bool quitRequested = false;

    while (true) {
      Report report = {};
      bool pinHostMemory = ppo->device.is_cuda();
      thread_local TensorPool hostPool(16);

      // Collect obs
      std::vector<float> allNewObs = {};
      std::vector<float> allOldObs = {};
      std::vector<uint8_t> allNewActionMasks = {};
      std::vector<uint8_t> allOldActionMasks = {};
      std::vector<int> allActionMaps = {};
      int stepsCollected;
      {
        RG_NO_GRAD;
        for (stepsCollected = 0; stepsCollected < tlConfig.batchSize;
             stepsCollected += envSet->state.numPlayers) {

          auto terminals = envSet->state.terminals; // Backup
          envSet->Reset();
          for (int i = 0; i < envSet->arenas.size();
               i++) // Manually reset old obs builders
            if (terminals[i])
              oldObsBuilders[i]->Reset(envSet->state.gameStates[i]);

          torch::Tensor tActions, tLogProbs;
          auto tStates = MakeHostTensorFromBlob<float>(
              envSet->state.obs.data.data(),
              {(int64_t)envSet->state.obs.size[0],
               (int64_t)envSet->state.obs.size[1]},
              torch::kFloat32, pinHostMemory,
              pinHostMemory ? &hostPool : nullptr);
          auto tActionMasks = MakeHostTensorFromBlob<uint8_t>(
              envSet->state.actionMasks.data.data(),
              {(int64_t)envSet->state.actionMasks.size[0],
               (int64_t)envSet->state.actionMasks.size[1]},
              torch::kUInt8, pinHostMemory,
              pinHostMemory ? &hostPool : nullptr);

          envSet->StepFirstHalf(true);

          allNewObs += envSet->state.obs.data;
          allNewActionMasks += envSet->state.actionMasks.data;

          // Run all old obs and old action parser on each player
          // TODO: Could be multithreaded
          for (int arenaIdx = 0; arenaIdx < envSet->arenas.size(); arenaIdx++) {
            auto &gs = envSet->state.gameStates[arenaIdx];
            for (auto &player : gs.players) {
              allOldObs += oldObsBuilders[arenaIdx]->BuildObs(player, gs);
              allOldActionMasks +=
                  oldActionParsers[arenaIdx]->GetActionMask(player, gs);

              if (tlConfig.mapActsFn) {
                auto curMap = tlConfig.mapActsFn(player, gs);
                if (curMap.size() != numActions)
                  RG_ERR_CLOSE(
                      "StartTransferLearn: Your action map must have the same "
                      "size as the new action parser's actions");
                allActionMaps += curMap;
              }
            }
          }

          ppo->InferActions(tStates.to(ppo->device, true),
                            tActionMasks.to(ppo->device, true), &tActions,
                            &tLogProbs);
          if (pinHostMemory) {
            hostPool.Release(std::move(tStates));
            hostPool.Release(std::move(tActionMasks));
          }

          auto curActions = TENSOR_TO_VEC<int>(tActions);

          envSet->Sync();
          envSet->StepSecondHalf(curActions, false);

          if (stepCallback)
            stepCallback(this, envSet->state.gameStates, report);
        }
      }

      uint64_t prevTimesteps = totalTimesteps;
      totalTimesteps += stepsCollected;
      report["Total Timesteps"] = totalTimesteps;
      report["Collected Timesteps"] = stepsCollected;
      totalIterations++;
      report["Total Iterations"] = totalIterations;

      // Make tensors
      int64_t newObsRows = static_cast<int64_t>(allNewObs.size() / obsSize);
      int64_t oldObsRows = static_cast<int64_t>(allOldObs.size() / oldObsSize);
      int64_t newMaskRows =
          static_cast<int64_t>(allNewActionMasks.size() / numActions);
      int64_t oldMaskRows =
          static_cast<int64_t>(allOldActionMasks.size() / oldNumActions);

      torch::Tensor tNewObs =
          MakeHostTensorFromVec<float>(allNewObs, {newObsRows, obsSize},
                                       torch::kFloat32, pinHostMemory)
              .to(ppo->device, true);
      torch::Tensor tOldObs =
          MakeHostTensorFromVec<float>(allOldObs, {oldObsRows, oldObsSize},
                                       torch::kFloat32, pinHostMemory)
              .to(ppo->device, true);
      torch::Tensor tNewActionMasks =
          MakeHostTensorFromVec<uint8_t>(allNewActionMasks,
                                         {newMaskRows, numActions},
                                         torch::kUInt8, pinHostMemory)
              .to(ppo->device, true);
      torch::Tensor tOldActionMasks =
          MakeHostTensorFromVec<uint8_t>(allOldActionMasks,
                                         {oldMaskRows, oldNumActions},
                                         torch::kUInt8, pinHostMemory)
              .to(ppo->device, true);

      torch::Tensor tActionMaps = {};
      if (!allActionMaps.empty()) {
        int64_t actionMapRows =
            static_cast<int64_t>(allActionMaps.size() / numActions);
        std::vector<int64_t> actionMaps64;
        actionMaps64.reserve(allActionMaps.size());
        for (int value : allActionMaps)
          actionMaps64.push_back(static_cast<int64_t>(value));
        tActionMaps = MakeHostTensorFromVec<int64_t>(
                          actionMaps64, {actionMapRows, numActions},
                          torch::kInt64, pinHostMemory)
                          .to(ppo->device, true);
      }

      // Transfer learn
      ppo->TransferLearn(oldModels, tNewObs, tOldObs, tNewActionMasks,
                         tOldActionMasks, tActionMaps, report, tlConfig);

      if (versionMgr)
        versionMgr->OnIteration(ppo, report, totalTimesteps, prevTimesteps);

      if (saveQueued.load()) {
        if (!config.checkpointFolder.empty())
          Save();
        quitRequested = true;
        break;
      }

      if (!config.checkpointFolder.empty()) {
        if (totalTimesteps / config.tsPerSave >
            prevTimesteps / config.tsPerSave) {
          // Auto-save
          Save();
        }
      }

      report.Finish();

      if (metricSender) {
        try {
          metricSender->Send(report);
        } catch (std::exception &e) {
          RG_LOG("MetricSender::Send failed: " << e.what());
        }
      }

      try {
        report.Display({"Transfer Learn Accuracy", "Transfer Learn Loss", "",
                        "Policy Entropy", "Old Policy Entropy",
                        "Policy Update Magnitude", "", "Collected Timesteps",
                        "Total Timesteps", "Total Iterations"});
      } catch (std::exception &e) {
        RG_LOG("Report::Display failed: " << e.what());
      }
      if (quitRequested)
        break;
    }

    quitThreadStop = true;
    if (keyPressThread.joinable())
      keyPressThread.join();
    if (quitRequested) {
      if (metricSender)
        metricSender->Flush();
      saveWorker->Flush();
      cleanupTransferLearn();
      std::exit(0);
    }

  } catch (std::exception &e) {
    quitThreadStop = true;
    if (keyPressThread.joinable())
      keyPressThread.join();
    if (metricSender)
      metricSender->Flush();
    saveWorker->Flush();
    cleanupTransferLearn();
    WriteCrashReport("transfer_learn_loop", runID, totalTimesteps,
                     totalIterations, config, &e);
    RG_ERR_CLOSE("Exception thrown during transfer learn loop: " << e.what());
  }
}

void GGL::Learner::Start() {
  if (pendingTransferLearn) {
    TransferLearnConfig tlConfig = config.transferLearnConfig;
    tlConfig.oldModelsPath = pendingTransferLearnPath;
    StartTransferLearn(tlConfig);
    return;
  }

  bool render = config.renderMode;

  RG_LOG("Learner::Start():");
  RG_LOG("\tObs size: " << obsSize);
  RG_LOG("\tAction amount: " << numActions);

  if (render)
    RG_LOG("\t(Render mode enabled)");

  std::atomic<bool> saveQueued(false);
  std::thread keyPressThread;
  try {
    StartQuitKeyThread(saveQueued, keyPressThread);
    bool quitRequested = false;

    ExperienceBuffer experience =
        ExperienceBuffer(config.randomSeed, torch::kCPU);

    int numPlayers = envSet->state.numPlayers;

    struct Trajectory {
      FList states, nextStates, rewards, logProbs;
      std::vector<uint8_t> actionMasks;
      std::vector<int8_t> terminals;
      std::vector<int32_t> actions;

      void Clear() { *this = Trajectory(); }
      void Reserve(size_t capacity, int obsSize, int numActions) {
        states.reserve(capacity * obsSize);
        nextStates.reserve(capacity * obsSize);
        rewards.reserve(capacity);
        logProbs.reserve(capacity);
        actionMasks.reserve(capacity * numActions);
        terminals.reserve(capacity);
        actions.reserve(capacity);
      }

      void Append(const Trajectory &other) {
        states += other.states;
        nextStates += other.nextStates;
        rewards += other.rewards;
        logProbs += other.logProbs;
        actionMasks += other.actionMasks;
        terminals += other.terminals;
        actions += other.actions;
      }

      size_t Length() const { return actions.size(); }
    };

    auto trajectories = std::vector<Trajectory>(numPlayers, Trajectory{});
    for (auto &traj : trajectories)
      traj.Reserve(config.ppo.tsPerItr / numPlayers * 2, obsSize, numActions);

    int maxEpisodeLength =
        (int)(config.ppo.maxEpisodeDuration * (120.f / config.tickSkip));

    while (true) {
      Report report = {};
      bool pinHostMemory = ppo->device.is_cuda();
      thread_local TensorPool hostPool(32);
      thread_local TensorPool devicePool(32);

      bool isFirstIteration = (totalTimesteps == 0);

      // TODO: Old version switching messes up the gameplay potentially
      GGL::PolicyVersion *oldVersion = NULL;
      std::vector<bool> oldVersionPlayerMask;
      std::vector<int> newPlayerIndices = {}, oldPlayerIndices = {};
      torch::Tensor tNewPlayerIndices, tOldPlayerIndices;

      for (int i = 0; i < numPlayers; i++)
        newPlayerIndices.push_back(i);

      if (config.trainAgainstOldVersions) {
        RG_ASSERT(config.trainAgainstOldChance >= 0 &&
                  config.trainAgainstOldChance <= 1);
        bool shouldTrainAgainstOld =
            (RocketSim::Math::RandFloat() < config.trainAgainstOldChance) &&
            !versionMgr->versions.empty() && !render;

        if (shouldTrainAgainstOld) {
          // Set up training against old versions

          int oldVersionIdx =
              RocketSim::Math::RandInt(0, versionMgr->versions.size());
          oldVersion = &versionMgr->versions[oldVersionIdx];

          Team oldVersionTeam = Team(RocketSim::Math::RandInt(0, 2));

          newPlayerIndices.clear();
          oldVersionPlayerMask.resize(numPlayers);
          int i = 0;
          for (auto &state : envSet->state.gameStates) {
            for (auto &player : state.players) {
              if (player.team == oldVersionTeam) {
                oldVersionPlayerMask[i] = true;
                oldPlayerIndices.push_back(i);
              } else {
                oldVersionPlayerMask[i] = false;
                newPlayerIndices.push_back(i);
              }
              i++;
            }
          }

          tNewPlayerIndices = torch::tensor(newPlayerIndices);
          tOldPlayerIndices = torch::tensor(oldPlayerIndices);
        }
      }

      int numRealPlayers =
          oldVersion ? newPlayerIndices.size() : envSet->state.numPlayers;

      int stepsCollected = 0;
      { // Generate experience

        // Only contains complete episodes
        auto combinedTraj = Trajectory();

        Timer collectionTimer = {};
        { // Collect timesteps
          RG_NO_GRAD;

          float inferTime = 0;
          float envStepTime = 0;

          size_t targetTimesteps = static_cast<size_t>(config.ppo.tsPerItr);
          for (int step = 0; combinedTraj.Length() < targetTimesteps || render;
               step++, stepsCollected += numRealPlayers) {
            Timer stepTimer = {};
            envSet->Reset();
            envStepTime += stepTimer.Elapsed();

            for (float f : envSet->state.obs.data)
              if (isnan(f) || isinf(f))
                RG_ERR_CLOSE("Obs builder produced a NaN/inf value");

            if (!render && obsStat) {
              // TODO: This samples from old versions too
              int numSamples =
                  RS_MAX(envSet->state.numPlayers, config.maxObsSamples);
              for (int i = 0; i < numSamples; i++) {
                int idx = Math::RandInt(0, envSet->state.numPlayers);
                obsStat->IncrementRow(&envSet->state.obs.At(idx, 0));
              }

              std::vector<double> mean = obsStat->GetMean();
              std::vector<double> std = obsStat->GetSTD();
              for (double &f : mean)
                f = RS_CLAMP(f, -config.maxObsMeanRange,
                             config.maxObsMeanRange);
              for (double &f : std)
                f = RS_MAX(f, config.minObsSTD);

              auto fnNormalize = [&](int i) {
                for (int j = 0; j < obsSize; j++) {
                  float &obsVal = envSet->state.obs.At(i, j);
                  obsVal = (obsVal - mean[j]) / std[j];
                }
              };
              g_ThreadPool.StartBatchedJobs(fnNormalize,
                                            envSet->state.numPlayers, false);
            }

            torch::Tensor tActions, tLogProbs;
            auto tStates = MakeHostTensorFromBlob<float>(
                envSet->state.obs.data.data(),
                {(int64_t)envSet->state.obs.size[0],
                 (int64_t)envSet->state.obs.size[1]},
                torch::kFloat32, pinHostMemory,
                pinHostMemory ? &hostPool : nullptr);
            auto tActionMasks = MakeHostTensorFromBlob<uint8_t>(
                envSet->state.actionMasks.data.data(),
                {(int64_t)envSet->state.actionMasks.size[0],
                 (int64_t)envSet->state.actionMasks.size[1]},
                torch::kUInt8, pinHostMemory,
                pinHostMemory ? &hostPool : nullptr);

            if (!render) {
              for (int newPlayerIdx : newPlayerIndices) {
                trajectories[newPlayerIdx].states +=
                    envSet->state.obs.GetRow(newPlayerIdx);
                trajectories[newPlayerIdx].actionMasks +=
                    envSet->state.actionMasks.GetRow(newPlayerIdx);
              }
            }

            envSet->StepFirstHalf(true);

            Timer inferTimer = {};

            if (oldVersion) {
              torch::Tensor tdNewStates =
                  tStates.index_select(0, tNewPlayerIndices)
                      .to(ppo->device, true);
              torch::Tensor tdOldStates =
                  tStates.index_select(0, tOldPlayerIndices)
                      .to(ppo->device, true);
              torch::Tensor tdNewActionMasks =
                  tActionMasks.index_select(0, tNewPlayerIndices)
                      .to(ppo->device, true);
              torch::Tensor tdOldActionMasks =
                  tActionMasks.index_select(0, tOldPlayerIndices)
                      .to(ppo->device, true);

              torch::Tensor tNewActions;
              torch::Tensor tOldActions;

              ppo->InferActions(tdNewStates, tdNewActionMasks, &tNewActions,
                                &tLogProbs);
              ppo->InferActions(tdOldStates, tdOldActionMasks, &tOldActions,
                                NULL, &oldVersion->models);

              tActions = torch::zeros(numPlayers, tNewActions.dtype());
              tActions.index_copy_(0, tNewPlayerIndices, tNewActions.cpu());
              tActions.index_copy_(0, tOldPlayerIndices, tOldActions.cpu());
            } else {
              torch::Tensor tdStates = devicePool.Acquire(
                  tStates.sizes(), tStates.options().device(ppo->device),
                  false);
              tdStates.copy_(tStates, true);

              torch::Tensor tdActionMasks = devicePool.Acquire(
                  tActionMasks.sizes(),
                  tActionMasks.options().device(ppo->device), false);
              tdActionMasks.copy_(tActionMasks, true);

              ppo->InferActions(tdStates, tdActionMasks, &tActions, &tLogProbs);
              tActions = tActions.cpu();

              devicePool.Release(std::move(tdStates));
              devicePool.Release(std::move(tdActionMasks));
            }
            if (pinHostMemory) {
              hostPool.Release(std::move(tStates));
              hostPool.Release(std::move(tActionMasks));
            }
            inferTime += inferTimer.Elapsed();

            auto curActions = TENSOR_TO_VEC<int>(tActions);
            FList newLogProbs;
            if (tLogProbs.defined() && !render)
              newLogProbs = TENSOR_TO_VEC<float>(tLogProbs);

            stepTimer.Reset();
            envSet->Sync(); // Make sure the first half is done
            envSet->StepSecondHalf(curActions, false);
            envStepTime += stepTimer.Elapsed();

            if (stepCallback)
              stepCallback(this, envSet->state.gameStates, report);

            if (render) {
              renderSender->Send(envSet->state.gameStates[0]);
              continue;
            }

            // Calc average rewards
            if (config.addRewardsToMetrics &&
                (Math::RandInt(0, config.rewardSampleRandInterval) == 0)) {
              int numSamples =
                  RS_MIN(envSet->arenas.size(), config.maxRewardSamples);
              std::unordered_map<std::string, AvgTracker> avgRewards = {};
              for (int i = 0; i < numSamples; i++) {
                int arenaIdx = Math::RandInt(0, envSet->arenas.size());
                auto &prevRewards = envSet->state.lastRewards[i];

                for (int j = 0; j < envSet->rewards[arenaIdx].size(); j++) {
                  std::string rewardName =
                      envSet->rewards[arenaIdx][j].reward->GetName();
                  avgRewards[rewardName] += prevRewards[j];
                }
              }

              for (auto &pair : avgRewards)
                report.AddAvg("Rewards/" + pair.first, pair.second.Get());
            }

            // Now that we've inferred and stepped the env, we can add that
            // stuff to the trajectories
            auto fnUpdateTraj = [&](int i) {
              int newPlayerIdx = newPlayerIndices[i];
              trajectories[newPlayerIdx].actions.push_back(
                  curActions[newPlayerIdx]);
              trajectories[newPlayerIdx].rewards +=
                  envSet->state.rewards[newPlayerIdx];
              trajectories[newPlayerIdx].logProbs += newLogProbs[i];
            };
            g_ThreadPool.StartBatchedJobs(fnUpdateTraj, newPlayerIndices.size(),
                                          false);

            auto curTerminals = std::vector<uint8_t>(numPlayers, 0);
            auto fnUpdateTerminals = [&](int idx) {
              uint8_t terminalType = envSet->state.terminals[idx];
              if (!terminalType)
                return;

              auto playerStartIdx = envSet->state.arenaPlayerStartIdx[idx];
              int playersInArena = envSet->state.gameStates[idx].players.size();
              for (int i = 0; i < playersInArena; i++)
                curTerminals[playerStartIdx + i] = terminalType;
            };
            g_ThreadPool.StartBatchedJobs(fnUpdateTerminals,
                                          envSet->arenas.size(), false);

            for (int newPlayerIdx : newPlayerIndices) {
              int8_t terminalType = curTerminals[newPlayerIdx];
              auto &traj = trajectories[newPlayerIdx];

              if (!terminalType && traj.Length() >= maxEpisodeLength) {
                // Episode is too long, truncate it here
                // This won't actually reset the env, but rather will just add
                // it to experience buffer as truncated
                terminalType = RLGC::TerminalType::TRUNCATED;
              }

              traj.terminals.push_back(terminalType);
              if (terminalType) {

                if (terminalType == RLGC::TerminalType::TRUNCATED) {
                  // Truncation requires an additional next state for the critic
                  traj.nextStates += envSet->state.obs.GetRow(newPlayerIdx);
                }

                combinedTraj.Append(traj);
                traj.Clear();
              }
            }
          }

          report["Inference Time"] = inferTime;
          report["Env Step Time"] = envStepTime;
        }
        float collectionTime = collectionTimer.Elapsed();

        Timer consumptionTimer = {};
        torch::Tensor tRewards, tTerminals, tNextTruncStates;
        torch::Tensor tStates, tActionMasks, tActions, tLogProbs;
        torch::Tensor tAdvantages, tTargetVals, tReturns;
        { // Process timesteps
          RG_NO_GRAD;

          // Make and transpose tensors
          int64_t trajRows =
              static_cast<int64_t>(combinedTraj.states.size() / obsSize);
          int64_t maskRows = static_cast<int64_t>(
              combinedTraj.actionMasks.size() / numActions);

          std::vector<int64_t> actions64;
          actions64.reserve(combinedTraj.actions.size());
          for (int32_t value : combinedTraj.actions)
            actions64.push_back(static_cast<int64_t>(value));

          // Parallelize tensor creation
          auto fnMakeTensors = [&](int idx) {
            if (idx == 0)
              tStates = MakeHostTensorFromVec<float>(
                  combinedTraj.states, {trajRows, obsSize}, torch::kFloat32,
                  pinHostMemory, pinHostMemory ? &hostPool : nullptr);
            else if (idx == 1)
              tActionMasks = MakeHostTensorFromVec<uint8_t>(
                  combinedTraj.actionMasks, {maskRows, numActions},
                  torch::kUInt8, pinHostMemory,
                  pinHostMemory ? &hostPool : nullptr);
            else if (idx == 2)
              tActions = MakeHostTensorFromVec<int64_t>(
                  actions64, {static_cast<int64_t>(actions64.size())},
                  torch::kInt64, pinHostMemory,
                  pinHostMemory ? &hostPool : nullptr);
            else if (idx == 3)
              tLogProbs = MakeHostTensorFromVec<float>(
                  combinedTraj.logProbs,
                  {static_cast<int64_t>(combinedTraj.logProbs.size())},
                  torch::kFloat32, pinHostMemory,
                  pinHostMemory ? &hostPool : nullptr);
            else if (idx == 4)
              tRewards = MakeHostTensorFromVec<float>(
                  combinedTraj.rewards,
                  {static_cast<int64_t>(combinedTraj.rewards.size())},
                  torch::kFloat32, pinHostMemory,
                  pinHostMemory ? &hostPool : nullptr);
            else if (idx == 5)
              tTerminals = MakeHostTensorFromVec<int8_t>(
                  combinedTraj.terminals,
                  {static_cast<int64_t>(combinedTraj.terminals.size())},
                  torch::kInt8, pinHostMemory,
                  pinHostMemory ? &hostPool : nullptr);
          };
          g_ThreadPool.StartBatchedJobs(fnMakeTensors, 6, false);

          // States we truncated at (there could be none)
          if (!combinedTraj.nextStates.empty()) {
            int64_t truncRows =
                static_cast<int64_t>(combinedTraj.nextStates.size() / obsSize);
            tNextTruncStates = MakeHostTensorFromVec<float>(
                combinedTraj.nextStates, {truncRows, obsSize}, torch::kFloat32,
                pinHostMemory, pinHostMemory ? &hostPool : nullptr);
          }

          report["Average Step Reward"] = tRewards.mean().item<float>();
          report["Collected Timesteps"] = stepsCollected;

          torch::Tensor tValPreds;
          torch::Tensor tTruncValPreds;

          if (ppo->device.is_cpu()) {
            // Predict values all at once
            tValPreds =
                ppo->InferCritic(tStates.to(ppo->device, true, true)).cpu();
            if (tNextTruncStates.defined())
              tTruncValPreds =
                  ppo->InferCritic(tNextTruncStates.to(ppo->device, true, true))
                      .cpu();
          } else {
            // Predict values using minibatching
            int64_t trajLength = static_cast<int64_t>(combinedTraj.Length());
            tValPreds = torch::zeros({trajLength});
            for (int64_t i = 0; i < trajLength;
                 i += ppo->config.miniBatchSize) {
              int64_t start = i;
              int64_t end = RS_MIN(i + ppo->config.miniBatchSize, trajLength);
              torch::Tensor tStatesPart = tStates.slice(0, start, end);

              auto valPredsPart =
                  ppo->InferCritic(tStatesPart.to(ppo->device, true, true))
                      .cpu();
              RG_ASSERT(valPredsPart.size(0) == (end - start));
              tValPreds.slice(0, start, end).copy_(valPredsPart, true);
            }

            if (tNextTruncStates.defined()) {
              // This really just should never happen
              // If this is ever actually a real problem in a legitimate use
              // case, ping Zealan in the dead of night
              RG_ASSERT(tNextTruncStates.size(0) <= ppo->config.miniBatchSize);

              tTruncValPreds =
                  ppo->InferCritic(tNextTruncStates.to(ppo->device, true, true))
                      .cpu();
            }
          }

          report["Episode Length"] =
              1.f / (tTerminals == 1).to(torch::kFloat32).mean().item<float>();

          Timer gaeTimer = {};
          // Run GAE
          float rewClipPortion;
          GAE::Compute(
              tRewards, tTerminals, tValPreds, tTruncValPreds, tAdvantages,
              tTargetVals, tReturns, rewClipPortion, config.ppo.gaeGamma,
              config.ppo.gaeLambda, returnStat ? returnStat->GetSTD() : 1,
              config.ppo.rewardClipRange, pinHostMemory ? &hostPool : nullptr);
          report["GAE Time"] = gaeTimer.Elapsed();
          report["Clipped Reward Portion"] = rewClipPortion;

          if (returnStat) {
            report["GAE/Returns STD"] = returnStat->GetSTD();

            int numToIncrement =
                RS_MIN(config.maxReturnSamples, tReturns.size(0));
            if (numToIncrement > 0) {
              auto selectedReturns = tReturns.index_select(
                  0,
                  torch::randint(tReturns.size(0), {(int64_t)numToIncrement}));
              returnStat->Increment(TENSOR_TO_VEC<float>(selectedReturns));
            }
          }
          report["GAE/Avg Return"] = tReturns.abs().mean().item<float>();
          report["GAE/Avg Advantage"] = tAdvantages.abs().mean().item<float>();
          report["GAE/Avg Val Target"] = tTargetVals.abs().mean().item<float>();

          // Set experience buffer
          experience.data.actions = tActions;
          experience.data.logProbs = tLogProbs;
          experience.data.actionMasks = tActionMasks;
          experience.data.states = tStates;
          experience.data.advantages = tAdvantages;
          experience.data.targetValues = tTargetVals;
        }

        // Free CUDA cache
#ifdef RG_CUDA_SUPPORT
        if (ppo->device.is_cuda()) {
          bool forceClear =
              (totalIterations % 10 ==
               0); // Force clear every 10 iterations for stability
          size_t freeBytes = 0;
          size_t totalBytes = 0;
          if (!forceClear &&
              cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess &&
              totalBytes > 0) {
            double freeRatio = static_cast<double>(freeBytes) /
                               static_cast<double>(totalBytes);
            if (freeRatio < 0.20) // Increased from 0.15 for better stability on
                                  // low-VRAM cards
              c10::cuda::CUDACachingAllocator::emptyCache();
          } else {
            c10::cuda::CUDACachingAllocator::emptyCache();
          }
        }
#endif

        // Learn
        Timer learnTimer = {};
        ppo->Learn(experience, report, isFirstIteration);
        report["PPO Learn Time"] = learnTimer.Elapsed();

        // Set metrics
        float consumptionTime = consumptionTimer.Elapsed();
        report["Collection Time"] = collectionTime;
        report["Consumption Time"] = consumptionTime;
        report["Collection Steps/Second"] = stepsCollected / collectionTime;
        report["Consumption Steps/Second"] = stepsCollected / consumptionTime;
        report["Overall Steps/Second"] =
            stepsCollected / (collectionTime + consumptionTime);

        uint64_t prevTimesteps = totalTimesteps;
        totalTimesteps += stepsCollected;
        report["Total Timesteps"] = totalTimesteps;
        totalIterations++;
        report["Total Iterations"] = totalIterations;

        if (versionMgr) {
          try {
            versionMgr->OnIteration(ppo, report, totalTimesteps, prevTimesteps);
          } catch (std::exception &e) {
            RG_LOG("PolicyVersionManager::OnIteration failed: " << e.what());
          }
        }

        if (saveQueued.load()) {
          if (!config.checkpointFolder.empty())
            Save();
          quitRequested = true;
          break;
        }

        if (!config.checkpointFolder.empty()) {
          if (totalTimesteps / config.tsPerSave >
              prevTimesteps / config.tsPerSave) {
            // Auto-save
            Save();
          }
        }

        report.Finish();

        if (metricSender) {
          try {
            metricSender->Send(report);
          } catch (std::exception &e) {
            RG_LOG("MetricSender::Send failed: " << e.what());
          }
        }

        try {
          report.Display({"Average Step Reward",
                          "Policy Entropy",
                          "KL Div Loss",
                          "First Accuracy",
                          "",
                          "Policy Update Magnitude",
                          "Critic Update Magnitude",
                          "Shared Head Update Magnitude",
                          "",
                          "Collection Steps/Second",
                          "Consumption Steps/Second",
                          "Overall Steps/Second",
                          "",
                          "Collection Time",
                          "-Inference Time",
                          "-Env Step Time",
                          "Consumption Time",
                          "-GAE Time",
                          "-PPO Learn Time"
                          "",
                          "Collected Timesteps",
                          "Total Timesteps",
                          "Total Iterations"});
        } catch (std::exception &e) {
          RG_LOG("Report::Display failed: " << e.what());
        }
        if (quitRequested)
          break;

        if (pinHostMemory) {
          hostPool.Release(std::move(experience.data.actions));
          hostPool.Release(std::move(experience.data.logProbs));
          hostPool.Release(std::move(experience.data.actionMasks));
          hostPool.Release(std::move(experience.data.states));
          hostPool.Release(std::move(experience.data.advantages));
          hostPool.Release(std::move(experience.data.targetValues));

          if (tNextTruncStates.defined())
            hostPool.Release(std::move(tNextTruncStates));

          // Also release the raw ones if they were pooled
          hostPool.Release(std::move(tRewards));
          hostPool.Release(std::move(tTerminals));
        }
      }
    }

    quitThreadStop = true;
    if (keyPressThread.joinable())
      keyPressThread.join();
    if (quitRequested) {
      if (metricSender)
        metricSender->Flush();
      saveWorker->Flush();
      std::exit(0);
    }

  } catch (std::exception &e) {
    quitThreadStop = true;
    if (keyPressThread.joinable())
      keyPressThread.join();
    if (metricSender)
      metricSender->Flush();
    saveWorker->Flush();
    WriteCrashReport("main_learner_loop", runID, totalTimesteps, totalIterations,
                     config, &e);
    RG_ERR_CLOSE("Exception thrown during main learner loop: " << e.what());
  } catch (...) {
    quitThreadStop = true;
    if (keyPressThread.joinable())
      keyPressThread.join();
    if (metricSender)
      metricSender->Flush();
    saveWorker->Flush();
    WriteCrashReport("main_learner_loop", runID, totalTimesteps, totalIterations,
                     config, nullptr);
    RG_ERR_CLOSE("Exception thrown during main learner loop: unknown error");
  }
}

GGL::Learner::~Learner() {
  if (saveWorker)
    saveWorker->Flush();
  delete envSet;
  delete returnStat;
  delete obsStat;
  delete ppo;
  delete versionMgr;
  delete metricSender;
  delete renderSender;
  pybind11::finalize_interpreter();
}
