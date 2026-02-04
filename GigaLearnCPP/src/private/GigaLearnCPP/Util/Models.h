#pragma once
#include "../FrameworkTorch.h"
#include <RLGymCPP/Framework.h>


#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/container/sequential.h>

#include <torch/optim/adagrad.h>
#include <torch/optim/adam.h>
#include <torch/optim/adamw.h>
#include <torch/optim/rmsprop.h>
#include <torch/optim/sgd.h>


#include "MagSGD.h"

#include <utility>

#include <GigaLearnCPP/PPO/PPOLearnerConfig.h>
#include <GigaLearnCPP/Util/ModelConfig.h>

namespace GGL {
bool ConsumeTransferLearnRequest();

inline void AddActivationFunc(torch::nn::Sequential &seq,
                              ModelActivationType type) {
  switch (type) {
  case ModelActivationType::RELU:
    seq->push_back(torch::nn::ReLU());
    return;
  case ModelActivationType::LEAKY_RELU:
    seq->push_back(torch::nn::LeakyReLU());
    return;
  case ModelActivationType::SIGMOID:
    seq->push_back(torch::nn::Sigmoid());
    return;
  case ModelActivationType::TANH:
    seq->push_back(torch::nn::Tanh());
    return;
  }

  RG_ERR_CLOSE("Unknown activation function type: " << (int)type);
}

inline torch::optim::Optimizer *
MakeOptimizer(ModelOptimType type, const std::vector<torch::Tensor> &parameters,
              float lr) {
  switch (type) {
  case ModelOptimType::ADAM:
    return new torch::optim::Adam(parameters, lr);
  case ModelOptimType::ADAMW:
    return new torch::optim::AdamW(parameters, lr);
  case ModelOptimType::ADAGRAD:
    return new torch::optim::Adagrad(parameters, lr);
  case ModelOptimType::RMSPROP:
    return new torch::optim::RMSprop(parameters, lr);
  case ModelOptimType::MAGSGD:
    return new MagSGD(parameters, lr);
  }

  RG_ERR_CLOSE("Unknown optimizer type: " << (int)type);
  return NULL;
}

inline void SetOptimizerLR(torch::optim::Optimizer *optimizer,
                           ModelOptimType type, float lr) {
  for (auto &group : optimizer->param_groups()) {
    switch (type) {
    case ModelOptimType::ADAM:
      static_cast<torch::optim::AdamOptions &>(group.options()).lr(lr);
      break;
    case ModelOptimType::ADAMW:
      static_cast<torch::optim::AdamWOptions &>(group.options()).lr(lr);
      break;
    case ModelOptimType::ADAGRAD:
      static_cast<torch::optim::AdagradOptions &>(group.options()).lr(lr);
      break;
    case ModelOptimType::RMSPROP:
      static_cast<torch::optim::RMSpropOptions &>(group.options()).lr(lr);
      break;
    case ModelOptimType::MAGSGD:
      static_cast<MagSGDOptions &>(group.options()).lr(lr);
      break;
    default:
      RG_ERR_CLOSE("Unknown optimizer type: " << (int)type);
    }
  }
}

//////////////////////////

class Model : public torch::nn::Module {
public:
  const char *modelName;
  torch::Device device;
  torch::nn::Sequential seq, seqHalf;
  bool _seqHalfOutdated = true;
  ModelConfig config;

  torch::optim::Optimizer *optim;

  Model()
      : config(PartialModelConfig{}), device({}), modelName(NULL), optim(NULL) {
  } // Uninitialized init

  Model(const char *modelName, ModelConfig config, torch::Device device);

  virtual torch::Tensor Forward(torch::Tensor input, bool halfPrec);

  void SetOptimLR(float newLR);

  void StepOptim();

  // Locale-independent ASCII uppercase conversion
  static char AsciiToUpper(char c) {
    return (c >= 'a' && c <= 'z') ? (c - 'a' + 'A') : c;
  }

  std::filesystem::path GetSuffixedSavePath(std::filesystem::path folder,
                                            std::string suffix) const {
    std::string filename = modelName + suffix;
    for (char &c : filename)
      c = AsciiToUpper(c);
    filename += ".lt";
    return folder / filename;
  }

  std::filesystem::path GetSavePath(std::filesystem::path folder) const {
    return GetSuffixedSavePath(folder, "");
  }

  std::filesystem::path GetOptimSavePath(std::filesystem::path folder) const {
    return GetSuffixedSavePath(folder, "_optim");
  }

  virtual void Save(std::filesystem::path folder, bool saveOptim = true);
  virtual void Load(std::filesystem::path folder, bool allowNotExist,
                    bool loadOptim = true);

  virtual torch::Tensor CopyParams() const;
  void CopyParamsFrom(const Model &src);
  Model *MakeCloneToDevice(torch::Device device) const;

  // NOTE: Resets parameters
  Model *MakeEmptyClone() { return new Model(modelName, config, device); }

  Model *MakeClone() {
    RG_NO_GRAD;

    Model *clone = MakeEmptyClone();
    auto fromParams = this->parameters();
    auto toParams = clone->parameters();
    for (int i = 0; i < fromParams.size(); i++)
      toParams[i].copy_(fromParams[i], true);
    return clone;
  }

  Model *MakeCloneWithOptim();

  uint64_t GetParamCount() {
    uint64_t total = 0;
    for (auto &param : this->parameters()) {
      if (!param.requires_grad())
        continue;
      total += param.numel();
    }

    return total;
  }

  virtual ~Model() { delete optim; }
};

class ModelSet {
public:
  std::map<std::string, Model *> map = {};
  bool ownsModels = false;

  ModelSet() = default;
  explicit ModelSet(bool owns) : ownsModels(owns) {}

  ModelSet(const ModelSet &other) : map(other.map), ownsModels(false) {}

  ModelSet &operator=(const ModelSet &other) {
    if (this != &other) {
      if (ownsModels)
        Free();
      map = other.map;
      ownsModels = false;
    }
    return *this;
  }

  ModelSet(ModelSet &&other) noexcept
      : map(std::move(other.map)), ownsModels(other.ownsModels) {
    other.ownsModels = false;
  }

  ModelSet &operator=(ModelSet &&other) noexcept {
    if (this != &other) {
      if (ownsModels)
        Free();
      map = std::move(other.map);
      ownsModels = other.ownsModels;
      other.ownsModels = false;
    }
    return *this;
  }

  ~ModelSet() {
    if (ownsModels)
      Free();
  }

  void SetOwning(bool owns) { ownsModels = owns; }

  Model *operator[](const std::string &name) {
    auto itr = map.find(name);
    if (itr == map.end()) {
      return NULL;
    } else {
      return map[name];
    }
  };

  void Add(Model *model) { map[model->modelName] = model; }

  // NOTE: Automatically zeros grad after
  void StepOptims() {
    for (Model *model : *this) {
      model->StepOptim();
    }
  }

  void Save(std::filesystem::path folder, bool saveOptims = true) {
    for (Model *model : *this)
      model->Save(folder, saveOptims);
  }

  void Load(std::filesystem::path folder, bool allowNotExist, bool loadOptims) {
    for (Model *model : *this)
      model->Load(folder, allowNotExist, loadOptims);
  }

  class ModelIterator
      : public std::iterator<std::forward_iterator_tag, typename Model *> {
  public:
    using MapItr = std::map<std::string, Model *>::iterator;
    MapItr _mapItr;

    ModelIterator(MapItr mapItr) : _mapItr(mapItr) {}

    ModelIterator &operator++() {
      ++_mapItr;
      return *this;
    }

    bool operator==(const ModelIterator &other) const {
      return _mapItr == other._mapItr;
    }
    bool operator!=(const ModelIterator &other) const {
      return _mapItr != other._mapItr;
    }

    typename Model *&operator*() const { return _mapItr->second; }
  };

  ModelIterator begin() { return map.begin(); }

  ModelIterator end() { return map.end(); }

  ModelSet CloneAll() const {
    ModelSet clone = *this;
    for (auto &pair : clone.map)
      pair.second = pair.second->MakeClone();
    clone.ownsModels = true;
    return clone;
  }

  ModelSet CloneAllWithOptims() const {
    ModelSet clone = *this;
    for (auto &pair : clone.map)
      pair.second = pair.second->MakeCloneWithOptim();
    clone.ownsModels = true;
    return clone;
  }

  ModelSet CloneToDevice(torch::Device device) const {
    ModelSet clone = *this;
    for (auto &pair : clone.map)
      pair.second = pair.second->MakeCloneToDevice(device);
    clone.ownsModels = true;
    return clone;
  }

  void CopyParamsFrom(const ModelSet &src) {
    for (auto &pair : map) {
      auto itr = src.map.find(pair.first);
      if (itr == src.map.end())
        RG_ERR_CLOSE("ModelSet::CopyParamsFrom: Missing model \""
                     << pair.first << "\" in source");
      pair.second->CopyParamsFrom(*itr->second);
    }
  }

  void ZeroGrad() {
    for (Model *model : *this) {
      if (model->optim)
        model->optim->zero_grad();
    }
  }

  void Free() {
    if (ownsModels) {
      for (Model *model : *this)
        delete model;
    }
    map.clear();
    ownsModels = false;
  }
};
} // namespace GGL
