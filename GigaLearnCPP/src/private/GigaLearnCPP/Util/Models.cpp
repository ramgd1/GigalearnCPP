#include "Models.h"

#include <iostream>
#include <sstream>
#include <torch/csrc/api/include/torch/nn/utils/convert_parameters.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include <torch/nn/modules/normalization.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

static bool g_transferLearnRequested = false;

bool GGL::ConsumeTransferLearnRequest() {
  bool requested = g_transferLearnRequested;
  g_transferLearnRequested = false;
  return requested;
}

GGL::Model::Model(const char *modelName, ModelConfig config,
                  torch::Device device)
    : modelName(modelName), device(device), seq({}), seqHalf({}),
      config(config) {

  if (!config.IsValid())
    RG_ERR_CLOSE("Failed to create model \"" << modelName
                                             << "\" with invalid config");

  int lastSize = config.numInputs;
  for (int i = 0; i < config.layerSizes.size(); i++) {
    seq->push_back(torch::nn::Linear(lastSize, config.layerSizes[i]));
    if (config.addLayerNorm)
      seq->push_back(torch::nn::LayerNorm(
          torch::nn::LayerNormOptions({(int64_t)config.layerSizes[i]})));
    lastSize = config.layerSizes[i];
    AddActivationFunc(seq, config.activationType);
  }

  if (config.addOutputLayer) {
    seq->push_back(torch::nn::Linear(lastSize, config.numOutputs));
  } else {
    config.numOutputs = config.layerSizes.back();
  }

  register_module("seq", seq);
  seq->to(device);
  optim = MakeOptimizer(config.optimType, this->parameters(), 0);
}

torch::Tensor GGL::Model::Forward(torch::Tensor input, bool halfPrec) {

  if (torch::GradMode::is_enabled())
    halfPrec = false;

  if (halfPrec) {

    if (_seqHalfOutdated) {
      _seqHalfOutdated = false;

      if (seqHalf->size() == 0) {
        for (auto &mod : *seq)
          seqHalf->push_back(mod.clone());
        seqHalf->to(RG_HALFPERC_TYPE, true);
      } else {
        auto fromParams = seq->parameters();
        auto toParams = seqHalf->parameters();
        for (int i = 0; i < fromParams.size(); i++) {
          auto scaledParams = fromParams[i].to(RG_HALFPERC_TYPE, true);
          toParams[i].copy_(scaledParams, true);
        }
      }
    }

    auto halfInput = input.to(RG_HALFPERC_TYPE);
    auto halfOutput = seqHalf->forward(halfInput);
    return halfOutput.to(torch::kFloat);
  } else {
    return seq->forward(input);
  }
}

// Get sizes of all parameters in a sequence
std::vector<uint64_t> GetSeqSizes(torch::nn::Sequential &seq) {
  std::vector<uint64_t> result = {};

  for (int i = 0; i < seq->size(); i++)
    for (auto param : seq[i]->parameters())
      result.push_back(param.numel());

  return result;
}

void GGL::Model::SetOptimLR(float newLR) {
  SetOptimizerLR(optim, config.optimType, newLR);
}

void GGL::Model::StepOptim() {
  optim->step();
  optim->zero_grad();
  _seqHalfOutdated = true;
}

GGL::Model *GGL::Model::MakeCloneWithOptim() {
  RG_NO_GRAD;

  Model *clone = MakeEmptyClone();
  auto fromParams = this->parameters();
  auto toParams = clone->parameters();
  for (int i = 0; i < fromParams.size(); i++)
    toParams[i].copy_(fromParams[i], true);

  torch::serialize::OutputArchive optimArchive;
  optim->save(optimArchive);

  std::ostringstream outStream;
  optimArchive.save_to(outStream);

  torch::serialize::InputArchive inArchive;
  std::istringstream inStream(outStream.str());
  inArchive.load_from(inStream, device);
  clone->optim->load(inArchive);

  return clone;
}

void GGL::Model::Save(std::filesystem::path folder, bool saveOptim) {
  std::filesystem::path path = GetSavePath(folder);
  auto streamOut = std::ofstream(path, std::ios::binary);

  if (!streamOut.is_open()) {
    RG_ERR_CLOSE("Model::Save: Failed to open file for writing: " << path);
  }

  if (config.saveHalfPrecision) {
    // Create a temporary clone, convert to half-precision, and save
    Model *clone = this->MakeClone();
    clone->to(RG_HALFPERC_TYPE);
    torch::save(clone->seq, streamOut);
    delete clone;
  } else {
    torch::save(seq, streamOut);
  }

  streamOut.flush();
  if (!streamOut.good()) {
    RG_ERR_CLOSE(
        "Model::Save: Failed to write model to file (disk full?): " << path);
  }
  streamOut.close();

  if (saveOptim) {
    torch::serialize::OutputArchive optimArchive;
    optim->save(optimArchive);
    optimArchive.save_to(GetOptimSavePath(folder).string());
  }
}

void GGL::Model::Load(std::filesystem::path folder, bool allowNotExist,
                      bool loadOptim) {
  std::filesystem::path path = GetSavePath(folder);

  if (!std::filesystem::exists(path)) {
    if (allowNotExist) {
      RG_LOG("Warning: Model \"" << modelName << "\" does not exist at " << path
                                 << " and will be reset");
      return;
    } else {
      RG_ERR_CLOSE("Model::Load: File not found: " << path);
    }
  }

  auto sizesBefore = GetSeqSizes(seq);

#ifdef _WIN32
  // Memory-mapped loading
  HANDLE hFile = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (hFile != INVALID_HANDLE_VALUE) {
    DWORD fileSize = GetFileSize(hFile, NULL);
    HANDLE hMapping = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (hMapping != NULL) {
      void *pData = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
      if (pData != NULL) {
        try {
          std::string dataStr((char *)pData, fileSize);
          std::istringstream streamIn(dataStr);
          torch::load(this->seq, streamIn, device);
        } catch (std::exception &e) {
          UnmapViewOfFile(pData);
          CloseHandle(hMapping);
          CloseHandle(hFile);
          RG_ERR_CLOSE("Failed to load model \""
                       << modelName << "\" via mmap: " << e.what());
        }
        UnmapViewOfFile(pData);
      }
      CloseHandle(hMapping);
    }
    CloseHandle(hFile);
  } else {
#endif
    auto streamIn = std::ifstream(path, std::ios::binary);
    streamIn >> std::noskipws;

    if (!streamIn.good())
      RG_ERR_CLOSE("Failed to load from "
                   << path << ", file does not exist or can't be accessed");

    try {
      torch::load(this->seq, streamIn, device);
    } catch (std::exception &e) {
      RG_ERR_CLOSE(
          "Failed to load model \""
          << modelName
          << ", checkpoint may be corrupt or of different model arch.\n"
          << "Exception: " << e.what());
    }
#ifdef _WIN32
  }
#endif

  // Torch will happily load in a model of a totally different size, then we
  // will crash when we try to use it So we need to manually check if it is the
  // same size
  auto sizesAfter = GetSeqSizes(seq);
  if (!std::equal(sizesBefore.begin(), sizesBefore.end(), sizesAfter.begin(),
                  sizesAfter.end())) {
    static bool sizeMismatchHandled = false;
    if (!sizeMismatchHandled) {
      sizeMismatchHandled = true;

      std::stringstream stream;
      stream << "Saved model has different size than current model from "
             << path << ":\n";
      for (int i = 0; i < 2; i++) {
        stream << " > " << (i ? "Saved model:   [ " : "Current model: [ ");
        for (uint64_t size : (i ? sizesAfter : sizesBefore))
          stream << size << ' ';
        stream << " ]";
        if (i == 0)
          stream << ",\n";
      }
      std::cout << stream.str() << std::endl;

      while (true) {
        std::cout << "Continue and transfer learn? [Y/N] (N = delete "
                     "checkpoint folder and exit): ";
        std::string input;
        if (!std::getline(std::cin, input))
          RG_ERR_CLOSE(
              "Failed to read user input for model size mismatch prompt");

        if (input.empty())
          continue;
        char c = static_cast<char>(toupper(input[0]));
        if (c == 'Y') {
          g_transferLearnRequested = true;
          return;
        }
        if (c == 'N') {
          std::filesystem::remove_all(folder);
          std::exit(0);
        }
      }
    } else {
      return;
    }
  }

  /////////////////////////////

  if (loadOptim) {
    std::filesystem::path optimPath = GetOptimSavePath(folder);

    if (std::filesystem::exists(optimPath)) {
      std::ifstream testStream =
          std::ifstream(optimPath, std::istream::ate | std::ios::binary);
      if (testStream.tellg() > 0) {
        torch::serialize::InputArchive optimArchive;
        optimArchive.load_from(optimPath.string(), device);
        optim->load(optimArchive);
      } else {
        RG_LOG("WARNING: Saved optimizer at "
               << optimPath << " is empty, optimizer will be reset");
      }
    } else {
      RG_LOG("WARNING: No optimizer found at " << optimPath
                                               << ", optimizer will be reset");
    }
  }
}

torch::Tensor GGL::Model::CopyParams() const {
  return torch::nn::utils::parameters_to_vector(parameters()).cpu();
}

void GGL::Model::CopyParamsFrom(const Model &src) {
  RG_NO_GRAD;
  auto fromParams = src.parameters();
  auto toParams = this->parameters();
  if (fromParams.size() != toParams.size())
    RG_ERR_CLOSE("Model::CopyParamsFrom: Parameter size mismatch for model \""
                 << modelName << "\"");

  for (int i = 0; i < fromParams.size(); i++) {
    toParams[i].copy_(fromParams[i].to(device), true);
  }
  _seqHalfOutdated = true;
}

GGL::Model *GGL::Model::MakeCloneToDevice(torch::Device device) const {
  RG_NO_GRAD;
  Model *clone = new Model(modelName, config, device);
  auto fromParams = this->parameters();
  auto toParams = clone->parameters();
  for (int i = 0; i < fromParams.size(); i++)
    toParams[i].copy_(fromParams[i].to(device), true);
  clone->_seqHalfOutdated = true;
  return clone;
}
