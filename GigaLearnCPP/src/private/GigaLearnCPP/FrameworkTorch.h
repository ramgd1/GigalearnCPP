#pragma once
#include <GigaLearnCPP/Framework.h>
#include <RLGymCPP/BasicTypes/Lists.h>

// Include torch
#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <torch/utils.h>

#include <cstddef>
#include <mutex>
#include <utility>
#include <vector>

#define RG_NO_GRAD torch::NoGradGuard _noGradGuard

#define RG_AUTOCAST_ON()                                                       \
  {                                                                            \
    at::autocast::set_enabled(true);                                           \
    at::autocast::set_autocast_gpu_dtype(torch::kBFloat16);                    \
    at::autocast::set_autocast_cpu_dtype(torch::kFloat);                       \
  }

#define RG_AUTOCAST_OFF()                                                      \
  {                                                                            \
    at::autocast::clear_cache();                                               \
    at::autocast::set_enabled(false);                                          \
  }

#define RG_HALFPERC_TYPE torch::ScalarType::BFloat16

namespace GGL {
class TensorPool {
public:
  explicit TensorPool(size_t maxEntries = 8) : maxEntries(maxEntries) {}

  torch::Tensor Acquire(at::IntArrayRef sizes,
                        const torch::TensorOptions &options,
                        bool requirePinned = false) {
    bool requirePinnedEffective = requirePinned && options.device().is_cpu();

    std::lock_guard<std::mutex> lock(mutex);
    for (auto itr = pool.begin(); itr != pool.end(); ++itr) {
      if (Matches(*itr, sizes, options, requirePinnedEffective)) {
        torch::Tensor tensor = std::move(*itr);
        pool.erase(itr);
        if (!SameSizes(tensor, sizes))
          tensor = tensor.view(sizes);
        return tensor;
      }
    }
    return torch::empty(sizes, options);
  }

  void Release(torch::Tensor tensor) {
    if (!tensor.defined())
      return;

    std::lock_guard<std::mutex> lock(mutex);
    if (pool.size() >= maxEntries)
      return;
    pool.push_back(std::move(tensor));
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mutex);
    pool.clear();
  }

private:
  static int64_t Numel(at::IntArrayRef sizes) {
    int64_t total = 1;
    for (auto size : sizes)
      total *= size;
    return total;
  }

  static bool SameSizes(const torch::Tensor &tensor, at::IntArrayRef sizes) {
    if (tensor.sizes().size() != sizes.size())
      return false;
    for (size_t i = 0; i < sizes.size(); i++) {
      if (tensor.size(static_cast<int64_t>(i)) != sizes[i])
        return false;
    }
    return true;
  }

  bool Matches(const torch::Tensor &tensor, at::IntArrayRef sizes,
               const torch::TensorOptions &options, bool requirePinned) const {
    if (!tensor.defined())
      return false;
    if (tensor.dtype() != options.dtype())
      return false;
    if (tensor.device() != options.device())
      return false;
    if (tensor.layout() != options.layout())
      return false;
    if (tensor.requires_grad() != options.requires_grad())
      return false;
    if (tensor.device().is_cpu() && tensor.is_pinned() != requirePinned)
      return false;
    if (tensor.numel() != Numel(sizes))
      return false;
    return tensor.is_contiguous();
  }

  size_t maxEntries;
  std::vector<torch::Tensor> pool;
  std::mutex mutex;
};

template <typename T>
inline torch::Tensor DIMLIST2_TO_TENSOR(const RLGC::DimList2<T> &list) {
  return torch::tensor(list.data).reshape(
      {(int64_t)list.size[0], (int64_t)list.size[1]});
}

template <typename T>
inline std::vector<T> TENSOR_TO_VEC(torch::Tensor tensor) {
  assert(tensor.dim() == 1);
  tensor = tensor.detach();
  if (tensor.device().is_cuda())
    tensor = tensor.cpu();
  auto desiredType = torch::CppTypeToScalarType<T>();
  if (tensor.scalar_type() != desiredType)
    tensor = tensor.to(desiredType);
  if (!tensor.is_contiguous())
    tensor = tensor.contiguous();
  T *data = tensor.data_ptr<T>();
  return std::vector<T>(data, data + tensor.size(0));
}
inline int64_t NumelFromSizes(at::IntArrayRef sizes) {
  int64_t total = 1;
  for (auto size : sizes)
    total *= size;
  return total;
}

template <typename T>
inline torch::Tensor
MakeHostTensorFromBlob(const T *data, at::IntArrayRef sizes,
                       torch::ScalarType dtype, bool pinMemory,
                       GGL::TensorPool *pool = nullptr) {
  auto options = torch::TensorOptions().dtype(dtype);
  if (!pinMemory) {
    return torch::from_blob(const_cast<T *>(data), sizes, options);
  }
  options = options.pinned_memory(true);
  torch::Tensor tensor;
  if (pool) {
    tensor = pool->Acquire(sizes, options, true);
  } else {
    tensor = torch::empty(sizes, options);
  }
  std::memcpy(tensor.data_ptr<T>(), data, sizeof(T) * NumelFromSizes(sizes));
  return tensor;
}

template <typename T>
inline torch::Tensor
MakeHostTensorFromVec(const std::vector<T> &data, at::IntArrayRef sizes,
                      torch::ScalarType dtype, bool pinMemory,
                      GGL::TensorPool *pool = nullptr) {
  auto options = torch::TensorOptions().dtype(dtype);
  torch::Tensor tensor;
  if (pinMemory) {
    options = options.pinned_memory(true);
    if (pool) {
      tensor = pool->Acquire(sizes, options, true);
    } else {
      tensor = torch::empty(sizes, options);
    }
  } else {
    tensor = torch::empty(sizes, options);
  }

  if (!data.empty())
    std::memcpy(tensor.data_ptr<T>(), data.data(), sizeof(T) * data.size());
  return tensor;
}
} // namespace GGL
