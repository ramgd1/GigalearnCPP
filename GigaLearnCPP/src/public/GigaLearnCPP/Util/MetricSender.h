#pragma once
#include "Report.h"
#include <condition_variable>
#include <mutex>
#include <pybind11/pybind11.h>
#include <queue>
#include <thread>

namespace GGL {
struct RG_IMEXPORT MetricSender {
  virtual void Send(const Report &report) = 0;
  virtual void Flush() = 0;
  virtual std::string GetRunID() const = 0;

  static MetricSender *Make(const struct LearnerConfig &config,
                            std::string runID);

  virtual ~MetricSender() {}
};

struct RG_IMEXPORT PythonMetricSender : public MetricSender {
  std::string curRunID;
  std::string projectName, groupName, runName;
  pybind11::module pyMod;
  bool enabled = true;

  PythonMetricSender(std::string projectName = {}, std::string groupName = {},
                     std::string runName = {}, std::string runID = {});

  RG_NO_COPY(PythonMetricSender);

  void Send(const Report &report) override;
  void Flush() override;
  std::string GetRunID() const override { return curRunID; }

  ~PythonMetricSender();

private:
  void Enqueue(Report report);
  void RunWorker();
  void StopWorker();

  std::mutex queueMutex;
  std::condition_variable queueCv;
  std::condition_variable flushCv;
  std::queue<Report> reportQueue;
  std::thread worker;
  bool stopWorkerFlag = false;
  bool busy = false;
};
} // namespace GGL
