#include "MetricSender.h"
#include "NativeMetricSender.h"
#include <GigaLearnCPP/LearnerConfig.h>

#include "Timer.h"
#include <pybind11/gil.h>

namespace py = pybind11;
using namespace GGL;

MetricSender *MetricSender::Make(const LearnerConfig &config,
                                 std::string runID) {
  if (config.metricsType == LearnerConfig::MetricsType::DISABLED) {
    return nullptr;
  } else if (config.metricsType == LearnerConfig::MetricsType::PYTHON_WANDB) {
    return new PythonMetricSender(config.metricsProjectName,
                                  config.metricsGroupName,
                                  config.metricsRunName, runID);
  } else {
    return new NativeMetricSender(config.metricsProjectName,
                                  config.metricsGroupName,
                                  config.metricsRunName, runID);
  }
}

GGL::PythonMetricSender::PythonMetricSender(std::string _projectName,
                                            std::string _groupName,
                                            std::string _runName,
                                            std::string runID)
    : projectName(_projectName), groupName(_groupName), runName(_runName) {

  RG_LOG("Initializing PythonMetricSender...");

  try {
    pyMod = py::module::import("python_scripts.metric_receiver");
  } catch (std::exception &e) {
    RG_LOG("PythonMetricSender: Failed to import metrics receiver, disabling "
           "metrics. Exception: "
           << e.what());
    enabled = false;
    return;
  }

  try {
    auto returedRunID = pyMod.attr("init")(PY_EXEC_PATH, projectName, groupName,
                                           runName, runID);
    curRunID = returedRunID.cast<std::string>();
    RG_LOG(" > " << (runID.empty() ? "Starting" : "Continuing")
                 << " run with ID : \"" << curRunID << "\"...");

  } catch (std::exception &e) {
    RG_LOG("PythonMetricSender: Failed to initialize, disabling metrics. "
           "Exception: "
           << e.what());
    enabled = false;
    return;
  }

  RG_LOG(" > PythonMetricSender initalized.");

  worker = std::thread([this]() { RunWorker(); });
}

void GGL::PythonMetricSender::Send(const Report &report) { Enqueue(report); }

void GGL::PythonMetricSender::Enqueue(Report report) {
  if (!enabled)
    return;
  {
    std::lock_guard<std::mutex> lock(queueMutex);
    reportQueue.push(std::move(report));
  }
  queueCv.notify_one();
}

void GGL::PythonMetricSender::Flush() {
  if (!enabled)
    return;
  std::unique_lock<std::mutex> lock(queueMutex);
  flushCv.wait(lock, [&]() { return reportQueue.empty() && !busy; });
}

void GGL::PythonMetricSender::RunWorker() {
  while (true) {
    Report report;
    {
      std::unique_lock<std::mutex> lock(queueMutex);
      queueCv.wait(lock,
                   [&]() { return stopWorkerFlag || !reportQueue.empty(); });
      if (stopWorkerFlag && reportQueue.empty())
        return;
      report = std::move(reportQueue.front());
      reportQueue.pop();
      busy = true;
    }

    try {
      py::gil_scoped_acquire gil;
      py::dict reportDict = {};

      for (auto &pair : report.data)
        reportDict[pair.first.c_str()] = pair.second;

      pyMod.attr("add_metrics")(reportDict);
    } catch (std::exception &e) {
      RG_LOG("PythonMetricSender: Failed to add metrics, disabling metrics. "
             "Exception: "
             << e.what());
      enabled = false;
      {
        std::lock_guard<std::mutex> lock(queueMutex);
        while (!reportQueue.empty())
          reportQueue.pop();
        stopWorkerFlag = true;
      }
      queueCv.notify_all();
    }

    {
      std::lock_guard<std::mutex> lock(queueMutex);
      busy = false;
      if (reportQueue.empty())
        flushCv.notify_all();
    }
  }
}

void GGL::PythonMetricSender::StopWorker() {
  if (!enabled)
    return;
  {
    std::lock_guard<std::mutex> lock(queueMutex);
    stopWorkerFlag = true;
  }
  queueCv.notify_all();
  if (worker.joinable())
    worker.join();
}

GGL::PythonMetricSender::~PythonMetricSender() {
  if (enabled) {
    Flush();
    StopWorker();
  }
}
