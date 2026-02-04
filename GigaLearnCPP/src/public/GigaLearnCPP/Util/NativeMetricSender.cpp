#include "NativeMetricSender.h"
#include <chrono>
#include <iostream>

using namespace GGL;

NativeMetricSender::NativeMetricSender(std::string _projectName,
                                       std::string _groupName,
                                       std::string _runName, std::string runID,
                                       std::filesystem::path outputDir)
    : projectName(_projectName), groupName(_groupName), runName(_runName),
      curRunID(runID) {
  if (curRunID.empty()) {
    auto now = std::chrono::system_clock::now();
    curRunID = std::to_string(
        std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch())
            .count());
  }

  std::filesystem::create_directories(outputDir);
  logPath = outputDir / (curRunID + ".gglm");
  file.open(logPath, std::ios::binary);

  if (!file.is_open()) {
    RG_LOG("NativeMetricSender: Failed to open log file at " << logPath
                                                             << ", disabling "
                                                                "metrics.");
    enabled = false;
    return;
  }

  WriteHeader();

  worker = std::thread([this]() { RunWorker(); });
}

void NativeMetricSender::Send(const Report &report) { Enqueue(report); }

void NativeMetricSender::Enqueue(Report report) {
  if (!enabled)
    return;
  {
    std::lock_guard<std::mutex> lock(queueMutex);
    reportQueue.push(std::move(report));
  }
  queueCv.notify_one();
}

void NativeMetricSender::Flush() {
  if (!enabled)
    return;
  std::unique_lock<std::mutex> lock(queueMutex);
  flushCv.wait(lock, [&]() { return reportQueue.empty() && !busy; });
}

void NativeMetricSender::RunWorker() {
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

    WriteReport(report);

    {
      std::lock_guard<std::mutex> lock(queueMutex);
      busy = false;
      if (reportQueue.empty())
        flushCv.notify_all();
    }
  }
}

void NativeMetricSender::StopWorker() {
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

void NativeMetricSender::WriteHeader() {
  const char magic[4] = {'G', 'G', 'L', 'M'};
  file.write(magic, 4);

  uint32_t version = 1;
  file.write((char *)&version, 4);

  auto writeString = [&](const std::string &s) {
    uint32_t len = (uint32_t)s.length();
    file.write((char *)&len, 4);
    file.write(s.data(), len);
  };

  writeString(projectName);
  writeString(groupName);
  writeString(runName);
  writeString(curRunID);
}

void NativeMetricSender::WriteReport(const Report &report) {
  auto now = std::chrono::system_clock::now();
  double timestamp =
      std::chrono::duration<double>(now.time_since_epoch()).count();
  file.write((char *)&timestamp, 8);

  uint32_t numMetrics = (uint32_t)report.data.size();
  file.write((char *)&numMetrics, 4);

  for (auto const &[key, val] : report.data) {
    uint32_t len = (uint32_t)key.length();
    file.write((char *)&len, 4);
    file.write(key.data(), len);
    file.write((char *)&val, 4);
  }
  file.flush();
}

NativeMetricSender::~NativeMetricSender() {
  if (enabled) {
    Flush();
    StopWorker();
    if (file.is_open())
      file.close();
  }
}
