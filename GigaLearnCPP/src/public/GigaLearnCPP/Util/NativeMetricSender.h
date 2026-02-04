#pragma once
#include "MetricSender.h"
#include "Report.h"

namespace GGL {
struct RG_IMEXPORT NativeMetricSender : public MetricSender {
  std::string curRunID;
  std::string projectName, groupName, runName;
  bool enabled = true;
  std::filesystem::path logPath;

  NativeMetricSender(std::string projectName = {}, std::string groupName = {},
                     std::string runName = {}, std::string runID = {},
                     std::filesystem::path outputDir = "metrics");

  RG_NO_COPY(NativeMetricSender);

  void Send(const Report &report) override;
  void Flush() override;
  std::string GetRunID() const override { return curRunID; }

  ~NativeMetricSender();

private:
  void Enqueue(Report report);
  void RunWorker();
  void StopWorker();

  void WriteHeader();
  void WriteReport(const Report &report);

  std::ofstream file;
  std::mutex queueMutex;
  std::condition_variable queueCv;
  std::condition_variable flushCv;
  std::queue<Report> reportQueue;
  std::thread worker;
  bool stopWorkerFlag = false;
  bool busy = false;
};
} // namespace GGL
