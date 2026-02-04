#pragma once
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace RLGC {
class AsyncLogger {
public:
  static void LogDirect(const std::string &s) { GetInstance().Enqueue(s); }

  static AsyncLogger &GetInstance() {
    static AsyncLogger instance;
    return instance;
  }

  AsyncLogger(const AsyncLogger &) = delete;
  AsyncLogger &operator=(const AsyncLogger &) = delete;

private:
  AsyncLogger() : stopWorker(false) {
    worker = std::thread(&AsyncLogger::RunWorker, this);
  }

  ~AsyncLogger() {
    {
      std::lock_guard<std::mutex> lock(queueMutex);
      stopWorker = true;
    }
    queueCv.notify_all();
    if (worker.joinable())
      worker.join();
  }

  void Enqueue(const std::string &s) {
    {
      std::lock_guard<std::mutex> lock(queueMutex);
      logQueue.push(s);
    }
    queueCv.notify_one();
  }

  void RunWorker() {
    while (true) {
      std::string s;
      {
        std::unique_lock<std::mutex> lock(queueMutex);
        queueCv.wait(lock, [this] { return stopWorker || !logQueue.empty(); });

        if (stopWorker && logQueue.empty())
          break;

        s = std::move(logQueue.front());
        logQueue.pop();
      }
      std::cout << s << std::endl;
    }
  }

  std::queue<std::string> logQueue;
  std::mutex queueMutex;
  std::condition_variable queueCv;
  std::thread worker;
  std::atomic<bool> stopWorker;
};
} // namespace RLGC
