#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <cstddef>

template <typename T>
class frame_Queue
{
public:
    explicit frame_Queue(size_t capacity_ = 4) : capacity(capacity_) {};

    bool push(T item)
    {
        std::unique_lock lock(mtx_); // LOCK
        not_full_cv_.wait(lock, [this]
                          { return stopAll || queue_.size() < capacity });
        if (stopAll == true)
            return false;

        queue_.push(std::move(item)); // push item
        not_empty_cv_.notify_one();   // wakeup one waitting consumer
        return true;
    }

    bool pop(T &recive)
    {
        std::unique_lock lock(mtx_); // lock
        not_empty_cv_.wait(lock, [this] { return stopAll || !queue_.empty(); });
        if (queue_.empty()) return false;

        recive = std::move(queue_.front()); //recive the poped data
        queue_.pop();
        not_full_cv_.notify_one(); // notify one waitting producer
        return true;
    }

    bool stop()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stopped_ = true;
        not_empty_cv_.notify_all();
        not_full_cv_.notify_all();
    }

    private : size_t capacity;
    bool stopAll;
    std::queue<T> queue_;
    std::mutex mtx_;
    std::condition_variable not_empty_cv_; // for consumer
    std::condition_variable not_full_cv_;  // for producer
};