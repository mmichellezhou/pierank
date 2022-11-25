//
// Created by Michelle Zhou on 11/24/22.
//

#ifndef PIERANK_TIME_UTILS_H_
#define PIERANK_TIME_UTILS_H_

#include "absl/time/clock.h"

namespace pierank {

class Timer {
public:
  Timer() = default;

  Timer(absl::Time start_time) : start_time_(start_time) {}

  void Start() { start_time_ = absl::Now(); }

  // Returns the cumulative elapsed time in milliseconds.
  double Stop() {
    elapsed_ += absl::Now() - start_time_;
    start_time_ = absl::Time();
    return absl::ToDoubleMilliseconds(elapsed_);
  }

  void Restart() {
    elapsed_ = absl::Duration();
    start_time_ = absl::Now();
  }

private:
  absl::Time start_time_;
  absl::Duration elapsed_;
};

}  // namespace pierank

#endif //PIERANK_TIME_UTILS_H_
