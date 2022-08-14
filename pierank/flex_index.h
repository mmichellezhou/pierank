//
// Created by Michelle Zhou on 2/22/22.
//

#ifndef PIERANK_FLEX_INDEX_H_
#define PIERANK_FLEX_INDEX_H_

#include <cstdint>
#include <string>

#include <glog/logging.h>

namespace pierank {

template<typename T>
class FlexIndex {
public:
  explicit FlexIndex(uint32_t bytes_per_elem = sizeof(T)) :
      bytes_per_elem_(bytes_per_elem) {}

  void Append(T val) {
    std::size_t old_size = vals_.size();
    for (int i = 0; i < bytes_per_elem_; i++)
      vals_.push_back(0);
    memcpy(&vals_[old_size], reinterpret_cast<void *>(&val), bytes_per_elem_);
  }

  T operator[](uint64_t idx) const {
    auto *ptr = vals_.data() + idx * bytes_per_elem_;

    T res = 0;
    memcpy(&res, ptr, bytes_per_elem_);

    return res;
  }

  uint64_t Size() const {
    DCHECK_EQ(vals_.size() % bytes_per_elem_, 0);
    return vals_.size() / bytes_per_elem_;
  }

private:
  uint32_t bytes_per_elem_;
  std::string vals_;
};

}  // namespace pierank

#endif //PIERANK_FLEX_INDEX_H_

