//
// Created by Michelle Zhou on 2/22/22.
//

#ifndef PIERANK_FLEX_INDEX_H_
#define PIERANK_FLEX_INDEX_H_

#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include <glog/logging.h>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"

#include "io/file_utils.h"

namespace pierank {

template<typename T>
class FlexIndex {
public:
  explicit FlexIndex(uint32_t item_size = sizeof(T)) : item_size_(item_size) {}

  void Append(T val) {
    std::size_t old_size = vals_.size();
    for (int i = 0; i < item_size_; i++)
      vals_.push_back(0);
    memcpy(&vals_[old_size], reinterpret_cast<void *>(&val), item_size_);
    if (max_val_ < val)
      max_val_ = val;
  }

  T operator[](uint64_t idx) const {
    auto *ptr = vals_.data() + idx * item_size_;

    T res = 0;
    memcpy(&res, ptr, item_size_);

    return res;
  }

  uint64_t NumItems() const {
    DCHECK_EQ(vals_.size() % item_size_, 0);
    return vals_.size() / item_size_;
  }

  uint32_t MinItemSize() const {
    uint64_t max_val64 = static_cast<uint64_t>(max_val_);
    if (max_val64 <= 0xFF) return 1;
    else if (max_val64 <= 0xFFFF) return 2;
    else if (max_val64 <= 0xFFFFFF) return 3;
    else if (max_val64 <= 0xFFFFFFFF) return 4;
    else if (max_val64 <= 0xFFFFFFFFFF) return 5;
    else if (max_val64 <= 0xFFFFFFFFFFFF) return 6;
    else if (max_val64 <= 0xFFFFFFFFFFFFFF) return 7;
    else return 8;
  }

  friend std::ostream &operator<<(std::ostream &os, const FlexIndex &index) {
    uint32_t min_item_size = index.MinItemSize();
    WriteUint32(os, min_item_size);
    uint64_t num_items = index.NumItems();
    WriteUint64(os, min_item_size * num_items);
    for (uint64_t i = 0; i < num_items; ++i)
      WriteInteger(os, index[i], min_item_size);

    return os;
  }

  friend std::istream &operator>>(std::istream &is, FlexIndex &index) {
    index.item_size_ = ReadUint32(is);
    index.vals_.resize(ReadUint64(is));
    is.read(index.vals_.data(), index.vals_.size());
    return is;
  }

  std::string DebugString(uint32_t indent = 0) const {
    std::string res =
        absl::StrFormat("FlexIndex@%x\n", reinterpret_cast<uint64_t>(this));
    std::string tab(indent, ' ');
    absl::StrAppend(&res, tab, "item_size: ", item_size_, "\n");
    absl::StrAppend(&res, tab, "vals[", NumItems(), "]:");
    for (uint64_t i = 0; i < NumItems(); ++i) {
      absl::StrAppend(&res, " ", (*this)[i]);
    }
    absl::StrAppend(&res, "\n");
    return res;
  }

private:
  uint32_t item_size_;
  T max_val_ = std::numeric_limits<T>::min();
  std::string vals_;
};

}  // namespace pierank

#endif //PIERANK_FLEX_INDEX_H_

