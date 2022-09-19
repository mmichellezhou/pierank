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
#include <type_traits>

#include <glog/logging.h>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"

#include "pierank/io/file_utils.h"

namespace pierank {

#define PRK_MEMCPY(dest, src, size)                                   \
  do {                                                                \
    if (size == 1)                                                    \
      memcpy(dest, src, 1);                                           \
    else if (size == 2)                                               \
      memcpy(dest, src, 2);                                           \
    else if (size == 3)                                               \
      memcpy(dest, src, 3);                                           \
    else if (size == 4)                                               \
      memcpy(dest, src, 4);                                           \
    else if (size == 5)                                               \
      memcpy(dest, src, 5);                                           \
    else if (size == 6)                                               \
      memcpy(dest, src, 6);                                           \
    else if (size == 7)                                               \
      memcpy(dest, src, 7);                                           \
    else {                                                            \
      DCHECK_EQ(size, 8);                                             \
      memcpy(dest, src, 8);                                           \
    }                                                                 \
  } while(0)


template<typename T>
class FlexIndex {
public:
  static uint32_t MinEncodeSize(uint64_t max_value) {
    if (max_value <= 0xFF) return 1;
    else if (max_value <= 0xFFFF) return 2;
    else if (max_value <= 0xFFFFFF) return 3;
    else if (max_value <= 0xFFFFFFFF) return 4;
    else if (max_value <= 0xFFFFFFFFFF) return 5;
    else if (max_value <= 0xFFFFFFFFFFFF) return 6;
    else if (max_value <= 0xFFFFFFFFFFFFFF) return 7;
    else return 8;
  }

  explicit FlexIndex(uint32_t item_size = sizeof(T)) : item_size_(item_size) {
    CHECK_GT(item_size_, 0);
    CHECK_LE(item_size_, sizeof(T));
    CHECK_LE(item_size_, 8);
  }

  FlexIndex(uint32_t item_size, uint64_t num_items) :
      item_size_(item_size), vals_(item_size * num_items, '\0') {}

  FlexIndex(const FlexIndex &) = delete;

  FlexIndex &operator=(const FlexIndex &) = delete;

  FlexIndex &operator=(FlexIndex &&) = default;

  void Append(T val) {
    std::size_t old_size = vals_.size();
    vals_.resize(old_size + item_size_);
    PRK_MEMCPY(&vals_[old_size], &val, item_size_);
    min_val_ = std::min(min_val_, val);
    max_val_ = std::max(max_val_, val);
  }

  T operator[](uint64_t idx) const {
    DCHECK(vals_.empty() || vals_mmap_.empty());
    auto *ptr = vals_mmap_.empty() ? vals_.data() : vals_mmap_.data();
    ptr += idx * item_size_;

    T res = 0;
    PRK_MEMCPY(&res, ptr, item_size_);

    if (shift_by_min_val_)
      res += min_val_;
    DCHECK_LE(res, max_val_) << "idx: " << idx;
    return res;
  }

  void SetItem(uint64_t idx, T value) {
    DCHECK_LE((idx + 1) * item_size_, vals_.size());
    auto *ptr = vals_.data() + idx * item_size_;
    PRK_MEMCPY(ptr, &value, item_size_);
    min_val_ = std::min(min_val_, value);
    max_val_ = std::max(max_val_, value);
  }

  T IncItem(uint64_t idx, T delta = 1) {
    DCHECK_GT(delta, 0);
    DCHECK_LE((idx + 1) * item_size_, vals_.size());
    T res = 0;
    auto ptr = vals_.data() + idx * item_size_;
    PRK_MEMCPY(&res, ptr, item_size_);
    res += delta;
    PRK_MEMCPY(ptr, &res, item_size_);
    max_val_ = std::max(max_val_, res);
    return res;
  }

  uint32_t ItemSize() const { return item_size_; }

  uint64_t NumItems() const {
    DCHECK_EQ(vals_.size() % item_size_, 0);
    return vals_.size() / item_size_;
  }

  T MinValue() const { return min_val_; }

  T MaxValue() const { return max_val_; }

  bool ShiftByMinValue() const { return shift_by_min_val_; }

  void Reset() { std::memset(vals_.data(), 0, vals_.size()); }

  std::pair<uint32_t, bool> MinEncode() const {
    uint32_t encode_size_without_shift = MinEncodeSize(max_val_);
    uint32_t encode_size_with_shift = MinEncodeSize(max_val_ - min_val_);
    DCHECK_LE(encode_size_with_shift, encode_size_without_shift);
    if (encode_size_with_shift < encode_size_without_shift) {
      return std::make_pair(encode_size_with_shift, true);
    }
    return std::make_pair(encode_size_without_shift, false);
  }

  friend std::ostream &operator<<(std::ostream &os, const FlexIndex &index) {
    auto [min_item_size, shift_by_min_val] = index.MinEncode();
    if (!WriteUint32(os, min_item_size)) return os;
    if (!ConvertAndWriteUint32(os, shift_by_min_val)) return os;
    CHECK(index.ShiftByMinValue() == shift_by_min_val || shift_by_min_val);
    if (!ConvertAndWriteUint64(os, index.MinValue())) return os;
    if (!ConvertAndWriteUint64(os, index.MaxValue())) return os;
    uint64_t num_items = index.NumItems();
    if (!WriteUint64(os, min_item_size * num_items)) return os;
    for (uint64_t i = 0; i < num_items; ++i) {
      T val = index[i];
      if (index.ShiftByMinValue() != shift_by_min_val) {
        DCHECK_GE(val, index.MinValue());
        val -= index.MinValue();
      }
      if (!WriteInteger(os, val, min_item_size)) return os;
    }

    return os;
  }

  friend std::istream &operator>>(std::istream &is, FlexIndex &index) {
    index.item_size_ = ReadUint32(is);
    if (!is) return is;
    index.shift_by_min_val_ = ReadUint32AndConvert<bool>(is);
    if (!is) return is;
    index.min_val_ = ReadUint64AndConvert<T>(is);
    if (!is) return is;
    index.max_val_ = ReadUint64AndConvert<T>(is);
    if (!is) return is;

    auto size = ReadUint64(is);
    if (!is) return is;
    index.vals_.resize(size);
    is.read(index.vals_.data(), size);

    return is;
  }

  absl::Status Mmap(const std::string &path, uint64_t *offset) {
    auto file = OpenReadFile(path);
    if (!file.ok()) return file.status();
    auto item_size = ReadUint32AtOffset(*file, offset);
    static_assert(std::is_same_v<decltype(item_size_), decltype(item_size)>);
    item_size_ = item_size;
    shift_by_min_val_ = ReadUint32AndConvert<bool>(*file, offset);
    min_val_ = ReadUint64AndConvert<T>(*file, offset);
    max_val_ = ReadUint64AndConvert<T>(*file, offset);
    auto size = ReadUint64(*file, offset);
    if (!(*file))
      return absl::InternalError(absl::StrCat("Error reading file: ", path));
    file->close();

    auto mmap = MmapReadOnlyFile(path, *offset, size);
    *offset += size;
    if (!mmap.ok()) return mmap.status();
    vals_mmap_ = std::move(mmap).value();

    return absl::OkStatus();
  }

  void UnMmap() {
    DCHECK(vals_.empty());
    if (vals_mmap_.size())
      vals_mmap_.unmap();
  }

  std::string DebugString(uint32_t indent = 0) const {
    std::string res =
        absl::StrFormat("FlexIndex@%x\n", reinterpret_cast<uint64_t>(this));
    std::string tab(indent, ' ');
    absl::StrAppend(&res, tab, "item_size: ", item_size_, "\n");
    absl::StrAppend(&res, tab, "shift_by_min_val: ", shift_by_min_val_, "\n");
    absl::StrAppend(&res, tab, "min_val: ", min_val_, "\n");
    absl::StrAppend(&res, tab, "max_val: ", max_val_, "\n");
    absl::StrAppend(&res, tab, "vals[", NumItems(), "]:");
    for (uint64_t i = 0; i < NumItems(); ++i)
      absl::StrAppend(&res, " ", (*this)[i]);
    absl::StrAppend(&res, "\n");
    return res;
  }

private:
  uint32_t item_size_;
  bool shift_by_min_val_ = false;
  T min_val_ = std::numeric_limits<T>::max();
  T max_val_ = std::numeric_limits<T>::min();
  std::string vals_;
  mio::mmap_source vals_mmap_;
};

}  // namespace pierank

#endif //PIERANK_FLEX_INDEX_H_

