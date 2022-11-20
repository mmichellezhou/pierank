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

  static std::pair<uint32_t, bool> MinEncode(T max_val, T min_val = 0) {
    uint32_t encode_size_without_shift = MinEncodeSize(max_val);
    uint32_t encode_size_with_shift = MinEncodeSize(max_val - min_val);
    DCHECK_LE(encode_size_with_shift, encode_size_without_shift);
    if (encode_size_with_shift < encode_size_without_shift) {
      return std::make_pair(encode_size_with_shift, true);
    }
    return std::make_pair(encode_size_without_shift, false);
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

  void Append(const FlexIndex<T> &other) {
    DCHECK(vals_mmap_.empty());
    DCHECK_EQ(item_size_, other.item_size_);
    DCHECK_EQ(shift_by_min_val_, other.shift_by_min_val_);
    min_val_ = std::min(min_val_, other.min_val_);
    max_val_ = std::max(max_val_, other.max_val_);
    vals_.append(other.vals_);
  }

  inline const char *Data() const {
    return vals_mmap_.empty() ? vals_.data() : vals_mmap_.data();
  }

  T operator[](uint64_t idx) const {
    DCHECK(vals_.empty() || vals_mmap_.empty());
    auto *ptr = Data();
    ptr += idx * item_size_;

    T res = 0;
    PRK_MEMCPY(&res, ptr, item_size_);

    if (shift_by_min_val_)
      res += min_val_;
    DCHECK_LE(res, max_val_) << "idx: " << idx;
    return res;
  }

  bool IsCompressed() const { return item_size_ < sizeof(T); }

  T At(uint64_t idx) const {
    DCHECK(!IsCompressed());
    return vals_mmap_.empty()
           ? reinterpret_cast<const T *>(vals_.data())[idx]
           : reinterpret_cast<const T *>(vals_mmap_.data())[idx];
  }

  void SetItemSize(uint32_t item_size) {
    DCHECK_GT(item_size, 0);
    item_size_ = item_size;
  }

  void SetItem(uint64_t idx, T value) {
    DCHECK(vals_mmap_.empty());
    DCHECK_LE((idx + 1) * item_size_, vals_.size());
    auto *ptr = vals_.data() + idx * item_size_;
    PRK_MEMCPY(ptr, &value, item_size_);
    min_val_ = std::min(min_val_, value);
    max_val_ = std::max(max_val_, value);
  }

  T IncItem(uint64_t idx, T delta = 1) {
    DCHECK(vals_mmap_.empty());
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

  uint64_t Size() const {
    if (vals_.size()) {
      DCHECK(vals_mmap_.empty());
      return vals_.size();
    }
    return vals_mmap_.size();
  }

  uint64_t NumItems() const {
    DCHECK_GT(item_size_, 0);
    DCHECK_EQ(Size() % item_size_, 0);
    return Size() / item_size_;
  }

  T MinValue() const { return min_val_; }

  T MaxValue() const { return max_val_; }

  void SetMinMaxValues(T min_val, T max_val) {
    min_val_ = min_val;
    max_val_ = max_val;
  }

  bool ShiftByMinValue() const { return shift_by_min_val_; }

  void Reset() {
    DCHECK(vals_mmap_.empty());
    std::memset(vals_.data(), 0, vals_.size());
  }

  std::pair<uint32_t, bool> MinEncode() const {
    return MinEncode(max_val_, min_val_);
  }

  friend bool operator==(const FlexIndex<T> &lhs, const FlexIndex<T> &rhs) {
    CHECK_EQ(lhs.shift_by_min_val_, rhs.shift_by_min_val_) << "Not supported";
    if (lhs.item_size_ == rhs.item_size_) {
      if (lhs.Size() != rhs.Size()) return false;
      return memcmp(lhs.Data(), rhs.Data(), lhs.Size()) == 0;
    }
    if (lhs.NumItems() != rhs.NumItems()) return false;
    for (uint64_t i = 0; i < lhs.NumItems(); ++i) {
      if (lhs[i] != rhs[i]) return false;
    }
    return true;
  }

  friend bool operator!=(const FlexIndex<T> &lhs, const FlexIndex<T> &rhs) {
    return !(lhs == rhs);
  }

  bool WriteAllButValues(std::ostream *os, uint32_t item_size,
                         bool shift_by_min_value) const {
    if (!WriteUint32(os, item_size)) return false;
    if (!ConvertAndWriteUint32(os, shift_by_min_value)) return false;
    CHECK(shift_by_min_val_ == shift_by_min_value || shift_by_min_value);
    if (!ConvertAndWriteUint64(os, min_val_)) return false;
    if (!ConvertAndWriteUint64(os, max_val_)) return false;
    return true;
  }

  template<typename OutputStreamType>
  bool WriteValues(OutputStreamType *os, uint32_t item_size,
                   bool shift_by_min_value) const {
    uint64_t num_items = NumItems();
    if (!WriteUint64(os, item_size * num_items)) return false;
    int shift;
    if (ShiftByMinValue() == shift_by_min_value)
      shift = 0;
    else if (shift_by_min_value)
      shift = -1;
    else
      shift = 1;
    for (uint64_t i = 0; i < num_items; ++i) {
      T val = (*this)[i];
      if (shift < 0) {
        DCHECK_GE(val, min_val_);
        val -= min_val_;
      } else if (shift > 0) {
        DCHECK_LE(val, std::numeric_limits<T>::max() - min_val_);
        val += min_val_;
      }
      if (!WriteInteger(os, val, item_size)) return false;
    }
    return true;
  }

  template<typename InputStreamType>
  bool ReadValues(InputStreamType *is, uint64_t *offset = nullptr) {
    auto size = ReadUint64(is);
    if (!*is) return false;
    uint64_t new_size = offset ? size + *offset : size;
    if (vals_.size() < new_size)
      vals_.resize(new_size);
    char *str = offset ? vals_.data() + *offset : vals_.data();
    if (offset) *offset += size;
    return ReadData<InputStreamType>(is, str, size);
  }

  friend std::ostream &operator<<(std::ostream &os, const FlexIndex &index) {
    auto [min_item_size, shift_by_min_val] = index.MinEncode();
    if (!index.WriteAllButValues(&os, min_item_size, shift_by_min_val))
      return os;
    if (!index.WriteValues(&os, min_item_size, shift_by_min_val))
      return os;
    return os;
  }

  friend std::istream &operator>>(std::istream &is, FlexIndex &index) {
    index.item_size_ = ReadUint32(&is);
    if (!is) return is;
    index.shift_by_min_val_ = ReadUint32AndConvert<bool>(&is);
    if (!is) return is;
    index.min_val_ = ReadUint64AndConvert<T>(&is);
    if (!is) return is;
    index.max_val_ = ReadUint64AndConvert<T>(&is);
    if (!is) return is;
    index.ReadValues(&is);

    return is;
  }

  absl::Status Mmap(const std::string &path, uint64_t *offset) {
    auto file_or = OpenReadFile(path);
    if (!file_or.ok()) return file_or.status();
    std::ifstream file = std::move(*file_or);
    auto item_size = ReadUint32AtOffset(&file, offset);
    static_assert(std::is_same_v<decltype(item_size_), decltype(item_size)>);
    item_size_ = item_size;
    shift_by_min_val_ = ReadUint32AndConvert<bool>(&file, offset);
    min_val_ = ReadUint64AndConvert<T>(&file, offset);
    max_val_ = ReadUint64AndConvert<T>(&file, offset);
    auto size = ReadUint64(&file, offset);
    if (!file)
      return absl::InternalError(absl::StrCat("Error reading file: ", path));
    file.close();

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

  std::string DebugString(uint64_t max_items = 0, uint32_t indent = 0) const {
    std::string res;
    std::string tab(indent, ' ');
    absl::StrAppend(&res, tab, "item_size: ", item_size_, "\n");
    absl::StrAppend(&res, tab, "shift_by_min_val: ", shift_by_min_val_, "\n");
    absl::StrAppend(&res, tab, "min_val: ", min_val_, "\n");
    absl::StrAppend(&res, tab, "max_val: ", max_val_, "\n");
    absl::StrAppend(&res, tab, "vals[", NumItems(), "]:");
    max_items = std::min(max_items, NumItems());
    for (uint64_t i = 0; i < max_items; ++i)
      absl::StrAppend(&res, " ", (*this)[i]);
    if (max_items < NumItems())
      absl::StrAppend(&res, " ...");
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

