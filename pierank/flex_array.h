//
// Created by Michelle Zhou on 2/22/22.
//

#ifndef PIERANK_FLEX_ARRAY_H_
#define PIERANK_FLEX_ARRAY_H_

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
#include "pierank/math_utils.h"

namespace pierank {

#ifndef PIERANK_USE_CONST_MEMCPY
#define PIERANK_CONST_MEMCPY(dest, src, size) memcpy(dest, src, size)
#else
#define PIERANK_CONST_MEMCPY(dest, src, size)                         \
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
#endif  // PIERANK_CONST_MEMCPY

template<typename T>
class FlexArray {
public:
  struct Iterator {
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = T;
    using pointer           = const FlexArray<T> *;  // not the ordinary T*
    using reference         = void;  // not the ordinary T&
    using const_reference   = void;  // not the ordinary T&
    using size_type         = std::size_t;

    Iterator() = default;

    Iterator(const Iterator &rhs) = default;

    Iterator(const FlexArray<T> &index, uint64_t idx) :
        index_(&index), idx_(idx) {}

    inline value_type operator*() const { return (*index_)[idx_]; }

    // NOTE: Not the ordinary -> operator semantics, use with care !!!
    inline pointer operator->() const { return index_; }

    inline value_type operator[](difference_type rhs) const {
      return (*index_)[idx_ + rhs];
    }

    inline Iterator &operator++() {
      ++idx_;
      return *this;
    }

    inline Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    inline Iterator &operator+=(difference_type rhs) {
      idx_ += rhs;
      return *this;
    }

    inline Iterator operator+(difference_type rhs) const {
      return Iterator(*index_, idx_ + rhs);
    }

    friend inline Iterator operator+(difference_type lhs, const Iterator &rhs) {
      return Iterator(*rhs.index_, lhs + rhs.idx_);
    }

    inline Iterator &operator--() {
      --idx_;
      return *this;
    }

    inline Iterator operator--(int) {
      Iterator tmp = *this;
      --(*this);
      return tmp;
    }

    inline Iterator& operator-=(difference_type rhs) {
      idx_ -= rhs;
      return *this;
    }

    inline Iterator operator-(difference_type rhs) const {
      return Iterator(*index_, idx_ - rhs);
    }

    inline difference_type operator-(const Iterator &rhs) const {
      DCHECK_EQ(index_, rhs.index_);
      return idx_ - rhs.idx_;
    }

    friend inline Iterator operator-(difference_type lhs, const Iterator &rhs) {
      return Iterator(*rhs.index_, lhs - rhs.idx_);
    }

    // Comparison operators
    friend bool operator==(const Iterator& lhs, const Iterator& rhs) {
      return (lhs.index_ == rhs.index_) && (lhs.idx_ == rhs.idx_);
    }

    friend bool operator!=(const Iterator& lhs, const Iterator& rhs) {
      return !(lhs == rhs);
    }

    friend bool operator<(const Iterator& lhs, const Iterator& rhs) {
      DCHECK_EQ(lhs.index_, rhs.index_);
      return (lhs.idx_ < rhs.idx_);
    }

    friend bool operator>(const Iterator& lhs, const Iterator& rhs) {
      DCHECK_EQ(lhs.index_, rhs.index_);
      return (lhs.idx_ > rhs.idx_);
    }

    friend bool operator<=(const Iterator& lhs, const Iterator& rhs) {
      DCHECK_EQ(lhs.index_, rhs.index_);
      return (lhs.idx_ <= rhs.idx_);
    }

    friend bool operator>=(const Iterator& lhs, const Iterator& rhs) {
      DCHECK_EQ(lhs.index_, rhs.index_);
      return (lhs.idx_ >= rhs.idx_);
    }

  private:
    const FlexArray<T> *index_ = nullptr;
    difference_type idx_ = 0;
  };

  Iterator begin() const { return Iterator(*this, /*idx=*/0); }
  Iterator end() const   { return Iterator(*this, /*idx=*/size()); }

  Iterator cbegin() const { return Iterator(*this, /*idx=*/0); }
  Iterator cend() const   { return Iterator(*this, /*idx=*/size()); }

  using SignedT = std::make_signed_t<T>;

  using value_type = T;

  static std::pair<uint32_t, bool> MinEncode(T max_val, T min_val = 0) {
    uint32_t encode_size_without_shift = MinEncodeSize(max_val);
    uint32_t encode_size_with_shift = MinEncodeSize(max_val - min_val);
    DCHECK_LE(encode_size_with_shift, encode_size_without_shift);
    if (encode_size_with_shift < encode_size_without_shift) {
      return std::make_pair(encode_size_with_shift, true);
    }
    return std::make_pair(encode_size_without_shift, false);
  }

  explicit FlexArray(uint32_t item_size = sizeof(T)) : item_size_(item_size) {
    CHECK_GT(item_size_, 0);
    CHECK_LE(item_size_, sizeof(T));
    CHECK_LE(item_size_, 8);
  }

  FlexArray(uint32_t item_size, uint64_t num_items) :
      item_size_(item_size), num_items_(num_items),
      vals_(item_size * num_items, '\0') {}

  FlexArray(const FlexArray &) = delete;

  FlexArray &operator=(const FlexArray &) = delete;

  FlexArray(FlexArray &&) = default;

  FlexArray &operator=(FlexArray &&) = default;

  bool HasSketch() const { return sketch_bits_ > 0; }

  void push_back(T val) {
    DCHECK(vals_mmap_.empty());
    std::size_t old_size = vals_.size();
    vals_.resize(old_size + item_size_);
    if (!shift_by_min_val_) {
      min_val_ = std::min(min_val_, val);
      max_val_ = std::max(max_val_, val);
    } else {
      DCHECK_LT(item_size_, sizeof(T));
      DCHECK_GT(min_val_, 0);
      DCHECK_GE(val, min_val_);
      DCHECK_LE(val, max_val_);
      val -= min_val_;
    }
    if (sketch_bits_) {
      uint64_t sketch_idx = num_items_ >> sketch_bits_;
      if (num_items_ & sketch_bit_mask_) {
        DCHECK_GE(val, sketch_[sketch_idx]);
        val -= sketch_[sketch_idx];
      } else {
        DCHECK_EQ(sketch_idx, sketch_.size());
        sketch_.push_back(val);
        val = 0;
      }
    }
    if (item_size_ == sizeof(T))
      *(reinterpret_cast<T *>(&vals_[old_size])) = val;
    else
      PIERANK_CONST_MEMCPY(&vals_[old_size], &val, item_size_);

    ++num_items_;
  }

  void Append(const FlexArray<T> &other) {
    DCHECK(vals_mmap_.empty());
    DCHECK_EQ(item_size_, other.item_size_);
    DCHECK_EQ(shift_by_min_val_, other.shift_by_min_val_);
    DCHECK(!sketch_bits_);
    DCHECK(!other.sketch_bits_);
    min_val_ = std::min(min_val_, other.min_val_);
    max_val_ = std::max(max_val_, other.max_val_);
    vals_.append(other.vals_);
    num_items_ += other.num_items_;
  }

  inline const char *Data() const {
    return vals_mmap_.empty() ? vals_.data() : vals_mmap_.data();
  }

  T operator[](uint64_t idx) const {
    DCHECK(vals_.empty() || vals_mmap_.empty());
    DCHECK_LT(idx, num_items_);
    DCHECK(item_size_ < sizeof(T) || !shift_by_min_val_);
    DCHECK(item_size_ < sizeof(T) || !sketch_bits_);
    if (item_size_ == sizeof(T)) {
      return vals_mmap_.empty()
             ? reinterpret_cast<const T *>(vals_.data())[idx]
             : reinterpret_cast<const T *>(vals_mmap_.data())[idx];
    }

    auto *ptr = Data();
    ptr += idx * item_size_;
    T res = 0;
    PIERANK_CONST_MEMCPY(&res, ptr, item_size_);

    if (sketch_bits_) {
      DCHECK_LT(idx >> sketch_bits_, sketch_.size());
      res += sketch_[idx >> sketch_bits_];
    }

    if (shift_by_min_val_)
      res += min_val_;
    DCHECK_LE(res, max_val_) << "idx: " << idx;
    return res;
  }

  auto operator()(uint64_t idx = 0) const { return Iterator(*this, idx); }

  void SetItemSize(uint32_t item_size) {
    DCHECK_GT(item_size, 0);
    item_size_ = item_size;
  }

  void SetItem(uint64_t idx, T value) {
    DCHECK(vals_mmap_.empty());
    DCHECK(!sketch_bits_);
    DCHECK_LT(idx, num_items_);
    DCHECK_LE((idx + 1) * item_size_, vals_.size());
    if (item_size_ == sizeof(T))
      reinterpret_cast<T *>(vals_.data())[idx] = value;
    else {
      auto *ptr = vals_.data() + idx * item_size_;
      PIERANK_CONST_MEMCPY(ptr, &value, item_size_);
    }
    min_val_ = std::min(min_val_, value);
    max_val_ = std::max(max_val_, value);
  }

  T IncItem(uint64_t idx, T delta = 1) {
    DCHECK(vals_mmap_.empty());
    DCHECK(!sketch_bits_);
    DCHECK_GT(delta, 0);
    DCHECK_LT(idx, num_items_);
    DCHECK_LE((idx + 1) * item_size_, vals_.size());
    T res = 0;
    if (item_size_ == sizeof(T))
      reinterpret_cast<T *>(vals_.data())[idx] += delta;
    else {
      DCHECK_LT(item_size_, sizeof(T));
      auto ptr = vals_.data() + idx * item_size_;
      PIERANK_CONST_MEMCPY(&res, ptr, item_size_);
      res += delta;
      PIERANK_CONST_MEMCPY(ptr, &res, item_size_);
    }
    max_val_ = std::max(max_val_, res);
    return res;
  }

  uint32_t ItemSize() const { return item_size_; }

  // Returns the # of bytes to store the items.
  uint64_t Bytes() const {
    if (vals_.size()) {
      DCHECK(vals_mmap_.empty());
      return vals_.size();
    }
    return vals_mmap_.size();
  }

  // Returns the # of items stored.
  inline std::size_t size() const {
    DCHECK_GT(item_size_, 0);
    DCHECK_EQ(Bytes() % item_size_, 0);
    DCHECK_EQ(num_items_, Bytes() / item_size_);
    return num_items_;
  }

  inline bool empty() const { return size() == 0; }

  inline void resize(std::size_t num_items) {
    CHECK(vals_mmap_.empty());
    CHECK(!sketch_bits_);
    vals_.resize(num_items * item_size_);
    num_items_ = num_items;
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
    std::memset(sketch_.data(), 0, sketch_.size());
    std::memset(vals_.data(), 0, vals_.size());
  }

  std::pair<uint32_t, bool> MinEncode() const {
    return MinEncode(max_val_, min_val_);
  }

  friend bool operator==(const FlexArray<T> &lhs, const FlexArray<T> &rhs) {
    if (lhs.num_items_ != rhs.num_items_) return false;
    if (lhs.item_size_ == rhs.item_size_ && lhs.Bytes() == rhs.Bytes()
        && lhs.sketch_bits_ == rhs.sketch_bits_) {
      return memcmp(lhs.Data(), rhs.Data(), lhs.Bytes()) == 0;
    }

    for (uint64_t i = 0; i < lhs.num_items_; ++i) {
      if (lhs[i] != rhs[i]) return false;
    }
    return true;
  }

  friend bool operator!=(const FlexArray<T> &lhs, const FlexArray<T> &rhs) {
    return !(lhs == rhs);
  }

  // Returns (-1, 0) if index values are not monotonically increasing.
  std::pair<uint64_t, uint32_t>  // <encode_size, new_item_size>
  EncodeSizeWithSketch(uint32_t sketch_bits) const {
    if (num_items_ == 0 || sketch_bits == 0) return std::make_pair(0, 0);

    uint64_t items_per_sketch = 1ULL << sketch_bits;
    T max_diff = 0;
    for (uint64_t i = 0; i < num_items_; i += items_per_sketch) {
      uint64_t sketch_end = std::min(i + items_per_sketch - 1, num_items_ - 1);
      if ((*this)[sketch_end] < (*this)[i])
        return std::make_pair(std::numeric_limits<uint64_t>::max(), 0);
      T diff = (*this)[sketch_end] - (*this)[i];
      max_diff = std::max(max_diff, diff);
    }

    uint64_t sketches = UnsignedDivideCeil(num_items_, items_per_sketch);
    auto [new_item_size, shift_by_min_val] = MinEncode(max_diff);
    return std::make_pair(num_items_ * new_item_size + sketches * sizeof(T),
                          new_item_size);
  }

  inline uint64_t SketchItems(uint32_t sketch_bits) const {
    if (!sketch_bits) return 0;
    uint64_t items_per_sketch = 1ULL << sketch_bits;
    return UnsignedDivideCeil(num_items_, items_per_sketch);
  }

  std::tuple<uint32_t, uint32_t, bool> // <sketch_bits, item_size, shift_min>
  FindBestEncoding() const {
    auto [new_item_size, shift_by_min_val] = MinEncode();
    if (new_item_size == 1)  // minimal size achieved without sketch
      return std::make_tuple(0, new_item_size, shift_by_min_val);

    uint64_t min_encode_size = num_items_ * new_item_size;
    uint32_t best_sketch_bits = 0;
    uint32_t min_item_size = new_item_size;
    // For sketch compression to save memory, it must:
    // #items * new_item_size + #sketches * sizeof(T) < #items * item_size
    // so, #sketches * sizeof(T) < #items * (item_size - new_item_size)
    // since new_item_size >= 1, item_size - new_item_size <= item_size - 1
    // Thus, we have:  #sketches < #items * (item_size - 1) / sizeof(T)
    // Also, trivially we have: #sketches >= 2.
    // In summary, # of sketch items must satisfy the following inequalities:
    // 2 <= #sketches < #items * (item_size - 1) / sizeof(T)
    uint64_t max_sketches =
        UnsignedDivideCeil(num_items_ * (item_size_ - 1), sizeof(T));
    for (uint32_t sketch_bits = 1; sketch_bits < 64; ++sketch_bits) {
      uint64_t sketches = SketchItems(sketch_bits);
      if (sketches >= max_sketches) continue;
      if (sketches < 2) break;
      auto [encode_size, item_size] = EncodeSizeWithSketch(sketch_bits);
      DCHECK_GT(encode_size, 0);
      if (encode_size == std::numeric_limits<uint64_t>::max()) {
        // index values are not monotonic -> cannot use any sketch
        return std::make_tuple(0, new_item_size, shift_by_min_val);
      }
      if (encode_size < min_encode_size) {
        min_encode_size = encode_size;
        best_sketch_bits = sketch_bits;
        min_item_size = item_size;
      }
    }
    DCHECK_LT(best_sketch_bits, 64);
    DCHECK_LE(min_item_size, new_item_size);
    return std::make_tuple(best_sketch_bits, min_item_size, shift_by_min_val);
  }

  template<typename OutputStreamType>
  bool
  WriteAllButValues(OutputStreamType *os, uint32_t item_size,
                    bool shift_by_min_value, uint32_t sketch_bits = 0) const {
    if (!WriteUint32(os, item_size)) return false;
    if (!ConvertAndWriteUint32(os, shift_by_min_value)) return false;
    CHECK(shift_by_min_val_ == shift_by_min_value || shift_by_min_value);
    if (!ConvertAndWriteUint64(os, min_val_)) return false;
    if (!ConvertAndWriteUint64(os, max_val_)) return false;
    if (!WriteUint32(os, sketch_bits)) return false;
    if (!WriteSketch(os, sketch_bits)) return false;
    return true;
  }

  template<typename OutputStreamType>
  bool WriteValues(OutputStreamType *os, uint32_t item_size,
                   bool shift_by_min_value, uint32_t sketch_bits = 0) const {
    if (!WriteUint64(os, item_size * num_items_)) return false;

    T sketch;
    uint64_t sketch_bit_mask = (1ULL << sketch_bits) - 1;
    for (uint64_t i = 0; i < num_items_; ++i) {
      T val = (*this)[i];
      if (shift_by_min_value) {
        DCHECK_GE(val, min_val_);
        val -= min_val_;
      }
      if (sketch_bits) {
        if (i & sketch_bit_mask) {
          DCHECK_GE(val, sketch);
          val -= sketch;
        }
        else {
          sketch = val;
          val = 0;
        }
      }
      if (!WriteInteger(os, val, item_size)) return false;
    }
    return true;
  }

  template<typename InputStreamType>
  bool ReadValues(InputStreamType *is, uint32_t item_size,
                  SignedT value_shift = 0) {
    auto size = ReadUint64(is);
    if (!*is) return false;
    CHECK_EQ(size % item_size, 0);
    num_items_ = size / item_size;
    uint64_t new_size = num_items_ * item_size_;
    if (vals_.size() < new_size)
      vals_.resize(new_size);
    if (value_shift == 0 && item_size == item_size_)
      return ReadData<InputStreamType>(is, vals_.data(), size);

    for (uint64_t i = 0; i < num_items_; ++i) {
      auto val = ReadInteger<T>(is, item_size);
      DCHECK_LT(static_cast<int64_t>(val) + value_shift,
                1ULL << item_size_ * 8);
      val += value_shift;
      SetItem(i, val);
    }
    return true;
  }

  inline void CheckSketchBits() const {
    CHECK(sketch_bits_ == 0 || item_size_ < sizeof(T));
    CHECK_LT(sketch_bits_, 8 * sizeof(T));
    CHECK_NE(sketch_bits_ > 0, sketch_.empty());
  }

  template<typename OutputStreamType>
  bool WriteSketch(OutputStreamType *os, uint32_t sketch_bits) const {
    CheckSketchBits();  // NOTE: This has nothing to do with `sketch_bits`.

    CHECK_LT(sketch_bits, 8 * sizeof(T));
    uint64_t sketches = SketchItems(sketch_bits);
    CHECK_EQ(sketch_bits == sketch_bits_, sketch_.size() == sketches);

    if (!WriteUint64(os, sketches)) return false;
    if (sketches == 0) return true;
    if (sketch_bits == sketch_bits_)
      return WriteData<OutputStreamType, T>(os, sketch_.data(), sketch_.size());

    uint64_t items_per_sketch = 1ULL << sketch_bits;
    for (uint64_t i = 0; i < num_items_; i += items_per_sketch)
      if (!WriteInteger(os, (*this)[i])) return false;

    return true;
  }

  template<typename OutputStreamType>
  absl::Status Write(OutputStreamType *os) const {
    auto [sketch_bits, item_size, shift_by_min_val] = FindBestEncoding();
    if (!WriteAllButValues(os, item_size, shift_by_min_val, sketch_bits))
      return absl::InternalError("Error writing all but values");
    if (!WriteValues(os, item_size, shift_by_min_val, sketch_bits))
      return absl::InternalError("Error writing values");
    return absl::OkStatus();
  }

  template<typename InputStreamType>
  bool ReadSketch(InputStreamType *is, uint64_t *offset = nullptr) {
    auto size = ReadUint64(is, offset);
    if (!*is) return false;
    sketch_.resize(size);
    if (offset)
      *offset += size * sizeof(T);
    CheckSketchBits();

    return size ? ReadData<InputStreamType, T>(is, sketch_.data(), size) : true;
  }

  friend std::ostream &operator<<(std::ostream &os, const FlexArray &index) {
    auto status = index.Write(&os);
    if (!status.ok())
      os << status.message();
    return os;
  }

  friend std::istream &operator>>(std::istream &is, FlexArray &index) {
    auto status = index.Read(&is);
    if (!status.ok())
      LOG(ERROR) << status.message();
    return is;
  }

  // NOTE: offset == nullptr implies mmap == false, which means the values
  // will be read as well; otherwise mmap == true, which means the values
  // will NOT be read.
  template<typename InputStreamType>
  absl::Status Read(InputStreamType *is, uint64_t *offset = nullptr) {
    auto item_size = ReadUint32(is, offset);
    static_assert(std::is_same_v<decltype(item_size_), decltype(item_size)>);
    item_size_ = item_size;
    shift_by_min_val_ = ReadUint32AndConvert<bool>(is, offset);
    min_val_ = ReadUint64AndConvert<T>(is, offset);
    max_val_ = ReadUint64AndConvert<T>(is, offset);
    sketch_bits_ = ReadUint32(is, offset);
    CHECK(sketch_bits_ == 0 || item_size_ < sizeof(T));
    CHECK_LT(sketch_bits_, 8 * sizeof(T));
    sketch_bit_mask_= (1ULL << sketch_bits_) - 1;
    if (!ReadSketch(is, offset))
      return absl::InternalError("Error reading sketch");
    CHECK_NE(sketch_bits_ > 0, sketch_.empty());

    // Only ReadValues if offset == nullptr, which means mmap == false
    if (!offset && !ReadValues(is, item_size_))
      return absl::InternalError("Error reading values");

    return absl::OkStatus();
  }

  absl::Status Mmap(const std::string &path, uint64_t *offset) {
    auto file_or = OpenReadFile(path);
    if (!file_or.ok()) return file_or.status();
    std::ifstream file = *std::move(file_or);
    if (!Seek(&file, *offset))
      return absl::InternalError(absl::StrCat("Error seeking file: ", path));

    auto status = Read(&file, offset);
    if (!status.ok()) {
      LOG(ERROR) << status.message();
      return absl::InternalError(absl::StrCat("Error reading file: ", path));
    }

    auto size = ReadUint64(&file, offset);
    if (!file) {
      LOG(ERROR) << "Error reading value size";
      return absl::InternalError(absl::StrCat("Error reading file: ", path));
    }
    CHECK_EQ(size % item_size_, 0);
    num_items_ = size / item_size_;
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

  void clear() {
    if (vals_mmap_.size())
      vals_mmap_.unmap();
    *this = std::move(FlexArray());
  }

  std::string DebugString(uint64_t max_items = 0, uint32_t indent = 0) const {
    std::string res;
    std::string tab(indent, ' ');
    absl::StrAppend(&res, tab, "item_size: ", item_size_, "\n");
    absl::StrAppend(&res, tab, "min_val: ", min_val_, "\n");
    absl::StrAppend(&res, tab, "max_val: ", max_val_, "\n");
    absl::StrAppend(&res, tab, "num_items: ", num_items_, "\n");
    absl::StrAppend(&res, tab, "shift_by_min_val: ", shift_by_min_val_, "\n");
    absl::StrAppend(&res, tab, "sketch_bits: ", sketch_bits_, "\n");
    absl::StrAppend(&res, tab, "sketch: [");
    uint64_t max_sketch_items = std::min((uint64_t)sketch_.size(), max_items);
    for (uint64_t i = 0; i < max_sketch_items; ++i) {
      if (i > 0) absl::StrAppend(&res, ", ");
      absl::StrAppend(&res, sketch_[i]);
    }
    if (max_sketch_items < sketch_.size()) {
      if (max_sketch_items > 0) absl::StrAppend(&res, ", ");
      absl::StrAppend(&res, "...");
    }
    absl::StrAppend(&res, "]\n");
    absl::StrAppend(&res, tab, "vals: ", VectorToString(*this, max_items));
    absl::StrAppend(&res, "\n");

    return res;
  }

private:
  uint32_t item_size_;
  bool shift_by_min_val_ = false;
  T min_val_ = std::numeric_limits<T>::max();
  T max_val_ = std::numeric_limits<T>::min();
  uint64_t num_items_ = 0;
  uint32_t sketch_bits_ = 0;
  uint64_t sketch_bit_mask_ = 0;
  std::vector<T> sketch_;
  std::string vals_;
  mio::mmap_source vals_mmap_;
};

}  // namespace pierank

#endif //PIERANK_FLEX_ARRAY_H_
