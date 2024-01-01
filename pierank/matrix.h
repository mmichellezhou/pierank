//
// Created by Michelle Zhou 9/17/23.
//

#ifndef PIERANK_MATRIX_H_
#define PIERANK_MATRIX_H_

#include <cstdint>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"

#include "pierank/data_type.h"
#include "pierank/flex_array.h"
#include "pierank/io/matrix_market_io.h"
#include "pierank/io/mio.h"

namespace pierank {

template<typename PosType, typename IdxType,
    typename DataContainerType = std::vector<double>>
class Matrix {
public:
  using PosSpan = absl::Span<const PosType>;

  using value_type = typename DataContainerType::value_type;

  Matrix() = default;

  virtual ~Matrix() = default;

  Matrix(MatrixType type,
         const std::vector<uint64_t> &shape,
         const std::vector<uint32_t> &order,
         uint32_t index_dim_order = 0) {
    Config(type, shape, order, index_dim_order);
    if constexpr (is_specialization_v<DataContainerType, FlexArray> ||
                  is_specialization_v<DataContainerType, std::vector>)
      data_.resize(elems_ * Depths());
  }

  Matrix(const Matrix &) = delete;

  Matrix &operator=(const Matrix &) = delete;

  Matrix(Matrix &&) = default;

  Matrix &operator=(Matrix &&) = default;

  const MatrixType &Type() const { return type_; }

  const std::vector<uint64_t> &Shape() const { return shape_; }

  const std::vector<uint32_t> &Order() const { return order_; }

  PosType Rows() const { return shape_[std::min(order_[0], order_[1])]; }

  PosType Cols() const { return shape_[std::max(order_[0], order_[1])]; }

  const value_type* data() const {
    if constexpr (is_specialization_v<DataContainerType, FlexArray>)
      return reinterpret_cast<const value_type*>(data_.data());
    else
      return data_.data();
  }

  uint32_t Depths() const { return shape_.back(); }

  // Returns the # of non-depth dimensions.
  // Since depth dim is always the last one in SparseMatrix, it's same as
  // # of non-depth dims.
  uint32_t NonDepthDims() const { return shape_.size() - 1; }

  bool SplitDepths() const { return stride_.back() > 1; }

  IdxType DepthStride() const { return stride_.back(); }

  IdxType ElemStride() const { return SplitDepths() ? 1 : Depths(); }

  uint32_t IndexDimOrder() const { return index_dim_order_; }

  uint32_t IndexDim() const { return order_[index_dim_order_]; }

  uint32_t NonIndexDim() const { return order_[non_index_dim_order_]; }

  PosType MaxDimSize() const {
    return *std::max_element(shape_.begin(), shape_.end());
  }

  PosType IndexDimSize() const { return shape_[order_[0]]; }

  PosType NonIndexDimSize() const { return shape_[order_[1]]; }

  bool IsRoot() const { return index_dim_order_ == 0; }

  bool IsLeaf() const { return index_dim_order_ + 3 == shape_.size(); }

  // Total number of elements (regardless of their values), where a single
  // element is a point with "depths" dimensions
  IdxType Elems() const { return elems_; }

  std::pair<std::vector<PosType>, uint32_t> IdxToPosAndDepth(IdxType idx) const {
    std::vector<PosType> pos(order_.size() - 1);
    uint32_t depth;
    for (auto it = order_.begin(); it != order_.end(); ++it) {
      if (*it < NonDepthDims())
        pos[*it] = idx / stride_[*it];
      else
        depth = static_cast<uint32_t>(idx / stride_[*it]);
      idx %= stride_[*it];
    }
    return std::make_pair(pos, depth);
  }

  void InitData() {
    if constexpr (is_specialization_v<DataContainerType, FlexArray>)
      data_.Reset();
    else if constexpr (is_specialization_v<DataContainerType, std::vector>)
      std::memset(data_.data(), 0, data_.size() * sizeof(data_[0]));
  }

  IdxType DataIndex(PosSpan pos, uint32_t depth = 0) const {
    DCHECK_EQ(pos.size() + 1, stride_.size());
    IdxType res = 0;
    for (size_t i = 0; i < pos.size(); ++i)
      res += pos[i] * stride_[i];
    res += depth * stride_.back();
    return res;
  }

  void Set(IdxType idx, value_type val) {
    if constexpr (is_specialization_v<DataContainerType, FlexArray>)
      return data_.SetItem(idx, val);
    else {
      if constexpr (!is_specialization_v<DataContainerType, std::vector>)
        CHECK(false);
      data_[idx] = val;
    }
  }

  void Set(value_type val, PosSpan pos, uint32_t depth = 0) {
    Set(DataIndex(pos, depth), val);
  }

  IdxType DataIndex(PosSpan pos, const std::vector<bool>& pos_mask,
                    uint32_t depth = 0) const {
    DCHECK_EQ(pos.size(), pos_mask.size());
    IdxType res = 0;
    std::size_t stride_idx = 0;
    for (std::size_t i = 0; i < pos.size(); ++i) {
      if (!pos_mask[i])
        res += pos[i] * stride_[stride_idx++];
    }
    DCHECK_EQ(stride_idx + 1, stride_.size());
    res += depth * stride_.back();
    return res;
  }

  void Set(value_type val, PosSpan pos, const std::vector<bool>& pos_mask,
           uint32_t depth = 0) {
    Set(DataIndex(pos, pos_mask, depth), val);
  }

  value_type operator()(PosSpan pos, uint32_t depth = 0) const {
    return data_[DataIndex(pos, depth)];
  }

  value_type operator()(PosType row, PosType col, uint32_t depth = 0) const {
    return (*this)({row, col}, depth);
  }

  bool ok() const { return status_.ok(); }

  absl::Status status() const { return status_; }

protected:
  virtual void Config(MatrixType type,
                      const std::vector<uint64_t> &shape,
                      const std::vector<uint32_t> &order,
                      uint32_t index_dim_order = 0) {
    CHECK_EQ(shape.size(), order.size());
    type_ = type;
    shape_ = shape;
    order_ = order;
    index_dim_order_ = index_dim_order;
    non_index_dim_order_ = index_dim_order + 1;

    elems_ = 1;
    for (std::size_t i = 0; i < shape.size() - 1; ++i)
      elems_ *= shape[i];
    CHECK_LE(elems_, std::numeric_limits<IdxType>::max());

    stride_.resize(order.size(), 1);
    uint64_t size = 1;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      stride_[*it] = size;
      size *= shape[*it];
    }
  }

  absl::Status status_;
  MatrixType type_ = MatrixType::kUnknown;
  // Size of each dim, including sub-matrix in data_, where the last dim
  // is ALWAYS the depth, eg, {#rows, #cols, 1} for scalar data_ or
  // {#rows, #cols, 2} for 2D data_
  std::vector<uint64_t> shape_;
  // Storage order of dims in shape_, eg, {0, 1, 2} for row-major or
  // {1, 0, 2} for column-major.
  // For SparseMatrix, depth dim is ALWAYS last in order_
  std::vector<uint32_t> order_;
  uint32_t index_dim_order_ = 0;  // which elem of order_ is the index dim
  uint32_t non_index_dim_order_ = 1;  // which elem of order_ is non-index dim
  // Stride for each dim, as determined by shape_ and order_
  std::vector<uint64_t> stride_;
  IdxType elems_ = 0;  // == shape_[0] * shape_[1]
  DataContainerType data_;
  mio::mmap_source data_mmap_;
};

}

#endif //PIERANK_MATRIX_H_
