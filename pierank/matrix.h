//
// Created by Michelle Zhou 9/17/23.
//

#ifndef PIERANK_MATRIX_H_
#define PIERANK_MATRIX_H_

#include <cstdint>
#include <limits>
#include <vector>

#include "absl/status/status.h"

#include "pierank/data_type.h"
#include "pierank/flex_array.h"
#include "pierank/io/matrix_market_io.h"
#include "pierank/io/mio.h"

namespace pierank {

template<typename PosType, typename IdxType,
    typename DataContainerType = std::vector<double>>
class Matrix {
public:
  using value_type = typename DataContainerType::value_type;

  Matrix() = default;

  Matrix(MatrixType type, const std::vector<uint64_t> &shape,
         const std::vector<uint32_t> &order) :
      type_(type), shape_(shape), order_(order) {
    CHECK_EQ(shape.size(), order.size());

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

  Matrix(const Matrix &) = delete;

  Matrix &operator=(const Matrix &) = delete;

  Matrix(Matrix &&) = default;

  Matrix &operator=(Matrix &&) = default;

  const MatrixType &Type() const { return type_; }

  const std::vector<uint64_t> &Shape() const { return shape_; }

  const std::vector<uint32_t> &Order() const { return order_; }

  PosType Rows() const { return shape_[std::min(order_[0], order_[1])]; }

  PosType Cols() const { return shape_[std::max(order_[0], order_[1])]; }

  uint32_t Depths() const { return shape_.back(); }

  uint32_t DepthDim() const { return shape_.size() - 1; }

  bool SplitDepths() const { return stride_.back() > 1; }

  IdxType DepthStride() const { return stride_.back(); }

  IdxType ElemStride() const { return SplitDepths() ? 1 : Depths(); }

  uint32_t IndexDim() const { return order_.front(); }

  PosType MaxDimSize() const {
    return *std::max_element(shape_.begin(), shape_.end());
  }

  PosType IndexDimSize() const { return shape_[order_[0]]; }

  PosType NonIndexDimSize() const { return shape_[order_[1]]; }

  // Total number of elements (regardless of their values), where a single
  // element is a point with "depths" dimensions
  IdxType Elems() const { return elems_; }

  std::tuple<PosType, PosType, PosType> IdxToPos(IdxType idx) const {
    std::vector<PosType> pos(order_.size());
    for (auto it = order_.begin(); it != order_.end(); ++it) {
      pos[*it] = idx / stride_[*it];
      idx %= stride_[*it];
    }
    return std::make_tuple(pos[0], pos[1], pos[2]);
  }

  void InitData() {
    data_.clear();
    data_.resize(elems_ * Depths());
  }

  void Set(IdxType idx, value_type val) {
    if constexpr (is_specialization_v<DataContainerType, FlexArray>)
      return data_.SetItem(idx, val);
    data_[idx] = val;
  }

  value_type operator()(PosType row, PosType col, uint32_t depth = 0) const {
    uint64_t idx = row * stride_[0] + col * stride_[1] + depth * stride_[2];
    return data_[idx];
  }

  bool ok() const { return status_.ok(); }

  absl::Status status() const { return status_; }

protected:
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
  // Stride for each dim, as determined by shape_ and order_
  std::vector<uint64_t> stride_;
  IdxType elems_ = 0;  // == shape_[0] * shape_[1]
  DataContainerType data_;
  mio::mmap_source data_mmap_;
};

}

#endif //PIERANK_MATRIX_H_
