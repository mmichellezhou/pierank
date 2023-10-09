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

  Matrix(MatrixType type, uint32_t data_dims, bool split_data_dims,
         PosType rows, PosType cols, uint32_t index_dim) :
      type_(type), elems_(rows * cols) {
    shape_ = {rows, cols, data_dims};
    CHECK_LT(index_dim, 2);
    order_ = {index_dim, 1 - index_dim, 2};
    data_dim_stride_ = split_data_dims ? elems_ : 1;
    CHECK_LE(rows * cols, std::numeric_limits<IdxType>::max());
  }

  Matrix(const Matrix &) = delete;

  Matrix &operator=(const Matrix &) = delete;

  Matrix(Matrix &&) = default;

  Matrix &operator=(Matrix &&) = default;

  const std::vector<uint64_t> &Shape() const { return shape_; }

  const std::vector<uint32_t> &Order() const { return order_; }

  PosType Rows() const { return shape_[std::min(order_[0], order_[1])]; }

  PosType Cols() const { return shape_[std::max(order_[0], order_[1])]; }

  const MatrixType &Type() const { return type_; }

  uint32_t DataDims() const { return shape_.back(); }

  bool SplitDataDims() const { return data_dim_stride_ > 1; }

  IdxType DataDimStride() const { return data_dim_stride_; }

  uint32_t IndexDim() const { return order_.front(); }

  PosType MaxDimSize() const {
    return std::max(shape_[order_[0]], shape_[order_[1]]);
  }

  // Total number of elements (regardless of their values), where a single
  // element is a point with `data_dims` dimensions
  IdxType Elems() const { return elems_; }

  std::pair<PosType, PosType> IdxToPos(IdxType idx) const {
    return order_[0] < order_[1]
           ? std::make_pair(idx / Cols(), idx % Cols())
           : std::make_pair(idx % Rows(), idx / Rows());
  }

  void InitData() {
    data_.clear();
    data_.resize(shape_[order_[0]] * shape_[order_[1]] * DataDims());
  }

  void Set(IdxType idx, value_type val) {
    if constexpr (is_specialization_v<DataContainerType, FlexArray>)
      return data_.SetItem(idx, val);
    data_[idx] = val;
  }

  value_type operator()(PosType row, PosType col, uint32_t data_dim = 0) const {
    typename DataContainerType::size_type idx =
        order_[0] < order_[1] ? row * Cols() + col : col * Rows() + row;
    if (data_dim_stride_ == 1) idx = idx * DataDims() + data_dim;
    else idx += data_dim * elems_;
    return data_[idx];
  }

  bool ok() const { return status_.ok(); }

  absl::Status status() const { return status_; }

protected:
  absl::Status status_;
  MatrixType type_ = MatrixType::kUnknown;
  // Size of each dim, including sub-matrix in data_, where the last dim
  // is ALWAYS the data dim, eg, {#rows, #cols, 1} for scalar data_ or
  // {#rows, #cols, 2} for 2D data_
  std::vector<uint64_t> shape_;
  // Storage order of dims in shape_, eg, {0, 1, 2} for row-major or
  // {1, 0, 2} for column-major.
  // For SparseMatrix, last (=data) dim is ALWAYS last in order_
  std::vector<uint32_t> order_;
  IdxType data_dim_stride_ = 1;  // distance of consecutive dims of a data point
  IdxType elems_ = 0;  // == shape_[order_[0]] * shape_[order_[1]]
  DataContainerType data_;
  mio::mmap_source data_mmap_;
};

}

#endif //PIERANK_MATRIX_H_
