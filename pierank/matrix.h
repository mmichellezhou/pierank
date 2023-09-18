//
// Created by Michelle Zhou 9/17/23.
//

#ifndef PIERANK_MATRIX_H_
#define PIERANK_MATRIX_H_

#include <cstdint>
#include <limits>

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
  using ValueType = typename DataContainerType::value_type;

  Matrix() = default;

  Matrix(MatrixType type, uint32_t data_dims, bool split_data_dims,
         PosType rows, PosType cols, uint32_t index_dim) :
      type_(type), data_dims_(data_dims), rows_(rows), cols_(cols),
      elems_(rows * cols), index_dim_(index_dim) {
    data_dim_stride_ = split_data_dims ? elems_ : 1;
    CHECK_LE(rows * cols, std::numeric_limits<IdxType>::max());
  }

  Matrix(const Matrix &) = delete;

  Matrix &operator=(const Matrix &) = delete;

  Matrix(Matrix &&) = default;

  Matrix &operator=(Matrix &&) = default;

  PosType Rows() const { return rows_; }

  PosType Cols() const { return cols_; }

  const MatrixType &Type() const { return type_; }

  uint32_t DataDims() const { return data_dims_; }

  IdxType DataDimStride() const { return data_dim_stride_; }

  uint32_t IndexDim() const { return index_dim_; }

  PosType MaxDimSize() const { return std::max(rows_, cols_); }

  // Total number of elements (regardless of their values), where a single
  // element is a point with `data_dims` dimensions
  IdxType Elems() const { return elems_; }

  void InitData() { data_.clear(); data_.resize(rows_ * cols_ * data_dims_); }

  void Set(IdxType idx, ValueType val) {
    if constexpr (is_specialization_v<DataContainerType, FlexArray>)
      return data_.SetItem(idx, val);
    data_[idx] = val;
  }

  ValueType operator()(PosType row, PosType col, uint32_t data_dim = 0) const {
    typename DataContainerType::size_type idx =
        index_dim_ == 0 ? row * cols_ + col : col * rows_ + row;
    if (data_dim_stride_ == 1) idx = idx * data_dims_ + data_dim;
    else idx += data_dim * elems_;
    return data_[idx];
  }

  bool ok() const { return status_.ok(); }

  absl::Status status() const { return status_; }

protected:
  absl::Status status_;
  MatrixType type_ = MatrixType::kUnknown;
  uint32_t data_dims_ = 1;  // dimensions for an element of data_
  IdxType data_dim_stride_ = 1;  // distance b/n 2 consecutive dims of an elem
  PosType rows_ = 0;
  PosType cols_ = 0;
  IdxType elems_ = 0;  // == rows_ * cols_
  uint32_t index_dim_ = 0;
  DataContainerType data_;
  mio::mmap_source data_mmap_;
};

}

#endif //PIERANK_MATRIX_H_
