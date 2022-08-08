//
// Created by Michelle Zhou on 2/26/22.
//

#ifndef PIERANK_SPARSE_MATRIX_H_
#define PIERANK_SPARSE_MATRIX_H_

#include <limits>
#include <glog/logging.h>

#include "flex_index.h"
#include "io/matrix_market_io.h"

template<typename PosType, typename IdxType>
class SparseMatrix {
public:
  using PosRange = std::pair<PosType, PosType>;

  using PosRanges = std::vector<std::pair<PosType, PosType>>;

  using FlexIdxType = FlexIndex<IdxType>;

  using FlexPosType = FlexIndex<PosType>;

  SparseMatrix(const std::string &matrix_market_file_path) {
    MatrixMarketIo mat(matrix_market_file_path);
    CHECK(mat.Ok()) << "Fail to open file: " << matrix_market_file_path;
    // DLOG(INFO) << "NumNonZeros: " << mat.NumNonZeros();

    rows_ = mat.Rows();
    cols_ = mat.Cols();

    PosType col = std::numeric_limits<PosType>::max();

    while (mat.HasNext()) {
      auto pos = mat.Next();
      DCHECK_GT(pos.first, 0);
      DCHECK_GT(pos.second, 0);
      --pos.first;
      --pos.second;
      if (col != pos.second) {
        index_.Append(nnz_);
        col = pos.second;
      }
      pos_.Append(pos.first);
      nnz_++;
    }
    index_.Append(nnz_);
  }

  const FlexIdxType &Index() const { return index_; }

  IdxType Index(PosType pos) const { return index_[pos]; }

  const FlexPosType &Pos() const { return pos_; }

  PosType Pos(IdxType idx) const { return pos_[idx]; }

  IdxType NumNonZeros() const { return nnz_; }

  PosType Rows() const { return rows_; }

  PosType Cols() const { return cols_; }

private:
  FlexIdxType index_;
  FlexPosType pos_;
  IdxType nnz_ = 0;
  PosType rows_ = 0;
  PosType cols_ = 0;
};

#endif //PIERANK_SPARSE_MATRIX_H_
