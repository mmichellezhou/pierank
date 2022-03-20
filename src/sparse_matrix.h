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
  SparseMatrix(const std::string &matrix_market_file_path) {
    MatrixMarketIo mat(matrix_market_file_path);
    CHECK(mat.Ok()) << "Fail to open file: " << matrix_market_file_path;
    DLOG(INFO) << "NumNonZeros: " << mat.NumNonZeros();

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

  PosType NumNonZeros() const { return nnz_; }

private:
  FlexIndex<IdxType> index_;
  FlexIndex<PosType> pos_;
  PosType nnz_ = 0;
};

#endif //PIERANK_SPARSE_MATRIX_H_
