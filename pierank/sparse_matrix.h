//
// Created by Michelle Zhou on 2/26/22.
//

#ifndef PIERANK_SPARSE_MATRIX_H_
#define PIERANK_SPARSE_MATRIX_H_

#include <limits>
#include <glog/logging.h>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"

#include "flex_index.h"
#include "io/file_utils.h"
#include "io/matrix_market_io.h"

namespace pierank {

/* Example SparseMatrix:
   *
   *     0 1 2 3 4
   *     - - - - -
   * 0 | 0 1 1 0 0 <- 0 is linked to 1 and 2
   * 1 | 0 0 0 1 0
   * 2 | 1 1 0 1 1
   * 3 | 0 0 1 0 0
   * 4 | 1 0 0 0 0
   *     ^
   *     2 and 4 are linked to 0
   *
   * pos = [2, 4, 0, 2, 0, 3, 1, 2, 2]
   * idx = [0, 2, 4, 6, 8, 9]
   */
template<typename PosType, typename IdxType>
class SparseMatrix {
public:
  static constexpr absl::string_view kPrmFileMagicNumbers = "#PRM";

  using PosRange = std::pair<PosType, PosType>;

  using PosRanges = std::vector <std::pair<PosType, PosType>>;

  using FlexIdxType = FlexIndex<IdxType>;

  using FlexPosType = FlexIndex<PosType>;

  SparseMatrix(const SparseMatrix &) = delete;

  SparseMatrix &operator=(const SparseMatrix &) = delete;

  SparseMatrix() = default;

  SparseMatrix(const std::string &prm_file_path) {
    status_ = this->ReadPrmFile(prm_file_path);
  }

  const FlexIdxType &Index() const { return index_; }

  IdxType Index(PosType pos) const { return index_[pos]; }

  const FlexPosType &Pos() const { return pos_; }

  PosType Pos(IdxType idx) const { return pos_[idx]; }

  IdxType NumNonZeros() const { return nnz_; }

  PosType Rows() const { return rows_; }

  PosType Cols() const { return cols_; }

  bool Ok() const { return status_.ok(); }

  friend std::ostream &
  operator<<(std::ostream &os, const SparseMatrix &matrix) {
    ConvertAndWriteUint64(os, matrix.rows_);
    ConvertAndWriteUint64(os, matrix.cols_);
    ConvertAndWriteUint64(os, matrix.nnz_);
    os << matrix.index_;
    os << matrix.pos_;
    return os;
  }

  friend std::istream &operator>>(std::istream &is, SparseMatrix &matrix) {
    matrix.rows_ = ReadUint64AndConvert<PosType>(is);
    matrix.cols_ = ReadUint64AndConvert<PosType>(is);
    matrix.nnz_ = ReadUint64AndConvert<IdxType>(is);
    is >> matrix.index_;
    is >> matrix.pos_;
    return is;
  }

  absl::Status WritePrmFile(const std::string &path) const {
    std::ofstream file(path);
    if (!file.is_open() || !file.good())
      return absl::InternalError(absl::StrCat("Error opening file: ", path));
    file << kPrmFileMagicNumbers;
    file << *this;
    file.close();
    if (file.fail())
      return absl::InternalError(absl::StrCat("Error writing file: ", path));
    return absl::OkStatus();
  }

  absl::Status ReadPrmFile(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open() || !file.good())
      return absl::InternalError(absl::StrCat("Error opening file: ", path));
    if (!EatString(file, kPrmFileMagicNumbers))
      return absl::InternalError(absl::StrCat("Bad file format: ", path));

    file >> *this;
    file.close();
    if (file.fail())
      return absl::InternalError(absl::StrCat("Error reading file: ", path));
    return absl::OkStatus();
  }

  absl::Status ReadMatrixMarketFile(const std::string &path,
                                    uint32_t bytes_per_pos = sizeof(PosType),
                                    uint32_t bytes_per_idx = sizeof(IdxType)) {
    DCHECK_LE(bytes_per_pos, sizeof(PosType));
    DCHECK_LE(bytes_per_idx, sizeof(IdxType));
    MatrixMarketIo mat(path);
    if (!mat.Ok()) {
      status_ = absl::InternalError(absl::StrCat("Fail to open file: ", path));
      return status_;
    }

    rows_ = mat.Rows();
    cols_ = mat.Cols();

    PosType prev_col = static_cast<PosType>(-1);
    while (mat.HasNext()) {
      auto pos = mat.Next();
      DCHECK_GT(pos.first, 0);
      DCHECK_GT(pos.second, 0);
      --pos.first;
      --pos.second;
      while (prev_col != pos.second) {
        index_.Append(nnz_);
        ++prev_col;
        DCHECK_LE(prev_col, pos.second);
        DCHECK_EQ(index_[prev_col], nnz_);
      }
      pos_.Append(pos.first);
      nnz_++;
    }
    index_.Append(nnz_);
    return absl::OkStatus();
  }

  std::string DebugString(uint32_t indent = 0) const {
    std::string res =
        absl::StrFormat("SparseMatrix@%x\n", reinterpret_cast<uint64_t>(this));
    std::string tab(indent, ' ');
    absl::StrAppend(&res, tab, "rows: ", rows_, "\n");
    absl::StrAppend(&res, tab, "cols: ", cols_, "\n");
    absl::StrAppend(&res, tab, "nnz: ", nnz_, "\n");
    indent += 2;
    absl::StrAppend(&res, tab, "index: ", index_.DebugString(indent));
    absl::StrAppend(&res, tab, "pos: ", pos_.DebugString(indent));
    return res;
  }


protected:
  absl::Status status_;

private:
  PosType rows_ = 0;
  PosType cols_ = 0;
  IdxType nnz_ = 0;
  FlexIdxType index_;
  FlexPosType pos_;
};

}  // namespace pierank

#endif //PIERANK_SPARSE_MATRIX_H_
