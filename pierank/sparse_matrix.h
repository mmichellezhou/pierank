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
#include "absl/strings/strip.h"

#include "pierank/flex_index.h"
#include "pierank/io/file_utils.h"
#include "pierank/io/matrix_market_io.h"

namespace pierank {

inline constexpr absl::string_view kPieRankMatrixFileMagicNumbers = "#PRM";

inline constexpr absl::string_view kPieRankMatrixFileExtension = ".prm";

inline constexpr absl::string_view kMatrixMarketFileExtension = ".mtx";

// Returns an empty string on error.
inline std::string MatrixMarketToPieRankMatrixPath(
    absl::string_view mtx_path, absl::string_view prm_dir = "") {
  if (!absl::ConsumeSuffix(&mtx_path, kMatrixMarketFileExtension))
    return "";
  if (prm_dir.empty())
    return absl::StrCat(mtx_path, kPieRankMatrixFileExtension);
  return absl::StrCat(prm_dir, kPathSeparator, FileNameInPath(mtx_path),
                      kPieRankMatrixFileExtension);
}

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
  using PosRange = std::pair<PosType, PosType>;

  using PosRanges = std::vector<std::pair<PosType, PosType>>;

  using FlexIdxType = FlexIndex<IdxType>;

  using FlexPosType = FlexIndex<PosType>;

  using UniquePtr = std::unique_ptr<SparseMatrix<PosType, IdxType>>;

  SparseMatrix() = default;

  SparseMatrix(const std::string &prm_file_path, bool mmap = false) {
    status_ = mmap
              ? this->MmapPieRankMatrixFile(prm_file_path)
              : this->ReadPieRankMatrixFile(prm_file_path);
  }

  SparseMatrix(const SparseMatrix &) = delete;

  SparseMatrix &operator=(const SparseMatrix &) = delete;

  const FlexIdxType &Index() const { return index_; }

  // Returns the pos AFTER the max index pos.
  const PosType IndexPosEnd() const {
    return static_cast<PosType>(index_.NumItems() - 1);
  }

  IdxType Index(PosType pos) const { return index_[pos]; }

  const FlexPosType &Pos() const { return pos_; }

  PosType Pos(IdxType idx) const { return pos_[idx]; }

  IdxType NumNonZeros() const { return nnz_; }

  PosType Rows() const { return rows_; }

  PosType Cols() const { return cols_; }

  uint32_t IndexDim() const { return index_dim_; }

  bool ok() const { return status_.ok(); }

  absl::Status status() const { return status_; }

  friend std::ostream &
  operator<<(std::ostream &os, const SparseMatrix &matrix) {
    os << kPieRankMatrixFileMagicNumbers;
    ConvertAndWriteUint64(os, matrix.rows_);
    ConvertAndWriteUint64(os, matrix.cols_);
    ConvertAndWriteUint64(os, matrix.nnz_);
    WriteUint32(os, matrix.index_dim_);
    os << matrix.index_;
    os << matrix.pos_;
    return os;
  }

  // Reads {rows, cols, nnz} from `is`
  uint64_t ReadPieRankMatrixFileHeader(std::istream &is) {
    uint64_t offset = 0;
    if (EatString(is, kPieRankMatrixFileMagicNumbers,&offset)) {
      rows_ = ReadUint64AndConvert<PosType>(is, &offset);
      cols_ = ReadUint64AndConvert<PosType>(is, &offset);
      nnz_ = ReadUint64AndConvert<IdxType>(is, &offset);
      index_dim_ = ReadUint32(is, &offset);
      if (!is)
        status_.Update(absl::InternalError("Error read PRM file header"));
    } else
      status_.Update(absl::InternalError("Bad file format"));
    return offset;
  }

  friend std::istream &operator>>(std::istream &is, SparseMatrix &matrix) {
    matrix.ReadPieRankMatrixFileHeader(is);
    is >> matrix.index_;
    is >> matrix.pos_;
    return is;
  }

  absl::Status WritePieRankMatrixFile(const std::string &path) const {
    auto file = OpenWriteFile(path);
    if (!file.ok()) return file.status();
    *file << *this;
    file->close();
    if (!(*file))
      return absl::InternalError(absl::StrCat("Error write file: ", path));
    return absl::OkStatus();
  }

  absl::Status ReadPieRankMatrixFile(const std::string &path) {
    auto file = OpenReadFile(path);
    if (!file.ok()) return file.status();

    *file >> *this;
    if (!(*file))
      return absl::InternalError(absl::StrCat("Error read file: ", path));

    return absl::OkStatus();
  }

  absl::Status MmapPieRankMatrixFile(const std::string &path) {
    auto file = OpenReadFile(path);
    if (!file.ok()) return file.status();
    uint64_t offset = ReadPieRankMatrixFileHeader(*file);
    status_.Update(index_.Mmap(path, &offset));
    status_.Update(pos_.Mmap(path, &offset));
    return status_;
  }

  void UnMmap() {
    index_.UnMmap();
    pos_.UnMmap();
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
    index_dim_ = 1;  // Matrix Market file is column-major
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

  PosRanges SplitIndexDim(uint32_t num_pieces) const {
    DCHECK(this->status_.ok());
    PosType num_index_pos = index_dim_ ? this->Cols() : this->Rows();
    DCHECK_GT(num_index_pos, 0);
    if (num_pieces == 1)
      return {std::make_pair(0, num_index_pos)};
    PosRanges res;
    PosType range_size = (num_index_pos + num_pieces - 1) / num_pieces;
    for (PosType first = 0; first < num_index_pos; first += range_size) {
      PosType last = std::min(first + range_size, num_index_pos);
      res.push_back(std::make_pair(first, last));
    }
    return res;
  }

  UniquePtr ChangeIndexDim() const {
    auto res = std::make_unique<SparseMatrix<PosType, IdxType>>();
    res->rows_ = rows_;
    res->cols_ = cols_;
    res->nnz_ = nnz_;
    res->index_dim_ = index_dim_ ? 0 : 1;

    PosType new_index_dim_size = index_dim_ ? Rows() : Cols();
    FlexPosType nbr(pos_.ItemSize(), new_index_dim_size);
    auto index_pos_end = IndexPosEnd();
    for (PosType p = 0; p < index_pos_end; ++p) {
      for (IdxType i = index_[p]; i < index_[p + 1]; ++i)
        nbr.IncItem(pos_[i]);
    }

    auto [min_item_size, shift_by_min_val] = index_.MinEncode();
    CHECK(!shift_by_min_val) << "Not yet supported";
    FlexIdxType idx(min_item_size);
    IdxType nnz = 0;
    for (PosType p = 0; p < new_index_dim_size && nnz < nnz_; ++p) {
      idx.Append(nnz);
      nnz += nbr[p];
    }
    DCHECK_EQ(nnz, nnz_);
    if (idx[idx.NumItems() - 1] != nnz)
      idx.Append(nnz);

    nbr.Reset();
    std::tie(min_item_size, shift_by_min_val) = pos_.MinEncode();
    CHECK(!shift_by_min_val) << "Not yet supported";
    FlexPosType pos(min_item_size, nnz);
    for (PosType p = 0; p < index_pos_end; ++p) {
      for (IdxType i = index_[p]; i < index_[p + 1]; ++i) {
        auto pos_i = pos_[i];
        pos.SetItem(idx[pos_i] + nbr[pos_i], p);
        nbr.IncItem(pos_i);
      }
    }

    res->index_ = std::move(idx);
    res->pos_ = std::move(pos);
    return res;
  }

  std::string DebugString(uint32_t indent = 0) const {
    std::string res =
        absl::StrFormat("SparseMatrix@%x\n", reinterpret_cast<uint64_t>(this));
    std::string tab(indent, ' ');
    absl::StrAppend(&res, tab, "rows: ", rows_, "\n");
    absl::StrAppend(&res, tab, "cols: ", cols_, "\n");
    absl::StrAppend(&res, tab, "nnz: ", nnz_, "\n");
    absl::StrAppend(&res, tab, "index_dim: ", index_dim_, "\n");
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
  uint32_t index_dim_ = 0;
  FlexIdxType index_;
  FlexPosType pos_;
};

}  // namespace pierank

#endif //PIERANK_SPARSE_MATRIX_H_
