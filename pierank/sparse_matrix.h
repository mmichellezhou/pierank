//
// Created by Michelle Zhou on 2/26/22.
//

#ifndef PIERANK_SPARSE_MATRIX_H_
#define PIERANK_SPARSE_MATRIX_H_

#include <cstdio>
#include <limits>
#include <glog/logging.h>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
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
    absl::string_view mtx_path, bool change_index_dim = false,
    absl::string_view prm_dir = "") {
  if (!absl::ConsumeSuffix(&mtx_path, kMatrixMarketFileExtension))
    return "";
  std::string_view index_extension = change_index_dim ? ".i0" : ".i1";
  if (prm_dir.empty())
    return absl::StrCat(mtx_path, index_extension, kPieRankMatrixFileExtension);
  return absl::StrCat(prm_dir, kPathSeparator, FileNameInPath(mtx_path),
                      index_extension, kPieRankMatrixFileExtension);
}

// Returns an empty string on error.
inline std::string
PieRankMatrixPathAfterIndexChange(absl::string_view prm_path) {
  auto prm_dir = DirectoryInPath(prm_path);
  auto prm_file_with_extensions = FileNameAndExtensionsInPath(prm_path);
  DCHECK_GE(prm_file_with_extensions.size(), 3);
  if (prm_file_with_extensions.size() < 3)
    return "";
  auto index_extension_idx = prm_file_with_extensions.size() - 2;
  if (prm_file_with_extensions[index_extension_idx] == "i0")
    prm_file_with_extensions[index_extension_idx] = "i1";
  else if (prm_file_with_extensions[index_extension_idx] == "i1")
    prm_file_with_extensions[index_extension_idx] = "i0";
  else
    return "";
  auto prm_file_name = absl::StrJoin(prm_file_with_extensions, ".");
  return absl::StrCat(prm_dir, kPathSeparator, prm_file_name);
}

// Returns (uint32_t)-1 on error.
inline uint32_t IndexDimInPieRankMatrixPath(absl::string_view prm_path) {
  auto prm_file_with_extensions = FileNameAndExtensionsInPath(prm_path);
  DCHECK_GE(prm_file_with_extensions.size(), 3);
  if (prm_file_with_extensions.size() < 3)
    return std::numeric_limits<uint32_t>::max();
  auto index_extension_idx = prm_file_with_extensions.size() - 2;
  if (prm_file_with_extensions[index_extension_idx] == "i0")
    return 0;
  if (prm_file_with_extensions[index_extension_idx] == "i1")
    return 1;
  return std::numeric_limits<uint32_t>::max();
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
  // <min_pos, max_pos, nnz>
  using PosRange = std::tuple<PosType, PosType, IdxType>;

  using PosRanges = std::vector<PosRange>;

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

  friend bool operator==(const SparseMatrix<PosType, IdxType> &lhs,
                         const SparseMatrix<PosType, IdxType> &rhs) {
    if (lhs.rows_ != rhs.rows_) return false;
    if (lhs.cols_ != rhs.cols_) return false;
    if (lhs.nnz_ != rhs.nnz_) return false;
    if (lhs.index_dim_ != rhs.index_dim_) return false;
    if (lhs.index_ != rhs.index_) return false;
    if (lhs.pos_ != rhs.pos_) return false;
    return true;
  }

  void WriteAllButPos(std::ostream *os) const {
    *os << kPieRankMatrixFileMagicNumbers;
    ConvertAndWriteUint64(os, rows_);
    ConvertAndWriteUint64(os, cols_);
    ConvertAndWriteUint64(os, nnz_);
    WriteUint32(os, index_dim_);
    *os << index_;
  }

  friend std::ostream &
  operator<<(std::ostream &os, const SparseMatrix &matrix) {
    matrix.WriteAllButPos(&os);
    os << matrix.pos_;
    return os;
  }

  // Reads {rows, cols, nnz} from `is`
  uint64_t ReadPieRankMatrixFileHeader(std::istream &is) {
    uint64_t offset = 0;
    if (EatString(&is, kPieRankMatrixFileMagicNumbers,&offset)) {
      rows_ = ReadUint64AndConvert<PosType>(&is, &offset);
      cols_ = ReadUint64AndConvert<PosType>(&is, &offset);
      nnz_ = ReadUint64AndConvert<IdxType>(&is, &offset);
      index_dim_ = ReadUint32(&is, &offset);
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

  static PosRanges
  SplitIndexDimByPos(const FlexIdxType &index, uint32_t num_ranges) {
    DCHECK_GT(num_ranges, 0);
    if (index.NumItems() == 0)
      return {std::make_tuple(0, 0, 0)};
    PosType num_index_pos = index.NumItems() - 1;
    auto range_nnz = std::numeric_limits<IdxType>::max();
    if (num_ranges == 1)
      return {std::make_tuple(0, num_index_pos, range_nnz)};
    PosRanges res;

    PosType range_size = (num_index_pos + num_ranges - 1) / num_ranges;
    for (PosType first = 0; first < num_index_pos; first += range_size) {
      PosType last = std::min(first + range_size, num_index_pos);
      res.push_back(std::make_tuple(first, last, range_nnz));
    }

    DCHECK_EQ(res.size(), num_ranges);
    return res;
  }

  static PosRanges SplitIndexDimByNnz(const FlexIdxType &index, IdxType nnz,
                                      uint32_t num_ranges) {
    DCHECK_GT(num_ranges, 0);
    if (index.NumItems() == 0) {
      DCHECK_EQ(nnz, 0);
      return {std::make_tuple(0, 0, 0)};
    }
    PosType num_index_pos = index.NumItems() - 1;
    if (num_ranges == 1)
      return {std::make_tuple(0, num_index_pos, nnz)};
    PosRanges res;
    IdxType max_nnz_per_range = (nnz + num_ranges - 1) / num_ranges;
    PosType avg_nnz_per_pos = (nnz + num_index_pos - 1) / num_index_pos;
    PosType pos_step_size = max_nnz_per_range / avg_nnz_per_pos;
    pos_step_size = std::min(pos_step_size, num_index_pos);
    pos_step_size = std::max(pos_step_size, static_cast<PosType>(1));
    PosType first = 0;
    while (res.size() < num_ranges - 1) {
      DCHECK_LT(first, num_index_pos);
      PosType last = first + 1;
      IdxType range_nnz = index[last] - index[first];
      if (range_nnz < max_nnz_per_range) {
        PosType step_size = pos_step_size;
        while (step_size > 0 && last < num_index_pos) {
          PosType new_last = std::min(last + step_size, num_index_pos);
          IdxType step_nnz = index[new_last] - index[last];
          if (range_nnz + step_nnz <= max_nnz_per_range) {
            range_nnz += step_nnz;
            last = new_last;
            if (step_size * 2 <= pos_step_size)
              step_size *= 2;
          } else
            step_size /= 2;
        }
      }
      res.push_back(std::make_tuple(first, last, range_nnz));
      first = last;
    }

    res.push_back(std::make_tuple(first, num_index_pos,
                                  index[num_index_pos] - index[first]));
    DCHECK_EQ(res.size(), num_ranges);
    return res;
  }

  PosRanges SplitIndexDimByPos(uint32_t num_ranges) const {
    DCHECK(this->status_.ok());
    return SplitIndexDimByPos(index_, num_ranges);
  }

  PosRanges SplitIndexDimByNnz(uint32_t num_ranges) const {
    DCHECK(this->status_.ok());
    return SplitIndexDimByNnz(index_, nnz_, num_ranges);
  }

  absl::StatusOr<UniquePtr> ChangeIndexDim(uint32_t num_ranges = 1) const {
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

    auto [idx_item_size, idx_shift_by_min_val] = index_.MinEncode();
    CHECK(!idx_shift_by_min_val) << "Not yet supported";
    FlexIdxType idx(idx_item_size);
    IdxType nnz = 0;
    for (PosType p = 0; p < new_index_dim_size && nnz < nnz_; ++p) {
      idx.Append(nnz);
      nnz += nbr[p];
    }
    DCHECK_EQ(nnz, nnz_);
    if (idx[idx.NumItems() - 1] != nnz)
      idx.Append(nnz);
    auto ranges = SplitIndexDimByNnz(idx, nnz, num_ranges);
    // std::cout << PosRangesDebugString(ranges);

    nbr.Reset();
    auto [pos_item_size, pos_shift_by_min_val] = pos_.MinEncode();
    CHECK(!pos_shift_by_min_val) << "Not yet supported";
    nnz = 0;

    for (uint32_t r = 0; r < num_ranges; ++r) {
      FlexPosType pos(pos_item_size, std::get<2>(ranges[r]));
      IdxType range_nnz = 0;
      auto [min_pos, max_pos, size] = ranges[r];
      for (PosType p = 0; p < index_pos_end; ++p) {
        for (IdxType i = index_[p]; i < index_[p + 1]; ++i) {
          auto pos_i = pos_[i];
          if (pos_i >= min_pos && pos_i < max_pos) {
            pos.SetItem(idx[pos_i] + nbr[pos_i] - nnz, p);
            nbr.IncItem(pos_i);
            ++range_nnz;
          }
        }
      }
      nnz += range_nnz;
      if (num_ranges == 1)
        res->pos_ = std::move(pos);
      else {
        CHECK(false) << "Not yet supported";
      }
    }
    DCHECK_EQ(nnz, nnz_);

    res->index_ = std::move(idx);
    if (num_ranges > 1) {
      CHECK(false) << "Not yet supported";
    }

    return res;
  }

  // Returns a memory-mapped SparseMatrix with its index dim changed.
  absl::StatusOr<UniquePtr> ChangeIndexDimByMmap(
      const std::string &path,
      uint64_t max_nnz_per_range = std::numeric_limits<uint32_t>::max()) const {
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

    auto [idx_item_size, idx_shift_by_min_val] = index_.MinEncode();
    CHECK(!idx_shift_by_min_val) << "Not yet supported";
    FlexIdxType idx(idx_item_size);
    IdxType nnz = 0;
    for (PosType p = 0; p < new_index_dim_size && nnz < nnz_; ++p) {
      idx.Append(nnz);
      nnz += nbr[p];
    }
    DCHECK_EQ(nnz, nnz_);
    if (idx[idx.NumItems() - 1] != nnz)
      idx.Append(nnz);
    uint32_t num_ranges = (nnz + max_nnz_per_range - 1) / max_nnz_per_range;
    auto ranges = SplitIndexDimByNnz(idx, nnz, num_ranges);
    // std::cout << PosRangesDebugString(ranges);

    nbr.Reset();
    auto [pos_item_size, pos_shift_by_min_val] = pos_.MinEncode();
    CHECK(!pos_shift_by_min_val) << "Not yet supported";
    nnz = 0;

    // <num_items, file_path> for each range's pos FlexIndex
    std::vector<std::pair<uint64_t, std::string>> pos_items_and_paths;
    for (uint32_t r = 0; r < num_ranges; ++r) {
      FlexPosType pos(pos_item_size, std::get<2>(ranges[r]));
      IdxType range_nnz = 0;
      auto [min_pos, max_pos, size] = ranges[r];
      for (PosType p = 0; p < index_pos_end; ++p) {
        for (IdxType i = index_[p]; i < index_[p + 1]; ++i) {
          auto pos_i = pos_[i];
          if (pos_i >= min_pos && pos_i < max_pos) {
            pos.SetItem(idx[pos_i] + nbr[pos_i] - nnz, p);
            nbr.IncItem(pos_i);
            ++range_nnz;
          }
        }
      }
      nnz += range_nnz;
      auto[fp, tmp_path] = OpenTmpFile(path);
      DCHECK_NOTNULL(fp);
      pos_items_and_paths.push_back(std::make_pair(pos.NumItems(), tmp_path));
      // TODO: Support shift_by_min_value
      if (!pos.WriteValues(fp, pos_item_size, /*shift_by_min_val=*/false))
        return absl::InternalError("Error write file: " + tmp_path);
      fclose(fp);
    }
    DCHECK_EQ(nnz, nnz_);

    DCHECK_EQ(pos_items_and_paths.size(), num_ranges);
    auto file_or = OpenWriteFile(path);
    if (!file_or.ok()) return file_or.status();
    auto ofs = std::move(file_or).value();
    res->index_ = std::move(idx);
    res->WriteAllButPos(&ofs);
    res->pos_.SetMinMaxValues(0, index_pos_end - 1);
    res->pos_.WriteAllButValues(&ofs, pos_item_size, /*shift_by_min=*/false);
    if (!WriteUint64(&ofs, pos_item_size * nnz))
      return absl::InternalError("Error write file: " + path);
    for (const auto &pos_items_and_path : pos_items_and_paths) {
      auto[pos_items, pos_path] = pos_items_and_path;
      auto tmp_file_or = OpenReadFile(pos_path);
      if (!tmp_file_or.ok()) return tmp_file_or.status();
      FlexPosType pos(pos_item_size, pos_items);
      pos.ReadValues(&*tmp_file_or);
      DCHECK_EQ(pos.NumItems(), pos_items);
      WriteData(&ofs, pos.Data(), pos_item_size * pos_items);
      std::remove(pos_path.c_str());
    }
    ofs.close();

    res.reset(new SparseMatrix<PosType, IdxType>(path, /*mmap=*/true));
    return res;
  }

  std::string DebugString(uint64_t max_items = 0, uint32_t indent = 0) const {
    std::string res;
    std::string tab(indent, ' ');
    if (!status_.ok()) {
      absl::StrAppend(&res, tab, status_.ToString(), "\n");
      return res;
    }
    absl::StrAppend(&res, tab, "rows: ", rows_, "\n");
    absl::StrAppend(&res, tab, "cols: ", cols_, "\n");
    absl::StrAppend(&res, tab, "nnz: ", nnz_, "\n");
    absl::StrAppend(&res, tab, "index_dim: ", index_dim_, "\n");
    indent += 2;
    absl::StrAppend(&res, tab, "index:\n",
                    index_.DebugString(max_items, indent));
    absl::StrAppend(&res, tab, "pos:\n", pos_.DebugString(max_items, indent));
    return res;
  }

  static std::string PosRangeDebugString(const PosRange &range) {
    std::string res;
    absl::StrAppend(&res, "min: ", std::get<0>(range));
    absl::StrAppend(&res, ", max: ", std::get<1>(range));
    absl::StrAppend(&res, ", size: ", std::get<2>(range));
    return res;
  }

  static std::string PosRangesDebugString(const PosRanges &ranges) {
    std::string res;
    for (const auto &r : ranges)
      absl::StrAppend(&res, PosRangeDebugString(r), "\n");
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
