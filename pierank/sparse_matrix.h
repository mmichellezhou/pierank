//
// Created by Michelle Zhou on 2/26/22.
//

#ifndef PIERANK_SPARSE_MATRIX_H_
#define PIERANK_SPARSE_MATRIX_H_

#include <algorithm>
#include <cstdio>
#include <limits>
#include <numeric>
#include <glog/logging.h>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"

#include "pierank/flex_index.h"
#include "pierank/math_utils.h"
#include "pierank/thread_pool.h"
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

  // PosRanges for InitRanges, UpdateRanges, and ReconcileRanges
  using TriplePosRanges = std::array<PosRanges, 3>;

  using FlexIdxType = FlexIndex<IdxType>;

  using FlexPosType = FlexIndex<PosType>;

  using UniquePtr = std::unique_ptr<SparseMatrix<PosType, IdxType>>;

  using UniqueIdxPtr = std::unique_ptr<FlexIdxType>;

  using UniquePosPtr = std::unique_ptr<FlexPosType>;

  using RangeFunc = void (SparseMatrix<PosType, IdxType>::*)(
      const PosRanges &ranges, uint32_t range_id);

  SparseMatrix() = default;

  SparseMatrix(const std::string &prm_file_path, bool mmap = false) {
    status_ = mmap
              ? this->MmapPieRankMatrixFile(prm_file_path)
              : this->ReadPieRankMatrixFile(prm_file_path);
  }

  SparseMatrix(const SparseMatrix &) = delete;

  SparseMatrix &operator=(const SparseMatrix &) = delete;

  const FlexIdxType &Index() const { return index_; }

  IdxType Index(PosType pos) const { return index_[pos]; }

  // Returns the pos AFTER the max index pos.
  const PosType IndexPosEnd() const {
    return static_cast<PosType>(index_.NumItems() - 1);
  }

  const FlexPosType &Pos() const { return pos_; }

  PosType Pos(IdxType idx) const { return pos_[idx]; }

  bool PosIsCompressed() const { return pos_.IsCompressed(); }

  IdxType NumNonZeros() const { return nnz_; }

  PosType Rows() const { return rows_; }

  PosType Cols() const { return cols_; }

  PosType MaxDimSize() const { return std::max(rows_, cols_); }

  bool Symmetric() const { return symmetric_; }

  uint32_t IndexDim() const { return index_dim_; }

  bool ok() const { return status_.ok(); }

  absl::Status status() const { return status_; }

  friend bool operator==(const SparseMatrix<PosType, IdxType> &lhs,
                         const SparseMatrix<PosType, IdxType> &rhs) {
    if (lhs.rows_ != rhs.rows_) return false;
    if (lhs.cols_ != rhs.cols_) return false;
    if (lhs.nnz_ != rhs.nnz_) return false;
    if (lhs.symmetric_ != rhs.symmetric_) return false;
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
    ConvertAndWriteUint32(os, symmetric_);
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
      symmetric_ = ReadUint32AndConvert<bool>(&is, &offset);
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
    symmetric_ = mat.Symmetric();
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

  static PosRanges ClonePosRange(uint32_t num_copies, PosType max_pos,
                                 PosType min_pos = 0) {
    auto range_nnz = std::numeric_limits<IdxType>::max();
    return PosRanges(num_copies, std::make_tuple(min_pos, max_pos, range_nnz));
  }

  static PosRanges SplitPosIntoRanges(PosType num_pos, uint32_t max_ranges) {
    DCHECK_GT(max_ranges, 0);
    if (num_pos == 0)
      return {std::make_tuple(0, 0, 0)};
    auto range_nnz = std::numeric_limits<IdxType>::max();
    if (max_ranges == 1)
      return {std::make_tuple(0, num_pos, range_nnz)};

    PosRanges res;
    PosType range_size = UnsignedDivideCeil(num_pos, max_ranges);
    DCHECK_GT(range_size, 0);
    for (PosType min_pos = 0; min_pos < num_pos; min_pos += range_size) {
      PosType max_pos = std::min(min_pos + range_size, num_pos);
      DCHECK_LT(min_pos, max_pos);
      res.push_back(std::make_tuple(min_pos, max_pos, range_nnz));
    }
    DCHECK_LE(res.size(), max_ranges);
    return res;
  }

  static PosRanges
  SplitIndexDimByPos(const FlexIdxType &index, uint32_t num_ranges) {
    if (index.NumItems() == 0)
      return {std::make_tuple(0, 0, 0)};
    return SplitPosIntoRanges(index.NumItems() - 1, num_ranges);
  }

  // Returned PosRanges.size() may be less than max_ranges.
  static PosRanges SplitIndexDimByNnz(const FlexIdxType &index, IdxType nnz,
                                      uint32_t max_ranges) {
    DCHECK_GT(max_ranges, 0);
    if (index.NumItems() == 0) {
      DCHECK_EQ(nnz, 0);
      return {std::make_tuple(0, 0, 0)};
    }
    PosType num_index_pos = index.NumItems() - 1;
    if (max_ranges == 1)
      return {std::make_tuple(0, num_index_pos, nnz)};
    PosRanges res;
    IdxType max_nnz_per_range = UnsignedDivideCeil(nnz, max_ranges);
    PosType avg_nnz_per_pos = UnsignedDivideCeil(nnz, num_index_pos);
    PosType pos_step_size = max_nnz_per_range / avg_nnz_per_pos;
    pos_step_size = std::min(pos_step_size, num_index_pos);
    pos_step_size = std::max(pos_step_size, static_cast<PosType>(1));
    PosType min_pos = 0;
    while ((res.size() < max_ranges - 1) && (min_pos < num_index_pos)) {
      PosType max_pos = min_pos + 1;
      IdxType range_nnz = index[max_pos] - index[min_pos];
      if (range_nnz < max_nnz_per_range) {
        PosType step_size = pos_step_size;
        while (step_size > 0 && max_pos < num_index_pos) {
          PosType new_max_pos = std::min(max_pos + step_size, num_index_pos);
          IdxType step_nnz = index[new_max_pos] - index[max_pos];
          if (range_nnz + step_nnz <= max_nnz_per_range) {
            range_nnz += step_nnz;
            max_pos = new_max_pos;
            if (step_size * 2 <= pos_step_size)
              step_size *= 2;
          } else
            step_size /= 2;
        }
      }
      res.push_back(std::make_tuple(min_pos, max_pos, range_nnz));
      min_pos = max_pos;
    }
    if (min_pos < num_index_pos) {
      res.push_back(std::make_tuple(min_pos, num_index_pos,
                                    index[num_index_pos] - index[min_pos]));
      DCHECK_EQ(res.size(), max_ranges);
    }
    DCHECK_LE(res.size(), max_ranges);
    return res;
  }

  PosRanges SplitIndexDimByPos(uint32_t max_ranges) const {
    DCHECK(this->status_.ok());
    return SplitIndexDimByPos(index_, max_ranges);
  }

  PosRanges SplitIndexDimByNnz(uint32_t max_ranges) const {
    DCHECK(this->status_.ok());
    return SplitIndexDimByNnz(index_, nnz_, max_ranges);
  }

  uint32_t
  MaxRanges(uint64_t max_nnz_per_range,
            uint32_t max_ranges = std::numeric_limits<uint32_t>::max()) const {
    return std::min(
        static_cast<uint32_t>(UnsignedDivideCeil(nnz_, max_nnz_per_range)),
        max_ranges);
  }

  uint32_t MaxRanges(uint64_t max_nnz_per_range,
                     std::shared_ptr<ThreadPool> pool) const {
    return MaxRanges(max_nnz_per_range, pool ? pool->Size() : 1);
  }

  // Counts the # of non-zeros (nnz) for each pos in non-index dim
  UniquePosPtr CountNonIndexDimNnz() const {
    PosType non_index_dim_size = index_dim_ ? Rows() : Cols();
    auto res = std::make_unique<FlexPosType>(pos_.ItemSize(),
                                             non_index_dim_size);
    auto index_pos_end = this->IndexPosEnd();
    for (PosType p = 0; p < index_pos_end; ++p) {
      for (IdxType i = index_[p]; i < index_[p + 1]; ++i)
        res->IncItem(pos_[i]);
    }

    return res;
  }

  UniqueIdxPtr CreateReverseIndex(const FlexPosType &nbr) const {
    auto [idx_item_size, idx_shift_by_min_val] = index_.MinEncode();
    CHECK(!idx_shift_by_min_val) << "Not yet supported";
    auto res = std::make_unique<FlexIdxType>(idx_item_size);
    IdxType nnz = 0;
    PosType new_index_dim_size = index_dim_ ? Rows() : Cols();
    for (PosType p = 0; p < new_index_dim_size && nnz < nnz_; ++p) {
      res->Append(nnz);
      nnz += nbr[p];
    }
    DCHECK_EQ(nnz, nnz_);
    if ((*res)[res->NumItems() - 1] != nnz)
      res->Append(nnz);

    return res;
  }

  std::vector<IdxType> RangeNnzOffsets(const PosRanges &ranges) const {
    std::vector<IdxType> res;
    // std::transform_exclusive_scan(
    //    ranges.begin(), ranges.end(),
    //    std::back_inserter(res), 0, std::plus<IdxType>{},
    //    [](const PosRange &range) { return std::get<2>(range); });
    res.push_back(0);
    for (std::size_t i = 0; i < ranges.size() - 1; ++i)
      res.push_back(res.back() + std::get<2>(ranges[i]));

    return res;
  }

  UniquePosPtr ReversePosInRange(const PosRanges &ranges,
                                 const std::vector<IdxType> &offsets,
                                 uint32_t range_id,
                                 const FlexIdxType &idx,
                                 FlexPosType *nbr) const {
    auto [min_pos, max_pos, range_size] = ranges[range_id];
    auto offset = offsets[range_id];
    auto index_pos_end = IndexPosEnd();
    uint32_t pos_item_size = MinEncodeSize(index_pos_end);
    auto res = std::make_unique<FlexPosType>(pos_item_size, range_size);

    for (PosType p = 0; p < index_pos_end; ++p) {
      for (IdxType i = index_[p]; i < index_[p + 1]; ++i) {
        auto pos_i = pos_[i];
        if (pos_i >= min_pos && pos_i < max_pos) {
          res->SetItem(idx[pos_i] + (*nbr)[pos_i] - offset, p);
          nbr->IncItem(pos_i);
        }
      }
    }
    return res;
  }

  absl::StatusOr<UniquePtr>
  ChangeIndexDim(std::shared_ptr<ThreadPool> pool = nullptr,
                 uint64_t max_nnz_per_thread = 8000000) const {
    auto res = std::make_unique<SparseMatrix<PosType, IdxType>>();
    res->rows_ = rows_;
    res->cols_ = cols_;
    res->nnz_ = nnz_;
    res->symmetric_ = symmetric_;
    res->index_dim_ = index_dim_ ? 0 : 1;

    auto nnz = CountNonIndexDimNnz();
    auto idx = CreateReverseIndex(*nnz);
    auto ranges =
        SplitIndexDimByNnz(*idx, nnz_, MaxRanges(max_nnz_per_thread, pool));
    // std::cout << PosRangesDebugString(ranges);

    auto offsets = RangeNnzOffsets(ranges);
    std::vector<UniquePosPtr> poses(ranges.size());
    nnz->Reset();
    if (ranges.size() == 1) {
      auto pos = ReversePosInRange(ranges, offsets, 0, *idx, nnz.get());
      res->pos_ = std::move(*pos.release());
    } else {
      DCHECK(pool);
      pool->ParallelFor(
          ranges.size(), /*items_per_thread=*/1,
          [&, this](uint64_t first, uint64_t last) {
            for (auto r = first; r < last; ++r) {
              auto pos = ReversePosInRange(ranges, offsets, r, *idx, nnz.get());
              poses[r] = std::move(pos);
            }
          });
      res->pos_.SetItemSize(poses.front()->ItemSize());
      for (auto &pos : poses) {
        res->pos_.Append(*pos);
        pos.reset();
      }
    }
    res->index_ = std::move(*idx.release());
    return res;
  }

  // Returns a memory-mapped SparseMatrix with its index dim changed.
  absl::StatusOr<UniquePtr> ChangeIndexDim(
      const std::string &path,
      uint64_t max_nnz_per_range = 64000000) const {
    auto res = std::make_unique<SparseMatrix<PosType, IdxType>>();
    res->rows_ = rows_;
    res->cols_ = cols_;
    res->nnz_ = nnz_;
    res->index_dim_ = index_dim_ ? 0 : 1;

    auto nnz = CountNonIndexDimNnz();
    auto idx = CreateReverseIndex(*nnz);
    auto ranges = SplitIndexDimByNnz(*idx, nnz_, MaxRanges(max_nnz_per_range));
    // std::cout << PosRangesDebugString(ranges);

    auto offsets = RangeNnzOffsets(ranges);
    // <num_items, item_size, file_path> for each range's pos FlexIndex
    std::vector<std::tuple<uint64_t, uint32_t, std::string>> tmp_pos_infos;
    nnz->Reset();
    auto pos_min = std::numeric_limits<PosType>::max();
    auto pos_max = std::numeric_limits<PosType>::min();
    for (uint32_t r = 0; r < ranges.size(); ++r) {
      auto pos = ReversePosInRange(ranges, offsets, r, *idx, nnz.get());
      pos_min = std::min(pos_min, pos->MinValue());
      pos_max = std::max(pos_max, pos->MaxValue());
      auto[fp, tmp_path] = OpenTmpFile(path);
      DCHECK(fp);
      tmp_pos_infos.push_back(
          std::make_tuple(pos->NumItems(), pos->ItemSize(), tmp_path));
      if (!pos->WriteValues(fp, pos->ItemSize(), /*shift_by_min_val=*/false))
        return absl::InternalError("Error write file: " + tmp_path);
      fclose(fp);
    }
    DCHECK_GE(pos_min, 0);
    DCHECK_LT(pos_max, res->index_dim_ ? res->rows_ : res->cols_);
    DCHECK_EQ(tmp_pos_infos.size(), ranges.size());

    auto file_or = OpenWriteFile(path);
    if (!file_or.ok()) return file_or.status();
    auto ofs = std::move(file_or).value();
    res->index_ = std::move(*idx.release());
    res->WriteAllButPos(&ofs);
    res->pos_.SetMinMaxValues(pos_min, pos_max);
    auto[pos_item_size, pos_shift_by_min_val] = FlexPosType::MinEncode(pos_max,
                                                                       pos_min);
    PosType pos_value_shift = pos_shift_by_min_val ? -pos_min : 0;
    res->pos_.WriteAllButValues(&ofs, pos_item_size, pos_shift_by_min_val);
    if (!WriteUint64(&ofs, pos_item_size * nnz_))
      return absl::InternalError("Error write file: " + path);
    for (const auto &tmp_pos_info : tmp_pos_infos) {
      auto[tmp_pos_items, tmp_pos_item_size, tmp_pos_path] = tmp_pos_info;
      auto tmp_file_or = OpenReadFile(tmp_pos_path);
      if (!tmp_file_or.ok()) return tmp_file_or.status();
      FlexPosType pos(pos_item_size, tmp_pos_items);
      pos.ReadValues(&*tmp_file_or, tmp_pos_item_size, pos_value_shift);
      DCHECK_EQ(pos.NumItems(), tmp_pos_items);
      WriteData(&ofs, pos.Data(), pos_item_size * tmp_pos_items);
      std::remove(tmp_pos_path.c_str());
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
    absl::StrAppend(&res, tab, "symmetric: ", symmetric_, "\n");
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

  virtual void InitRanges(const PosRanges &ranges, uint32_t range_id) {}

  virtual void UpdateRanges(const PosRanges &ranges, uint32_t range_id) {}

  virtual void ReconcileRanges(const PosRanges &ranges, uint32_t range_id) {}

  virtual bool Stop() const { return true; }

  std::pair<PosType, PosType>
  RangeMinMaxPos(const PosRanges &ranges, uint32_t range_id) const {
    DCHECK_LT(range_id, ranges.size());
    const auto &range = ranges[range_id];
    auto min_pos = std::get<0>(range);
    auto max_pos = std::get<1>(range);
    DCHECK_LT(min_pos, max_pos);
    return std::make_pair(min_pos, max_pos);
  }

  void RunRangeFunc(RangeFunc func, const PosRanges &ranges,
                    std::shared_ptr<ThreadPool> pool = nullptr) {
    if (ranges.size() == 1) {
      (this->*func)(ranges, 0);
    } else {
      DCHECK(pool);
      pool->ParallelFor(ranges.size(), /*items_per_thread=*/1,
                        [this, func, &ranges](uint64_t first, uint64_t last) {
                          for (auto r = first; r < last; ++r)
                            (this->*func)(ranges, /*range_id=*/r);
                        });
    }
  }

  auto RangeFuncs() const {
    return std::make_tuple(&SparseMatrix<PosType, IdxType>::InitRanges,
                           &SparseMatrix<PosType, IdxType>::UpdateRanges,
                           &SparseMatrix<PosType, IdxType>::ReconcileRanges);
  }

  bool ProcessRanges(uint32_t max_ranges = 1,
                     std::shared_ptr<ThreadPool> pool = nullptr) {
    DCHECK_GT(max_ranges, 0);
    if (!this->status_.ok()) return false;
    max_ranges = std::min(max_ranges, pool ? pool->Size() : 1);
    const auto pos_balanced_ranges =
        this->SplitPosIntoRanges(MaxDimSize(), max_ranges);
    const auto nnz_balanced_ranges = this->SplitIndexDimByNnz(max_ranges);

    auto[init, update, reconcile] = RangeFuncs();
    RunRangeFunc(init, pos_balanced_ranges, pool);
    while (!Stop()) {
      RunRangeFunc(update, nnz_balanced_ranges, pool);
      RunRangeFunc(reconcile, pos_balanced_ranges, pool);
    }
    return true;
  }

  bool ProcessRanges(const TriplePosRanges &ranges,
                     std::shared_ptr <ThreadPool> pool = nullptr) {
    if (!this->status_.ok()) return false;

    auto[init, update, reconcile] = RangeFuncs();
    RunRangeFunc(init, ranges[0], pool);
    while (!Stop()) {
      RunRangeFunc(update, ranges[1], pool);
      RunRangeFunc(reconcile, ranges[2], pool);
    }
    return true;
  }

private:
  PosType rows_ = 0;
  PosType cols_ = 0;
  IdxType nnz_ = 0;
  bool symmetric_ = false;
  uint32_t index_dim_ = 0;
  FlexIdxType index_;
  FlexPosType pos_;
};

}  // namespace pierank

#endif //PIERANK_SPARSE_MATRIX_H_
