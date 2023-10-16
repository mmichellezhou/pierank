//
// Created by Michelle Zhou on 2/26/22.
//

#ifndef PIERANK_SPARSE_MATRIX_H_
#define PIERANK_SPARSE_MATRIX_H_

#include <algorithm>
#include <cstdio>
#include <limits>
#include <numeric>
#include <type_traits>
#include <variant>
#include <vector>

#include <glog/logging.h>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"

#include "pierank/data_type.h"
#include "pierank/flex_array.h"
#include "pierank/math_utils.h"
#include "pierank/matrix.h"
#include "pierank/string_utils.h"
#include "pierank/thread_pool.h"
#include "pierank/io/file_utils.h"
#include "pierank/io/matrix_market_io.h"

namespace pierank {

inline constexpr absl::string_view kPieRankMatrixFileMagicNumbers = "#PRM";

inline constexpr absl::string_view kPieRankMatrixFileExtension = ".prm";

inline constexpr absl::string_view kMatrixMarketFileExtension = ".mtx";

// Returns an empty string on error.
inline std::string MatrixMarketToPieRankMatrixPath(
    absl::string_view mtx_path, uint32_t index_dim = 1,
    absl::string_view prm_dir = "") {
  if (!absl::ConsumeSuffix(&mtx_path, kMatrixMarketFileExtension))
    return "";

  std::string index_extension = absl::StrFormat(".i%u", index_dim);
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

inline absl::StatusOr<std::pair<MatrixType, DataType>>
PieRankFileTypes(const std::string &prm_path) {
  auto file_or = OpenReadFile(prm_path);
  if (!file_or.ok()) return file_or.status();
  auto file = *std::move(file_or);
  if (EatString(&file, kPieRankMatrixFileMagicNumbers)) {
    MatrixType matrix_type;
    auto status = matrix_type.Read(&file);
    if (!status.ok()) return status;
    DataType data_type;
    status.Update(data_type.Read(&file));
    if (!status.ok()) return status;
    return std::make_pair(matrix_type, data_type);
  } else
    return absl::InternalError("Bad file format");
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
template<typename PosType, typename IdxType,
    typename DataContainerType = std::vector<double>>
class SparseMatrix : public Matrix<PosType, IdxType, DataContainerType> {
public:
  // <min_pos, max_pos, nnz>
  using PosRange = std::tuple<PosType, PosType, IdxType>;

  using PosRanges = std::vector<PosRange>;

  // PosRanges for InitRanges, UpdateRanges, and SyncRanges
  using TriplePosRanges = std::array<PosRanges, 3>;

  using FlexIdxType = FlexArray<IdxType>;

  using FlexIdxIterator = typename FlexIdxType::Iterator;

  using FlexPosType = FlexArray<PosType>;

  using FlexPosIterator = typename FlexPosType::Iterator;

  using value_type = typename DataContainerType::value_type;

  using DenseType = Matrix<PosType, IdxType, DataContainerType>;

  using Entry = MatrixMarketIo::Entry;

  using DataFamily = MatrixType::DataFamily;

  using UniquePtr =
      std::unique_ptr<SparseMatrix<PosType, IdxType, DataContainerType>>;

  using RangeFunc = void (SparseMatrix<PosType, IdxType>::*)(
      const PosRanges &ranges, uint32_t range_id);

  SparseMatrix() = default;

  SparseMatrix(const std::string &prm_file_path, bool mmap = false) {
    this->status_ = mmap
              ? this->MmapPieRankMatrixFile(prm_file_path)
              : this->ReadPieRankMatrixFile(prm_file_path);
  }

  SparseMatrix(const DenseType &dense) :
      Matrix<PosType, IdxType, DataContainerType>(dense.Type(),
                                                  dense.Shape(),
                                                  dense.Order()) {
    if (dense.SplitDepths()) {
      // SparseMatrix can't split data dims, which thus must be last in order_
      auto it =
          std::find(this->order_.begin(), this->order_.end(), this->DepthDim());
      CHECK(it != this->order_.end());
      std::rotate(it, it + 1, this->order_.end());
    }

    const uint32_t depths = dense.Depths();
    const uint64_t elem_stride = dense.ElemStride();
    for (IdxType i = 0; i < dense.Elems(); ++i) {
      auto && [row, col, depth] = dense.IdxToPos(i * elem_stride);
      DCHECK_EQ(depth, 0);
      bool all_zeros = true;
      for (uint32_t d = 0; d < depths && all_zeros; ++d) {
        if (dense(row, col, d) != 0) all_zeros = false;
      }
      if (!all_zeros) {
        std::vector<MatrixMarketIo::Var> vars;
        for (uint32_t d = 0; d < depths; ++d)
          vars.push_back(dense(row, col, d));
        push_back({{row, col}, vars});
      }
    }
    ++index_pos_end_;
    index_.push_back(nnz_);
  }

  SparseMatrix(const SparseMatrix &) = delete;

  SparseMatrix &operator=(const SparseMatrix &) = delete;

  SparseMatrix(SparseMatrix &&) = default;

  SparseMatrix &operator=(SparseMatrix &&) = default;

  inline static constexpr DataType StaticDataType() {
    if constexpr (is_specialization_v<DataContainerType, FlexArray>)
      return DataType::kFlex;
    return DataType::FromValueType<value_type>();
  }

  DenseType ToDense(bool split_depths = false) const {
    std::vector<uint64_t> shape = this->Shape();
    std::vector<uint32_t> order = this->Order();
    if (split_depths)
      std::rotate(order.begin(), order.begin() + this->DepthDim(), order.end());
    DenseType res(this->type_, shape, order);
    res.InitData();
    PosType items_per_index_pos = this->NonIndexDimSize();
    if (!split_depths) items_per_index_pos *= this->Depths();
    uint32_t elem_stride = split_depths ? 1 : this->Depths();
    IdxType data_idx = 0;
    const bool has_data = !this->type_.IsPattern();
    const IdxType depth_stride = res.DepthStride();
    for (PosType p = 0; p < index_.size() - 1; ++p) {
      for (IdxType i = this->Index(p); i < this->Index(p + 1); ++i) {
        IdxType idx0 = p * items_per_index_pos + this->Pos(i) * elem_stride;
        for (uint32_t j = 0; j < this->Depths(); ++j) {
          IdxType idx = idx0 + j * depth_stride;
          res.Set(idx, has_data ? this->data_[data_idx++] : 1);
        }
      }
    }
    DCHECK(!has_data || data_idx == nnz_ * this->Depths());
    return res;
  }

  value_type operator()(PosType row, PosType col, uint32_t depth = 0) const {
    PosType non_idx_pos;
    FlexPosIterator first, last;
    if (this->IndexDim() == 0) {
      if (row + 1 >= index_.size()) return 0;
      first = pos_(index_[row]);
      last = pos_(index_[row + 1]);
      non_idx_pos = col;
    } else {
      if (col + 1 >= index_.size()) return 0;
      first = pos_(index_[col]);
      last = pos_(index_[col + 1]);
      non_idx_pos = row;
    }
    auto it = std::lower_bound(first, last, non_idx_pos);
    if (it != last && *it == non_idx_pos) {
      if (!this->type_.IsPattern()) return At(it - pos_(), depth);
      return 1;
    } else
      return 0;
  }

  const FlexIdxType &Index() const { return index_; }

  IdxType Index(PosType pos) const { return index_[pos]; }

  // Returns the pos AFTER the max index pos.
  const PosType IndexPosEnd() const {
    return static_cast<PosType>(index_.size() - 1);
  }

  const FlexPosType &Pos() const { return pos_; }

  PosType Pos(IdxType idx) const { return pos_[idx]; }

  IdxType NumNonZeros() const { return nnz_; }

  friend bool
  operator==(const SparseMatrix<PosType, IdxType, DataContainerType> &lhs,
             const SparseMatrix<PosType, IdxType, DataContainerType> &rhs) {
    if (lhs.type_ != rhs.type_) return false;
    if (lhs.shape_ != rhs.shape_) return false;
    if (lhs.order_ != rhs.order_) return false;
    if (lhs.nnz_ != rhs.nnz_) return false;
    if (lhs.index_ != rhs.index_) return false;
    if (lhs.pos_ != rhs.pos_) return false;
    if (lhs.data_ != rhs.data_) return false;
    return true;
  }

  void WriteAllButPosAndData(std::ostream *os) const {
    *os << kPieRankMatrixFileMagicNumbers;
    auto status = this->type_.Write(os);
    if (!status.ok()) *os << status.message();
    status = StaticDataType().Write(os);
    if (!status.ok()) *os << status.message();
    WriteUint64Vector(os, this->shape_);
    WriteUint32Vector(os, this->order_);
    ConvertAndWriteUint64(os, nnz_);
    *os << index_;
  }

  friend std::ostream &
  operator<<(std::ostream &os, const SparseMatrix &matrix) {
    matrix.WriteAllButPosAndData(&os);
    os << matrix.pos_;
    WriteUint64(&os, matrix.data_.size());
    if constexpr (is_specialization_v<DataContainerType, FlexArray>) {
      auto status = matrix.data_.Write(&os);
      if (!status.ok()) LOG(FATAL) << status.message();
    } else {
      if constexpr (!is_specialization_v<DataContainerType, std::vector>)
        LOG(WARNING) << "Unknown data container type";
      WriteData<std::ostream, value_type>(&os, matrix.data_.data(),
                                          matrix.data_.size());
    }

    return os;
  }

  // Reads {rows, cols, nnz} from `is`
  uint64_t ReadPieRankMatrixFileHeader(std::istream &is) {
    uint64_t offset = 0;
    if (EatString(&is, kPieRankMatrixFileMagicNumbers, &offset)) {
      auto status = this->type_.Read(&is, &offset);
      if (!status.ok()) LOG(FATAL) << status.message();
      DataType data_type;
      status.Update(data_type.Read(&is, &offset));
      if (!status.ok()) LOG(FATAL) << status.message();
      CHECK_EQ(data_type, StaticDataType());
      this->shape_ = ReadUint64Vector(&is, &offset);
      this->order_ = ReadUint32Vector(&is, &offset);
      nnz_ = ReadUint64AndConvert<IdxType>(&is, &offset);
      CHECK_LT(this->IndexDim(), 2);
      if (!is)
        this->status_.Update(absl::InternalError("Error read PRM file header"));
    } else
      this->status_.Update(absl::InternalError("Bad file format"));
    return offset;
  }

  friend std::istream &operator>>(std::istream &is, SparseMatrix &matrix) {
    matrix.ReadPieRankMatrixFileHeader(is);
    is >> matrix.index_;
    is >> matrix.pos_;
    auto size = ReadUint64(&is);
    if constexpr (is_specialization_v<DataContainerType, FlexArray>) {
      auto status = matrix.data_.Read(&is);
      if (!status.ok()) LOG(FATAL) << status.message();
    } else {
      if constexpr (!is_specialization_v<DataContainerType, std::vector>)
        LOG(WARNING) << "Unknown data container type";
      matrix.data_.resize(size);
      ReadData(&is, matrix.data_.data(), size);
    }
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
    auto file_or = OpenReadFile(path);
    if (!file_or.ok()) return file_or.status();
    auto file = *std::move(file_or);
    uint64_t offset = ReadPieRankMatrixFileHeader(file);
    this->status_.Update(index_.Mmap(path, &offset));
    this->status_.Update(pos_.Mmap(path, &offset));
    auto size = ReadUint64AtOffset(&file, &offset);
    if (size) {
      CHECK(!this->type_.IsPattern()) << "Only non-pattern matrices have data";
      if (!file) {
        LOG(ERROR) << "Error reading matrix data size";
        return absl::InternalError(absl::StrCat("Error reading file: ", path));
      }
      size *= sizeof(value_type);
      auto mmap = MmapReadOnlyFile(path, offset, size);
      offset += size;
      if (!mmap.ok()) return mmap.status();
      this->data_mmap_ = *std::move(mmap);
    } else
      CHECK(this->type_.IsPattern()) << "Only pattern matrices have no data";
    return this->status_;
  }

  void UnMmap() {
    index_.UnMmap();
    pos_.UnMmap();
    if (this->data_mmap_.size()) this->data_mmap_.unmap();
  }

  void push_back(const Entry &entry) {
    const auto & [pos, vars] = entry;
    auto family = this->Type().Family();
    DCHECK(family == MatrixType::kBoolFamily || vars.size() == this->Depths());
    PosType index_pos = pos[this->IndexDim()];
    if (!vars.empty() && MatrixMarketIo::AreVarsZero(vars)) return;
    while (index_pos_end_ != index_pos) {
      index_.push_back(nnz_);
      ++index_pos_end_;
      DCHECK_LE(index_pos_end_, index_pos);
      DCHECK_EQ(index_[index_pos_end_], nnz_);
    }
    pos_.push_back(this->IndexDim() ? pos[0] : pos[1]);
    for (const auto & var : vars) {
      if (family == MatrixType::kIntegerFamily) {
        if constexpr (std::is_integral_v<value_type>) {
          DCHECK_LE(std::get<int64_t>(var),
                    std::numeric_limits<value_type>::max());
          DCHECK_GE(std::get<int64_t>(var),
                    std::numeric_limits<value_type>::lowest());
        }
        this->data_.push_back(std::get<int64_t>(var));
      } else if (family == MatrixType::kRealFamily) {
        if constexpr (std::is_floating_point_v<value_type>) {
          DCHECK_LE(std::get<double>(var),
                    std::numeric_limits<value_type>::max());
          DCHECK_GE(std::get<double>(var),
                    std::numeric_limits<value_type>::lowest());
        }
        this->data_.push_back(std::get<double>(var));
      } else if (family == MatrixType::kComplexFamily) {
        if constexpr (std::is_same_v<value_type, std::complex<double>>) {
          for (const auto &var : vars)
            this->data_.push_back(std::get<std::complex<double>>(var));
        } else if constexpr (std::is_same_v<value_type, std::complex<float>>) {
          for (const auto &var : vars) {
            this->data_.emplace_back(
                std::get<std::complex<double>>(var).real(),
                std::get<std::complex<double>>(var).imag());
          }
        } else
          CHECK(false) << "Complex matrix must have floating point data type";
      }
    }
    nnz_++;
  }

  absl::Status ReadMatrixMarketFile(const std::string &path) {
    DCHECK(this->data_mmap_.empty());
    MatrixMarketIo mat(path);
    if (!mat.ok()) {
      this->status_ =
          absl::InternalError(absl::StrCat("Fail to open file: ", path));
      return this->status_;
    }
    this->type_ = mat.Type();
    this->shape_ = mat.Shape();
    this->order_ = mat.Order();
    DCHECK_GT(this->Depths(), 0);
    while (mat.HasNext()) push_back(mat.Next());
    index_.push_back(nnz_);
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
    if (index.size() == 0)
      return {std::make_tuple(0, 0, 0)};
    return SplitPosIntoRanges(index.size() - 1, num_ranges);
  }

  // Returned PosRanges.size() may be less than max_ranges.
  static PosRanges SplitIndexDimByNnz(const FlexIdxType &index, IdxType nnz,
                                      uint32_t max_ranges) {
    DCHECK_GT(max_ranges, 0);
    if (index.size() == 0) {
      DCHECK_EQ(nnz, 0);
      return {std::make_tuple(0, 0, 0)};
    }
    PosType num_index_pos = index.size() - 1;
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

  absl::StatusOr<SparseMatrix<PosType, IdxType, DataContainerType>>
  ChangeIndexDim(std::shared_ptr<ThreadPool> pool = nullptr,
                 uint64_t max_nnz_per_thread = 8000000) const {
    auto res = CopyOnlyDimInfo(/*change_index_dim=*/true);

    auto nnz = CountNonIndexDimNnz();
    auto idx = ReverseIndex(nnz);
    auto ranges =
        SplitIndexDimByNnz(idx, nnz_, MaxRanges(max_nnz_per_thread, pool));
    // std::cout << PosRangesDebugString(ranges);

    auto offsets = RangeNnzOffsets(ranges);
    nnz.Reset();

    if (ranges.size() == 1) {
      ChangeIndexInRange(ranges, offsets, 0, idx, &nnz, &res.pos_, &res.data_);
    } else {
      std::vector<FlexPosType> poses(ranges.size());
      std::vector<DataContainerType> datas(ranges.size());
      DCHECK(pool);
      pool->ParallelFor(
          ranges.size(), /*items_per_thread=*/1,
          [&, this](uint64_t first, uint64_t last) {
            for (auto r = first; r < last; ++r) {
              ChangeIndexInRange(ranges, offsets, r, idx, &nnz, &poses[r],
                                 &datas[r]);
            }
          });
      res.pos_.SetItemSize(poses.front().ItemSize());
      for (auto &pos : poses) {
        res.pos_.Append(pos);
        pos.clear();
      }
      for (auto &data : datas) {
        if constexpr (is_specialization_v<DataContainerType, FlexArray>)
          res.data_.Append(data);
        else
          res.data_.insert(res.data_.end(), data.begin(), data.end());
        data.clear();
      }
    }
    res.index_ = std::move(idx);
    return std::move(res);
  }

  // Returns a memory-mapped SparseMatrix with its index dim changed.
  absl::StatusOr<SparseMatrix<PosType, IdxType, DataContainerType>>
  ChangeIndexDim(
      const std::string &path,
      uint64_t max_nnz_per_range = 64000000) const {
    auto mat = CopyOnlyDimInfo(/*change_index_dim=*/true);

    auto nnz = CountNonIndexDimNnz();
    auto idx = ReverseIndex(nnz);
    auto ranges = SplitIndexDimByNnz(idx, nnz_, MaxRanges(max_nnz_per_range));
    // std::cout << PosRangesDebugString(ranges);

    auto offsets = RangeNnzOffsets(ranges);
    // <num_items, pos_item_size, pos_file, data_file> for each range
    std::vector<std::tuple<uint64_t, uint32_t, std::string, std::string>>
        tmp_file_infos;
    nnz.Reset();
    auto pos_min = std::numeric_limits<PosType>::max();
    auto pos_max = std::numeric_limits<PosType>::min();
    bool has_data = !this->type_.IsPattern();
    for (uint32_t r = 0; r < ranges.size(); ++r) {
      FlexPosType pos;
      DataContainerType data;
      ChangeIndexInRange(ranges, offsets, r, idx, &nnz, &pos, &data);
      pos_min = std::min(pos_min, pos.MinValue());
      pos_max = std::max(pos_max, pos.MaxValue());
      auto[fpos, pos_file] = OpenTmpFile(path);
      CHECK(fpos);
      if (!pos.WriteValues(fpos, pos.ItemSize(), /*shift_by_min_val=*/false))
        return absl::InternalError("Error write file: " + pos_file);
      fclose(fpos);

      std::string data_file;
      if (has_data) {
        FILE *fval;
        std::tie(fval, data_file) = OpenTmpFile(path);
        CHECK(fval);
        ConvertAndWriteUint64(fval, data.size());
        if constexpr (is_specialization_v<DataContainerType, FlexArray>) {
          auto status = data.Write(fval);
          if (!status.ok()) return status;
        }
        else if (!WriteData(fval, data.data(), data.size()))
          return absl::InternalError("Error write file: " + data_file);
        fclose(fval);
      } else
        DCHECK(data.empty());

      tmp_file_infos.push_back(
          std::make_tuple(pos.size(), pos.ItemSize(), pos_file, data_file));
    }
    DCHECK_GE(pos_min, 0);
    DCHECK_LT(pos_max, mat.NonIndexDimSize());
    DCHECK_EQ(tmp_file_infos.size(), ranges.size());

    auto file_or = OpenWriteFile(path);
    if (!file_or.ok()) return file_or.status();
    auto ofs = *std::move(file_or);
    mat.index_ = std::move(idx);
    mat.WriteAllButPosAndData(&ofs);
    mat.pos_.SetMinMaxValues(pos_min, pos_max);
    auto[pos_item_size, pos_shift_by_min_val] = FlexPosType::MinEncode(pos_max,
                                                                       pos_min);
    PosType pos_value_shift = pos_shift_by_min_val ? -pos_min : 0;
    mat.pos_.WriteAllButValues(&ofs, pos_item_size, pos_shift_by_min_val);
    if (!WriteUint64(&ofs, pos_item_size * nnz_))
      return absl::InternalError("Error write file: " + path);
    uint64_t total_items = 0;
    for (const auto &tmp_file_info : tmp_file_infos) {
      auto[num_items, pos_item_size, pos_file, data_file] = tmp_file_info;
      auto pos_file_or = OpenReadFile(pos_file);
      if (!pos_file_or.ok()) return pos_file_or.status();
      FlexPosType pos(pos_item_size, num_items);
      if (!pos.ReadValues(&*pos_file_or, pos_item_size, pos_value_shift))
        return absl::InternalError("Error read file: " + pos_file);
      CHECK_EQ(pos.size(), num_items);
      total_items += num_items;
      if (!WriteData(&ofs, pos.Data(), pos_item_size * num_items))
        return absl::InternalError("Error write file: " + path);
      std::remove(pos_file.c_str());
    }
    CHECK_EQ(total_items, nnz_);

    auto data_size = has_data ? nnz_ : 0;  // 0 for pattern matrices
    // Write size of mat.data_
    if (!WriteUint64(&ofs, data_size))
      return absl::InternalError("Error write file: " + path);
    if (has_data) {
      total_items = 0;
      for (const auto &tmp_file_info : tmp_file_infos) {
        auto[num_items, pos_item_size, pos_file, data_file] = tmp_file_info;
        auto data_file_or = OpenReadFile(data_file);
        if (!data_file_or.ok()) return data_file_or.status();
        auto size = ReadUint64(&*data_file_or);
        CHECK_EQ(size, num_items);
        total_items += num_items;
        DataContainerType data;
        data.resize(size);
        if constexpr (is_specialization_v<DataContainerType, FlexArray>) {
          auto status = data.Read(&*data_file_or);
          if (!status.ok()) return status;
          status.Update(data.Write(&ofs));
          if (!status.ok()) return status;
        }
        else {
          if (!ReadData(&*data_file_or, data.data(), size))
            return absl::InternalError("Error read file: " + data_file);
          if (!WriteData(&ofs, data.data(), size))
            return absl::InternalError("Error write file: " + path);
        }
        std::remove(data_file.c_str());
      }
      CHECK_EQ(total_items, nnz_);
    }
    ofs.close();

    auto out = new SparseMatrix<PosType, IdxType, DataContainerType>(
        path,/*mmap=*/true);
    return std::move(*out);
  }

  std::string DebugString(uint64_t max_items = 0, uint32_t indent = 0) const {
    std::string res;
    std::string tab(indent, ' ');
    if (!this->status_.ok()) {
      absl::StrAppend(&res, tab, this->status_.ToString(), "\n");
      return res;
    }
    absl::StrAppend(&res, tab, "matrix_type: \"", this->type_.ToString(),
                    "\"\n");
    absl::StrAppend(&res, tab, "data_type: \"", StaticDataType().ToString(),
                    "\"\n");
    absl::StrAppend(&res, tab, "shape: ", VectorToString(this->Shape(), -1),
                    "\n");
    absl::StrAppend(&res, tab, "order: ", VectorToString(this->Order(), -1),
                    "\n");
    absl::StrAppend(&res, tab, "nnz: ", nnz_, "\n");
    indent += 2;
    absl::StrAppend(&res, tab, "index {\n",
                    index_.DebugString(max_items, indent));
    absl::StrAppend(&res, tab, "}\n");
    absl::StrAppend(&res, tab, "pos {\n", pos_.DebugString(max_items, indent));
    absl::StrAppend(&res, tab, "}\n");
    if constexpr (is_specialization_v<DataContainerType, FlexArray>) {
      CHECK(this->data_mmap_.empty());
      absl::StrAppend(&res, tab, "data {\n",
                      this->data_.DebugString(max_items, indent), "}\n");
    } else {
      if constexpr (!is_specialization_v<DataContainerType, std::vector>)
        LOG(WARNING) << "Unknown data container type";
      absl::StrAppend(&res, tab, "data: ",
                      this->data_mmap_.empty()
                      ? VectorToString(this->data_, max_items)
                      : VectorToString<decltype(this->data_mmap_), value_type>(
                          this->data_mmap_, max_items));
    }
    absl::StrAppend(&res, "\n");

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
  value_type At(IdxType idx, uint32_t depth = 0) const {
    idx = idx * this->Depths() + depth;
    if constexpr (is_specialization_v<DataContainerType, FlexArray>) {
      DCHECK(this->data_mmap_.empty());
      return this->data_[idx];
    } else {
      return this->data_mmap_.empty()
             ? reinterpret_cast<const value_type*>(this->data_.data())[idx]
             : reinterpret_cast<const value_type*>(this->data_mmap_.data())[idx];
    }
  }

  virtual void InitRanges(const PosRanges &ranges, uint32_t range_id) {}

  virtual void UpdateRanges(const PosRanges &ranges, uint32_t range_id) {}

  virtual void SyncRanges(const PosRanges &ranges, uint32_t range_id) {}

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
                           &SparseMatrix<PosType, IdxType>::SyncRanges);
  }

  bool ProcessRanges(uint32_t max_ranges = 1,
                     std::shared_ptr<ThreadPool> pool = nullptr) {
    DCHECK_GT(max_ranges, 0);
    if (!this->status_.ok()) return false;
    max_ranges = std::min(max_ranges, pool ? pool->Size() : 1);
    const auto pos_balanced_ranges =
        this->SplitPosIntoRanges(this->MaxDimSize(), max_ranges);
    const auto nnz_balanced_ranges = this->SplitIndexDimByNnz(max_ranges);

    auto[init, update, sync] = RangeFuncs();
    RunRangeFunc(init, pos_balanced_ranges, pool);
    while (!Stop()) {
      RunRangeFunc(update, nnz_balanced_ranges, pool);
      RunRangeFunc(sync, pos_balanced_ranges, pool);
    }
    return true;
  }

  bool ProcessRanges(const TriplePosRanges &ranges,
                     std::shared_ptr <ThreadPool> pool = nullptr) {
    if (!this->status_.ok()) return false;

    auto[init, update, sync] = RangeFuncs();
    RunRangeFunc(init, ranges[0], pool);
    while (!Stop()) {
      RunRangeFunc(update, ranges[1], pool);
      RunRangeFunc(sync, ranges[2], pool);
    }
    return true;
  }

  SparseMatrix<PosType, IdxType, DataContainerType>
  CopyOnlyDimInfo(bool change_index_dim = false) const {
    SparseMatrix<PosType, IdxType, DataContainerType> res;
    res.type_ = this->type_;
    res.shape_ = this->shape_;
    res.order_ = this->order_;
    if (change_index_dim)
      std::swap(res.order_[0], res.order_[1]);
    res.nnz_ = nnz_;

    return res;
  }

  FlexPosType CountNonIndexDimNnz() const {
    PosType non_index_dim_size = this->NonIndexDimSize();
    FlexPosType pos(pos_.ItemSize(), non_index_dim_size);
    auto index_pos_end = this->IndexPosEnd();
    for (PosType p = 0; p < index_pos_end; ++p) {
      for (IdxType i = index_[p]; i < index_[p + 1]; ++i)
        pos.IncItem(pos_[i]);
    }
    return pos;
  }

  FlexIdxType ReverseIndex(const FlexPosType &nbr) const {
    auto [idx_item_size, idx_shift_by_min_val] = index_.MinEncode();
    CHECK(!idx_shift_by_min_val) << "Not yet supported";
    FlexIdxType idx(idx_item_size);
    IdxType nnz = 0;
    PosType new_index_dim_size = this->NonIndexDimSize();
    for (PosType p = 0; p < new_index_dim_size && nnz < nnz_; ++p) {
      idx.push_back(nnz);
      nnz += nbr[p];
    }
    DCHECK_EQ(nnz, nnz_);
    if (idx[idx.size() - 1] != nnz)
      idx.push_back(nnz);

    return idx;
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

  void ChangeIndexInRange(const PosRanges &ranges,
                          const std::vector <IdxType> &offsets,
                          uint32_t range_id,
                          const FlexIdxType &idx,
                          FlexPosType *nbr,
                          FlexPosType *pos,
                          DataContainerType *data) const {
    auto [min_pos, max_pos, range_size] = ranges[range_id];
    auto offset = offsets[range_id];
    auto index_pos_end = IndexPosEnd();
    uint32_t pos_item_size = MinEncodeSize(index_pos_end);
    *pos = std::move(FlexPosType(pos_item_size, range_size));

    DCHECK(data->empty());
    bool has_data = !this->type_.IsPattern();
    data->resize(has_data ? range_size : 0);

    for (PosType p = 0; p < index_pos_end; ++p) {
      for (IdxType i = index_[p]; i < index_[p + 1]; ++i) {
        auto pos_i = pos_[i];
        if (pos_i >= min_pos && pos_i < max_pos) {
          auto i_new = idx[pos_i] + (*nbr)[pos_i] - offset;
          pos->SetItem(i_new, p);
          if (has_data) {
            if constexpr (is_specialization_v<DataContainerType, FlexArray>)
              data->SetItem(i_new, this->data_[i]);
            else
              (*data)[i_new] = this->data_[i];
          }
          nbr->IncItem(pos_i);
        }
      }
    }
  }

private:
  IdxType nnz_ = 0;
  PosType index_pos_end_ = static_cast<PosType>(-1);
  FlexIdxType index_;
  FlexPosType pos_;
};

template<typename PosType, typename IdxType>
class SparseMatrixVar {
public:
  enum Enum : uint32_t {
    kFlexInt64,
    kDouble,
    kComplexDouble
  };

  using SparseMatrixFlexInt64 =
      SparseMatrix<PosType, IdxType, FlexArray<int64_t>>;

  using SparseMatrixDouble =
      SparseMatrix<PosType, IdxType, std::vector<double>>;

  using SparseMatrixComplexDouble =
      SparseMatrix<PosType, IdxType, std::vector<std::complex<double>>>;

  SparseMatrixVar() = default;

  SparseMatrixVar(const SparseMatrixVar &) = delete;

  SparseMatrixVar &operator=(const SparseMatrixVar &) = delete;

  SparseMatrixVar(SparseMatrixVar &&) = default;

  SparseMatrixVar &operator=(SparseMatrixVar &&) = default;

  SparseMatrixVar(SparseMatrixFlexInt64 &&other) {
    type_ = kFlexInt64;
    var_.template emplace<kFlexInt64>(
        std::forward<SparseMatrixFlexInt64>(other));
  }

  SparseMatrixVar(SparseMatrixDouble &&other) {
    type_ = kDouble;
    var_.template emplace<kDouble>(std::forward<SparseMatrixDouble>(other));
  }

  SparseMatrixVar(SparseMatrixComplexDouble &&other) {
    type_ = kComplexDouble;
    var_.template emplace<kComplexDouble>(
        std::forward<SparseMatrixComplexDouble>(other));
  }

  SparseMatrixVar(const std::string &prm_path, bool mmap = false) {
    status_ = mmap ? this->MmapPieRankMatrixFile(prm_path)
                   : this->ReadPieRankMatrixFile(prm_path);
  }

  bool ok() const { return status_.ok(); }

  absl::Status status() const { return status_; }

  void SetType(Enum type) {
    if (type == kFlexInt64)
      var_.template emplace<kFlexInt64>();
    else if (type == kDouble)
      var_.template emplace<kDouble>();
    else
      var_.template emplace<kComplexDouble>();
    type_ = type;
  }

  void SetType(MatrixType type, bool flex = true) {
    if (!type.IsComplex()) {
      if (flex && type.IsInteger()) SetType(kFlexInt64);
      else SetType(kDouble);
    }
    else
      SetType(kComplexDouble);
  }

  absl::Status SetTypeFromMatrixMarketFile(const std::string &mtx_path) {
    auto matrix_type = MatrixMarketFileMatrixType(mtx_path);
    if (matrix_type == MatrixType::kUnknown)
      return absl::InternalError("Bad or missing matrix file: " + mtx_path);
    SetType(matrix_type);
    return absl::OkStatus();
  }

  absl::Status SetTypeFromPieRankMatrixFile(const std::string &prm_path) {
    auto types = PieRankFileTypes(prm_path);
    if (!types.ok()) return types.status();
    auto[matrix_type, data_type] = *std::move(types);
    SetType(matrix_type, data_type.IsFlex());
    return absl::OkStatus();
  }

  absl::Status ReadMatrixMarketFile(const std::string &mtx_path) {
    auto status = SetTypeFromMatrixMarketFile(mtx_path);
    if (!status.ok()) return status;

    auto idx = var_.index();
    if (idx == kFlexInt64)
      return std::get<kFlexInt64>(var_).ReadMatrixMarketFile(mtx_path);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).ReadMatrixMarketFile(mtx_path);
    else
      return std::get<kComplexDouble>(var_).ReadMatrixMarketFile(mtx_path);
  }

  absl::Status ReadPieRankMatrixFile(const std::string &prm_path) {
    auto status = SetTypeFromPieRankMatrixFile(prm_path);
    if (!status.ok()) return status;

    auto idx = var_.index();
    if (idx == kFlexInt64)
      return std::get<kFlexInt64>(var_).ReadPieRankMatrixFile(prm_path);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).ReadPieRankMatrixFile(prm_path);
    else
      return std::get<kComplexDouble>(var_).ReadPieRankMatrixFile(prm_path);
  }

  absl::Status WritePieRankMatrixFile(const std::string &prm_path) const {
    auto idx = var_.index();
    if (idx == kFlexInt64)
      return std::get<kFlexInt64>(var_).WritePieRankMatrixFile(prm_path);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).WritePieRankMatrixFile(prm_path);
    else
      return std::get<kComplexDouble>(var_).WritePieRankMatrixFile(prm_path);
  }

  absl::StatusOr<SparseMatrixVar<PosType, IdxType>>
  ChangeIndexDim(std::shared_ptr<ThreadPool> pool = nullptr,
                 uint64_t max_nnz_per_thread = 8000000) const {
    auto idx = var_.index();
    if (idx == kFlexInt64)
      return std::get<kFlexInt64>(var_).ChangeIndexDim(pool, max_nnz_per_thread);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).ChangeIndexDim(pool, max_nnz_per_thread);
    else
      return std::get<kComplexDouble>(var_).ChangeIndexDim(pool,
                                                           max_nnz_per_thread);
  }

  absl::StatusOr<SparseMatrixVar<PosType, IdxType>>
  ChangeIndexDim(const std::string &path,
                 uint64_t max_nnz_per_range = 64000000) const {
    auto idx = var_.index();
    if (idx == kFlexInt64)
      return std::get<kFlexInt64>(var_).ChangeIndexDim(path, max_nnz_per_range);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).ChangeIndexDim(path, max_nnz_per_range);
    else
      return std::get<kComplexDouble>(var_).ChangeIndexDim(path,
                                                           max_nnz_per_range);
  }

  absl::Status MmapPieRankMatrixFile(const std::string &prm_path) {
    auto status = SetTypeFromPieRankMatrixFile(prm_path);
    if (!status.ok()) return status;

    auto idx = var_.index();
    if (idx == kFlexInt64)
      return std::get<kFlexInt64>(var_).MmapPieRankMatrixFile(prm_path);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).MmapPieRankMatrixFile(prm_path);
    else
      return std::get<kComplexDouble>(var_).MmapPieRankMatrixFile(prm_path);
  }

  std::string DebugString(uint64_t max_items = 0, uint32_t indent = 0) const {
    auto idx = var_.index();
    if (idx == kFlexInt64)
      return std::get<kFlexInt64>(var_).DebugString(max_items, indent);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).DebugString(max_items, indent);
    else
      return std::get<kComplexDouble>(var_).DebugString(max_items, indent);
  }

protected:
  absl::Status status_;

private:
  Enum type_ = kFlexInt64;
  std::variant<
      SparseMatrix<PosType, IdxType, FlexArray<int64_t>>,
      SparseMatrix<PosType, IdxType, std::vector<double>>,
      SparseMatrix<PosType, IdxType, std::vector<std::complex<double>>>> var_;
};

}  // namespace pierank

#endif //PIERANK_SPARSE_MATRIX_H_
