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
#include "absl/types/span.h"

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

// Tuple: <MatrixType, DataType, shape, order, index_dim_order, nnz>
inline absl::StatusOr<std::tuple<MatrixType, DataType, std::vector<uint64_t>,
                                 std::vector<uint32_t>, uint32_t, uint64_t>>
PieRankFileInfo(const std::string &prm_path) {
  auto file_or = OpenReadFile(prm_path);
  if (!file_or.ok()) return file_or.status();
  auto file = *std::move(file_or);
  if (!EatString(&file, kPieRankMatrixFileMagicNumbers))
    return absl::InternalError("Bad file format");
  MatrixType matrix_type;
  auto status = matrix_type.Read(&file);
  if (!status.ok()) return status;
  DataType data_type;
  status.Update(data_type.Read(&file));
  if (!status.ok()) return status;
  auto shape = ReadUint64Vector(&file);
  auto order = ReadUint32Vector(&file);
  auto index_dim_order = ReadUint32(&file);
  auto nnz = ReadUint64(&file);

  return
   std::make_tuple(matrix_type, data_type, shape, order, index_dim_order, nnz);
}

inline absl::StatusOr<std::pair<MatrixType, DataType>>
PieRankFileTypes(const std::string &prm_path) {
  auto info = PieRankFileInfo(prm_path);
  if (!info.ok()) {
    LOG(ERROR) << info.status().message();
    return info.status();
  }
  auto [matrix_type, data_type, shape, order, index_dim_order, nnz] =
    *std::move(info);
  return std::make_pair(matrix_type, data_type);
}

inline uint32_t PieRankFileDims(const std::string &prm_path) {
  auto info = PieRankFileInfo(prm_path);
  if (!info.ok()) {
    LOG(ERROR) << info.status().message();
    return 0;
  }
  auto [matrix_type, data_type, shape, order, index_dim_order, nnz] =
    *std::move(info);
  return shape.size();
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

  using PosSpan = absl::Span<const PosType>;

  using PosSpanMutable = absl::Span<PosType>;

  // PosRanges for InitRanges, UpdateRanges, and SyncRanges
  using TriplePosRanges = std::array<PosRanges, 3>;

  using FlexIdxType = FlexArray<IdxType>;

  using FlexIdxIterator = typename FlexIdxType::Iterator;

  using FlexPosType = FlexArray<PosType>;

  using FlexPosIterator = typename FlexPosType::Iterator;

  using value_type = typename DataContainerType::value_type;

  using DenseDataType = typename std::conditional<
    is_specialization<DataContainerType, SparseMatrix>{},
    std::vector<value_type>,
    DataContainerType>::type;

  using DenseType = typename std::conditional<
    is_specialization<DataContainerType, SparseMatrix>{},
    Matrix<PosType, IdxType, DenseDataType>,
    Matrix<PosType, IdxType, DataContainerType>>::type;

  using Var = MatrixMarketIo::Var;

  using DataFamily = MatrixType::DataFamily;

  using RangeFunc = void (SparseMatrix<PosType, IdxType>::*)(
      const PosRanges &ranges, uint32_t range_id);

  SparseMatrix() = default;

  SparseMatrix(const std::string &prm_file_path, bool mmap = false) {
    this->status_ = mmap
              ? this->MmapPieRankMatrixFile(prm_file_path)
              : this->ReadPieRankMatrixFile(prm_file_path);
  }

  SparseMatrix(const DenseType &dense) {
    auto order = dense.Order();
    if (dense.SplitDepths()) {
      // SparseMatrix can't split data dims, which thus must be last in order.
      auto it = std::find(order.begin(), order.end(), dense.NonDepthDims());
      CHECK(it != order.end());
      std::rotate(it, it + 1, order.end());
    }
    Config(dense.Type(), dense.Shape(), order);

    const uint32_t depths = dense.Depths();
    const uint64_t elem_stride = dense.ElemStride();
    for (IdxType i = 0; i < dense.Elems(); ++i) {
      auto && [pos, depth] = dense.IdxToPosAndDepth(i * elem_stride);
      DCHECK_EQ(depth, 0);
      bool all_zeros = true;
      for (uint32_t d = 0; d < depths && all_zeros; ++d) {
        if (dense(pos, d) != 0) all_zeros = false;
      }
      if (!all_zeros) {
        std::vector<MatrixMarketIo::Var> vars;
        for (uint32_t d = 0; d < depths; ++d)
          vars.push_back(dense(pos, d));
        push_back(pos, vars);
      }
    }
    ++index_pos_end_;
    MarkEndofMatrix();
  }

  SparseMatrix(const SparseMatrix &) = delete;

  SparseMatrix &operator=(const SparseMatrix &) = delete;

  SparseMatrix(SparseMatrix &&) = default;

  SparseMatrix &operator=(SparseMatrix &&) = default;

  void Config(MatrixType type,
              const std::vector<uint64_t> &shape,
              const std::vector<uint32_t> &order,
              uint32_t index_dim_order = 0) override {
    Matrix<PosType, IdxType, DataContainerType>::Config(type, shape, order,
                                                        index_dim_order);
    if constexpr (is_specialization_v<DataContainerType, SparseMatrix>)
      this->data_.Config(type, shape, order, index_dim_order + 1);
  }

  inline static constexpr DataType StaticDataType() {
    if constexpr (is_specialization_v<DataContainerType, FlexArray>)
      return DataType::kFlex;
    return DataType::FromValueType<value_type>();
  }

  // Returns true if all non-zeros are processed; otherwise false.
  bool ForAllNonZeros(std::function<bool(PosSpan, IdxType)> func,
                      PosSpanMutable *pos = nullptr,
                      PosSpanMutable *zpos = nullptr) const {
    std::unique_ptr<PosType[]> posa = nullptr;
    PosSpanMutable poss;
    if (!pos) {
      posa = std::make_unique<PosType[]>(this->NonDepthDims());
      poss = PosSpanMutable(posa.get(), this->NonDepthDims());
      pos = &poss;
    }

    std::unique_ptr<PosType[]> zposa = nullptr;
    PosSpanMutable zposs;
    if (!zpos && !this->IsLeaf()) {
      zposa = std::make_unique<PosType[]>(this->NonDepthDims());
      zposs = PosSpanMutable(zposa.get(), this->NonDepthDims());
      zpos = &zposs;
    }

    uint32_t index_dim = this->IndexDim();
    uint32_t non_index_dim = this->NonIndexDim();
    PosSpanMutable mpos = this->index_dim_order_ == 0 ? *pos : *zpos;
    for (PosType p = 0; p < index_.size() - 1; ++p) {
      mpos[index_dim] = p;
      for (IdxType i = this->Index(p); i < this->Index(p + 1); ++i) {
        (*pos)[non_index_dim] = this->Pos(i);
        if constexpr (is_specialization_v<DataContainerType, SparseMatrix>) {
          DCHECK_LE(i, std::numeric_limits<PosType>::max());
          (*zpos)[non_index_dim] = static_cast<PosType>(i);
          if (!this->data_.ForNonZerosAtIndexPos(func, *pos, zpos))
            return false;
        } else {
          DCHECK(this->IsLeaf());
          if (!func(mpos, i)) return false;
        }
      }
    }
    return true;
  }

  // Returns true if all non-zeros for an index pos are processed; else false.
  bool ForNonZerosAtIndexPos(std::function<bool(PosSpan, IdxType)> func,
                             PosSpanMutable pos,
                             PosSpanMutable *zpos = nullptr) const {
    std::unique_ptr<PosType[]> zposa = nullptr;
    PosSpanMutable zposs;
    if (!zpos && !this->IsLeaf()) {
      zposa = std::make_unique<PosType[]>(this->NonDepthDims());
      zposs = PosSpanMutable(zposa.get(), this->NonDepthDims());
      zpos = &zposs;
    }

    uint32_t idx_dim = this->IndexDim();
    uint32_t non_idx_dim = this->NonIndexDim();
    PosType idx_pos = this->index_dim_order_ == 0 ? pos[idx_dim]
                                                  : (*zpos)[idx_dim];
    for (IdxType i = this->Index(idx_pos); i < this->Index(idx_pos + 1); ++i) {
      pos[non_idx_dim] = this->Pos(i);
      if constexpr (is_specialization_v<DataContainerType, SparseMatrix>) {
        DCHECK_LE(i, std::numeric_limits<PosType>::max());
        (*zpos)[non_idx_dim] = static_cast<PosType>(i);
        if (!this->data_.ForNonZerosAtIndexPos(func, pos, zpos)) return false;
      } else {
        DCHECK(this->IsLeaf());
        if (!func(pos, i)) return false;
      }
    }
    return true;
  }

  std::string NonZeroPosDebugString() const {
    std::string res;
    this->ForAllNonZeros([this, &res](PosSpan pos, IdxType unused) {
      (void)unused;
      for (const auto p : pos)
        absl::StrAppend(&res, " ", p);
      absl::StrAppend(&res, "\n");
      return true;
    });
    return res;
  }

  DenseType DenseMatrix(bool split_depths = false,
                        bool omit_idx_dim = false) const {
    std::vector<uint64_t> shape = this->Shape();
    std::vector<uint32_t> order = this->Order();
    if (omit_idx_dim) {
      std::copy(shape.begin() + order[0] + 1, shape.end(),
                shape.begin() + order[0]);  // Remove the index dim from shape
      shape.pop_back();
      for (std::size_t i = 1; i < order.size(); ++i)
        if (order[i] > order[0]) --order[i];  // Make sure min(order[i]) == 0
      order.erase(order.begin()); // Remove the index dim from order.
    }
    if (split_depths)
      std::rotate(order.rbegin(), order.rbegin() + 1, order.rend());
    DenseType res(this->type_, shape, order);
    res.InitData();
    return res;
  }

  void GetDense(DenseType *dense, bool split_depths = false) const {
    DCHECK_EQ(dense->Shape().size(), this->Shape().size());
    DCHECK_EQ(dense->Order().size(), this->Order().size());
    IdxType data_idx = 0;
    const bool has_data = !this->type_.IsPattern();
    this->ForAllNonZeros([=, &data_idx](PosSpan pos, IdxType unused) {
      (void)unused;
      for (uint32_t d = 0; d < this->Depths(); ++d)
        dense->Set(has_data ? this->data_[data_idx++] : 1, pos, d);
      return true;
    });
    DCHECK(!has_data || data_idx >= nnz_ * this->Depths());
  }

  DenseType ToDense(bool split_depths = false) const {
    auto res = DenseMatrix(split_depths);
    GetDense(&res, split_depths);
    return res;
  }

  void GetDenseSlice(DenseType *dense, PosType idx_pos,
                     bool split_depths = false,
                     bool omit_idx_dim = true) const {
    if (omit_idx_dim) {
      DCHECK_EQ(dense->Shape().size() + 1, this->Shape().size());
      DCHECK_EQ(dense->Order().size() + 1, this->Order().size());
    } else {
      DCHECK_EQ(dense->Shape().size(), this->Shape().size());
      DCHECK_EQ(dense->Order().size(), this->Order().size());
    }
    const bool has_data = !this->type_.IsPattern();
    auto posa = std::make_unique<PosType[]>(this->NonDepthDims());
    auto pos = PosSpanMutable(posa.get(), this->NonDepthDims());
    pos[this->IndexDim()] = idx_pos;
    std::vector<bool> pos_mask(this->NonDepthDims());
    if (omit_idx_dim) pos_mask[this->IndexDim()] = true;
    this->ForNonZerosAtIndexPos([=, &pos_mask](PosSpan pos, IdxType idx) {
      std::size_t data_idx = idx * this->Depths();
      for (uint32_t d = 0; d < this->Depths(); ++d)
        dense->Set(has_data ? this->data_[data_idx++] : 1, pos, pos_mask, d);
      return true;
    }, pos);
  }

  DenseType DenseSlice(PosType idx_pos, bool split_depths = false,
                       bool omit_idx_dim = true) const {
    auto res = DenseMatrix(split_depths, omit_idx_dim);
    GetDenseSlice(&res, idx_pos, split_depths, omit_idx_dim);
    return res;
  }

  value_type operator[](std::size_t idx) const { return this->data_[idx]; }

  value_type operator()(PosSpan pos, uint32_t depth = 0,
                        PosSpanMutable *zpos = nullptr) const {
    std::unique_ptr<PosType[]> zposa = nullptr;
    PosSpanMutable zposs;
    if (!zpos && !this->IsLeaf()) {
      zposa = std::make_unique<PosType[]>(pos.size());
      zposs = PosSpanMutable(zposa.get(), pos.size());
      zpos = &zposs;
    }

    PosType index_pos = this->index_dim_order_ == 0 ? pos[this->IndexDim()]
                                                    : (*zpos)[this->IndexDim()];
    if (index_pos + 1 >= index_.size()) return 0;
    FlexPosIterator first = pos_(index_[index_pos]);
    FlexPosIterator last = pos_(index_[index_pos + 1]);
    PosType non_idx_pos = pos[this->NonIndexDim()];

    auto it = std::lower_bound(first, last, non_idx_pos);
    if (it != last && *it == non_idx_pos) {
      if constexpr (is_specialization_v<DataContainerType, SparseMatrix>) {
        (*zpos)[this->NonIndexDim()] = it - pos_();
        return this->data_(pos, depth, zpos);
      } else if (!this->type_.IsPattern())
        return At(it - pos_(), depth);
      else
        return 1;
    } else
      return 0;
  }

  value_type operator()(PosType row, PosType col, uint32_t depth = 0) const {
    DCHECK_EQ(this->shape_.size(), 3);
    return (*this)({row, col}, depth);
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

  friend bool
  operator!=(const SparseMatrix<PosType, IdxType, DataContainerType> &lhs,
             const SparseMatrix<PosType, IdxType, DataContainerType> &rhs) {
    return !(lhs == rhs);
  }

  void WriteAllButPosAndData(std::ostream *os) const {
    if (this->IsRoot()) {
      *os << kPieRankMatrixFileMagicNumbers;
      auto status = this->type_.Write(os);
      if (!status.ok()) *os << status.message();
      status = StaticDataType().Write(os);
      if (!status.ok()) *os << status.message();
    }
    WriteUint64Vector(os, this->shape_);
    WriteUint32Vector(os, this->order_);
    WriteUint32(os, this->index_dim_order_);
    ConvertAndWriteUint64(os, nnz_);
    *os << index_;
  }

  friend std::ostream &
  operator<<(std::ostream &os, const SparseMatrix &matrix) {
    matrix.WriteAllButPosAndData(&os);
    os << matrix.pos_;
    if constexpr (is_specialization_v<DataContainerType, FlexArray>) {
      auto status = matrix.data_.Write(&os);
      if (!status.ok()) LOG(FATAL) << status.message();
    } else if constexpr (is_specialization_v<DataContainerType, std::vector>) {
      WriteUint64(&os, matrix.data_.size());
      WriteData<std::ostream, value_type>(&os, matrix.data_.data(),
                                          matrix.data_.size());
    } else if constexpr (is_specialization_v<DataContainerType, SparseMatrix>) {
      os << matrix.data_;
    } else
      LOG(FATAL) << "Unsupported data container type";

    return os;
  }

  // Reads {rows, cols, nnz} from `is`
  uint64_t ReadPieRankMatrixFileHeader(std::istream &is) {
    uint64_t offset = 0;
    if (this->IsRoot()) {
      if (EatString(&is, kPieRankMatrixFileMagicNumbers, &offset)) {
        auto status = this->type_.Read(&is, &offset);
        if (!status.ok()) LOG(FATAL) << status.message();
        DataType data_type;
        status.Update(data_type.Read(&is, &offset));
        if (!status.ok()) LOG(FATAL) << status.message();
        CHECK_EQ(data_type, StaticDataType());
    } else
      this->status_.Update(absl::InternalError("Bad file format"));
    }
    this->shape_ = ReadUint64Vector(&is, &offset);
    this->order_ = ReadUint32Vector(&is, &offset);
    if (this->IsRoot())
      Config(this->Type(), this->Shape(), this->Order());
    this->index_dim_order_ = ReadUint32(&is, &offset);
    this->non_index_dim_order_ = this->index_dim_order_ + 1;
    nnz_ = ReadUint64AndConvert<IdxType>(&is, &offset);
    CHECK_LT(this->IndexDim(), this->NonDepthDims());
    if (!is)
      this->status_.Update(absl::InternalError("Error read PRM file header"));

    return offset;
  }

  friend std::istream &operator>>(std::istream &is, SparseMatrix &matrix) {
    matrix.ReadPieRankMatrixFileHeader(is);
    is >> matrix.index_;
    is >> matrix.pos_;
    if constexpr (is_specialization_v<DataContainerType, FlexArray>) {
      auto status = matrix.data_.Read(&is);
      if (!status.ok()) LOG(FATAL) << status.message();
    } else if constexpr (is_specialization_v<DataContainerType, std::vector>) {
      auto size = ReadUint64(&is);
      matrix.data_.resize(size);
      ReadData(&is, matrix.data_.data(), size);
    } else if constexpr (is_specialization_v<DataContainerType, SparseMatrix>) {
      is >> matrix.data_;
    } else
      LOG(FATAL) << "Unsupported data container type";

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
      CHECK(!this->type_.IsPattern() || !this->IsLeaf())
          << "Only non-pattern or non-leaf matrices have data";
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

  void push_back(PosSpan pos, const std::vector<Var> vars,
                 PosSpanMutable *zpos = nullptr) {
    static PosType last_non_index_pos = std::numeric_limits<PosType>::max();

    auto family = this->Type().Family();
    DCHECK(family == MatrixType::kBoolFamily || vars.size() == this->Depths());
    std::unique_ptr<PosType[]> zposa = nullptr;
    PosSpanMutable zposs;
    if (!zpos && !this->IsLeaf()) {
      zposa = std::make_unique<PosType[]>(pos.size());
      zposs = PosSpanMutable(zposa.get(), pos.size());
      zpos = &zposs;
    }
    if (!vars.empty() && MatrixMarketIo::AreVarsZero(vars)) return;
    PosType index_pos = this->index_dim_order_ == 0 ? pos[this->IndexDim()]
                                                    : (*zpos)[this->IndexDim()];
    while (index_pos_end_ != index_pos) {
      index_.push_back(nnz_);
      ++index_pos_end_;
      last_non_index_pos = std::numeric_limits<PosType>::max();
      DCHECK_LE(index_pos_end_, index_pos);
      DCHECK_EQ(index_[index_pos_end_], nnz_);
    }
    const auto non_index_dim = this->NonIndexDim();
    const auto non_index_pos = pos[non_index_dim];
    bool new_pos = last_non_index_pos == std::numeric_limits<PosType>::max();
    if (!new_pos) {
      CHECK_LE(last_non_index_pos, non_index_pos) << "Unsorted element";
      new_pos = last_non_index_pos != non_index_pos;
    }
    last_non_index_pos = non_index_pos;
    if (new_pos) pos_.push_back(non_index_pos);
    if constexpr (is_specialization_v<DataContainerType, SparseMatrix>) {
      DCHECK(!this->IsLeaf()) << "Unexpected leaf matrix";
      CHECK_LE(nnz_, std::numeric_limits<PosType>::max());
      (*zpos)[non_index_dim] = static_cast<PosType>(new_pos ? nnz_ : nnz_ - 1);
      this->data_.push_back(pos, vars, zpos);
      if (new_pos) ++nnz_;
      return;
    } else if (!this->IsLeaf()) {
      CHECK(false) << "Unexpected non-leaf matrix";
      return;
    }
    CHECK(new_pos) << "Duplicate element found in leaf matrix";
    for (const auto & var : vars) {
      if (family == MatrixType::kIntegerFamily) {
        if constexpr (std::is_integral_v<value_type>) {
          DCHECK_LE(std::get<int64_t>(var),
                    std::numeric_limits<value_type>::max());
          DCHECK_GE(std::get<int64_t>(var),
                    std::numeric_limits<value_type>::lowest());
        }
        if constexpr (!is_specialization_v<DataContainerType, SparseMatrix>)
          this->data_.push_back(std::get<int64_t>(var));
      } else if (family == MatrixType::kRealFamily) {
        if constexpr (std::is_floating_point_v<value_type>) {
          DCHECK_LE(std::get<double>(var),
                    std::numeric_limits<value_type>::max());
          DCHECK_GE(std::get<double>(var),
                    std::numeric_limits<value_type>::lowest());
        }
        if constexpr (!is_specialization_v<DataContainerType, SparseMatrix>)
          this->data_.push_back(std::get<double>(var));
      } else if (family == MatrixType::kComplexFamily) {
        if constexpr (std::is_same_v<value_type, std::complex<double>>) {
          if constexpr (!is_specialization_v<DataContainerType, SparseMatrix>) {
            for (const auto &var : vars)
              this->data_.push_back(std::get<std::complex<double>>(var));
          }
        } else if constexpr (std::is_same_v<value_type, std::complex<float>>) {
          if constexpr (!is_specialization_v<DataContainerType, SparseMatrix>) {
            for (const auto &var : vars)
              this->data_.emplace_back(
                  std::get<std::complex<double>>(var).real(),
                  std::get<std::complex<double>>(var).imag());
          }
        } else
          CHECK(false) << "Complex matrix must have floating point data type";
      }
    }
    ++nnz_;
  }

  void push_back(PosType row, PosType col, const std::vector<Var> vars) {
    DCHECK_EQ(this->shape_.size(), 3);
    push_back({row, col}, vars);
  }

  void MarkEndofMatrix() {
    index_.push_back(nnz_);
    if constexpr (is_specialization_v<DataContainerType, SparseMatrix>)
      this->data_.MarkEndofMatrix();
  }

  absl::Status ReadMatrixMarketFile(const std::string &path) {
    DCHECK(this->data_mmap_.empty());
    MatrixMarketIo mat(path);
    if (!mat.ok()) {
      this->status_ =
          absl::InternalError(absl::StrCat("Fail to read file: ", path));
      return this->status_;
    }
    Config(mat.Type(), mat.Shape(), mat.Order());
    DCHECK_GT(this->Depths(), 0);
    while (mat.HasNext()) {
      const auto & [pos, vars] = mat.Next();
      std::vector<PosType> posv;
      for (const auto & p : pos) {
        if (p > std::numeric_limits<PosType>::max())
          return absl::InternalError(absl::StrCat("Pos too big: ", p));
        posv.push_back(static_cast<PosType>(p));
      }
      push_back(posv, vars);
    }
    MarkEndofMatrix();
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
    absl::StrAppend(&res, tab, "index_dim_order: ", this->IndexDimOrder(), "\n");
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
      if constexpr (is_specialization_v<DataContainerType, std::vector>) {
        absl::StrAppend(&res, tab, "data: ",
                      this->data_mmap_.empty()
                      ? VectorToString(this->data_, max_items)
                      : VectorToString<decltype(this->data_mmap_), value_type>(
                          this->data_mmap_, max_items));
      }
      else if constexpr (is_specialization_v<DataContainerType, SparseMatrix>) {
        absl::StrAppend(&res, tab, "data {\n",
                        this->data_.DebugString(max_items, indent));
        absl::StrAppend(&res, tab, "}\n");
      } else
        LOG(WARNING) << "Unknown data container type";
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

}  // namespace pierank

#endif //PIERANK_SPARSE_MATRIX_H_
