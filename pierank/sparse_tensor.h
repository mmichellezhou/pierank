#ifndef PIERANK_SPARSE_TENSOR_H_
#define PIERANK_SPARSE_TENSOR_H_

#include <complex>
#include <variant>
#include <vector>

#include "pierank/sparse_matrix.h"

namespace pierank {

template<typename PosType, typename IdxType, typename DataContainerType>
using SparseTensor2d = SparseMatrix<PosType, IdxType, DataContainerType>;

template<typename PosType, typename IdxType, typename DataContainerType>
using SparseTensor3d =
  SparseMatrix<PosType, IdxType, SparseTensor2d<PosType, IdxType,
                                                DataContainerType>>;

template<typename PosType, typename IdxType, typename DataContainerType>
using SparseTensor4d =
  SparseMatrix<PosType, IdxType, SparseTensor3d<PosType, IdxType,
                                                DataContainerType>>;

template<typename Int64TensorType,
         typename FloatTensorType, typename DoubleTensorType,
         typename ComplexFloatTensorType, typename ComplexDoubleTensorType>
class SparseTensorVar {
public:
  enum Type : uint32_t {
    kUnknown,
    kInt64,
    kFloat,
    kDouble,
    kComplexFloat,
    kComplexDouble
  };

  using PosType = typename Int64TensorType::PosT;

  using IdxType = typename Int64TensorType::IdxT;

  using PosSpan = typename Int64TensorType::PosSpan;

  using PosSpanMutable = typename Int64TensorType::PosSpanMutable;

  using FlexPosType = FlexArray<PosType>;

  using FlexIdxType = FlexArray<IdxType>;

  using DenseVar = std::variant<
      std::monostate,
      Matrix<PosType, IdxType, std::vector<int64_t>>,
      Matrix<PosType, IdxType, std::vector<float>>,
      Matrix<PosType, IdxType, std::vector<double>>,
      Matrix<PosType, IdxType, std::vector<std::complex<float>>>,
      Matrix<PosType, IdxType, std::vector<std::complex<double>>>>;

  using SparseVar = std::variant<
      std::monostate,
      Int64TensorType,
      FloatTensorType,
      DoubleTensorType,
      ComplexFloatTensorType,
      ComplexDoubleTensorType>;

  using ValueVar = std::variant<
      std::monostate,
      typename Int64TensorType::value_type,
      typename FloatTensorType::value_type,
      typename DoubleTensorType::value_type,
      typename ComplexFloatTensorType::value_type,
      typename ComplexDoubleTensorType::value_type>;

  SparseTensorVar() = default;

  SparseTensorVar(const SparseTensorVar &) = delete;

  SparseTensorVar &operator=(const SparseTensorVar &) = delete;

  SparseTensorVar(SparseTensorVar &&) = default;

  SparseTensorVar &operator=(SparseTensorVar &&) = default;

  SparseTensorVar(Int64TensorType &&other) {
    var_.template emplace<kInt64>(
        std::forward<Int64TensorType>(other));
  }

  SparseTensorVar(FloatTensorType &&other) {
    var_.template emplace<kFloat>(std::forward<FloatTensorType>(other));
  }

  SparseTensorVar(DoubleTensorType &&other) {
    var_.template emplace<kDouble>(std::forward<DoubleTensorType>(other));
  }

  SparseTensorVar(ComplexFloatTensorType &&other) {
    var_.template emplace<kComplexFloat>(
        std::forward<ComplexFloatTensorType>(other));
  }

  SparseTensorVar(ComplexDoubleTensorType &&other) {
    var_.template emplace<kComplexDouble>(
        std::forward<ComplexDoubleTensorType>(other));
  }

  SparseTensorVar(const std::string &prm_path, bool mmap = false) {
    status_ = mmap ? this->MmapPieRankMatrixFile(prm_path)
                   : this->ReadPieRankMatrixFile(prm_path);
  }

  SparseTensorVar(const DenseVar &dense) {
    auto idx = dense.index();
    if (idx == kInt64)
      var_.template emplace<kInt64>(std::get<kInt64>(dense));
    else if (idx == kFloat)
      var_.template emplace<kFloat>(std::get<kFloat>(dense));
    else if (idx == kDouble)
      var_.template emplace<kDouble>(std::get<kDouble>(dense));
    else if (idx == kComplexFloat)
      var_.template emplace<kComplexFloat>(std::get<kComplexFloat>(dense));
    else {
      DCHECK_EQ(idx, kComplexDouble);
      var_.template emplace<kComplexDouble>(std::get<kComplexDouble>(dense));
    }
  }

  static uint32_t DenseNonDepthDims(const DenseVar &dense) {
    auto idx = dense.index();
    if (idx == kInt64)
      return std::get<kInt64>(dense).NonDepthDims();
    else if (idx == kFloat)
      return std::get<kFloat>(dense).NonDepthDims();
    else if (idx == kDouble)
      return std::get<kDouble>(dense).NonDepthDims();
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(dense).NonDepthDims();
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(dense).NonDepthDims();
    }
  }

  bool ok() const { return status_.ok(); }

  absl::Status status() const { return status_; }

  const auto& Var() const { return var_; }

  const auto& VarI64() const { return std::get<kInt64>(var_); }

  const auto& VarF32() const { return std::get<kFloat>(var_); }

  const auto& VarF64() const { return std::get<kDouble>(var_); }

  const auto& VarC32() const { return std::get<kComplexFloat>(var_); }

  const auto& VarC64() const { return std::get<kComplexDouble>(var_); }

  void SetType(Type type) {
    if (type == kInt64)
      var_.template emplace<kInt64>();
    else if (type == kFloat)
      var_.template emplace<kFloat>();
    else if (type == kDouble)
      var_.template emplace<kDouble>();
    else if (type == kComplexFloat)
      var_.template emplace<kComplexFloat>();
    else {
      DCHECK_EQ(type, kComplexDouble);
      var_.template emplace<kComplexDouble>();
    }
  }

  void InferType(MatrixType type, const std::string &type_hint) {
    if (!type.IsComplex()) {
      if (type.IsInteger()) SetType(kInt64);
      else if (type_hint == "auto" || type_hint == "f64") SetType(kDouble);
      else {
        DCHECK_EQ(type_hint, "f32");
        SetType(kFloat);
      }
    }
    else if (type_hint == "auto" || type_hint == "f64") SetType(kComplexDouble);
    else {
      DCHECK_EQ(type_hint, "f32");
      SetType(kComplexFloat);
    }
  }

  void InferType(DataType type) {
    if (type.IsInteger()) SetType(kInt64);
    else if (type == DataType::kFloat) SetType(kFloat);
    else if (type == DataType::kDouble) SetType(kDouble);
    else if (type == DataType::kComplexFloat) SetType(kComplexFloat);
    else {
      DCHECK_EQ(type, DataType::kComplexDouble);
      SetType(kComplexDouble);
    }
  }

  absl::Status InferTypeFromMatrixMarketFile(
      const std::string &mtx_path,
      const std::string &type_hint = "auto") {
    auto matrix_type = MatrixMarketFileMatrixType(mtx_path);
    if (matrix_type == MatrixType::kUnknown)
      return absl::InternalError("Bad or missing matrix file: " + mtx_path);
    InferType(matrix_type, type_hint);
    return absl::OkStatus();
  }

  absl::Status InferTypeFromPieRankMatrixFile(const std::string &prm_path) {
    auto types = PieRankFileTypes(prm_path);
    if (!types.ok()) return types.status();
    auto[matrix_type, data_type] = *std::move(types);
    InferType(data_type);
    return absl::OkStatus();
  }

  absl::Status ReadMatrixMarketFile(const std::string &mtx_path,
                                    const std::string &type_hint = "auto") {
    auto status = InferTypeFromMatrixMarketFile(mtx_path, type_hint);
    if (!status.ok()) return status;

    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_).ReadMatrixMarketFile(mtx_path);
    else if (idx == kFloat)
      return std::get<kFloat>(var_).ReadMatrixMarketFile(mtx_path);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).ReadMatrixMarketFile(mtx_path);
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_).ReadMatrixMarketFile(mtx_path);
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_).ReadMatrixMarketFile(mtx_path);
    }
  }

  absl::Status ReadPieRankMatrixFile(const std::string &prm_path) {
    auto status = InferTypeFromPieRankMatrixFile(prm_path);
    if (!status.ok()) return status;

    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_).ReadPieRankMatrixFile(prm_path);
    else if (idx == kFloat)
      return std::get<kFloat>(var_).ReadPieRankMatrixFile(prm_path);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).ReadPieRankMatrixFile(prm_path);
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_).ReadPieRankMatrixFile(prm_path);
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_).ReadPieRankMatrixFile(prm_path);
    }
  }

  absl::Status WritePieRankMatrixFile(const std::string &prm_path) const {
    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_).WritePieRankMatrixFile(prm_path);
    else if (idx == kFloat)
      return std::get<kFloat>(var_).WritePieRankMatrixFile(prm_path);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).WritePieRankMatrixFile(prm_path);
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_).WritePieRankMatrixFile(prm_path);
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_).WritePieRankMatrixFile(prm_path);
    }
  }

  inline DataType GetDataType() {
    auto idx = var_.index();
    if (idx == kInt64)
      return DataType::kInt64;
    else if (idx == kFloat)
      return DataType::kFloat;
    else if (idx == kDouble)
      return DataType::kDouble;
    else if (idx == kComplexFloat)
      return DataType::kComplexFloat;
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return DataType::kComplexDouble;
    }
  }

  bool ForAllNonZeros(std::function<bool(PosSpan, IdxType)> func,
                      PosSpanMutable *pos = nullptr,
                      PosSpanMutable *zpos = nullptr) const {
    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_).ForAllNonZeros(func, pos, zpos);
    else if (idx == kFloat)
      return std::get<kFloat>(var_).ForAllNonZeros(func, pos, zpos);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).ForAllNonZeros(func, pos, zpos);
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_).ForAllNonZeros(func, pos, zpos);
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_).ForAllNonZeros(func, pos, zpos);
    }
  }

  bool ForNonZerosAtIndexPos(std::function<bool(PosSpan, IdxType)> func,
                             PosSpanMutable pos,
                             PosSpanMutable *zpos = nullptr) const {
    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_).ForNonZerosAtIndexPos(func, pos, zpos);
    else if (idx == kFloat)
      return std::get<kFloat>(var_).ForNonZerosAtIndexPos(func, pos, zpos);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).ForNonZerosAtIndexPos(func, pos, zpos);
    else if (idx == kComplexFloat)
      return
        std::get<kComplexFloat>(var_).ForNonZerosAtIndexPos(func, pos, zpos);
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return
        std::get<kComplexDouble>(var_).ForNonZerosAtIndexPos(func, pos, zpos);
    }
  }

  std::string NonZeroPosDebugString() const {
    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_).NonZeroPosDebugString();
    else if (idx == kFloat)
      return std::get<kFloat>(var_).NonZeroPosDebugString();
    else if (idx == kDouble)
      return std::get<kDouble>(var_).NonZeroPosDebugString();
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_).NonZeroPosDebugString();
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_).NonZeroPosDebugString();
    }
  }

  DenseVar Dense(bool split_depths = false, bool omit_idx_dim = false) const {
    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_).Dense(split_depths, omit_idx_dim);
    else if (idx == kFloat)
      return std::get<kFloat>(var_).Dense(split_depths, omit_idx_dim);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).Dense(split_depths, omit_idx_dim);
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_).Dense(split_depths, omit_idx_dim);
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_).Dense(split_depths, omit_idx_dim);
    }
  }

  void GetDense(DenseVar *dense, bool split_depths = false) const {
    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_).GetDense(&std::get<kInt64>(*dense),
                                             split_depths);
    else if (idx == kFloat)
      return std::get<kFloat>(var_).GetDense(&std::get<kFloat>(*dense),
                                             split_depths);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).GetDense(&std::get<kDouble>(*dense),
                                              split_depths);
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_).GetDense(
        &std::get<kComplexFloat>(*dense), split_depths);
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_).GetDense(
        &std::get<kComplexDouble>(*dense), split_depths);
    }
 }

  DenseVar ToDense(bool split_depths = false) const {
    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_).ToDense(split_depths);
    else if (idx == kFloat)
      return std::get<kFloat>(var_).ToDense(split_depths);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).ToDense(split_depths);
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_).ToDense(split_depths);
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_).ToDense(split_depths);
    }
  }

  void GetDenseSlice(DenseVar *dense, PosType idx_pos,
                     bool split_depths = false,
                     bool omit_idx_dim = true) const {
    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_).GetDenseSlice(
        &std::get<kInt64>(*dense), idx_pos, split_depths, omit_idx_dim);
    else if (idx == kFloat)
      return std::get<kFloat>(var_).GetDenseSlice(
        &std::get<kFloat>(*dense), idx_pos, split_depths, omit_idx_dim);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).GetDenseSlice(
        &std::get<kDouble>(*dense), idx_pos, split_depths, omit_idx_dim);
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_).GetDenseSlice(
        &std::get<kComplexFloat>(*dense), idx_pos, split_depths, omit_idx_dim);
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_).GetDenseSlice(
        &std::get<kComplexDouble>(*dense), idx_pos, split_depths, omit_idx_dim);
    }
  }

  DenseVar DenseSlice(PosType idx_pos, bool split_depths = false,
                       bool omit_idx_dim = true) const {
    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_).DenseSlice(
        idx_pos, split_depths, omit_idx_dim);
    else if (idx == kFloat)
      return std::get<kFloat>(var_).DenseSlice(
        idx_pos, split_depths, omit_idx_dim);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).DenseSlice(
        idx_pos, split_depths, omit_idx_dim);
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_).DenseSlice(
        idx_pos, split_depths, omit_idx_dim);
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_).DenseSlice(
        idx_pos, split_depths, omit_idx_dim);
    }
  }

  ValueVar operator[](std::size_t index) const {
    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_)[index];
    else if (idx == kFloat)
      return std::get<kFloat>(var_)[index];
    else if (idx == kDouble)
      return std::get<kDouble>(var_)[index];
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_)[index];
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_)[index];
    }
  }

  ValueVar operator()(PosSpan pos, uint32_t depth = 0,
                      PosSpanMutable *zpos = nullptr) const {
    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_)(pos, depth, zpos);
    else if (idx == kFloat)
      return std::get<kFloat>(var_)(pos, depth, zpos);
    else if (idx == kDouble)
      return std::get<kDouble>(var_)(pos, depth, zpos);
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_)(pos, depth, zpos);
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_)(pos, depth, zpos);
    }
  }

  ValueVar operator()(PosType row, PosType col, uint32_t depth = 0) const {
    return (*this)({row, col}, depth);
  }

  const FlexIdxType &Index() const {
    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_).Index();
    else if (idx == kFloat)
      return std::get<kFloat>(var_).Index();
    else if (idx == kDouble)
      return std::get<kDouble>(var_).Index();
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_).Index();
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_).Index();
    }
  }

  IdxType Index(PosType pos) const {
    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_).Index(pos);
    else if (idx == kFloat)
      return std::get<kFloat>(var_).Index(pos);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).Index(pos);
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_).Index(pos);
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_).Index(pos);
    }
  }

  const FlexPosType &Pos() const {
    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_).Pos();
    else if (idx == kFloat)
      return std::get<kFloat>(var_).Pos();
    else if (idx == kDouble)
      return std::get<kDouble>(var_).Pos();
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_).Pos();
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_).Pos();
    }
  }

  PosType Pos(IdxType index) const {
    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_).Pos(index);
    else if (idx == kFloat)
      return std::get<kFloat>(var_).Pos(index);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).Pos(index);
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_).Pos(index);
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_).Pos(index);
    }
  }

  IdxType NumNonZeros() const {
    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_).NumNonZeros();
    else if (idx == kFloat)
      return std::get<kFloat>(var_).NumNonZeros();
    else if (idx == kDouble)
      return std::get<kDouble>(var_).NumNonZeros();
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_).NumNonZeros();
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_).NumNonZeros();
    }
  }

  uint32_t Depths() const {
    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_).Depths();
    else if (idx == kFloat)
      return std::get<kFloat>(var_).Depths();
    else if (idx == kDouble)
      return std::get<kDouble>(var_).Depths();
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_).Depths();
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_).Depths();
    }
  }

  friend bool
  operator==(const SparseTensorVar<Int64TensorType,
                                   FloatTensorType,
                                   DoubleTensorType,
                                   ComplexFloatTensorType,
                                   ComplexDoubleTensorType> &lhs,
             const SparseTensorVar<Int64TensorType,
                                   FloatTensorType,
                                   DoubleTensorType,
                                   ComplexFloatTensorType,
                                   ComplexDoubleTensorType> &rhs) {
    auto idx = lhs.var_.index();
    if (idx != rhs.var_.index()) return false;
    if (idx == kInt64)
      return std::get<kInt64>(lhs.var_) == std::get<kInt64>(rhs.var_);
    else if (idx == kFloat)
      return std::get<kFloat>(lhs.var_) == std::get<kFloat>(rhs.var_);
    else if (idx == kDouble)
      return std::get<kDouble>(lhs.var_) == std::get<kDouble>(rhs.var_);
    else if (idx == kComplexFloat)
      return
        std::get<kComplexFloat>(lhs.var_) == std::get<kComplexFloat>(rhs.var_);
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return
        std::get<kComplexDouble>(lhs.var_) == std::get<kComplexDouble>(rhs.var_);
    }
  }

  friend bool
  operator!=(const SparseTensorVar<Int64TensorType,
                                   FloatTensorType,
                                   DoubleTensorType,
                                   ComplexFloatTensorType,
                                   ComplexDoubleTensorType> &lhs,
             const SparseTensorVar<Int64TensorType,
                                   FloatTensorType,
                                   DoubleTensorType,
                                   ComplexFloatTensorType,
                                   ComplexDoubleTensorType> &rhs) {
    return !(lhs == rhs);
  }

  absl::StatusOr<SparseTensorVar<Int64TensorType, FloatTensorType,
                                 DoubleTensorType, ComplexFloatTensorType,
                                 ComplexDoubleTensorType>>
  ChangeIndexDim(std::shared_ptr<ThreadPool> pool = nullptr,
                 uint64_t max_nnz_per_thread = 8000000) const {
    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_).ChangeIndexDim(pool, max_nnz_per_thread);
    else if (idx == kFloat)
      return std::get<kFloat>(var_).ChangeIndexDim(pool, max_nnz_per_thread);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).ChangeIndexDim(pool, max_nnz_per_thread);
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_).ChangeIndexDim(pool,
                                                          max_nnz_per_thread);
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_).ChangeIndexDim(pool,
                                                           max_nnz_per_thread);
    }
  }

  absl::StatusOr<SparseTensorVar<Int64TensorType, FloatTensorType,
                                 DoubleTensorType, ComplexFloatTensorType,
                                 ComplexDoubleTensorType>>
  ChangeIndexDim(const std::string &path,
                 uint64_t max_nnz_per_range = 64000000) const {
    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_).ChangeIndexDim(path, max_nnz_per_range);
    else if (idx == kFloat)
      return std::get<kFloat>(var_).ChangeIndexDim(path, max_nnz_per_range);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).ChangeIndexDim(path, max_nnz_per_range);
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_).ChangeIndexDim(path,
                                                          max_nnz_per_range);
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_).ChangeIndexDim(path,
                                                           max_nnz_per_range);
    }
  }

  absl::Status MmapPieRankMatrixFile(const std::string &prm_path) {
    auto status = InferTypeFromPieRankMatrixFile(prm_path);
    if (!status.ok()) return status;

    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_).MmapPieRankMatrixFile(prm_path);
    else if (idx == kFloat)
      return std::get<kFloat>(var_).MmapPieRankMatrixFile(prm_path);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).MmapPieRankMatrixFile(prm_path);
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_).MmapPieRankMatrixFile(prm_path);
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_).MmapPieRankMatrixFile(prm_path);
    }
  }

  std::string DebugString(uint64_t max_items = 0, uint32_t indent = 0) const {
    auto idx = var_.index();
    if (idx == kInt64)
      return std::get<kInt64>(var_).DebugString(max_items, indent);
    else if (idx == kFloat)
      return std::get<kFloat>(var_).DebugString(max_items, indent);
    else if (idx == kDouble)
      return std::get<kDouble>(var_).DebugString(max_items, indent);
    else if (idx == kComplexFloat)
      return std::get<kComplexFloat>(var_).DebugString(max_items, indent);
    else {
      DCHECK_EQ(idx, kComplexDouble);
      return std::get<kComplexDouble>(var_).DebugString(max_items, indent);
    }
  }

protected:
  absl::Status status_;

private:
  SparseVar var_;
};

using SparseMatrixInt64 = SparseMatrix<uint32_t, uint64_t, FlexArray<int64_t>>;
using SparseMatrixFloat = SparseMatrix<uint32_t, uint64_t, std::vector<float>>;
using SparseMatrixDouble = SparseMatrix<uint32_t, uint64_t, std::vector<double>>;
using SparseMatrixComplexFloat =
  SparseMatrix<uint32_t, uint64_t, std::vector<std::complex<float>>>;
using SparseMatrixComplexDouble =
  SparseMatrix<uint32_t, uint64_t, std::vector<std::complex<double>>>;
using SparseMatrixVar = SparseTensorVar<SparseMatrixInt64,
                                        SparseMatrixFloat,
                                        SparseMatrixDouble,
                                        SparseMatrixComplexFloat,
                                        SparseMatrixComplexDouble>;

using SparseTensor3dInt64 =
  SparseTensor3d<uint32_t, uint64_t, FlexArray<int64_t>>;
using SparseTensor3dFloat =
  SparseTensor3d<uint32_t, uint64_t, std::vector<float>>;
using SparseTensor3dDouble =
  SparseTensor3d<uint32_t, uint64_t, std::vector<double>>;
using SparseTensor3dComplexFloat =
    SparseTensor3d<uint32_t, uint64_t, std::vector<std::complex<float>>>;
using SparseTensor3dComplexDouble =
    SparseTensor3d<uint32_t, uint64_t, std::vector<std::complex<double>>>;
using SparseTensor3dVar = SparseTensorVar<SparseTensor3dInt64,
                                          SparseTensor3dFloat,
                                          SparseTensor3dDouble,
                                          SparseTensor3dComplexFloat,
                                          SparseTensor3dComplexDouble>;

using SparseTensor4dInt64 = SparseTensor4d<uint32_t, uint64_t, FlexArray<int64_t>>;
using SparseTensor4dFloat = SparseTensor4d<uint32_t, uint64_t, std::vector<float>>;
using SparseTensor4dDouble =
  SparseTensor4d<uint32_t, uint64_t, std::vector<double>>;
using SparseTensor4dComplexFloat =
    SparseTensor4d<uint32_t, uint64_t, std::vector<std::complex<float>>>;
using SparseTensor4dComplexDouble =
    SparseTensor4d<uint32_t, uint64_t, std::vector<std::complex<double>>>;
using SparseTensor4dVar = SparseTensorVar<SparseTensor4dInt64,
                                          SparseTensor4dFloat,
                                          SparseTensor4dDouble,
                                          SparseTensor4dComplexFloat,
                                          SparseTensor4dComplexDouble>;

class ScalarVar {
public:
enum Type : uint32_t {
    kUnknown,
    kInt64,
    kFloat,
    kDouble,
    kComplexFloat,
    kComplexDouble
  };

  ScalarVar() = default;

  ScalarVar(const ScalarVar &) = default;

  ScalarVar &operator=(const ScalarVar &) = default;

  ScalarVar(ScalarVar &&) = default;

  ScalarVar &operator=(ScalarVar &&) = default;

  ScalarVar(int64_t other) { var_ = other; }

  ScalarVar(float other) { var_ = other; }

  ScalarVar(double other) { var_ = other; }

  ScalarVar(std::complex<float> other) { var_ = other; }

  ScalarVar(std::complex<double> other) { var_ = other; }

private:
  std::variant<
      std::monostate,
      int64_t, float, double, std::complex<float>, std::complex<double>> var_;
};

class SparseTensor {
public:
  using PosType = uint32_t;

  using IdxType = uint64_t;

  using PosSpan = absl::Span<const PosType>;

  using PosSpanMutable = absl::Span<PosType>;

  using DenseVar = typename SparseMatrixVar::DenseVar;

  using SparseVar = std::variant<std::monostate, ScalarVar, SparseMatrixVar,
                                 SparseTensor3dVar, SparseTensor4dVar>;

  using ValueVar = typename SparseMatrixVar::ValueVar;

  using FlexPosType = FlexArray<PosType>;

  using FlexIdxType = FlexArray<IdxType>;

  using Type = SparseMatrixVar::Type;

  SparseTensor() = default;

  SparseTensor(const SparseTensor &) = delete;

  SparseTensor &operator=(const SparseTensor &) = delete;

  SparseTensor(SparseTensor &&) = default;

  SparseTensor &operator=(SparseTensor &&) = default;

  SparseTensor(SparseMatrixVar &&other) {
    var_.template emplace<2>(
        std::forward<SparseMatrixVar>(other));
  }

  SparseTensor(SparseTensor3dVar &&other) {
    var_.template emplace<3>(std::forward<SparseTensor3dVar>(other));
  }

  SparseTensor(SparseTensor4dVar &&other) {
    var_.template emplace<4>(std::forward<SparseTensor4dVar>(other));
  }

  SparseTensor(const std::string &prm_path, bool mmap = false) {
    status_ = mmap ? this->MmapPieRankMatrixFile(prm_path)
                   : this->ReadPieRankMatrixFile(prm_path);
  }

  SparseTensor(const DenseVar &dense) {
    auto dims = SparseMatrixVar::DenseNonDepthDims(dense);
    if (dims == 2)
      var_.emplace<SparseMatrixVar>(dense);
    else if (dims == 3)
      var_.emplace<SparseTensor3dVar>(dense);
    else {
      DCHECK_EQ(dims, 4);
      var_.emplace<SparseTensor4dVar>(dense);
    }
  }

  bool ok() const { return status_.ok(); }

  absl::Status status() const { return status_; }

  const SparseVar &Var() const { return var_; }

  const auto &Var2d() const { return std::get<2>(var_); }

  const auto &Var3d() const { return std::get<3>(var_); }

  const auto &Var4d() const { return std::get<4>(var_); }

  void SetDims(uint32_t dims) {
    if (dims == 1)
      var_.emplace<ScalarVar>();
    else if (dims == 2)
      var_.emplace<SparseMatrixVar>();
    else if (dims == 3)
      var_.emplace<SparseTensor3dVar>();
    else {
      DCHECK_EQ(dims, 4);
      var_.emplace<SparseTensor4dVar>();
    }
  }

  absl::Status GetDimsFromMatrixMarketFile(
      const std::string &mtx_path,
      const std::string &type_hint = "auto") {
    auto dims = MatrixMarketFileMatrixDims(mtx_path);
    if (dims < 2)
      return absl::InternalError("Bad or missing matrix file: " + mtx_path);
    --dims;  // to account for the extra dim in Matrix Market file
    SetDims(dims);
    return absl::OkStatus();
  }

  absl::Status GetDimsFromPieRankMatrixFile(const std::string &prm_path) {
    auto dims = PieRankFileDims(prm_path);
    if (dims < 2)
      return absl::InternalError("Bad or missing matrix file: " + prm_path);
    --dims;  // to account for the extra dense dim
    SetDims(dims);
    return absl::OkStatus();
  }

  absl::Status ReadMatrixMarketFile(const std::string &mtx_path,
                                    const std::string &type_hint = "auto") {
    auto status = GetDimsFromMatrixMarketFile(mtx_path, type_hint);
    if (!status.ok()) return status;

    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).ReadMatrixMarketFile(mtx_path, type_hint);
    else if (dims == 3)
      return std::get<3>(var_).ReadMatrixMarketFile(mtx_path, type_hint);
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_).ReadMatrixMarketFile(mtx_path, type_hint);
    }
  }

  absl::Status ReadPieRankMatrixFile(const std::string &prm_path) {
    auto status = GetDimsFromPieRankMatrixFile(prm_path);
    if (!status.ok()) return status;

    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).ReadPieRankMatrixFile(prm_path);
    else if (dims == 3)
      return std::get<3>(var_).ReadPieRankMatrixFile(prm_path);
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_).ReadPieRankMatrixFile(prm_path);
    }
  }

  absl::Status WritePieRankMatrixFile(const std::string &prm_path) const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).WritePieRankMatrixFile(prm_path);
    else if (dims == 3)
      return std::get<3>(var_).WritePieRankMatrixFile(prm_path);
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_).WritePieRankMatrixFile(prm_path);
    }
  }

  inline DataType GetDataType() {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).GetDataType();
    else if (dims == 3)
      return std::get<3>(var_).GetDataType();
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_).GetDataType();
    }
  }

  bool ForAllNonZeros(std::function<bool(PosSpan, IdxType)> func,
                      PosSpanMutable *pos = nullptr,
                      PosSpanMutable *zpos = nullptr) const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).ForAllNonZeros(func, pos, zpos);
    else if (dims == 3)
      return std::get<3>(var_).ForAllNonZeros(func, pos, zpos);
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_).ForAllNonZeros(func, pos, zpos);
    }
  }

  bool ForNonZerosAtIndexPos(std::function<bool(PosSpan, IdxType)> func,
                             PosSpanMutable pos,
                             PosSpanMutable *zpos = nullptr) const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).ForNonZerosAtIndexPos(func, pos, zpos);
    else if (dims == 3)
      return std::get<3>(var_).ForNonZerosAtIndexPos(func, pos, zpos);
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_).ForNonZerosAtIndexPos(func, pos, zpos);
    }
  }

  std::string NonZeroPosDebugString() const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).NonZeroPosDebugString();
    else if (dims == 3)
      return std::get<3>(var_).NonZeroPosDebugString();
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_).NonZeroPosDebugString();
    }
  }

  DenseVar Dense(bool split_depths = false, bool omit_idx_dim = false) const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).Dense(split_depths, omit_idx_dim);
    else if (dims == 3)
      return std::get<3>(var_).Dense(split_depths, omit_idx_dim);
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_).Dense(split_depths, omit_idx_dim);
    }
  }

  void GetDense(DenseVar *dense, bool split_depths = false) const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).GetDense(dense, split_depths);
    else if (dims == 3)
      return std::get<3>(var_).GetDense(dense, split_depths);
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_).GetDense(dense, split_depths);
    }
  }

  DenseVar ToDense(bool split_depths = false) const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).ToDense(split_depths);
    else if (dims == 3)
      return std::get<3>(var_).ToDense(split_depths);
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_).ToDense(split_depths);
    }
  }

  void GetDenseSlice(DenseVar *dense, PosType idx_pos,
                     bool split_depths = false,
                     bool omit_idx_dim = true) const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).GetDenseSlice(dense, idx_pos, split_depths,
                                             omit_idx_dim);
    else if (dims == 3)
      return std::get<3>(var_).GetDenseSlice(dense, idx_pos, split_depths,
                                             omit_idx_dim);
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_).GetDenseSlice(dense, idx_pos, split_depths,
                                             omit_idx_dim);
    }
  }

  DenseVar DenseSlice(PosType idx_pos, bool split_depths = false,
                      bool omit_idx_dim = true) const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).DenseSlice(idx_pos, split_depths, omit_idx_dim);
    else if (dims == 3)
      return std::get<3>(var_).DenseSlice(idx_pos, split_depths, omit_idx_dim);
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_).DenseSlice(idx_pos, split_depths, omit_idx_dim);
    }
  }

  ValueVar operator[](std::size_t index) const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_)[index];
    else if (dims == 3)
      return std::get<3>(var_)[index];
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_)[index];
    }
  }

  ValueVar operator()(PosSpan pos, uint32_t depth = 0,
                      PosSpanMutable *zpos = nullptr) const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_)(pos, depth, zpos);
    else if (dims == 3)
      return std::get<3>(var_)(pos, depth, zpos);
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_)(pos, depth, zpos);
    }
  }

  ValueVar operator()(PosType row, PosType col, uint32_t depth = 0) const {
    return (*this)({row, col}, depth);
  }

  const FlexIdxType &Index() const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).Index();
    else if (dims == 3)
      return std::get<3>(var_).Index();
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_).Index();
    }
  }

  IdxType Index(PosType pos) const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).Index(pos);
    else if (dims == 3)
      return std::get<3>(var_).Index(pos);
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_).Index(pos);
    }
  }

  const FlexPosType &Pos() const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).Pos();
    else if (dims == 3)
      return std::get<3>(var_).Pos();
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_).Pos();
    }
  }

  PosType Pos(IdxType index) const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).Pos(index);
    else if (dims == 3)
      return std::get<3>(var_).Pos(index);
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_).Pos(index);
    }
  }

  IdxType NumNonZeros() const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).NumNonZeros();
    else if (dims == 3)
      return std::get<3>(var_).NumNonZeros();
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_).NumNonZeros();
    }
  }

  uint32_t Depths() const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).Depths();
    else if (dims == 3)
      return std::get<3>(var_).Depths();
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_).Depths();
    }
  }

  friend bool
  operator==(const SparseTensor &lhs, const SparseTensor &rhs) {
    if (lhs.var_.index() != rhs.var_.index()) return false;
    auto dims = lhs.var_.index();
    if (dims == 2)
      return std::get<2>(lhs.var_) == std::get<2>(rhs.var_);
    else if (dims == 3)
      return std::get<3>(lhs.var_) == std::get<3>(rhs.var_);
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(lhs.var_) == std::get<4>(rhs.var_);
    }
  }

  friend bool
  operator!=(const SparseTensor &lhs, const SparseTensor &rhs) {
    return !(lhs == rhs);
  }

  absl::StatusOr<SparseTensor>
  ChangeIndexDim(std::shared_ptr<ThreadPool> pool = nullptr,
                 uint64_t max_nnz_per_thread = 8000000) const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).ChangeIndexDim(pool, max_nnz_per_thread);
    // else if (dims == 3)
    //   return std::get<3>(var_).ChangeIndexDim(pool, max_nnz_per_thread);
    // else {
    //   DCHECK_EQ(dims, 4);
    //   return std::get<4>(var_).ChangeIndexDim(pool, max_nnz_per_thread);
    // }

    return absl::InternalError("Internal error");
  }

  absl::StatusOr<SparseTensor>
  ChangeIndexDim(const std::string &path,
                 uint64_t max_nnz_per_range = 64000000) const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).ChangeIndexDim(path, max_nnz_per_range);
    // else if (dims == 3)
    //   return std::get<3>(var_).ChangeIndexDim(path, max_nnz_per_range);
    // else {
    //   DCHECK_EQ(dims, 4);
    //   return std::get<4>(var_).ChangeIndexDim(path, max_nnz_per_range);
    // }

    return absl::InternalError("Internal error");
  }

  absl::Status MmapPieRankMatrixFile(const std::string &prm_path) {
    auto status = GetDimsFromPieRankMatrixFile(prm_path);
    if (!status.ok()) return status;

    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).MmapPieRankMatrixFile(prm_path);
    else if (dims == 3)
      return std::get<3>(var_).MmapPieRankMatrixFile(prm_path);
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_).MmapPieRankMatrixFile(prm_path);
    }
  }

  std::string DebugString(uint64_t max_items = 0, uint32_t indent = 0) const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).DebugString(max_items, indent);
    else if (dims == 3)
      return std::get<3>(var_).DebugString(max_items, indent);
    else {
      DCHECK_EQ(dims, 4);
      return std::get<4>(var_).DebugString(max_items, indent);
    }
  }

protected:
  absl::Status status_;

private:
  SparseVar var_;
};

inline auto& DenseI64(const SparseTensor::DenseVar &var) {
  return std::get<SparseTensor::Type::kInt64>(var);
}

inline auto& DenseF32(const SparseTensor::DenseVar &var) {
  return std::get<SparseTensor::Type::kFloat>(var);
}

inline auto& DenseF64(const SparseTensor::DenseVar &var) {
  return std::get<SparseTensor::Type::kDouble>(var);
}

inline auto& DenseC32(const SparseTensor::DenseVar &var) {
  return std::get<SparseTensor::Type::kComplexFloat>(var);
}

inline auto& DenseC64(const SparseTensor::DenseVar &var) {
  return std::get<SparseTensor::Type::kComplexDouble>(var);
}

inline auto ValueI64(const SparseTensor::ValueVar &var) {
  return std::get<SparseTensor::Type::kInt64>(var);
}

inline auto ValueF32(const SparseTensor::ValueVar &var) {
  return std::get<SparseTensor::Type::kFloat>(var);
}

inline auto ValueF64(const SparseTensor::ValueVar &var) {
  return std::get<SparseTensor::Type::kDouble>(var);
}

inline auto ValueC32(const SparseTensor::ValueVar &var) {
  return std::get<SparseTensor::Type::kComplexFloat>(var);
}

inline auto ValueC64(const SparseTensor::ValueVar &var) {
  return std::get<SparseTensor::Type::kComplexDouble>(var);
}

}  // namespace pierank

#endif //PIERANK_SPARSE_TENSOR_H_
