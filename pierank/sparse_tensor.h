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

  SparseTensorVar() = default;

  SparseTensorVar(const SparseTensorVar &) = delete;

  SparseTensorVar &operator=(const SparseTensorVar &) = delete;

  SparseTensorVar(SparseTensorVar &&) = default;

  SparseTensorVar &operator=(SparseTensorVar &&) = default;

  SparseTensorVar(Int64TensorType &&other) {
    type_ = kInt64;
    var_.template emplace<kInt64>(
        std::forward<Int64TensorType>(other));
  }

  SparseTensorVar(FloatTensorType &&other) {
    type_ = kFloat;
    var_.template emplace<kFloat>(std::forward<FloatTensorType>(other));
  }

  SparseTensorVar(DoubleTensorType &&other) {
    type_ = kDouble;
    var_.template emplace<kDouble>(std::forward<DoubleTensorType>(other));
  }

  SparseTensorVar(ComplexFloatTensorType &&other) {
    type_ = kComplexFloat;
    var_.template emplace<kComplexFloat>(
        std::forward<ComplexFloatTensorType>(other));
  }

  SparseTensorVar(ComplexDoubleTensorType &&other) {
    type_ = kComplexDouble;
    var_.template emplace<kComplexDouble>(
        std::forward<ComplexDoubleTensorType>(other));
  }

  SparseTensorVar(const std::string &prm_path, bool mmap = false) {
    status_ = mmap ? this->MmapPieRankMatrixFile(prm_path)
                   : this->ReadPieRankMatrixFile(prm_path);
  }

  bool ok() const { return status_.ok(); }

  absl::Status status() const { return status_; }

  void SetType(Type type) {
    if (type == kInt64)
      var_.template emplace<kInt64>();
    else if (type == kFloat)
      var_.template emplace<kFloat>();
    else if (type == kDouble)
      var_.template emplace<kDouble>();
    else if (type == kComplexFloat)
      var_.template emplace<kComplexFloat>();
    else if (type == kComplexDouble)
      var_.template emplace<kComplexDouble>();
    else
      CHECK(false);
    type_ = type;
  }

  void InferType(MatrixType type, const std::string &type_hint) {
    if (!type.IsComplex()) {
      if (type.IsInteger()) SetType(kInt64);
      else if (type_hint == "auto" || type_hint == "f64") SetType(kDouble);
      else if (type_hint == "f32") SetType(kFloat);
      else CHECK(false);
    }
    else if (type_hint == "auto" || type_hint == "f64") SetType(kComplexDouble);
    else if (type_hint == "f32") SetType(kComplexFloat);
    else CHECK(false);
  }

  void InferType(DataType type) {
    if (type.IsInteger()) SetType(kInt64);
    else if (type == DataType::kFloat) SetType(kFloat);
    else if (type == DataType::kDouble) SetType(kDouble);
    else if (type == DataType::kComplexFloat) SetType(kComplexFloat);
    else if (type == DataType::kComplexDouble) SetType(kComplexDouble);
    else CHECK(false);
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
    else if (idx == kComplexDouble)
      return std::get<kComplexDouble>(var_).ReadMatrixMarketFile(mtx_path);
    else
      CHECK(false);
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
    else if (idx == kComplexDouble)
      return std::get<kComplexDouble>(var_).ReadPieRankMatrixFile(prm_path);
    else
      CHECK(false);
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
    else if (idx == kComplexDouble)
      return std::get<kComplexDouble>(var_).WritePieRankMatrixFile(prm_path);
    else
      CHECK(false);
  }

  absl::StatusOr<SparseTensorVar<Int64TensorType, FloatTensorType,
                                 DoubleTensorType, ComplexFloatTensorType,
                                 ComplexDoubleTensorType>>
  ChangeIndexDim(std::shared_ptr<ThreadPool> pool = nullptr,
                 uint64_t max_nnz_per_thread = 8000000) const
  {
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
    else if (idx == kComplexDouble)
      return std::get<kComplexDouble>(var_).ChangeIndexDim(pool,
                                                           max_nnz_per_thread);
    else
      CHECK(false);
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
    else if (idx == kComplexDouble)
      return std::get<kComplexDouble>(var_).ChangeIndexDim(path,
                                                           max_nnz_per_range);
    else
      CHECK(false);
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
    else if (idx == kComplexDouble)
      return std::get<kComplexDouble>(var_).MmapPieRankMatrixFile(prm_path);
    else
      CHECK(false);
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
    else if (idx == kComplexDouble)
      return std::get<kComplexDouble>(var_).DebugString(max_items, indent);
    else
      CHECK(false);
  }

protected:
  absl::Status status_;

private:
  Type type_ = kUnknown;
  std::variant<
      std::monostate,
      Int64TensorType,
      FloatTensorType, DoubleTensorType,
      ComplexFloatTensorType, ComplexDoubleTensorType> var_;
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

  ScalarVar(int64_t other) {
    type_ = kInt64;
    var_ = other;
  }

  ScalarVar(float other) {
    type_ = kFloat;
    var_ = other;
  }

  ScalarVar(double other) {
    type_ = kDouble;
    var_ = other;
  }

  ScalarVar(std::complex<float> other) {
    type_ = kComplexFloat;
    var_ = other;
  }

  ScalarVar(std::complex<double> other) {
    type_ = kComplexDouble;
    var_ = other;
  }

private:
  Type type_ = kUnknown;
  std::variant<
      std::monostate,
      int64_t, float, double, std::complex<float>, std::complex<double>> var_;
};

class SparseTensor {
public:
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

  bool ok() const { return status_.ok(); }

  absl::Status status() const { return status_; }

  void SetDims(uint32_t dims) {
    if (dims == 1)
      var_.emplace<ScalarVar>();
    else if (dims == 2)
      var_.emplace<SparseMatrixVar>();
    else if (dims == 3)
      var_.emplace<SparseTensor3dVar>();
    else if (dims == 4)
      var_.emplace<SparseTensor4dVar>();
    else
      CHECK(false);
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
    else if (dims == 4)
      return std::get<4>(var_).ReadMatrixMarketFile(mtx_path, type_hint);
    else
      CHECK(false);
  }

  absl::Status ReadPieRankMatrixFile(const std::string &prm_path) {
    auto status = GetDimsFromPieRankMatrixFile(prm_path);
    if (!status.ok()) return status;

    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).ReadPieRankMatrixFile(prm_path);
    else if (dims == 3)
      return std::get<3>(var_).ReadPieRankMatrixFile(prm_path);
    else if (dims == 4)
      return std::get<4>(var_).ReadPieRankMatrixFile(prm_path);
    else
      CHECK(false);
  }

  absl::Status WritePieRankMatrixFile(const std::string &prm_path) const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).WritePieRankMatrixFile(prm_path);
    else if (dims == 3)
      return std::get<3>(var_).WritePieRankMatrixFile(prm_path);
    else if (dims == 4)
      return std::get<4>(var_).WritePieRankMatrixFile(prm_path);
    else
      CHECK(false);
  }

  absl::StatusOr<SparseTensor>
  ChangeIndexDim(std::shared_ptr<ThreadPool> pool = nullptr,
                 uint64_t max_nnz_per_thread = 8000000) const
  {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).ChangeIndexDim(pool, max_nnz_per_thread);
    // else if (dims == 3)
    //   return std::get<3>(var_).ChangeIndexDim(pool, max_nnz_per_thread);
    // else if (dims == 4)
    //   return std::get<4>(var_).ChangeIndexDim(pool, max_nnz_per_thread);
    else
      CHECK(false);
  }

  absl::StatusOr<SparseTensor>
  ChangeIndexDim(const std::string &path,
                 uint64_t max_nnz_per_range = 64000000) const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).ChangeIndexDim(path, max_nnz_per_range);
    // else if (dims == 3)
    //   return std::get<3>(var_).ChangeIndexDim(path, max_nnz_per_range);
    // else if (dims == 4)
    //   return std::get<4>(var_).ChangeIndexDim(path, max_nnz_per_range);
    else
      CHECK(false);
  }

  absl::Status MmapPieRankMatrixFile(const std::string &prm_path) {
    auto status = GetDimsFromPieRankMatrixFile(prm_path);
    if (!status.ok()) return status;

    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).MmapPieRankMatrixFile(prm_path);
    else if (dims == 3)
      return std::get<3>(var_).MmapPieRankMatrixFile(prm_path);
    else if (dims == 4)
      return std::get<4>(var_).MmapPieRankMatrixFile(prm_path);
    else
      CHECK(false);
  }

  std::string DebugString(uint64_t max_items = 0, uint32_t indent = 0) const {
    auto dims = var_.index();
    if (dims == 2)
      return std::get<2>(var_).DebugString(max_items, indent);
    else if (dims == 3)
      return std::get<3>(var_).DebugString(max_items, indent);
    else if (dims == 4)
      return std::get<4>(var_).DebugString(max_items, indent);
    else
      CHECK(false);
  }

protected:
  absl::Status status_;

private:
  std::variant<std::monostate, ScalarVar, SparseMatrixVar, SparseTensor3dVar,
      SparseTensor4dVar> var_;
};

}  // namespace pierank

#endif //PIERANK_SPARSE_TENSOR_H_
