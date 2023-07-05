//
// Created by Michelle Zhou on 2/12/22.
//

#ifndef PIERANK_IO_MATRIX_MARKET_IO_H_
#define PIERANK_IO_MATRIX_MARKET_IO_H_

#include <complex>
#include <cstdint>
#include <fstream>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"

namespace pierank {

class MatrixType {
public:
  constexpr static std::size_t kStringLength = 3;

  enum Enum : uint32_t {
    kUnknown,
    kArrayComplexGeneral,
    kArrayRealGeneral,
    kCoordinateComplexGeneral,
    kCoordinateComplexHermitian,
    kCoordinateComplexSymmetric,
    kCoordinateIntegerGeneral,
    kCoordinateIntegerSymmetric,
    kCoordinatePatternGeneral,
    kCoordinatePatternSymmetric,
    kCoordinateRealGeneral,
    kCoordinateRealSkewSymmetric,
    kCoordinateRealSymmetric
  };

  static Enum FromString(absl::string_view str) {
    if (str == "UNK") return kUnknown;
    if (str == "ACG") return kArrayComplexGeneral;
    if (str == "ARG") return kArrayRealGeneral;
    if (str == "CCG") return kCoordinateComplexGeneral;
    if (str == "CCH") return kCoordinateComplexHermitian;
    if (str == "CCS") return kCoordinateComplexSymmetric;
    if (str == "CIG") return kCoordinateIntegerGeneral;
    if (str == "CIS") return kCoordinateIntegerSymmetric;
    if (str == "CPG") return kCoordinatePatternGeneral;
    if (str == "CPS") return kCoordinatePatternSymmetric;
    if (str == "CRG") return kCoordinateRealGeneral;
    if (str == "CRK") return kCoordinateRealSkewSymmetric;
    if (str == "CRS") return kCoordinateRealSymmetric;
    return kUnknown;
  }

  static Enum FromLongString(absl::string_view str) {
    if (str == "unknown") return kUnknown;
    if (str == "array complex general") return kArrayComplexGeneral;
    if (str == "array real general") return kArrayRealGeneral;
    if (str == "coordinate complex general") return kCoordinateComplexGeneral;
    if (str == "coordinate complex Hermitian")
      return kCoordinateComplexHermitian;
    if (str == "coordinate complex symmetric")
      return kCoordinateComplexSymmetric;
    if (str == "coordinate integer general") return kCoordinateIntegerGeneral;
    if (str == "coordinate integer symmetric")
      return kCoordinateIntegerSymmetric;
    if (str == "coordinate pattern general") return kCoordinatePatternGeneral;
    if (str == "coordinate pattern symmetric")
      return kCoordinatePatternSymmetric;
    if (str == "coordinate real general") return kCoordinateRealGeneral;
    if (str == "coordinate real skew-symmetric")
      return kCoordinateRealSkewSymmetric;
    if (str == "coordinate real symmetric") return kCoordinateRealSymmetric;
    return kUnknown;
  }

  MatrixType() = default;

  MatrixType(Enum value) : val_(value) {}

  MatrixType(absl::string_view str) : val_(FromString(str)) {}

  inline bool IsPattern() const {
    return val_ == kCoordinatePatternGeneral ||
           val_ == kCoordinatePatternSymmetric;
  }

  inline bool IsInteger() const {
    return val_ == kCoordinateIntegerGeneral ||
           val_ == kCoordinateIntegerSymmetric;
  }

  inline bool IsReal() const {
    return val_ == kArrayRealGeneral ||
           val_ == kCoordinateRealGeneral ||
           val_ == kCoordinateRealSkewSymmetric ||
           val_ == kCoordinateRealSymmetric;
  }

  inline bool IsComplex() const {
    return val_ == kArrayComplexGeneral ||
           val_ == kCoordinateComplexGeneral ||
           val_ == kCoordinateComplexHermitian ||
           val_ == kCoordinateComplexSymmetric;
  }

  inline bool Symmetric() const {
    return val_ == kCoordinateIntegerSymmetric ||
           val_ == kCoordinatePatternSymmetric ||
           val_ == kCoordinateRealSymmetric;
  }

  constexpr operator Enum() const { return val_; }

  std::string ToString() const {
    switch (val_) {
      case kUnknown: return "UNK";
      case kArrayComplexGeneral: return "ACG";
      case kArrayRealGeneral: return "ARG";
      case kCoordinateComplexGeneral: return "CCG";
      case kCoordinateComplexHermitian: return "CCH";
      case kCoordinateComplexSymmetric: return "CCS";
      case kCoordinateIntegerGeneral: return "CIG";
      case kCoordinateIntegerSymmetric: return "CIS";
      case kCoordinatePatternGeneral: return "CPG";
      case kCoordinatePatternSymmetric: return "CPS";
      case kCoordinateRealGeneral: return "CRG";
      case kCoordinateRealSkewSymmetric: return "CRK";
      case kCoordinateRealSymmetric: return "CRS";
      default: return "UDF";
    }
  }

  std::string ToLongString() const {
    switch (val_) {
      case kUnknown: return "unknown";
      case kArrayComplexGeneral: return "array complex general";
      case kArrayRealGeneral: return "array real general";
      case kCoordinateComplexGeneral: return "coordinate complex general";
      case kCoordinateComplexHermitian: return "coordinate complex Hermitian";
      case kCoordinateComplexSymmetric: return "coordindate complex symmetric";
      case kCoordinateIntegerGeneral: return "coordinate integer general";
      case kCoordinateIntegerSymmetric: return "coordinate integer symmetric";
      case kCoordinatePatternGeneral: return "coordinate pattern general";
      case kCoordinatePatternSymmetric: return "coordinate pattern symmetric";
      case kCoordinateRealGeneral: return "coordinate real general";
      case kCoordinateRealSkewSymmetric:
        return "coordinate real skew-symmetric";
      case kCoordinateRealSymmetric: return "coordinate real symmetric";
      default: return "undefined";
    }
  }

  // Reads a null-terminated "C" string as a MatrixType string.
  template<typename InputStreamType>
  absl::Status Read(InputStreamType *is, uint64_t *offset = nullptr) {
    char buf[kStringLength + 1];
    if (!ReadData(is, buf, sizeof(buf)))
      return absl::InternalError("Error reading matrix type string");
    if (strlen(buf) != kStringLength)
      return absl::InternalError("Bad matrix type string: " + std::string(buf));
    if (offset) *offset += kStringLength + 1; // +1 for the null terminator
    val_ = MatrixType(FromString(buf));
    return absl::OkStatus();
  }

  friend std::istream &operator>>(std::istream &is, MatrixType &type) {
    auto status = type.Read(&is, nullptr);
    if (!status.ok())
      LOG(ERROR) << status.message();
    return is;
  }

  // Writes a MatrixType string as a null-terminated "C" string.
  template<typename OutputStreamType>
  absl::Status Write(OutputStreamType *os) const {
    auto str = ToString();
    DCHECK_EQ(str.size(), kStringLength);
    const char *buf = str.c_str();
    DCHECK_EQ(buf[kStringLength], '\0');
    if (!WriteData(os, buf, kStringLength + 1))
      return absl::InternalError("Error writing matrix type string");
    return absl::OkStatus();
  }

private:
  Enum val_;
};

class MatrixMarketIo {
public:
  using Var =
      std::variant<std::monostate, int64_t, double, std::complex<double>>;

  using Entry = std::tuple<uint32_t, uint32_t, Var>;

  inline static bool HasMtxFileExtension(absl::string_view path) {
    return absl::EndsWith(path, ".mtx");
  }

  inline static bool IsVarZero(const Var &var) {
    if (std::holds_alternative<std::monostate>(var)) return false;
    if (std::holds_alternative<int64_t>(var)) return !std::get<int64_t>(var);
    if (std::holds_alternative<double>(var)) return !std::get<double>(var);
    if (std::holds_alternative<std::complex<double>>(var))
      return !std::get<std::complex<double>>(var).real() &&
             !std::get<std::complex<double>>(var).imag();
    CHECK(false);
  }

  MatrixMarketIo(const std::string &file_path) : is_(file_path) {
    if (static_cast<bool>(is_)) {
      type_ = ReadBanner();
      if (type_ != MatrixType::kUnknown) {
        SkipComments();
        is_ >> rows_ >> cols_ >> nnz_;
      }
    }
  }

  void SkipComments() {
    std::string line;
    while (true) {
      auto pos = is_.tellg();
      getline(is_, line);
      if (line[0] != '%') {
        is_.seekg(pos, std::ios_base::beg);
        return;
      }
    }
  }

  MatrixType ReadBanner() {
    std::vector<std::string> words = {"%%MatrixMarket", "matrix"};
    MatrixType res = MatrixType::kUnknown;
    for (const auto &word : words) {
      std::string str;
      is_ >> str;
      if (str != word) return res;
    }
    std::string mtype, dtype, subtype;
    is_ >> mtype >> dtype >> subtype;
    if (mtype == "array") {
      if (dtype == "complex") {
        if (subtype == "general") res = MatrixType::kArrayComplexGeneral;
      } else if (dtype == "real") {
        if (subtype == "general") res = MatrixType::kArrayRealGeneral;
      }
    } else if (mtype == "coordinate") {
      if (dtype == "complex") {
        if (subtype == "general") res = MatrixType::kCoordinateComplexGeneral;
        else if (subtype == "Hermitian")
          res = MatrixType::kCoordinateComplexHermitian;
        else if (subtype == "symmetric")
          res = MatrixType::kCoordinateComplexSymmetric;
      } else if (dtype == "integer") {
        if (subtype == "general") res = MatrixType::kCoordinateIntegerGeneral;
        else if (subtype == "symmetric")
          res = MatrixType::kCoordinateIntegerSymmetric;
      } else if (dtype == "pattern") {
        if (subtype == "general") res = MatrixType::kCoordinatePatternGeneral;
        else if (subtype == "symmetric")
          res = MatrixType::kCoordinatePatternSymmetric;
      } else if (dtype == "real") {
        if (subtype == "general") res = MatrixType::kCoordinateRealGeneral;
        else if (subtype == "skew-symmetric")
          res = MatrixType::kCoordinateRealSkewSymmetric;
        else if (subtype == "symmetric")
          res = MatrixType::kCoordinateRealSymmetric;
      }
    }
    is_.ignore();  // skip the new line character
    return res;
  }

  bool HasNext() const { return count_ < nnz_; }

  Entry Next() {
    uint32_t first, second;
    is_ >> first >> second;
    Var var;
    if (type_.IsInteger()) {
      int64_t value;
      is_ >> value;
      var = value;
    } else if (type_.IsReal()) {
      double value;
      is_ >> value;
      var = value;
    } else if (type_.IsComplex()) {
      double re, im;
      is_ >> re >> im;
      var = std::complex<double>(re, im);
    }
    count_++;
    return std::make_tuple(first, second, var);
  }

  uint32_t Rows() const { return rows_; }

  uint32_t Cols() const { return cols_; }

  uint64_t NumNonZeros() const { return nnz_; }

  bool ok() const { return type_ != MatrixType::kUnknown; }

  const MatrixType &Type() const { return type_; }

private:
  uint32_t rows_;
  uint32_t cols_;
  uint64_t nnz_;
  std::ifstream is_;
  MatrixType type_ = MatrixType::kUnknown;
  uint64_t count_ = 0;
};

// Tuple: <MatrixType, #rows, #cols, #nnz>
inline absl::StatusOr<std::tuple<MatrixType, uint64_t, uint64_t, uint64_t>>
MatrixMarketFileInfo(const std::string &mtx_path) {
  std::tuple<MatrixType, uint64_t, uint64_t, uint64_t> res;
  MatrixMarketIo mat(mtx_path);
  if (!mat.ok()) return absl::InternalError("Bad matrix market file");
  return std::make_tuple(mat.Type(), mat.Rows(), mat.Cols(), mat.NumNonZeros());
}

}  // namespace pierank

#endif //PIERANK_IO_MATRIX_MARKET_IO_H_
