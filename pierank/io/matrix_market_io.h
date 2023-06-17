//
// Created by Michelle Zhou on 2/12/22.
//

#ifndef PIERANK_IO_MATRIX_MARKET_IO_H_
#define PIERANK_IO_MATRIX_MARKET_IO_H_

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"

namespace pierank {

class MatrixMarketIo {
public:
  enum class MatrixType {
    kUnknown,
    kIntegerGeneral,
    kIntegerSymmetric,
    kPatternGeneral,
    kPatternSymmetric,
    kRealGeneral,
    kRealSymmetric,
  };

  inline static bool HasMtxFileExtension(absl::string_view path) {
    return absl::EndsWith(path, ".mtx");
  }

  MatrixMarketIo(const std::string &file_path) : is_(file_path) {
    if (static_cast<bool>(is_)) {
      matrix_type_ = ReadBanner();
      if (matrix_type_ != MatrixType::kUnknown) {
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
    std::vector<std::string> words = {"%%MatrixMarket", "matrix", "coordinate"};
    MatrixType res = MatrixType::kUnknown;
    for (const auto &word : words) {
      std::string str;
      is_ >> str;
      if (str != word) return res;
    }
    std::string dtype, subtype;
    is_ >> dtype >> subtype;
    if (dtype == "integer") {
      if (subtype == "general") res = MatrixType::kIntegerGeneral;
      else if (subtype == "symmetric") res = MatrixType::kIntegerSymmetric;
    } else if (dtype == "pattern") {
      if (subtype == "general") res = MatrixType::kPatternGeneral;
      else if (subtype == "symmetric") res = MatrixType::kPatternSymmetric;
    } else if (dtype == "real") {
      if (subtype == "general") res = MatrixType::kRealGeneral;
      else if (subtype == "symmetric") res = MatrixType::kRealSymmetric;
    }
    is_.ignore();  // skip the new line character
    return res;
  }

  bool HasNext() const { return count_ < nnz_; }

  std::pair<uint32_t, uint32_t> Next() {
    std::pair<uint32_t, uint32_t> ret;
    is_ >> ret.first >> ret.second;
    count_++;
    return ret;
  }

  uint32_t Rows() const { return rows_; }

  uint32_t Cols() const { return cols_; }

  uint64_t NumNonZeros() const { return nnz_; }

  bool Symmetric() const {
    return matrix_type_ == MatrixType::kPatternSymmetric;
  }

  bool Ok() const { return matrix_type_ != MatrixType::kUnknown; }

private:
  uint32_t rows_;
  uint32_t cols_;
  uint64_t nnz_;
  std::ifstream is_;
  MatrixType matrix_type_ = MatrixType::kUnknown;
  uint64_t count_ = 0;
};

}  // namespace pierank

#endif //PIERANK_IO_MATRIX_MARKET_IO_H_
