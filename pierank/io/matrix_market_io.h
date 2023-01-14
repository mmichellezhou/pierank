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
    kPatternNone,
    kPatternGeneral,
    kPatternSymmetric
  };

  inline static bool HasMtxFileExtension(absl::string_view path) {
    return absl::EndsWith(path, ".mtx");
  }

  MatrixMarketIo(const std::string &file_path) : is_(file_path) {
    if (static_cast<bool>(is_)) {
      matrix_type_ = CheckBanner();
      if (matrix_type_ != MatrixType::kPatternNone) {
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

  MatrixType CheckBanner() {
    std::vector<std::string> words = {"%%MatrixMarket", "matrix", "coordinate",
                                      "pattern"};
    MatrixType res = MatrixType::kPatternNone;
    std::string str;
    for (const auto &word : words) {
      is_ >> str;
      if (str != word) return res;
    }
    is_ >> str;
    if (str == "general")
      res = MatrixType::kPatternGeneral;
    else if (str == "symmetric")
      res = MatrixType::kPatternSymmetric;
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

  bool Ok() const { return matrix_type_ != MatrixType::kPatternNone; }

private:
  uint32_t rows_;
  uint32_t cols_;
  uint64_t nnz_;
  std::ifstream is_;
  MatrixType matrix_type_ = MatrixType::kPatternNone;
  uint64_t count_ = 0;
};

}  // namespace pierank

#endif //PIERANK_IO_MATRIX_MARKET_IO_H_
