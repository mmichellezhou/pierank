//
// Created by Michelle Zhou on 2/12/22.
//
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

class MatrixMarketIo {
public:
  MatrixMarketIo(const std::string &file_path) : is_(file_path) {
    ok_ = static_cast<bool>(is_) && CheckBanner();
    if (ok_)
      is_ >> rows_ >> cols_ >> nnz_;
  }

  bool CheckBanner() {
    std::vector<std::string> words = {"%%MatrixMarket", "matrix", "coordinate", "pattern", "general"};
    for (const auto &word : words) {
      std::string temp;
      is_ >> temp;
      if (temp != word) return false;
    }

    return true;
  }

  bool HasNext() const {
    if (count_ >= nnz_)
      return false;
    return true;
  }

  std::pair<uint32_t, uint32_t> Next() {
    std::pair<uint32_t, uint32_t> ret;
    is_ >> ret.first >> ret.second;
    count_++;
    return ret;
  }

  uint32_t Rows() const {
    return rows_;
  }

  uint32_t Cols() const {
    return cols_;
  }

  uint64_t NNZ() const {
    return nnz_;
  }

  bool Ok() const {
    return ok_;
  }

private:
  uint32_t rows_;
  uint32_t cols_;
  uint64_t nnz_;
  std::ifstream is_;
  bool ok_;
  uint64_t count_;
};