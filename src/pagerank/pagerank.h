//
// Created by Michelle Zhou on 1/17/22.
//

#ifndef PIERANK_PAGERANK_H_
#define PIERANK_PAGERANK_H_

#include <limits>
#include <glog/logging.h>

#include "sparse_matrix.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

template<typename T = double, typename PosType = uint32_t, typename IdxType = uint64_t>
class PageRank : public SparseMatrix<PosType, IdxType> {
public:
  PageRank(const std::string &file_path, T damping_factor = 0.85,
           uint32_t max_iterations = 30, T epsilon = 1e-6) :
           SparseMatrix<PosType, IdxType>(file_path),
      damping_factor_(damping_factor),
      num_pages_(std::max(this->Rows(), this->Cols())),
      one_minus_d_over_n_((1 - damping_factor_) / num_pages_),
      scores_(num_pages_, one_minus_d_over_n_), out_degree_(this->Rows()),
      max_iterations_(max_iterations), epsilon_(epsilon) {
    NumOutboundLinks();
  }

  /* Example SparseMatrix:
   *
   *     0 1 2 3 4
   *     - - - - -
   * 0 | 0 1 1 0 0 (0 is pointing to 1 and 2)
   * 1 | 0 0 0 1 0
   * 2 | 1 1 0 1 1
   * 3 | 0 0 1 0 0
   * 4 | 1 0 0 0 0
   *    (2 and 4 are pointing to 0)
   *
   * pos = [2, 4, 0, 2, 0, 3, 1, 2, 2]
   * idx = [0, 2, 4, 6, 8, 9]
   */
  void NumOutboundLinks() {
    for (PosType nz = 0; nz < this->NumNonZeros(); nz++) {
      out_degree_[this->Pos(nz)]++;
    }
  }

  // Returns <epsilon, num_iterations> pair
  std::pair<T, uint32_t> Run() {
    T epsilon;
    uint32_t iter;

    for (iter = 0; iter < max_iterations_; ++iter) {
      epsilon = 0.0;

      for (uint64_t p = 0; p < this->Cols(); ++p) {
        T sum = 0.0;

        for (uint64_t i = this->Index(p); i < this->Index(p + 1); ++i) {
          DCHECK_GT(out_degree_[this->Pos(i)], 0);
          DCHECK_LT(this->Pos(i), scores_.size());
          sum += scores_[this->Pos(i)] / out_degree_[this->Pos(i)];
        }

        T score = one_minus_d_over_n_ + damping_factor_ * sum;
        epsilon = std::max(epsilon, std::fabs(scores_[p] - score));
        scores_[p] = score;
      }

      if (epsilon < epsilon_)
        break;
    }

    return std::make_pair(epsilon, std::min(iter + 1, max_iterations_));
  }

  std::vector<std::pair<T, uint32_t>> TopK(uint32_t k = 100) {
    auto score_and_page_vec = ScoreAndPageVector();
    std::sort(score_and_page_vec.begin(), score_and_page_vec.end(),
        [](const std::pair<T, uint32_t> &a, const std::pair<T, uint32_t> &b){
      return a.first > b.first;
    });

    if (k > num_pages_)
      k = num_pages_;

    score_and_page_vec.resize(k);

    // for (int i = 0; i < k; ++i) {
      // auto pair = score_and_page[num_pages_ - i - 1];
      // std::cout << "Page: " << pair.second << " Score: " << pair.first << "\n";
    // }

    return score_and_page_vec;
  }

  const std::vector<T> &Scores() const { return scores_; }

protected:
  std::vector<std::pair<T, uint32_t>> ScoreAndPageVector() {
    std::vector<std::pair<T, uint32_t>> res;

    for (int i = 0; i < scores_.size(); ++i) {
      res.push_back(std::make_pair(scores_[i], i));
    }

    return res;
  }

private:
  T damping_factor_;
  uint64_t num_pages_;
  T one_minus_d_over_n_;
  std::vector<T> scores_;
  std::vector<uint32_t> out_degree_;
  uint32_t max_iterations_;
  T epsilon_;
};

#endif //PIERANK_PAGERANK_H_
