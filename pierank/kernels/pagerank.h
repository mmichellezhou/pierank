//
// Created by Michelle Zhou on 1/17/22.
//

#ifndef PIERANK_KERNELS_PAGERANK_H_
#define PIERANK_KERNELS_PAGERANK_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "absl/status/status.h"
#include "pierank/sparse_matrix.h"
#include "pierank/thread_pool.h"

namespace pierank {

template<typename T = double, typename PosType = uint32_t, typename IdxType = uint64_t>
class PageRank : public SparseMatrix<PosType, IdxType> {
public:
  using PosRange = typename SparseMatrix<PosType, IdxType>::PosRange;

  using PosRanges = typename SparseMatrix<PosType, IdxType>::PosRanges;

  PageRank(const std::string &file_path, T damping_factor = 0.85,
           uint32_t max_iterations = 30, T epsilon = 1e-6) :
      damping_factor_(damping_factor), max_iterations_(max_iterations),
      epsilon_(epsilon) {
    this->status_ = MatrixMarketIo::HasMtxFileExtension(file_path)
                    ? this->ReadMatrixMarketFile(file_path)
                    : this->ReadPrmFile(file_path);
    if (!this->status_.ok())
      return;
    num_pages_ = std::max(this->Rows(), this->Cols());
    one_minus_d_over_n_ = (1 - damping_factor_) / num_pages_;
    scores_.resize(num_pages_, one_minus_d_over_n_);
    out_degree_.resize(this->Rows());

    NumOutboundLinks();
  }

  // Returns <epsilon, num_iterations> pair or <+infinity, 0> on error
  std::pair<T, uint32_t> Run(std::shared_ptr<ThreadPool> pool = nullptr) {
    if (!this->status_.ok())
      return std::make_pair(std::numeric_limits<T>::max(), 0);

    const auto ranges = SplitCols(pool);
    epsilons_.resize(ranges.size(), std::numeric_limits<T>::max());

    uint32_t iter;
    for (iter = 0; iter < max_iterations_; ++iter) {
      if (!pool) {
        DCHECK_EQ(ranges.size(), 1);
        DoRange(ranges[0], 0);
      } else {
        pool->ParallelFor(ranges.size(), /*items_per_thread=*/1,
          [this, &ranges](uint64_t first, uint64_t last) {
            for (auto r = first; r < last; ++r)
              DoRange(ranges[r], /*range_id=*/r);
        });
      }
      if (MaxEpsilon() < epsilon_)
        break;
    }

    return std::make_pair(MaxEpsilon(), std::min(iter + 1, max_iterations_));
  }

  // Returns an emtpy vector on error.
  std::vector<std::pair<T, uint32_t>> TopK(uint32_t k = 100) {
    std::vector<std::pair<T, uint32_t>> score_and_page_vec;
    if (!this->status_.ok()) return score_and_page_vec;

    score_and_page_vec = std::move(ScoreAndPageVector());
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
  void NumOutboundLinks() {
    DCHECK(this->status_.ok());
    for (PosType nz = 0; nz < this->NumNonZeros(); nz++) {
      out_degree_[this->Pos(nz)]++;
    }
  }

  std::vector<std::pair<T, uint32_t>> ScoreAndPageVector() {
    DCHECK(this->status_.ok());
    std::vector<std::pair<T, uint32_t>> res;

    for (int i = 0; i < scores_.size(); ++i) {
      res.push_back(std::make_pair(scores_[i], i));
    }

    return res;
  }

  void DoRange(const PosRange &range, uint32_t range_id) {
    DCHECK(this->status_.ok());
    const auto first = range.first;
    const auto last = range.second;
    DCHECK_LT(first, last);
    T epsilon = 0.0;
    for (PosType p = first; p < last; ++p) {
      T sum = 0.0;

      for (IdxType i = this->Index(p); i < this->Index(p + 1); ++i) {
        DCHECK_LT(i, this->NumNonZeros());
        DCHECK_LT(this->Pos(i), this->Rows());
        DCHECK_LT(this->Pos(i), scores_.size());
        DCHECK_GT(out_degree_[this->Pos(i)], 0);
        sum += scores_[this->Pos(i)] / out_degree_[this->Pos(i)];
      }

      T score = one_minus_d_over_n_ + damping_factor_ * sum;
      epsilon = std::max(epsilon, std::fabs(scores_[p] - score));
      scores_[p] = score;
    }

    epsilons_[range_id] = epsilon;
  }

  T MaxEpsilon() const {
    DCHECK(this->status_.ok());
    return *std::max_element(epsilons_.begin(), epsilons_.end());
  }

  PosRanges SplitCols(std::shared_ptr<ThreadPool> pool = nullptr) const {
    DCHECK(this->status_.ok());
    DCHECK_GT(this->Cols(), 0);
    if (!pool)
      return {std::make_pair(0, this->Cols())};
    PosRanges res;
    PosType range_size = (this->Cols() + pool->Size() - 1) / pool->Size();
    for (PosType first = 0; first < this->Cols(); first += range_size) {
      PosType last = std::min(first + range_size, this->Cols());
      res.push_back(std::make_pair(first, last));
    }
    return res;
  }

private:
  T damping_factor_;
  uint64_t num_pages_;
  T one_minus_d_over_n_;
  std::vector<T> scores_;
  std::vector<T> epsilons_;
  std::vector<uint32_t> out_degree_;
  uint32_t max_iterations_;
  T epsilon_;
};

}  // namespace pierank

#endif //PIERANK_KERNELS_PAGERANK_H_
