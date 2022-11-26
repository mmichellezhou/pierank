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
  
  using PageScore = std::pair<PosType, T>;

  using PageScores = std::vector<PageScore>;

  PageRank(const std::string &file_path, bool mmap_prm_file = false,
           T damping_factor = 0.85, uint32_t max_iterations = 100,
           T epsilon = 1e-6) :
      damping_factor_(damping_factor), max_iterations_(max_iterations),
      epsilon_(epsilon) {
    if (MatrixMarketIo::HasMtxFileExtension(file_path))
      this->status_ = this->ReadMatrixMarketFile(file_path);
    else if (mmap_prm_file)
      this->status_ = this->MmapPieRankMatrixFile(file_path);
    else
      this->status_ = this->ReadPieRankMatrixFile(file_path);
    if (!this->status_.ok())
      return;
    CHECK_EQ(this->IndexDim(), 1);
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

    const auto ranges = this->SplitIndexDimByNnz(pool ? pool->Size() : 1);
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
      if (SumEpsilons() < epsilon_)
        break;
    }

    return std::make_pair(SumEpsilons(), std::min(iter + 1, max_iterations_));
  }

  // Returns an emtpy vector on error.
  PageScores TopK(PosType k = 100) {
    PageScores page_scores;
    if (!this->status_.ok()) return page_scores;

    InitPageScores(&page_scores, k);
    std::sort(page_scores.begin(), page_scores.end(),
        [](const PageScore &a, const PageScore &b){
      return a.second > b.second;
    });

    if (k < num_pages_) {
      for (auto i = k; i < num_pages_; ++i) {
        auto it = std::upper_bound(page_scores.begin(),
                                   page_scores.end(),
                                   scores_[i],
                                   [](T score, const PageScore &p) {
                                     return score > p.second;
                                   });
        if (it != page_scores.end()) {
          page_scores.back() = std::make_pair(i, scores_[i]);
          std::rotate(it, page_scores.end() - 1, page_scores.end());
        }
      }
    }

    return page_scores;
  }

  const std::vector<T> &Scores() const { return scores_; }

protected:
  void NumOutboundLinks() {
    DCHECK(this->status_.ok());
    for (PosType nz = 0; nz < this->NumNonZeros(); nz++) {
      out_degree_[this->Pos(nz)]++;
    }
  }

  void InitPageScores(
      PageScores *pairs,
      PosType max_pairs = std::numeric_limits<PosType>::max()) {
    DCHECK(this->status_.ok());
    DCHECK_EQ(scores_.size(), num_pages_);
    DCHECK_LE(scores_.size(), std::numeric_limits<PosType>::max());
    PosType size = (scores_.size() < max_pairs) ? scores_.size() : max_pairs;

    DCHECK(pairs->empty());
    pairs->clear();
    for (PosType i = 0; i < size; ++i) {
      pairs->push_back(std::make_pair(i, scores_[i]));
    }
  }

  void DoRange(const PosRange &range, uint32_t range_id) {
    DCHECK(this->status_.ok());
    auto first = std::get<0>(range);
    auto last = std::get<1>(range);
    DCHECK_LT(first, last);
    last = std::min(last, this->IndexPosEnd());

    T epsilon = 0.0;
    for (PosType p = first; p < last; ++p) {
      T sum = 0.0;

      if (this->PosIsCompressed()) {
        for (IdxType i = this->Index(p); i < this->Index(p + 1); ++i) {
          DCHECK_LT(i, this->NumNonZeros()) << p;
          DCHECK_LT(this->Pos(i), this->Rows());
          DCHECK_LT(this->Pos(i), scores_.size());
          DCHECK_GT(out_degree_[this->Pos(i)], 0);
          sum += scores_[this->Pos(i)] / out_degree_[this->Pos(i)];
        }
      } else {
        for (IdxType i = this->Index(p); i < this->Index(p + 1); ++i)
          sum += scores_[this->PosAt(i)] / out_degree_[this->PosAt(i)];
      }

      T score = one_minus_d_over_n_ + damping_factor_ * sum;
      epsilon += std::fabs(scores_[p] - score);
      scores_[p] = score;
    }

    epsilons_[range_id] = epsilon;
  }

  T SumEpsilons() const {
    DCHECK(this->status_.ok());
    return std::accumulate(epsilons_.begin(), epsilons_.end(), 0.0);
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
