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
           T max_residual = 1e-6) :
      damping_factor_(damping_factor), max_iterations_(max_iterations),
      max_residual_(max_residual) {
    if (MatrixMarketIo::HasMtxFileExtension(file_path))
      this->status_ = this->ReadMatrixMarketFile(file_path);
    else if (mmap_prm_file)
      this->status_ = this->MmapPieRankMatrixFile(file_path);
    else
      this->status_ = this->ReadPieRankMatrixFile(file_path);
    if (!this->status_.ok())
      return;
    CHECK_EQ(this->IndexDim(), 1);
    num_pages_ = this->MaxDimSize();
    scores_[0].resize(num_pages_);
    if (!update_score_in_place_) scores_[1].resize(num_pages_);
    one_minus_d_over_n_ = (1 - damping_factor_) / num_pages_;
    NumOutboundLinks(); // accounts for 99% of the time taken by this Ctor.
  }

  // Returns <residual, num_iterations> pair or <+infinity, 0> on error
  std::pair<T, uint32_t> Run(std::shared_ptr <ThreadPool> pool = nullptr,
                             bool update_score_in_place = false) {
    update_score_in_place_ = update_score_in_place;
    uint32_t max_ranges = pool ? pool->Size() : 1;
    residual_ = std::numeric_limits<T>::max();
    // No need for score-update thread safety if there is only a single range.
    if (max_ranges == 1) update_score_in_place_ = true;
    if (!this->ProcessRanges(max_ranges, pool))
      return std::make_pair(std::numeric_limits<T>::max(), 0);
    auto &scores =
        update_score_in_place_ ? scores_[0] : scores_[num_iterations_ % 2];
    // Compute the original PageRank scores (without out-degree adjustment)
    for (PosType p = 0; p < num_pages_; ++p) {
      if (out_degree_[p] > 1)
        scores[p] *= out_degree_[p];
    }
    return std::make_pair(residual_, num_iterations_);
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

    const auto &score = Scores();
    if (k < num_pages_) {
      for (auto i = k; i < num_pages_; ++i) {
        auto it = std::upper_bound(page_scores.begin(),
                                   page_scores.end(),
                                   score[i],
                                   [](T score, const PageScore &p) {
                                     return score > p.second;
                                   });
        if (it != page_scores.end()) {
          page_scores.back() = std::make_pair(i, score[i]);
          std::rotate(it, page_scores.end() - 1, page_scores.end());
        }
      }
    }

    return page_scores;
  }

  T Residual() const { return residual_; }

  const std::vector<T> &Scores() const {
    return update_score_in_place_ ? scores_[0] : scores_[num_iterations_ % 2];
  }

protected:
  void NumOutboundLinks() {
    DCHECK(this->status_.ok());
    auto nnz = this->NumNonZeros();
    out_degree_.clear();
    DCHECK_EQ(num_pages_, this->MaxDimSize());
    out_degree_.resize(num_pages_);
    for (IdxType i = 0; i < nnz; ++i) {
      ++out_degree_[this->Pos(i)];
    }
  }

  void InitRanges(const PosRanges &ranges, uint32_t range_id) override {
    DCHECK_LT(range_id, ranges.size());
    if (range_id == 0) {
      num_iterations_ = 0;
      residuals_.clear();
      residuals_.resize(ranges.size());
    }
    const auto[min_pos, max_pos] = this->RangeMinMaxPos(ranges, range_id);
    T init_prob = 1.0 / static_cast<T>(num_pages_);
    for (PosType p = min_pos; p < max_pos; ++p) {
      T prob = out_degree_[p] > 1 ? init_prob / out_degree_[p] : init_prob;
      scores_[0][p] = prob;
      if (!update_score_in_place_) scores_[1][p] = prob;
    }
  }

  void UpdateRanges(const PosRanges &ranges, uint32_t range_id) override {
    DCHECK(this->status_.ok());
    const auto[min_pos, max_pos] = this->RangeMinMaxPos(ranges, range_id);
    T residual = 0.0;
    const auto &old_score = Scores();
    auto &new_score =
        update_score_in_place_ ? scores_[0] : scores_[1 - num_iterations_ % 2];
    for (PosType p = min_pos; p < max_pos; ++p) {
      T sum = 0.0;

      for (IdxType i = this->Index(p); i < this->Index(p + 1); ++i) {
        DCHECK_LT(i, this->NumNonZeros()) << p;
        auto pos = this->Pos(i);
        DCHECK_LT(pos, this->Rows());
        DCHECK_LT(pos, old_score.size());
        DCHECK_GT(out_degree_[pos], 0);
        sum += old_score[pos];
      }

      T score = one_minus_d_over_n_ + damping_factor_ * sum;
      if (out_degree_[p] > 1) {
        // Undo out-degree adjustment for old score before computing residual.
        residual += std::fabs(old_score[p] * out_degree_[p] - score);
        new_score[p] = score / out_degree_[p];
      } else {
        residual += std::fabs(old_score[p] - score);
        new_score[p] = score;
      }
    }

    residuals_[range_id] = residual;
  }

  void ReconcileRanges(const PosRanges &ranges, uint32_t range_id) override {
    if (range_id == 0) {
      ++num_iterations_;
      residual_ = std::accumulate(residuals_.begin(), residuals_.end(), 0.0);
    }
  }

  bool Stop() const override {
    return residual_ <= max_residual_ || num_iterations_ >= max_iterations_;
  }

  void InitPageScores(PageScores *pairs,
                      PosType max_pairs = std::numeric_limits<PosType>::max()) {
    DCHECK(this->status_.ok());
    const auto &score = Scores();
    DCHECK_EQ(score.size(), num_pages_);
    DCHECK_LE(score.size(), std::numeric_limits<PosType>::max());
    PosType size = (score.size() < max_pairs) ? score.size() : max_pairs;

    DCHECK(pairs->empty());
    pairs->clear();
    for (PosType i = 0; i < size; ++i) {
      pairs->push_back(std::make_pair(i, score[i]));
    }
  }

private:
  T damping_factor_;
  uint64_t num_pages_;
  T one_minus_d_over_n_;
  bool update_score_in_place_ = false;  // not thread safe but saves RAM
  std::array<std::vector<T>, 2> scores_;
  std::vector<T> residuals_;
  std::vector<uint32_t> out_degree_;
  uint32_t num_iterations_;
  uint32_t max_iterations_;
  T residual_ = std::numeric_limits<T>::max();
  T max_residual_;
};

}  // namespace pierank

#endif //PIERANK_KERNELS_PAGERANK_H_
