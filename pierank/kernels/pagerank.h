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
    num_pages_ = std::max(this->Rows(), this->Cols());
    one_minus_d_over_n_ = (1 - damping_factor_) / num_pages_;
    out_degree_.resize(this->Rows());
    NumOutboundLinks();
  }

  // Returns <residual, num_iterations> pair or <+infinity, 0> on error
  std::pair<T, uint32_t> Run(std::shared_ptr<ThreadPool> pool = nullptr) {
    const auto ranges = this->SplitIndexDimByNnz(pool ? pool->Size() : 1);
    residuals_.resize(ranges.size(), std::numeric_limits<T>::max());

    if (!this->ProcessRanges(ranges, pool))
      return std::make_pair(std::numeric_limits<T>::max(), 0);
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

  T Residual() const { return residual_; }

  const std::vector<T> &Scores() const { return scores_; }

protected:
  void NumOutboundLinks() {
    DCHECK(this->status_.ok());
    auto nnz = this->NumNonZeros();
    for (IdxType i = 0; i < nnz; ++i) {
      ++out_degree_[this->Pos(i)];
    }
  }

  void InitRanges(const PosRanges &ranges, uint32_t range_id) override {
    DCHECK_LT(range_id, ranges.size());
    if (range_id == 0) {
      num_iterations_ = 0;
      residual_ = std::numeric_limits<T>::max();
      std::fill(residuals_.begin(), residuals_.end(), residual_);
      scores_.resize(num_pages_);
      std::fill(scores_.begin(), scores_.end(), one_minus_d_over_n_);
    }
  }

  void UpdateRanges(const PosRanges &ranges, uint32_t range_id) override {
    DCHECK(this->status_.ok());
    auto[min_pos, max_pos] = this->RangeMinMaxPos(ranges, range_id);
    T residual = 0.0;
    for (PosType p = min_pos; p < max_pos; ++p) {
      T sum = 0.0;

      for (IdxType i = this->Index(p); i < this->Index(p + 1); ++i) {
        DCHECK_LT(i, this->NumNonZeros()) << p;
        auto pos = this->Pos(i);
        DCHECK_LT(pos, this->Rows());
        DCHECK_LT(pos, scores_.size());
        DCHECK_GT(out_degree_[pos], 0);
        sum += scores_[pos] / out_degree_[pos];
      }

      T score = one_minus_d_over_n_ + damping_factor_ * sum;
      residual += std::fabs(scores_[p] - score);
      scores_[p] = score;
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

private:
  T damping_factor_;
  uint64_t num_pages_;
  T one_minus_d_over_n_;
  std::vector<T> scores_;
  std::vector<T> residuals_;
  std::vector<uint32_t> out_degree_;
  uint32_t num_iterations_;
  uint32_t max_iterations_;
  T residual_;
  T max_residual_;
};

}  // namespace pierank

#endif //PIERANK_KERNELS_PAGERANK_H_
