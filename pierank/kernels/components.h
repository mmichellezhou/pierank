//
// Created by Michelle Zhou on 11/27/22.
//

#ifndef PIERANK_KERNELS_COMPONENTS_H_
#define PIERANK_KERNELS_COMPONENTS_H_

#include <algorithm>

#include "absl/status/status.h"
#include "pierank/sparse_matrix.h"
#include "pierank/thread_pool.h"

namespace pierank {

// Computes weakly connected components of a graph.
template<typename PosType = uint32_t, typename IdxType = uint64_t>
class ConnectedComponents : public SparseMatrix<PosType, IdxType> {
public:
  using PosRange = typename SparseMatrix<PosType, IdxType>::PosRange;

  using PosRanges = typename SparseMatrix<PosType, IdxType>::PosRanges;

  ConnectedComponents(const std::string &file_path, bool mmap_prm_file = false,
                      uint32_t max_iterations = 100) :
      max_iterations_(max_iterations) {
    if (MatrixMarketIo::HasMtxFileExtension(file_path))
      this->status_ = this->ReadMatrixMarketFile(file_path);
    else if (mmap_prm_file)
      this->status_ = this->MmapPieRankMatrixFile(file_path);
    else
      this->status_ = this->ReadPieRankMatrixFile(file_path);
    if (!this->status_.ok())
      return;
    num_nodes_ = std::max(this->Rows(), this->Cols());
  }

  // Returns <num_iterations, converged> pair or <0, false> on error
  std::tuple<uint32_t, bool> Run(std::shared_ptr<ThreadPool> pool = nullptr) {
    const auto ranges = this->SplitIndexDimByNnz(pool ? pool->Size() : 1);
    propagations_.resize(ranges.size());
    labels_.resize(ranges.size());
    for (auto &labels : labels_)
      labels.resize(num_nodes_);

    if (!this->ProcessRanges(ranges, pool))
      return std::make_pair(0, false);
    bool converged = (num_propagations_ == 0);
    return std::make_pair(num_iterations_, converged);
  }

  const std::vector<PosType> &Labels() const { return labels_[0]; }

  IdxType NumPropagations() const {
    DCHECK(this->status_.ok());
    return num_propagations_;
  }

  PosType NumComponents() const {
    auto labels = labels_[0];
    std::sort(labels.begin(), labels.end());
    auto it = std::unique(labels.begin(), labels.end());
    return std::distance(labels.begin(), it);
  }

protected:
  void InitRanges(const PosRanges &ranges, uint32_t range_id) override {
    if (range_id == 0) {
      num_iterations_ = 0;
      num_propagations_ = std::numeric_limits<IdxType>::max();
      std::fill(propagations_.begin(), propagations_.end(), num_propagations_);
    }
    DCHECK_LT(range_id, ranges.size());
    auto &labels = labels_[range_id];
    for (PosType n = 0; n < num_nodes_; ++n)
      labels[n] = n;
  }

  void UpdateRanges(const PosRanges &ranges, uint32_t range_id) override {
    DCHECK(this->status_.ok());
    DCHECK_LT(range_id, ranges.size());
    const auto &range = ranges[range_id];
    auto first = std::get<0>(range);
    auto last = std::get<1>(range);
    DCHECK_LT(first, last);
    last = std::min(last, this->IndexPosEnd());

    PosType num_props = 0;
    auto &labels = labels_[range_id];
    for (PosType p = first; p < last; ++p) {
      for (IdxType i = this->Index(p); i < this->Index(p + 1); ++i) {
        auto pos = this->Pos(i);
        if (labels[p] < labels[pos]) {
          labels[pos] = labels[p];
          ++num_props;
        }
        else if (labels[p] > labels[pos]) {
          labels[p] = labels[pos];
          ++num_props;
        }
      }
    }
    propagations_[range_id] = num_props;
  }

  void ReconcileRanges(const PosRanges &ranges, uint32_t range_id) override {
    DCHECK_LT(range_id, ranges.size());
    if (range_id == 0) {
      ++num_iterations_;
      num_propagations_ = std::accumulate(propagations_.begin(),
                                          propagations_.end(), 0);
    }
    if (ranges.size() == 1) return;

    const auto &range = ranges[range_id];
    auto first = std::get<0>(range);
    auto last = std::get<1>(range);
    DCHECK_LT(first, last);
    last = std::min(last, this->IndexPosEnd());

    auto num_ranges = ranges.size();
    auto &labels0 = labels_[0];
    for (PosType p = first; p < last; ++p) {
      PosType min_label = labels0[p];
      uint32_t min_label_range = 0;
      bool back_prop = false;
      for (uint32_t r = 1; r < num_ranges; ++r) {
        PosType label = labels_[r][p];
        if (min_label > label) {
          min_label = label;
          back_prop = true;
          min_label_range = r;
        } else if (min_label < label)
          labels_[r][p] = min_label;
      }
      for (uint32_t r = 0; back_prop && r < min_label_range; ++r)
        labels_[r][p] = min_label;
    }
  }

  bool Stop() const override {
    return num_propagations_ == 0 || num_iterations_ >= max_iterations_;
  }

private:
  uint64_t num_nodes_;
  std::vector<std::vector<PosType>> labels_;  // # labels per thread: num_nodes_
  std::vector<IdxType> propagations_;
  IdxType num_propagations_;
  uint32_t num_iterations_;
  uint32_t max_iterations_;
};

}  // namespace pierank

#endif //PIERANK_KERNELS_COMPONENTS_H_
