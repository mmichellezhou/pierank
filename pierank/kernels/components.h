//
// Created by Michelle Zhou on 11/27/22.
//

#ifndef PIERANK_KERNELS_COMPONENTS_H_
#define PIERANK_KERNELS_COMPONENTS_H_

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
    labels_.resize(num_nodes_);
  }

  // Returns <num_iterations, converged> pair or <0, false> on error
  std::tuple<uint32_t, bool> Run(std::shared_ptr <ThreadPool> pool = nullptr) {
    CHECK_EQ(labels_.size(), num_nodes_);
    if (!this->status_.ok())
      return std::make_pair(0, false);

    const auto ranges = this->SplitIndexDimByNnz(pool ? pool->Size() : 1);
    num_propagations_.resize(ranges.size());

    for (PosType n = 0; n < num_nodes_; ++n)
      labels_[n] = n;

    bool converged = false;
    uint32_t iter;
    for (iter = 0; iter < max_iterations_ && !converged; ++iter) {
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
      if (SumNumPropagations() == 0)
        converged = true;
    }

    return std::make_pair(std::min(iter + 1, max_iterations_), converged);
  }

  const std::vector<PosType> &Labels() const { return labels_; }

protected:
  void DoRange(const PosRange &range, uint32_t range_id) {
    DCHECK(this->status_.ok());
    auto first = std::get<0>(range);
    auto last = std::get<1>(range);
    DCHECK_LT(first, last);
    last = std::min(last, this->IndexPosEnd());

    PosType num_props = 0;
    for (PosType p = first; p < last; ++p) {
      for (IdxType i = this->Index(p); i < this->Index(p + 1); ++i) {
        if (labels_[p] < labels_[i]) {
          labels_[i] = labels_[p];
          ++num_props;
        }
        else if (labels_[p] > labels_[i]) {
          labels_[p] = labels_[i];
          ++num_props;
        }
      }
    }

    num_propagations_[range_id] = num_props;
  }

  PosType SumNumPropagations() const {
    DCHECK(this->status_.ok());
    return std::accumulate(num_propagations_.begin(), num_propagations_.end(),
                           0);
  }

private:
  uint64_t num_nodes_;
  std::vector<PosType> labels_;
  std::vector<PosType> num_propagations_;
  uint32_t max_iterations_;
};

}  // namespace pierank

#endif //PIERANK_KERNELS_COMPONENTS_H_
