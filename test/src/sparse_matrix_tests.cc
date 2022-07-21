//
// Created by Michelle Zhou on 2/26/22.
//

#include "sparse_matrix.h"
#include "gtest/gtest.h"

using namespace std;

TEST(SparseMatrix, Read) {
  SparseMatrix<uint32_t, uint64_t> mat("../../data/ash219.mtx");
  EXPECT_EQ(mat.NumNonZeros(), 438);
  auto index = mat.Index();
  auto pos = mat.Pos();

  // test the list of non-zero rows for 1st column
  for (uint64_t idx = 0; idx < index[1]; ++idx)
    EXPECT_EQ(pos[index[0] + idx], idx);

  // test the list of non-zero rows for the last column (85th)
  vector<uint32_t> row_ids = {164, 171, 218};
  for (uint64_t idx = index[84]; idx < index[85]; ++idx)
    EXPECT_EQ(pos[idx], row_ids[idx - index[84]]);
}