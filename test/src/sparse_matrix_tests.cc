//
// Created by Michelle Zhou on 2/26/22.
//

#include "sparse_matrix.h"
#include "gtest/gtest.h"

using namespace std;

TEST(SparseMatrix, Read) {
  SparseMatrix<uint32_t, uint64_t> mat("../../data/ash219.mtx");
  EXPECT_EQ(mat.NumNonZeros(), 438);
}