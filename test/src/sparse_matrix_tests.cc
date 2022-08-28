//
// Created by Michelle Zhou on 2/26/22.
//

#include "gtest/gtest.h"
#include "pierank/pierank.h"
#include "status_matcher.h"

using namespace std;

using namespace pierank;

TEST(SparseMatrix, Read) {
  SparseMatrix<uint32_t, uint64_t> mat;
  EXPECT_OK(mat.ReadMatrixMarketFile("../../data/ash219.mtx"));
  EXPECT_EQ(mat.NumNonZeros(), 438);
  auto& index = mat.Index();
  auto& pos = mat.Pos();

  // test the list of non-zero rows for 1st column
  for (uint64_t idx = 0; idx < index[1]; ++idx)
    EXPECT_EQ(pos[index[0] + idx], idx);

  // test the list of non-zero rows for the last column (85th)
  vector<uint32_t> row_ids = {164, 171, 218};
  for (uint64_t idx = index[84]; idx < index[85]; ++idx)
    EXPECT_EQ(pos[idx], row_ids[idx - index[84]]);

  EXPECT_OK(mat.WritePrmFile("ash219.prm"));
  EXPECT_OK(mat.ReadPrmFile("ash219.prm"));
  SparseMatrix<uint32_t, uint64_t> mat2("ash219.prm");
  EXPECT_EQ(mat2.NumNonZeros(), 438);
  auto& index2 = mat2.Index();
  auto& pos2 = mat2.Pos();

  // std::cout << mat2.DebugString();
  // test the list of non-zero rows for 1st column
  for (uint64_t idx = 0; idx < index2[1]; ++idx)
    EXPECT_EQ(pos2[index2[0] + idx], idx);

  // test the list of non-zero rows for the last column (85th)
  for (uint64_t idx = index2[84]; idx < index2[85]; ++idx)
    EXPECT_EQ(pos2[idx], row_ids[idx - index2[84]]);
}