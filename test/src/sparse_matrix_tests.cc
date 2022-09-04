//
// Created by Michelle Zhou on 2/26/22.
//

#include "gtest/gtest.h"
#include "pierank/pierank.h"
#include "status_matcher.h"

using namespace std;

using namespace pierank;

void CheckAsh219(const SparseMatrix<uint32_t, uint64_t> &mat) {
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
}

TEST(SparseMatrix, Read) {
  SparseMatrix<uint32_t, uint64_t> mat;
  EXPECT_OK(mat.ReadMatrixMarketFile("../../data/ash219.mtx"));
  CheckAsh219(mat);

  EXPECT_OK(mat.WritePieRankMatrixFile("ash219.prm"));
  EXPECT_OK(mat.ReadPieRankMatrixFile("ash219.prm"));
  SparseMatrix<uint32_t, uint64_t> mat2("ash219.prm");
  CheckAsh219(mat2);

  SparseMatrix<uint32_t, uint64_t> mat3("ash219.prm", /*mmap=*/true);
  CHECK(mat3.ok());
  CheckAsh219(mat3);
}