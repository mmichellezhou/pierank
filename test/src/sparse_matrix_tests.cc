//
// Created by Michelle Zhou on 2/26/22.
//

#include "gtest/gtest.h"
#include "pierank/pierank.h"
#include "status_matcher.h"
#include "test_utils.h"

using namespace std;

using namespace pierank;

static bool kGeneratePieRankMatrixFile = false;

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
  auto mtx_path = TestDataFilePath("ash219.mtx");
  EXPECT_OK(mat.ReadMatrixMarketFile(mtx_path));
  CheckAsh219(mat);

  auto prm_path = TestDataFilePath("ash219.prm");
  if (kGeneratePieRankMatrixFile)
    EXPECT_OK(mat.WritePieRankMatrixFile(prm_path));
  EXPECT_OK(mat.ReadPieRankMatrixFile(prm_path));
  SparseMatrix<uint32_t, uint64_t> mat2(prm_path);
  CheckAsh219(mat2);

  SparseMatrix<uint32_t, uint64_t> mat3(prm_path, /*mmap=*/true);
  CHECK(mat3.ok());
  CheckAsh219(mat3);
}