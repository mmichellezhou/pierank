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

void CheckAsh219Common(const SparseMatrix<uint32_t, uint64_t> &mat) {
  EXPECT_EQ(mat.Rows(), 219);
  EXPECT_EQ(mat.Cols(), 85);
  EXPECT_EQ(mat.NumNonZeros(), 438);
  EXPECT_FALSE(mat.Index().ShiftByMinValue());
  EXPECT_EQ(mat.Index().MinValue(), 0);
  EXPECT_EQ(mat.Index().MaxValue(), 438);
  EXPECT_FALSE(mat.Pos().ShiftByMinValue());
  EXPECT_EQ(mat.Pos().MinValue(), 0);
}

void CheckAsh219ColIndex(const SparseMatrix<uint32_t, uint64_t> &mat) {
  CheckAsh219Common(mat);
  EXPECT_EQ(mat.IndexDim(), 1);
  EXPECT_EQ(mat.Pos().MaxValue(), 218);

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

void CheckAsh219RowIndex(const SparseMatrix<uint32_t, uint64_t> &mat) {
  CheckAsh219Common(mat);
  EXPECT_EQ(mat.IndexDim(), 0);
  EXPECT_EQ(mat.Pos().MaxValue(), 84);

  auto& index = mat.Index();
  auto& pos = mat.Pos();

  // test the list of non-zero columns for 1st row
  for (uint64_t idx = 0; idx < index[1]; ++idx)
    EXPECT_EQ(pos[index[0] + idx], idx);

  // test the list of non-zero columns for the last row (219th)
  vector<uint32_t> col_ids = {83, 84};
  for (uint64_t idx = index[218]; idx < index[219]; ++idx)
    EXPECT_EQ(pos[idx], col_ids[idx - index[218]]);
}

class SparseMatrixTestFixture : public ::testing::TestWithParam<std::string> {
protected:
  void Run(const std::string &file_path) {
    SparseMatrix<uint32_t, uint64_t> mat;
    if (MatrixMarketIo::HasMtxFileExtension(file_path)) {
      EXPECT_OK(mat.ReadMatrixMarketFile(file_path));
      EXPECT_EQ(mat.Index().ItemSize(), 8);
      EXPECT_EQ(mat.Pos().ItemSize(), 4);
      if (kGeneratePieRankMatrixFile)
        EXPECT_OK(mat.WritePieRankMatrixFile(file_path));
    }
    else {
      EXPECT_OK(mat.ReadPieRankMatrixFile(file_path));
      EXPECT_EQ(mat.Index().ItemSize(), 2);
      EXPECT_EQ(mat.Pos().ItemSize(), 1);
      SparseMatrix<uint32_t, uint64_t> mat_mmap(file_path, /*mmap=*/true);
      EXPECT_TRUE(mat_mmap.ok());
      CheckAsh219ColIndex(mat_mmap);
    }
    CheckAsh219ColIndex(mat);
    auto mat_by_row_or = mat.ChangeIndexDim();
    EXPECT_OK(mat_by_row_or);
    CheckAsh219RowIndex(*mat_by_row_or->get());
  }
};

TEST_P(SparseMatrixTestFixture, RankAsh219) {
  std::string file_path = GetParam();
  Run(file_path);
}

INSTANTIATE_TEST_SUITE_P(SparseMatrixTests, SparseMatrixTestFixture,
    ::testing::Values(TestDataFilePath("ash219.mtx"),
                      TestDataFilePath("ash219.prm"))
);
