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

auto Transpose(const SparseMatrix <uint32_t, uint64_t> &mat) {
  constexpr uint32_t kMaxThreads = 4;
  constexpr uint64_t kMaxNnzPerRange = 32;
  auto pool = std::make_shared<ThreadPool>(kMaxThreads);
  auto mat_inverse_or = mat.ChangeIndexDim(pool, kMaxNnzPerRange);
  EXPECT_OK(mat_inverse_or);
  return std::move(mat_inverse_or).value();

}

TEST(SparseMatrixTests, ReadAsh219MtxFile) {
  auto file_path = TestDataFilePath("ash219.mtx");
  CHECK(MatrixMarketIo::HasMtxFileExtension(file_path));
  SparseMatrix<uint32_t, uint64_t> mat;
  EXPECT_OK(mat.ReadMatrixMarketFile(file_path));
  EXPECT_EQ(mat.Index().ItemSize(), 8);
  EXPECT_EQ(mat.Pos().ItemSize(), 4);
  CheckAsh219ColIndex(mat);

  auto prm_path = MatrixMarketToPieRankMatrixPath(file_path);
  if (kGeneratePieRankMatrixFile) {
    EXPECT_OK(mat.WritePieRankMatrixFile(prm_path));
    auto mat_inverse = Transpose(mat);
    std::string inverse_prm_path = PieRankMatrixPathAfterIndexChange(prm_path);
    EXPECT_OK(mat_inverse->WritePieRankMatrixFile(inverse_prm_path));
  }
  SparseMatrix<uint32_t, uint64_t> mat0;
  EXPECT_OK(mat0.ReadPieRankMatrixFile(prm_path));
  EXPECT_EQ(mat, mat0);
}

class PieRankMatrixTestFixture : public ::testing::TestWithParam<std::string> {
protected:
  void Run(const std::string &file_path) {
    SparseMatrix<uint32_t, uint64_t> mat;
    uint32_t index_dim = IndexDimInPieRankMatrixPath(file_path);
    EXPECT_OK(mat.ReadPieRankMatrixFile(file_path));
    EXPECT_EQ(mat.Index().ItemSize(), 1);
    EXPECT_EQ(mat.Pos().ItemSize(), 1);
    EXPECT_EQ(mat.IndexDim(), index_dim);
    CHECK_LT(index_dim, 2);
    if (index_dim) CheckAsh219ColIndex(mat);
    else CheckAsh219RowIndex(mat);

    SparseMatrix<uint32_t, uint64_t> mat_mmap(file_path, /*mmap=*/true);
    EXPECT_OK(mat_mmap.MmapPieRankMatrixFile(file_path));
    EXPECT_TRUE(mat_mmap.ok());
    if (index_dim) CheckAsh219ColIndex(mat_mmap);
    else CheckAsh219RowIndex(mat_mmap);

    SparseMatrix<uint32_t, uint64_t> mat_inverse0;
    std::string inverse_prm_file = PieRankMatrixPathAfterIndexChange(file_path);
    EXPECT_OK(mat_inverse0.ReadPieRankMatrixFile(inverse_prm_file));

    auto mat_inverse = Transpose(mat);
    if (index_dim) CheckAsh219RowIndex(*mat_inverse);
    else CheckAsh219ColIndex(*mat_inverse);
    EXPECT_EQ(*mat_inverse, mat_inverse0);

    auto tmp_dir = MakeTmpDir("./");
    CHECK(!tmp_dir.empty());
    auto tmp_path = absl::StrCat(tmp_dir, kPathSeparator,
                                 FileNameInPath(inverse_prm_file));
    constexpr uint64_t kMaxNnzPerRange = 32;
    auto mat_inverse_mmap_or = mat.ChangeIndexDim(tmp_path, kMaxNnzPerRange);
    EXPECT_OK(mat_inverse_mmap_or);
    auto mat_inverse_mmap = std::move(mat_inverse_mmap_or).value();
    if (index_dim) CheckAsh219RowIndex(*mat_inverse_mmap);
    else CheckAsh219ColIndex(*mat_inverse_mmap);
    EXPECT_EQ(*mat_inverse_mmap, mat_inverse0);

    SparseMatrix<uint32_t, uint64_t> mat_inverse2(tmp_path);
    EXPECT_TRUE(mat_inverse2.ok());
    EXPECT_EQ(mat_inverse2, mat_inverse0);

    std::remove(tmp_path.c_str());
    rmdir(tmp_dir.c_str());
  }
};

TEST_P(PieRankMatrixTestFixture, RankAsh219) {
  std::string file_path = GetParam();
  Run(file_path);
}

INSTANTIATE_TEST_SUITE_P(SparseMatrixTests, PieRankMatrixTestFixture,
    ::testing::Values(TestDataFilePath("ash219.i0.prm"),
                      TestDataFilePath("ash219.i1.prm"))
);

TEST(SparseMatrixTests, ReadB1ssMtxFile) {
  auto file_path = TestDataFilePath("b1_ss.mtx");
  CHECK(MatrixMarketIo::HasMtxFileExtension(file_path));
  SparseMatrix<uint32_t, uint64_t> mat;
  EXPECT_OK(mat.ReadMatrixMarketFile(file_path));

  EXPECT_EQ(mat(0, 0), 0);
  EXPECT_DOUBLE_EQ(mat(4, 0), -0.03599942);
  EXPECT_DOUBLE_EQ(mat(5, 0), -0.0176371);
  EXPECT_DOUBLE_EQ(mat(6, 0), -0.007721779);
  EXPECT_DOUBLE_EQ(mat(0, 1), 1);
  EXPECT_DOUBLE_EQ(mat(1, 1), -1);
  EXPECT_DOUBLE_EQ(mat(6, 6), 1);
  auto prm_path = MatrixMarketToPieRankMatrixPath(file_path);
  if (kGeneratePieRankMatrixFile) {
    EXPECT_OK(mat.WritePieRankMatrixFile(prm_path));
  }
  SparseMatrix<uint32_t, uint64_t> mat0;
  EXPECT_OK(mat0.ReadPieRankMatrixFile(prm_path));
  EXPECT_EQ(mat, mat0);
}

TEST(SparseMatrixTests, ReadBcsstm01MtxFile) {
  auto file_path = TestDataFilePath("bcsstm01.mtx");
  CHECK(MatrixMarketIo::HasMtxFileExtension(file_path));
  SparseMatrix<uint32_t, uint64_t> mat;
  EXPECT_OK(mat.ReadMatrixMarketFile(file_path));

  EXPECT_EQ(mat(0, 1), 0);
  EXPECT_EQ(mat(0, 0), 100);
  EXPECT_EQ(mat(10, 10), 0);
  EXPECT_EQ(mat(25, 25), 200);
  EXPECT_EQ(mat(32, 32), 200);
  EXPECT_EQ(mat(41, 41), 0);
  EXPECT_EQ(mat(42, 42), 200);
  EXPECT_EQ(mat(47, 47), 0);
  auto prm_path = MatrixMarketToPieRankMatrixPath(file_path);
  if (kGeneratePieRankMatrixFile) {
    EXPECT_OK(mat.WritePieRankMatrixFile(prm_path));
  }
  SparseMatrix<uint32_t, uint64_t> mat0;
  EXPECT_OK(mat0.ReadPieRankMatrixFile(prm_path));
  EXPECT_EQ(mat, mat0);
}

TEST(SparseMatrixTests, ReadBcsstm02MtxFile) {
  auto file_path = TestDataFilePath("bcsstm02.mtx");
  CHECK(MatrixMarketIo::HasMtxFileExtension(file_path));
  SparseMatrix<uint32_t, uint64_t> mat;
  EXPECT_OK(mat.ReadMatrixMarketFile(file_path));

  EXPECT_EQ(mat(0, 1), 0);
  EXPECT_EQ(mat(0, 0), 0.09213858051);
  EXPECT_EQ(mat(3, 3), 0.137995737983);
  EXPECT_EQ(mat(9, 9), 0.09213858051);
  EXPECT_EQ(mat(12, 12), 0.172828573455);
  EXPECT_EQ(mat(15, 15), 0.0852383576022);
  EXPECT_EQ(mat(22, 22), 0.172828573455);
  EXPECT_EQ(mat(25, 25), 0.0617332189107);
  EXPECT_EQ(mat(65, 65), 0.019746938665);
  EXPECT_EQ(mat(65, 65), 0.019746938665);
  auto prm_path = MatrixMarketToPieRankMatrixPath(file_path);
  if (kGeneratePieRankMatrixFile) {
    EXPECT_OK(mat.WritePieRankMatrixFile(prm_path));
  }
  SparseMatrix<uint32_t, uint64_t> mat0;
  EXPECT_OK(mat0.ReadPieRankMatrixFile(prm_path));
  EXPECT_EQ(mat, mat0);
}

TEST(SparseMatrixTests, ReadCan24MtxFile) {
  auto file_path = TestDataFilePath("can_24.mtx");
  CHECK(MatrixMarketIo::HasMtxFileExtension(file_path));
  SparseMatrix<uint32_t, uint64_t> mat;
  EXPECT_OK(mat.ReadMatrixMarketFile(file_path));

  EXPECT_EQ(mat(1, 0), 0);
  EXPECT_EQ(mat(0, 0), 1);
  EXPECT_EQ(mat(6, 0), 1);
  EXPECT_EQ(mat(10, 0), 0);
  EXPECT_EQ(mat(100, 0), 0);
  EXPECT_EQ(mat(17, 1), 1);
  EXPECT_EQ(mat(22, 2), 1);
  EXPECT_EQ(mat(22, 3), 0);
  EXPECT_EQ(mat(23, 3), 0);
  EXPECT_EQ(mat(3, 3), 1);
  EXPECT_EQ(mat(18, 3), 1);
  EXPECT_EQ(mat(22, 22), 1);
  EXPECT_EQ(mat(23, 23), 1);
  EXPECT_EQ(mat(24, 24), 0);
  auto prm_path = MatrixMarketToPieRankMatrixPath(file_path);
  if (kGeneratePieRankMatrixFile) {
    EXPECT_OK(mat.WritePieRankMatrixFile(prm_path));
  }
  SparseMatrix<uint32_t, uint64_t> mat0;
  EXPECT_OK(mat0.ReadPieRankMatrixFile(prm_path));
  EXPECT_EQ(mat, mat0);
}

TEST(SparseMatrixTests, ReadDwg961aMtxFile) {
  auto file_path = TestDataFilePath("dwg961a.mtx");
  CHECK(MatrixMarketIo::HasMtxFileExtension(file_path));
  using ValueContainer = std::vector<std::complex<double>>;
  SparseMatrix<uint32_t, uint64_t, ValueContainer> mat;
  EXPECT_OK(mat.ReadMatrixMarketFile(file_path));

  EXPECT_EQ(mat(0, 1), std::complex(0.0));
  EXPECT_EQ(mat(0, 0), std::complex(-49.1537, 4797.68));
  EXPECT_EQ(mat(1, 0), std::complex(-24.0559, 992.499));
  EXPECT_EQ(mat(4, 1), std::complex(-24.0559, 992.5));
  EXPECT_EQ(mat(62, 3), std::complex(-12.5474, 1902.7));
  EXPECT_EQ(mat(8, 7), std::complex(-24.0572, 992.373));
  EXPECT_EQ(mat(12, 12), std::complex(35891.2, 4797.69));
  EXPECT_EQ(mat(48, 47), std::complex(-36022.7, 0.0));
  EXPECT_EQ(mat(94, 47), std::complex(-36022.6, 0.0));
  EXPECT_EQ(mat(95, 47), std::complex(35986.0, 0.0));
  EXPECT_EQ(mat(48, 48), std::complex(35891.2, 4797.7));
  EXPECT_EQ(mat(49, 48), std::complex(-24.0559, 992.508));
  EXPECT_EQ(mat(108, 63), std::complex(-12.5475, 1902.7));
  EXPECT_EQ(mat(69, 69), std::complex(-25.1009, 3804.95));
  EXPECT_EQ(mat(703, 703), std::complex(-42.4361, 0.24016));
  EXPECT_EQ(mat(704, 703), std::complex(-9.34981, -0.234424));
  EXPECT_EQ(mat(704, 704), std::complex(-54.292, 0.0881584));
  EXPECT_EQ(mat(705, 705), std::complex(0.0));
  auto prm_path = MatrixMarketToPieRankMatrixPath(file_path);
  if (kGeneratePieRankMatrixFile) {
    EXPECT_OK(mat.WritePieRankMatrixFile(prm_path));
  }
  SparseMatrix<uint32_t, uint64_t, ValueContainer> mat0;
  EXPECT_OK(mat0.ReadPieRankMatrixFile(prm_path));
  EXPECT_EQ(mat, mat0);
}
