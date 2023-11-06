//
// Created by Michelle Zhou on 2/26/22.
//

#include <set>
#include <vector>

#include "absl/strings/str_replace.h"
#include "gtest/gtest.h"
#include "pierank/pierank.h"
#include "status_matcher.h"
#include "test_utils.h"

using namespace std;

using namespace pierank;

static constexpr bool kGeneratePieRankMatrixFile = false;

template<typename M, typename V>
void CheckSparseMatrix(const M &mat,
                       const vector<pair<vector<uint32_t>, V>> &entries) {
  for (auto && [pos, value] : entries)
    EXPECT_EQ(mat(pos), value);
}

template<typename M, typename V>
void
CheckSparseMatrix(const M &mat,
                  const vector<pair<vector<uint32_t>, vector<V>>> &entries) {
  uint32_t depths = mat.Depths();
  for (auto && [pos, values] : entries) {
    EXPECT_EQ(values.size(), depths);
    for (size_t d = 0; d < values.size(); ++d)
      EXPECT_EQ(mat(pos, d), values[d]);
  }
}

template<typename M, typename V>
void
CheckDenseMatrix(const M &mat,
                 const vector<pair<vector<uint32_t>, vector<V>>> &entries) {
  set<vector<uint32_t>> checked;
  uint32_t depths = mat.Depths();
  for (auto && [pos, values] : entries) {
    EXPECT_EQ(values.size(), depths);
    checked.insert(pos);
    for (size_t d = 0; d < values.size(); ++d)
      EXPECT_EQ(mat(pos, d), values[d]);
  }

  const uint64_t elem_stride = mat.ElemStride();
  for (uint64_t i = 0; i < mat.Elems(); ++i) {
    auto && [pos, depth] = mat.IdxToPosAndDepth(i * elem_stride);
    if (checked.find(pos) != checked.end()) continue;
    for (size_t d = 0; d < depths; ++d)
      EXPECT_EQ(mat(pos, d), 0);
  }
}

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
    EXPECT_OK(mat_inverse.WritePieRankMatrixFile(inverse_prm_path));
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
    if (index_dim) CheckAsh219RowIndex(mat_inverse);
    else CheckAsh219ColIndex(mat_inverse);
    EXPECT_EQ(mat_inverse, mat_inverse0);

    auto tmp_dir = MakeTmpDir("./");
    CHECK(!tmp_dir.empty());
    auto tmp_path = absl::StrCat(tmp_dir, kPathSeparator,
                                 FileNameInPath(inverse_prm_file));
    constexpr uint64_t kMaxNnzPerRange = 32;
    auto mat_inverse_mmap_or = mat.ChangeIndexDim(tmp_path, kMaxNnzPerRange);
    EXPECT_OK(mat_inverse_mmap_or);
    auto mat_inverse_mmap = std::move(mat_inverse_mmap_or).value();
    if (index_dim) CheckAsh219RowIndex(mat_inverse_mmap);
    else CheckAsh219ColIndex(mat_inverse_mmap);
    EXPECT_EQ(mat_inverse_mmap, mat_inverse0);

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

vector<pair<vector<uint32_t>, double>> B1ssMatrixTestEntries() {
  return { {{0, 0}, 0}, {{4, 0}, -0.03599942}, {{5, 0}, -0.0176371},
           {{6, 0}, -0.007721779}, {{0, 1}, 1}, {{1, 1}, -1}, {{6, 6}, 1} };
}

TEST(SparseMatrixTests, ReadB1ssMtxFile) {
  auto file_path = TestDataFilePath("b1_ss.mtx");
  CHECK(MatrixMarketIo::HasMtxFileExtension(file_path));
  SparseMatrix<uint32_t, uint64_t> mat;
  EXPECT_OK(mat.ReadMatrixMarketFile(file_path));

  CheckSparseMatrix(mat, B1ssMatrixTestEntries());
  auto prm_path = MatrixMarketToPieRankMatrixPath(file_path);
  if (kGeneratePieRankMatrixFile) {
    EXPECT_OK(mat.WritePieRankMatrixFile(prm_path));
  }
  SparseMatrix<uint32_t, uint64_t> mat0;
  EXPECT_OK(mat0.ReadPieRankMatrixFile(prm_path));
  EXPECT_EQ(mat, mat0);
}

vector<pair<vector<uint32_t>, double>> Bcsstm01MatrixTestEntries() {
  return { {{0, 1}, 0}, {{0, 0}, 100}, {{10, 10}, 0}, {{25, 25}, 200},
           {{32, 32}, 200}, {{41, 41}, 0}, {{42, 42}, 200}, {{47, 47}, 0} };
}

TEST(SparseMatrixTests, ReadBcsstm01MtxFile) {
  auto file_path = TestDataFilePath("bcsstm01.mtx");
  CHECK(MatrixMarketIo::HasMtxFileExtension(file_path));
  SparseMatrix<uint32_t, uint64_t> mat;
  EXPECT_OK(mat.ReadMatrixMarketFile(file_path));

  CheckSparseMatrix(mat, Bcsstm01MatrixTestEntries());
  auto prm_path = MatrixMarketToPieRankMatrixPath(file_path);
  if (kGeneratePieRankMatrixFile) {
    EXPECT_OK(mat.WritePieRankMatrixFile(prm_path));
  }
  SparseMatrix<uint32_t, uint64_t> mat0;
  EXPECT_OK(mat0.ReadPieRankMatrixFile(prm_path));
  EXPECT_EQ(mat, mat0);
}

vector<pair<vector<uint32_t>, double>> Bcsstm02MatrixTestEntries() {
  return { {{0, 1}, 0}, {{0, 0}, 0.09213858051}, {{3, 3}, 0.137995737983},
           {{9, 9}, 0.09213858051}, {{12, 12}, 0.172828573455},
           {{15, 15}, 0.0852383576022}, {{22, 22}, 0.172828573455},
           {{25, 25}, 0.0617332189107}, {{65, 65}, 0.019746938665} };
}

TEST(SparseMatrixTests, ReadBcsstm02MtxFile) {
  auto file_path = TestDataFilePath("bcsstm02.mtx");
  CHECK(MatrixMarketIo::HasMtxFileExtension(file_path));
  SparseMatrix<uint32_t, uint64_t> mat;
  EXPECT_OK(mat.ReadMatrixMarketFile(file_path));

  CheckSparseMatrix(mat, Bcsstm02MatrixTestEntries());
  auto prm_path = MatrixMarketToPieRankMatrixPath(file_path);
  if (kGeneratePieRankMatrixFile) {
    EXPECT_OK(mat.WritePieRankMatrixFile(prm_path));
  }
  SparseMatrix<uint32_t, uint64_t> mat0;
  EXPECT_OK(mat0.ReadPieRankMatrixFile(prm_path));
  EXPECT_EQ(mat, mat0);
}

vector<pair<vector<uint32_t>, double>> Can24MatrixTestEntries() {
  return { {{1, 0}, 0}, {{0, 0}, 1}, {{6, 0}, 1}, {{10, 0}, 0},
           {{100, 0}, 0}, {{17, 1}, 1}, {{22, 2}, 1}, {{22, 3}, 0},
           {{23, 3}, 0}, {{3, 3}, 1}, {{18, 3}, 1}, {{22, 22}, 1},
           {{23, 23}, 1}, {{24, 24}, 0} };
}

TEST(SparseMatrixTests, ReadCan24MtxFile) {
  auto file_path = TestDataFilePath("can_24.mtx");
  CHECK(MatrixMarketIo::HasMtxFileExtension(file_path));
  SparseMatrix<uint32_t, uint64_t> mat;
  EXPECT_OK(mat.ReadMatrixMarketFile(file_path));

  CheckSparseMatrix(mat, Can24MatrixTestEntries());
  auto prm_path = MatrixMarketToPieRankMatrixPath(file_path);
  if (kGeneratePieRankMatrixFile) {
    EXPECT_OK(mat.WritePieRankMatrixFile(prm_path));
  }
  SparseMatrix<uint32_t, uint64_t> mat0;
  EXPECT_OK(mat0.ReadPieRankMatrixFile(prm_path));
  EXPECT_EQ(mat, mat0);
}

vector<pair<vector<uint32_t>, complex<double>>> Dwg961aMatrixTestEntries() {
  return { {{0, 1}, {0.0}}, {{0, 0}, {-49.1537, 4797.68}},
           {{1, 0}, {-24.0559, 992.499}}, {{4, 1}, {-24.0559, 992.5}},
           {{62, 3}, {-12.5474, 1902.7}}, {{8, 7}, {-24.0572, 992.373}},
           {{12, 12}, {35891.2, 4797.69}}, {{48, 47}, {-36022.7, 0.0}},
           {{94, 47}, {-36022.6, 0.0}}, {{95, 47}, {35986.0, 0.0}},
           {{48, 48}, {35891.2, 4797.7}}, {{49, 48}, {-24.0559, 992.508}},
           {{108, 63}, {-12.5475, 1902.7}}, {{69, 69}, {-25.1009, 3804.95}},
           {{703, 703}, {-42.4361, 0.24016}},
           {{704, 703}, {-9.34981, -0.234424}},
           {{704, 704}, {-54.292, 0.0881584}}, {{705, 705}, {0.0}} };
}

TEST(SparseMatrixTests, ReadDwg961aMtxFile) {
  auto file_path = TestDataFilePath("dwg961a.mtx");
  CHECK(MatrixMarketIo::HasMtxFileExtension(file_path));
  using ValueContainer = std::vector<std::complex<double>>;
  SparseMatrix<uint32_t, uint64_t, ValueContainer> mat;
  EXPECT_OK(mat.ReadMatrixMarketFile(file_path));

  CheckSparseMatrix(mat, Dwg961aMatrixTestEntries());
  auto prm_path = MatrixMarketToPieRankMatrixPath(file_path);
  if (kGeneratePieRankMatrixFile) {
    EXPECT_OK(mat.WritePieRankMatrixFile(prm_path));
  }
  SparseMatrix<uint32_t, uint64_t, ValueContainer> mat0;
  EXPECT_OK(mat0.ReadPieRankMatrixFile(prm_path));
  EXPECT_EQ(mat, mat0);
}

vector<pair<vector<uint32_t>, double>> FarmMatrixTestEntries() {
  return { {{3, 3}, 0}, {{0, 0}, 1}, {{1, 1}, 1}, {{1, 5}, 1}, {{2, 5}, 20},
           {{3, 6}, 40}, {{0, 11}, 250}, {{0, 14}, 125}, {{3, 14}, 10},
           {{2, 15}, 1}, {{3, 16}, 1}, {{4, 16}, 0} };
}

TEST(SparseMatrixTests, ReadFarmMtxFile) {
  auto file_path = TestDataFilePath("farm.mtx");
  CHECK(MatrixMarketIo::HasMtxFileExtension(file_path));
  SparseMatrix<uint32_t, uint64_t> mat;
  EXPECT_OK(mat.ReadMatrixMarketFile(file_path));

  CheckSparseMatrix(mat, FarmMatrixTestEntries());
  auto prm_path = MatrixMarketToPieRankMatrixPath(file_path);
  // Change name of matrix to "farmd", as "farm" is used by "flex" test below.
  prm_path = absl::StrReplaceAll(prm_path, {{"farm", "farmd"}});
  if (kGeneratePieRankMatrixFile) {
    EXPECT_OK(mat.WritePieRankMatrixFile(prm_path));
  }
  SparseMatrix<uint32_t, uint64_t> mat0;
  EXPECT_OK(mat0.ReadPieRankMatrixFile(prm_path));
  EXPECT_EQ(mat, mat0);
}

TEST(SparseMatrixTests, ReadFarmMtxFileFlex) {
  auto file_path = TestDataFilePath("farm.mtx");
  CHECK(MatrixMarketIo::HasMtxFileExtension(file_path));
  using ValueContainer = pierank::FlexArray<uint8_t>;
  SparseMatrix<uint32_t, uint64_t, ValueContainer> mat;
  EXPECT_OK(mat.ReadMatrixMarketFile(file_path));

  CheckSparseMatrix(mat, FarmMatrixTestEntries());
  auto prm_path = MatrixMarketToPieRankMatrixPath(file_path);
  if (kGeneratePieRankMatrixFile) {
    EXPECT_OK(mat.WritePieRankMatrixFile(prm_path));
  }
  SparseMatrix<uint32_t, uint64_t, ValueContainer> mat0;
  EXPECT_OK(mat0.ReadPieRankMatrixFile(prm_path));
  EXPECT_EQ(mat, mat0);
}

vector<pair<vector<uint32_t>, complex<float>>> Mhd1280bMatrixTestEntries() {
  return { {{0, 1}, {0.0f}}, {{0, 0}, {2.0f, 0.0f}},
           {{1, 1}, {0.252505826f, 0.0f}},
           {{3, 1}, {0.000144380768f, -1.11464849e-18f}},
           {{32, 1}, {0.101002561f, 0.0f}},
           {{39, 5}, {1.76503909e-8f, 2.42039935e-23f}},
           {{52, 27}, {4.84969212e-7f, 4.38176119e-10f}},
           {{1275, 1273}, {-5.2735779e-9f, 1.52774623e-23f}},
           {{1278, 1278}, {0.00153411731f, 0.0f}},
           {{1279, 1278}, {-4.21835721e-6f, 0.0f}},
           {{1279, 1279}, {1.49705588e-8f, 0.0f}}, {{1280, 1280}, {0.0f}} };
}

TEST(SparseMatrixTests, ReadMhd1280bMtxFile) {
  auto file_path = TestDataFilePath("mhd1280b.mtx");
  CHECK(MatrixMarketIo::HasMtxFileExtension(file_path));
  using ValueContainer = std::vector<std::complex<float>>;
  SparseMatrix<uint32_t, uint64_t, ValueContainer> mat;
  EXPECT_OK(mat.ReadMatrixMarketFile(file_path));

  CheckSparseMatrix(mat, Mhd1280bMatrixTestEntries());
  auto prm_path = MatrixMarketToPieRankMatrixPath(file_path);
  if (kGeneratePieRankMatrixFile) {
    EXPECT_OK(mat.WritePieRankMatrixFile(prm_path));
  }
  SparseMatrix<uint32_t, uint64_t, ValueContainer> mat0;
  EXPECT_OK(mat0.ReadPieRankMatrixFile(prm_path));
  EXPECT_EQ(mat, mat0);
}

vector<pair<vector<uint32_t>, float>> Plskz362MatrixTestEntries() {
  return { {{0, 0}, 0}, {{130, 0}, 0.1789438674667032f},
           {{246, 0}, 0.18205396002412488f},
           {{130, 1}, -0.1789438674667032f},
           {{268, 20}, 0.24528693972315807f},
           {{279, 42}, -0.2881268089697375f},
           {{195, 70}, 0.262563672245518f},
           {{218, 97}, -0.3188461370613087f},
           {{357, 125}, -0.28935985888075244f},
           {{323, 202}, -0.01902841376331865f},
           {{331, 218}, -0.020751353961338204f},
           {{359, 243}, -0.0293547623841256f},
           {{360, 244}, -0.02559133359087351f},
           {{361, 244}, -0.023915340143661593f},
           {{361, 245}, 0}, {{362, 244}, 0}, {{362, 245}, 0} };
}

TEST(SparseMatrixTests, ReadPlskz362MtxFile) {
  auto file_path = TestDataFilePath("plskz362.mtx");
  CHECK(MatrixMarketIo::HasMtxFileExtension(file_path));
  using ValueContainer = std::vector<float>;
  SparseMatrix<uint32_t, uint64_t, ValueContainer> mat;
  EXPECT_OK(mat.ReadMatrixMarketFile(file_path));

  CheckSparseMatrix(mat, Plskz362MatrixTestEntries());
  auto prm_path = MatrixMarketToPieRankMatrixPath(file_path);
  if (kGeneratePieRankMatrixFile) {
    EXPECT_OK(mat.WritePieRankMatrixFile(prm_path));
  }
  SparseMatrix<uint32_t, uint64_t, ValueContainer> mat0;
  EXPECT_OK(mat0.ReadPieRankMatrixFile(prm_path));
  EXPECT_EQ(mat, mat0);
}

vector<pair<vector<uint32_t>, complex<double>>> Young2cMatrixTestEntries() {
  return { {{841, 0}, {0.0}}, {{0, 841}, {0.0}}, {{2, 0}, {0.0}},
           {{0, 0}, {-218.46, 0.0}}, {{1, 0}, {64.0, 0.0}},
           {{29, 0}, {64.0, 0.0}}, {{35, 6}, {64.0, 0.0}},
           {{7, 8}, {64.0, 0.0}}, {{8, 8}, {-218.46, 0.0}},
           {{9, 9}, {-218.46, 0.0}}, {{47, 18}, {64.0, 0.0}},
           {{24, 24}, {-218.46, 0.0}}, {{11, 40}, {64.0, 0.0}},
           {{811, 840}, {64.0, 0.0}}, {{839, 840}, {64.0, 0.0}},
           {{840, 840}, {-218.46, 0.0}}, {{840, 841}, {0.0}},
           {{841, 840}, {0.0}}, {{841, 841}, {0.0}} };
}

TEST(SparseMatrixTests, ReadYoung2cMtxFile) {
  auto file_path = TestDataFilePath("young2c.mtx");
  CHECK(MatrixMarketIo::HasMtxFileExtension(file_path));
  using ValueContainer = std::vector<std::complex<double>>;
  SparseMatrix<uint32_t, uint64_t, ValueContainer> mat;
  EXPECT_OK(mat.ReadMatrixMarketFile(file_path));

  CheckSparseMatrix(mat, Young2cMatrixTestEntries());
  auto prm_path = MatrixMarketToPieRankMatrixPath(file_path);
  if (kGeneratePieRankMatrixFile) {
    EXPECT_OK(mat.WritePieRankMatrixFile(prm_path));
  }
  SparseMatrix<uint32_t, uint64_t, ValueContainer> mat0;
  EXPECT_OK(mat0.ReadPieRankMatrixFile(prm_path));
  EXPECT_EQ(mat, mat0);
}

vector<pair<vector<uint32_t>, vector<double>>> Real3dTestMatrixTestEntries() {
  return { {{0, 0}, {0, 0}},
           {{2, 0}, {3.11, 3.12}}, {{4, 0}, {5.11, 5.12}},
           {{0, 1}, {1.21, 1.22}}, {{2, 1}, {3.21, 3.22}},
           {{0, 2}, {1.31, 1.32}}, {{3, 2}, {4.31, 4.32}},
           {{1, 3}, {2.41, 2.42}}, {{2, 3}, {3.41, 3.42}},
           {{2, 4}, {3.51, 3.52}}};
}

TEST(SparseMatrixTests, ReadReal3dTestMtxFile) {
  auto file_path = TestDataFilePath("real_3d_test.mtx");
  CHECK(MatrixMarketIo::HasMtxFileExtension(file_path));
  SparseMatrix<uint32_t, uint64_t> mat;
  EXPECT_OK(mat.ReadMatrixMarketFile(file_path));

  CheckSparseMatrix(mat, Real3dTestMatrixTestEntries());
  auto prm_path = MatrixMarketToPieRankMatrixPath(file_path);
  if (kGeneratePieRankMatrixFile) {
    EXPECT_OK(mat.WritePieRankMatrixFile(prm_path));
  }
  SparseMatrix<uint32_t, uint64_t> mat0;
  EXPECT_OK(mat0.ReadPieRankMatrixFile(prm_path));
  EXPECT_EQ(mat, mat0);
}

TEST(SparseMatrixTests, ToDenseReal3dTestMtxFile) {
  auto file_path = TestDataFilePath("real_3d_test.mtx");
  CHECK(MatrixMarketIo::HasMtxFileExtension(file_path));
  SparseMatrix<uint32_t, uint64_t> mat;
  EXPECT_OK(mat.ReadMatrixMarketFile(file_path));
  CheckSparseMatrix(mat, Real3dTestMatrixTestEntries());

  auto dense = mat.ToDense(/*split_depths=*/false);
  CheckDenseMatrix(dense, Real3dTestMatrixTestEntries());

  SparseMatrix<uint32_t, uint64_t> mat0(dense);
  EXPECT_EQ(mat, mat0);

  dense = mat.ToDense(/*split_depths=*/true);
  CheckDenseMatrix(dense, Real3dTestMatrixTestEntries());

  SparseMatrix<uint32_t, uint64_t> mat1(dense);
  EXPECT_EQ(mat, mat1);
}

vector<pair<vector<uint32_t>, vector<double>>> Real4dTestMatrixTestEntries() {
  return { {{0, 0, 0}, {0, 0}},
           {{0, 1, 0}, {1.211, 1.212}}, {{0, 1, 2}, {1.231, 1.232}},
           {{0, 2, 1}, {1.321, 1.322}}, {{0, 2, 2}, {1.331, 1.332}},
           {{0, 2, 3}, {1.341, 1.342}}, {{1, 3, 2}, {2.431, 2.432}},
           {{2, 0, 1}, {3.121, 3.122}}, {{2, 1, 0}, {3.211, 3.212}},
           {{2, 1, 3}, {3.241, 3.242}}, {{2, 3, 0}, {3.411, 3.412}},
           {{2, 3, 3}, {3.441, 3.442}}, {{2, 4, 1}, {3.521, 3.522}},
           {{2, 4, 3}, {3.541, 3.542}}, {{4, 0, 0}, {5.111, 5.112}},
           {{4, 0, 3}, {5.141, 5.142}}};
}

TEST(SparseMatrixTests, ReadReal4dTestMtxFile) {
  auto file_path = TestDataFilePath("real_4d_test.mtx");
  CHECK(MatrixMarketIo::HasMtxFileExtension(file_path));
  using SubMat = SparseMatrix<uint32_t, uint64_t>;
  SparseMatrix<uint32_t, uint64_t, SubMat> mat;
  EXPECT_OK(mat.ReadMatrixMarketFile(file_path));
  CheckSparseMatrix(mat, Real4dTestMatrixTestEntries());

  // std::cout << mat.NonZeroPosDebugString();
  auto dense = mat.ToDense(/*split_depths=*/false);
  CheckDenseMatrix(dense, Real4dTestMatrixTestEntries());

  // SparseMatrix<uint32_t, uint64_t, SubMat> mat0(dense);
  // EXPECT_EQ(mat, mat0);

  dense = mat.ToDense(/*split_depths=*/true);
  CheckDenseMatrix(dense, Real4dTestMatrixTestEntries());

  // SparseMatrix<uint32_t, uint64_t, SubMat> mat1(dense);
  // EXPECT_EQ(mat, mat1);
}
