//
// Created by Michelle Zhou on 2/19/22.
//

#include "gtest/gtest.h"
#include "pierank/pierank.h"
#include "test_utils.h"

using namespace std;

using namespace pierank;

TEST(MatrixMarketIo, ReadHeaderAsh219) {
  auto file_path = TestDataFilePath("ash219.mtx");
  MatrixMarketIo mm(file_path);
  EXPECT_TRUE(mm.ok());
  EXPECT_EQ(mm.Rows(), 219);
  EXPECT_EQ(mm.Cols(), 85);
  EXPECT_EQ(mm.NumNonZeros(), 438);
  EXPECT_EQ(mm.Type(), MatrixType("CPG"));
}

TEST(MatrixMarketIo, ReadHeaderB1ss) {
  auto file_path = TestDataFilePath("b1_ss.mtx");
  MatrixMarketIo mm(file_path);
  EXPECT_TRUE(mm.ok());
  EXPECT_EQ(mm.Rows(), 7);
  EXPECT_EQ(mm.Cols(), 7);
  EXPECT_EQ(mm.NumNonZeros(), 15);
  EXPECT_EQ(mm.Type(), MatrixType("CRG"));
}

TEST(MatrixMarketIo, ReadHeaderBcsstm01) {
  auto file_path = TestDataFilePath("bcsstm01.mtx");
  MatrixMarketIo mm(file_path);
  EXPECT_TRUE(mm.ok());
  EXPECT_EQ(mm.Rows(), 48);
  EXPECT_EQ(mm.Cols(), 48);
  EXPECT_EQ(mm.NumNonZeros(), 48);
  EXPECT_EQ(mm.Type(), MatrixType("CIS"));
}

TEST(MatrixMarketIo, ReadHeaderBcsstm02) {
  auto file_path = TestDataFilePath("bcsstm02.mtx");
  MatrixMarketIo mm(file_path);
  EXPECT_TRUE(mm.ok());
  EXPECT_EQ(mm.Rows(), 66);
  EXPECT_EQ(mm.Cols(), 66);
  EXPECT_EQ(mm.NumNonZeros(), 66);
  EXPECT_EQ(mm.Type(), MatrixType("CRS"));
}

TEST(MatrixMarketIo, ReadHeaderCan24) {
  auto file_path = TestDataFilePath("can_24.mtx");
  MatrixMarketIo mm(file_path);
  EXPECT_TRUE(mm.ok());
  EXPECT_EQ(mm.Rows(), 24);
  EXPECT_EQ(mm.Cols(), 24);
  EXPECT_EQ(mm.NumNonZeros(), 92);
  EXPECT_EQ(mm.Type(), MatrixType("CPS"));
}

TEST(MatrixMarketIo, ReadHeaderDwg961a) {
  auto file_path = TestDataFilePath("dwg961a.mtx");
  MatrixMarketIo mm(file_path);
  EXPECT_TRUE(mm.ok());
  EXPECT_EQ(mm.Rows(), 961);
  EXPECT_EQ(mm.Cols(), 961);
  EXPECT_EQ(mm.NumNonZeros(), 2055);
  EXPECT_EQ(mm.Type(), MatrixType("CCS"));
}

TEST(MatrixMarketIo, ReadHeaderFarm) {
  auto file_path = TestDataFilePath("farm.mtx");
  MatrixMarketIo mm(file_path);
  EXPECT_TRUE(mm.ok());
  EXPECT_EQ(mm.Rows(), 7);
  EXPECT_EQ(mm.Cols(), 17);
  EXPECT_EQ(mm.NumNonZeros(), 41);
  EXPECT_EQ(mm.Type(), MatrixType("CIG"));
}

TEST(MatrixMarketIo, ReadHeaderMhd1280b) {
  auto file_path = TestDataFilePath("mhd1280b.mtx");
  MatrixMarketIo mm(file_path);
  EXPECT_TRUE(mm.ok());
  EXPECT_EQ(mm.Rows(), 1280);
  EXPECT_EQ(mm.Cols(), 1280);
  EXPECT_EQ(mm.NumNonZeros(), 12029);
  EXPECT_EQ(mm.Type(), MatrixType("CCH"));
}

TEST(MatrixMarketIo, ReadHeaderPlskz362) {
  auto file_path = TestDataFilePath("plskz362.mtx");
  MatrixMarketIo mm(file_path);
  EXPECT_TRUE(mm.ok());
  EXPECT_EQ(mm.Rows(), 362);
  EXPECT_EQ(mm.Cols(), 362);
  EXPECT_EQ(mm.NumNonZeros(), 880);
  EXPECT_EQ(mm.Type(), MatrixType("CRK"));
}

TEST(MatrixMarketIo, ReadHeaderYoung2c) {
  auto file_path = TestDataFilePath("young2c.mtx");
  MatrixMarketIo mm(file_path);
  EXPECT_TRUE(mm.ok());
  EXPECT_EQ(mm.Rows(), 841);
  EXPECT_EQ(mm.Cols(), 841);
  EXPECT_EQ(mm.NumNonZeros(), 4089);
  EXPECT_EQ(mm.Type(), MatrixType("CCG"));
}
