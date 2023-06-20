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
}