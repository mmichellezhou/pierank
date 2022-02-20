//
// Created by Michelle Zhou on 2/19/22.
//
#include "matrix_market_io.h"
#include "gtest/gtest.h"

using namespace std;

TEST(MatrixMarketIo, ReadHeaderAsh219) {
  string file_path = "data/ash219.mtx";
  MatrixMarketIo mm(file_path);
  EXPECT_TRUE(mm.Ok());
  EXPECT_EQ(mm.Rows(), 219);
  EXPECT_EQ(mm.Cols(), 85);
  EXPECT_EQ(mm.NNZ(), 438);
}