//
// Created by Michelle Zhou on 2/24/22.
//
#include "flex_index.h"
#include "gtest/gtest.h"

using namespace std;

TEST(FlexIndex, AddElems) {
  FlexIndex<uint32_t> index(3);
  index.Append(123);
  EXPECT_EQ(index[0], 123);
  index.Append(234);
  EXPECT_EQ(index[1], 234);
  index.Append(345);
  EXPECT_EQ(index[2], 345);
}