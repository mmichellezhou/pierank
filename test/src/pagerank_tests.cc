//
// Created by Michelle Zhou on 1/17/22.
//

#include <pagerank.h>
#include "gtest/gtest.h"

using namespace std;

class PagerankTest : public ::testing::Test {

protected:
  double kDampingFactor = 0.85;

  virtual void SetUp() {
  };

  virtual void TearDown() {
  };

  virtual void verify() {
    double score = 0.85;
    EXPECT_FLOAT_EQ(score, kDampingFactor);
  }
};

TEST_F(PagerankTest, Dummy) {
  verify();
}


