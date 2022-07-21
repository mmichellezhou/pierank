//
// Created by Michelle Zhou on 1/17/22.
//

#include <pagerank.h>
#include "gtest/gtest.h"

using namespace std;

TEST(PageRankTest, RankAsh219) {
  string file_path = "../../data/ash219.mtx";

  PageRank pr0(file_path);
  auto[epsilon0, num_iterations0] = pr0.Run();
  EXPECT_NEAR(epsilon0, 7.39947e-05, 1e-06);
  EXPECT_EQ(num_iterations0, 30);

  PageRank pr(file_path, /*damping_factor=*/0.85, /*max_iterations=*/50);
  auto[epsilon, num_iterations] = pr.Run();
  EXPECT_NEAR(epsilon, 9.75642e-07, 1e-06);
  EXPECT_EQ(num_iterations, 48);

  // std::vector<double> scores = pr.Scores();
  // std::cout << "\n";
  // for (int i = 0; i < scores.size(); ++i) {
  //   std::cout << i << " " << scores[i] << "\n";
  // }

  auto score_and_page = pr.TopK(10);

  std::vector<uint32_t> top_pages = {0, 1, 7, 9, 2, 3, 6, 5, 8, 4};
  EXPECT_EQ(score_and_page.size(), 10);
  for (int i = 0; i < score_and_page.size(); ++i) {
    auto pair = score_and_page[i];
    EXPECT_EQ(pair.second, top_pages[i]);
    // std::cout << "Page: " << pair.second << " Score: " << pair.first << "\n";
  }
}


