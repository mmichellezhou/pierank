//
// Created by Michelle Zhou on 1/17/22.
//

#include "gtest/gtest.h"

#include "pierank/pierank.h"

using namespace std;

using namespace pierank;

class PageRankTestFixture : public ::testing::TestWithParam<std::string> {
protected:
  void Run(const std::string &file_path) {
    PageRank pr0(file_path);
    CHECK(pr0.ok());
    auto[epsilon, num_iterations] = pr0.Run();
    EXPECT_NEAR(epsilon, 7.39947e-05, 1e-06);
    EXPECT_EQ(num_iterations, 30);

    PageRank pr1(file_path, /*mmap=*/false, /*df=*/0.85, /*max_iters=*/50);
    CHECK(pr1.ok());
    std::tie(epsilon, num_iterations) = pr1.Run();
    EXPECT_NEAR(epsilon, 9.75642e-07, 1e-06);
    EXPECT_EQ(num_iterations, 48);

    // std::vector<double> scores = pr.Scores();
    // std::cout << "\n";
    // for (int i = 0; i < scores.size(); ++i) {
    //   std::cout << i << " " << scores[i] << "\n";
    // }

    bool mmap = !MatrixMarketIo::HasMtxFileExtension(file_path);
    PageRank pr4(file_path, mmap, /*df=*/0.85, /*max_iters=*/50);
    CHECK(pr4.ok());
    constexpr uint32_t kMaxThreads = 4;
    auto pool = std::make_shared<ThreadPool>(kMaxThreads);
    std::tie(epsilon, num_iterations) = pr4.Run(pool);
    EXPECT_NEAR(epsilon, 9.75642e-07, 1e-06);
    EXPECT_EQ(num_iterations, 48);

    auto score_and_page = pr4.TopK(10);
    std::vector <uint32_t> top_pages = {0, 1, 7, 9, 2, 3, 6, 5, 8, 4};
    EXPECT_EQ(score_and_page.size(), 10);
    for (int i = 0; i < score_and_page.size(); ++i) {
      auto pair = score_and_page[i];
      EXPECT_EQ(pair.second, top_pages[i]);
      // std::cout << "Page: " << pair.second << " Score: " << pair.first
      //   << "\n";
    }
  }
};

TEST_P(PageRankTestFixture, RankAsh219) {
  std::string file_path = GetParam();
  Run(file_path);
}

INSTANTIATE_TEST_SUITE_P(PageRankTests, PageRankTestFixture,
    ::testing::Values("../../data/ash219.mtx", "../../data/ash219.prm"));
