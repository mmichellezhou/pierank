//
// Created by Michelle Zhou on 1/17/22.
//

#include "gtest/gtest.h"

#include "pierank/pierank.h"
#include "test_utils.h"

using namespace std;

using namespace pierank;

class PageRankTestFixture : public ::testing::TestWithParam<std::string> {
protected:
  void Run(const std::string &file_path) {
    constexpr double kPrecision = 1e-08;
    PageRank pr0(file_path);
    CHECK(pr0.ok());
    auto[residual, num_iterations] = pr0.Run();
    EXPECT_NEAR(residual, 9.8e-07, kPrecision);
    EXPECT_EQ(num_iterations, 52);

    PageRank pr1(file_path, /*mmap=*/false, /*df=*/0.85, /*max_iters=*/50);
    CHECK(pr1.ok());
    std::tie(residual, num_iterations) = pr1.Run();
    EXPECT_NEAR(residual, 1.6e-06, kPrecision);
    EXPECT_EQ(num_iterations, 50);

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
    std::tie(residual, num_iterations) = pr4.Run(pool);
    if (mmap) pr4.UnMmap();
    EXPECT_NEAR(residual, 4.436e-05, kPrecision);
    EXPECT_EQ(num_iterations, 50);

    auto page_scores = pr4.TopK(10);
    std::vector<uint32_t> top_pages = {0, 1, 7, 9, 2, 3, 6, 5, 8, 4};
    EXPECT_EQ(page_scores.size(), 10);
    for (int i = 0; i < page_scores.size(); ++i) {
      auto pair = page_scores[i];
      EXPECT_EQ(pair.first, top_pages[i]);
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
    ::testing::Values(TestDataFilePath("ash219.mtx"),
                      TestDataFilePath("ash219.i1.prm"))
);
