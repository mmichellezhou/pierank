#include <numeric>
#include <utility>

#include "gtest/gtest.h"
#include "pierank/pierank.h"

using namespace std;

using namespace pierank;

TEST(ThreadPool, SimpleIntegerFunc) {
  ThreadPool pool(4);

  // Enqueue and store future
  auto result = pool.Enqueue([](int answer) { return answer; }, 42);

  // Get result from future
  EXPECT_EQ(result.get(), 42);
}

TEST(ThreadPool, SquareFuncs) {
  ThreadPool pool(4);
  std::vector<std::future<int>> results;

  for(int i = 0; i < 8; ++i) {
    results.emplace_back(
      pool.Enqueue([i] {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return i*i;
      }));
  }

  for(int i = 0; i < 8; ++i) {
    EXPECT_EQ(results[i].get(), i*i);
  }
}

TEST(ThreadPool, ParallelForIncrement) {
  ThreadPool pool(4);
  constexpr uint32_t kNumItems = 40;
  constexpr uint32_t kItemsPerThread = 7;
  std::vector<int32_t> vec(kNumItems);
  std::iota(vec.begin(), vec.end(), 0);
  EXPECT_TRUE(std::is_sorted(vec.begin(), vec.end()));
  pool.ParallelFor(kNumItems, kItemsPerThread,
    [&vec](uint64_t first, uint64_t last) {
      for (uint64_t i = first; i < last; ++i) ++vec[i];
  });
  int32_t sum = std::accumulate(vec.begin(), vec.end(), 0);
  EXPECT_EQ(sum, (1 + kNumItems) * kNumItems / 2);
  EXPECT_TRUE(std::is_sorted(vec.begin(), vec.end()));
}
