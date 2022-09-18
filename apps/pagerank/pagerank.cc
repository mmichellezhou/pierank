//
// Created by Michelle Zhou on 1/17/22.
//

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/time/clock.h"

#include "pierank/pierank.h"

ABSL_FLAG(double, damping_factor, 0.85, "Damping factor");
ABSL_FLAG(std::string, matrix_file, "",
  "Input Matrix Market (.mtx) or PieRank Matrix (.prm) file");
ABSL_FLAG(int32_t, max_iterations, 100, "Maximum number of iterations");
ABSL_FLAG(uint32_t, max_threads, 16,
  "Maximum number of concurrent threads (<= 256)");
ABSL_FLAG(bool, mmap_prm_file, false, "Memory map .prm file");
ABSL_FLAG(uint64_t, print_top_k, 10,
  "Print top-k pages with max PageRank scores or -1 to print all");
ABSL_FLAG(double, tolerance, 1E-06, "Error tolerance to check for convergence");

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);

  absl::Time start_time = absl::Now();
  pierank::PageRank pr(absl::GetFlag(FLAGS_matrix_file),
                       absl::GetFlag(FLAGS_mmap_prm_file),
                       absl::GetFlag(FLAGS_damping_factor),
                       absl::GetFlag(FLAGS_max_iterations),
                       absl::GetFlag(FLAGS_tolerance));
  absl::Duration duration = absl::Now() - start_time;
  std::cout << "matrix_read_time=" << duration << "\n";

  start_time = absl::Now();
  auto pool =
      std::make_shared<pierank::ThreadPool>(absl::GetFlag(FLAGS_max_threads));
  auto [epsilon, iterations] = pr.Run(pool);
  duration = absl::Now() - start_time;
  std::cout << "pagerank_time: " << duration << std::endl;
  std::cout << "epsilon: " << epsilon << std::endl;
  std::cout << "iterations: " << iterations << std::endl;
  auto page_scores = pr.TopK(absl::GetFlag(FLAGS_print_top_k));
  std::cout << "# Page,Score" << std::endl;
  for (uint32_t i = 0; i < page_scores.size(); ++i) {
    const auto &pair = page_scores[i];
    std::cout << pair.first << "," << pair.second << std::endl;
  }
}