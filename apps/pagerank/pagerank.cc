//
// Created by Michelle Zhou on 1/17/22.
//

#include <gflags/gflags.h>

#include "absl/time/clock.h"
#include "pierank/pierank.h"

DEFINE_double(damping_factor, 0.85, "Damping factor");
DEFINE_string(matrix_file, "", "Input matrix file");
DEFINE_int32(max_iterations, 100, "Maximum number of iterations");
DEFINE_uint32(max_threads, 16, "Maximum number of concurrent threads (<= 256)");
DEFINE_uint64(print_top_k, 10, "Print top-k pages with max PageRank scores "
"or -1 to print all");
DEFINE_double(tolerance, 1E-06, "Error tolerance to check for convergence");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  absl::Time start_time = absl::Now();
  pierank::PageRank pr(FLAGS_matrix_file, FLAGS_damping_factor,
                       FLAGS_max_iterations, FLAGS_tolerance);
  absl::Duration duration = absl::Now() - start_time;
  std::cout << "matrix_read_time=" << duration << "\n";

  start_time = absl::Now();
  auto pool = std::make_shared<pierank::ThreadPool>(FLAGS_max_threads);
  auto [epsilon, iterations] = pr.Run(pool);
  duration = absl::Now() - start_time;
  std::cout << "pagerank_time=" << duration << std::endl;
  std::cout << "epsilon=" << epsilon << "\n";
  std::cout << "iterations=" << iterations << "\n";
  auto score_and_page = pr.TopK(FLAGS_print_top_k);
  std::cout << "# Page,Score\n";
  for (int i = 0; i < score_and_page.size(); ++i) {
    const auto &pair = score_and_page[i];
    std::cout << pair.second << "," << pair.first << "\n";
  }
}