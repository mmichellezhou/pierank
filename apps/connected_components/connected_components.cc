//
// Created by Michelle Zhou on 11/27/22.
//

#include <glog/logging.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include "pierank/pierank.h"

ABSL_FLAG(std::string, matrix_file, "",
  "Input Matrix Market (.mtx) or PieRank Matrix (.prm) file");
ABSL_FLAG(int32_t, max_iterations, 200, "Maximum number of iterations");
ABSL_FLAG(uint32_t, max_threads, 4, "Maximum number of concurrent threads");
ABSL_FLAG(bool, mmap_prm_file, false, "Memory map .prm file");
ABSL_FLAG(int64_t, print_first_k, 0,
  "Print component IDs for the first k nodes or -1 to print all");

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);

  pierank::Timer timer(absl::Now());
  pierank::ConnectedComponents cc(absl::GetFlag(FLAGS_matrix_file),
                                  absl::GetFlag(FLAGS_mmap_prm_file),
                                  absl::GetFlag(FLAGS_max_iterations));
  CHECK(cc.ok()) << cc.status();
  std::cout << "matrix_read_time_ms: " << timer.Stop() << "\n";

  timer.Restart();
  auto pool =
      std::make_shared<pierank::ThreadPool>(absl::GetFlag(FLAGS_max_threads));
  auto [iterations, converged] = cc.Run(pool);
  std::cout << "connected_component_time_ms: " << timer.Stop() << std::endl;
  std::cout << "connected_components: " << cc.NumComponents() << std::endl;
  std::cout << "iterations: " << iterations << std::endl;
  std::cout << "converged: " << converged << std::endl;
  const auto &labels = cc.Labels();
  int64_t size = absl::GetFlag(FLAGS_print_first_k);
  size = size < 0 ? labels.size()
                  : std::min(size, static_cast<int64_t>(labels.size()));
  if (size) std::cout << "# Node,ComponentId" << std::endl;
  for (int64_t i = 0; i < size; ++i)
    std::cout << i << "," << labels[i] << std::endl;
}
