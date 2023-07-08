//
// Created by Michelle Zhou on 9/24/22.
//

#include <glog/logging.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include "pierank/pierank.h"

ABSL_FLAG(int64_t, max_items, 0, "Print up to this many items per array");
ABSL_FLAG(bool, mmap, true, "Use memory map to read in PieRank matrix file");
ABSL_FLAG(std::string, prm_file, "", "PieRank Matrix (.prm) file");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  absl::ParseCommandLine(argc, argv);

  uint64_t max_items = static_cast<uint64_t>(absl::GetFlag(FLAGS_max_items));
  std::string prm_file = absl::GetFlag(FLAGS_prm_file);
  CHECK(!prm_file.empty()) << "Path to .prm file is missing";

  bool mmap = absl::GetFlag(FLAGS_mmap);
  pierank::SparseMatrixVar<uint32_t, uint64_t> mat(prm_file, mmap);
  std::cout << mat.DebugString(max_items);
}
