//
// Created by Michelle Zhou on 9/24/22.
//

#include <glog/logging.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include "pierank/pierank.h"

ABSL_FLAG(int64_t, max_items, 0, "Print up to this many items per array");
ABSL_FLAG(std::string, prm_file, "", "PieRank Matrix (.prm) file");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  absl::ParseCommandLine(argc, argv);

  uint64_t max_items = static_cast<uint64_t>(absl::GetFlag(FLAGS_max_items));
  std::string prm_file = absl::GetFlag(FLAGS_prm_file);
  CHECK(!prm_file.empty()) << "Path to .prm file is missing";
  auto types = pierank::PieRankFileTypes(prm_file);
  if (!types.ok())
    LOG(FATAL) << types.status().message();
  auto [matrix_type, data_type] = *std::move(types);
  if (!matrix_type.IsComplex()) {
    pierank::SparseMatrix <uint32_t, uint64_t> mat(prm_file, /*mmap=*/true);
    std::cout << mat.DebugString(max_items);
  } else {
    using DataContainer = std::vector<std::complex<double>>;
    pierank::SparseMatrix <uint32_t, uint64_t, DataContainer>
        mat(prm_file, /*mmap=*/true);
    std::cout << mat.DebugString(max_items);
  }
}
