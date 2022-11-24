//
// Created by Michelle Zhou on 9/3/22.
//

#include <glog/logging.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/time/clock.h"

#include "pierank/pierank.h"

ABSL_FLAG(std::string, mtx_file, "", "Input Matrix Market (.mtx) file");
ABSL_FLAG(std::string, output_dir, "",
  "Output directory if not same as input directory");
ABSL_FLAG(bool, output_row_major, false, "Output in row major order");

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);

  pierank::SparseMatrix<uint32_t, uint64_t> mat;
  std::string mtx_file = absl::GetFlag(FLAGS_mtx_file);
  std::cout << "mtx_file: " << mtx_file << std::endl;
  std::string output_dir = absl::GetFlag(FLAGS_output_dir);
  bool change_index_dim = absl::GetFlag(FLAGS_output_row_major);
  std::string prm_file = pierank::MatrixMarketToPieRankMatrixPath(
      mtx_file, change_index_dim, output_dir);
  CHECK(!prm_file.empty()) << "Bad .mtx file: " << mtx_file;
  std::cout << "prm_file: " << prm_file << std::endl;

  pierank::Timer timer(absl::Now());
  CHECK_OK(mat.ReadMatrixMarketFile(mtx_file));
  std::cout << "mtx_read_time_ms: " << timer.Stop() << std::endl;

  timer.Restart();
  if (change_index_dim) {
    CHECK_OK(mat.ChangeIndexDim(prm_file));
    std::cout << "index_and_prm_write_time_ms: " << timer.Stop() << std::endl;
  } else {
    CHECK_OK(mat.WritePieRankMatrixFile(prm_file));
    std::cout << "prm_write_time_ms: " << timer.Stop() << std::endl;
  }
}