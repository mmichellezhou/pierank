//
// Created by Michelle Zhou on 9/9/22.
//

#ifndef PIERANK_TEST_UTILS_H_
#define PIERANK_TEST_UTILS_H_

#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

#include "pierank/io/file_utils.h"

namespace pierank {

inline std::string TestDataFilePath(absl::string_view file_name) {
  std::vector<absl::string_view> path = { "..", "..", "data"};
  path.push_back(file_name);
  return absl::StrJoin(path, kPathSeparator);
}

}  // namespace pierank

#endif //PIERANK_TEST_UTILS_H_
