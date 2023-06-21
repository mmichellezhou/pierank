//
// Created by Michelle Zhou on 6/20/23.
//

#ifndef PIERANK_STRING_UTILS_H_
#define PIERANK_STRING_UTILS_H_

#include <string>

#include "absl/strings/str_cat.h"

namespace pierank {

template<typename T>
std::string VectorToString(const T &data, std::size_t max_items = 0) {
  std::string res;
  absl::StrAppend(&res, "[");
  std::size_t max_size = std::min(max_items, data.size());
  for (std::size_t i = 0; i < max_size; ++i) {
    if (i > 0) absl::StrAppend(&res, ", ");
    absl::StrAppend(&res, data[i]);
  }
  if (max_size < data.size()) {
    if (max_size > 0) absl::StrAppend(&res, ", ");
    absl::StrAppend(&res, "...");
  }
  absl::StrAppend(&res, "]");
  return res;
}

}  // namespace pierank

#endif //PIERANK_STRING_UTILS_H_
