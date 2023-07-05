//
// Created by Michelle Zhou on 6/20/23.
//

#ifndef PIERANK_STRING_UTILS_H_
#define PIERANK_STRING_UTILS_H_

#include <complex>
#include <sstream>
#include <string>

#include "absl/strings/str_cat.h"

namespace pierank {

template<typename T>
inline std::string ToString(const T &data) { return std::to_string(data); }

template<>
inline std::string ToString(const double &data) {
  std::ostringstream oss;
  oss << std::noshowpoint << data;
  return oss.str();
}

template<>
inline std::string ToString(const float &data) {
  std::ostringstream oss;
  oss << std::noshowpoint << data;
  return oss.str();
}

template<>
inline std::string ToString(const std::complex<double> &data) {
  std::ostringstream oss;
  oss << std::noshowpoint << "(" << data.real() << ", " << data.imag() << ")";
  return oss.str();
}

template<>
inline std::string ToString(const std::complex<float> &data) {
  std::ostringstream oss;
  oss << std::noshowpoint << "(" << data.real() << ", " << data.imag() << ")";
  return oss.str();
}

template<typename T>
inline std::string VectorToString(const T &data, std::size_t max_items = 0) {
  std::string res;
  absl::StrAppend(&res, "[");
  std::size_t max_size = std::min(max_items, data.size());
  for (std::size_t i = 0; i < max_size; ++i) {
    if (i > 0) absl::StrAppend(&res, ", ");
    absl::StrAppend(&res, ToString(data[i]));
  }
  if (max_size < data.size()) {
    if (max_size > 0) absl::StrAppend(&res, ", ");
    absl::StrAppend(&res, "...");
  }
  absl::StrAppend(&res, "]");
  return res;
}

template<typename SrcType, typename DestType>
inline std::string
VectorToString(const SrcType &src, std::size_t max_items = 0) {
  std::string res;
  absl::StrAppend(&res, "[");
  DCHECK_EQ(src.size() * sizeof(src[0]) % sizeof(DestType), 0);
  std::size_t size = src.size() * sizeof(src[0]) / sizeof(DestType);
  std::size_t max_size = std::min(max_items, size);
  for (std::size_t i = 0; i < max_size; ++i) {
    if (i > 0) absl::StrAppend(&res, ", ");
    absl::StrAppend(
        &res, ToString(reinterpret_cast<const DestType *>(src.data())[i]));
  }
  if (max_size < size) {
    if (max_size > 0) absl::StrAppend(&res, ", ");
    absl::StrAppend(&res, "...");
  }
  absl::StrAppend(&res, "]");
  return res;
}

}  // namespace pierank

#endif //PIERANK_STRING_UTILS_H_
