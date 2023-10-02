//
// Created by Michelle Zhou on 6/20/23.
//

#ifndef PIERANK_STRING_UTILS_H_
#define PIERANK_STRING_UTILS_H_

#include <charconv>
#include <complex>
#include <map>
#include <sstream>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"

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

inline void RemoveWhiteSpaces(std::string &str) {
  str.erase(std::remove_if(str.begin(), str.end(),
                           [](char x) { return std::isspace(x); }),
            str.end());
}

inline absl::Status RemovePrefixAndSuffix(absl::string_view *str,
                                          absl::string_view prefix,
                                          absl::string_view suffix) {
  if (!absl::ConsumePrefix(str, prefix))
    return absl::InternalError(
        absl::StrCat("Prefix '", prefix, "' not found in: ", *str));
  if (!absl::ConsumeSuffix(str, suffix))
    return absl::InternalError(
        absl::StrCat("Suffix '", suffix, "' not found in: ", *str));
  return absl::OkStatus();
}

inline absl::StatusOr<std::map<std::string, std::string>>
StringToDict(absl::string_view str,
             absl::string_view key_delim = ",",
             absl::string_view value_delim = ":",
             absl::string_view prefix = "{",
             absl::string_view suffix = "}") {
  auto status = RemovePrefixAndSuffix(&str, prefix, suffix);
  if (!status.ok()) return status;
  std::map<std::string, std::string> res;
  absl::string_view::size_type start_pos = 0;
  while (start_pos < str.size()) {
    auto key_delim_pos = str.find(key_delim, start_pos);
    auto kv_pair = str.substr(start_pos, key_delim_pos - start_pos);
    auto value_delim_pos = kv_pair.find(value_delim);
    if (value_delim_pos == absl::string_view::npos)
      return absl::InternalError(absl::StrCat("No value in '", kv_pair, "'"));
    res[std::string(kv_pair.substr(0, value_delim_pos))] =
        std::string(kv_pair.substr(value_delim_pos + 1));
    if (key_delim_pos == absl::string_view::npos) break;
    start_pos = key_delim_pos + 1;
  }
  return res;
}

template<typename T>
inline absl::StatusOr<std::vector<T>>
StringToVector(absl::string_view str,
               absl::string_view separator = ",",
               absl::string_view prefix = "[",
               absl::string_view suffix = "]") {
  auto status = RemovePrefixAndSuffix(&str, prefix, suffix);
  if (!status.ok()) return status;
  auto && svs = absl::StrSplit(str, separator);
  std::vector<T> res;
  for (const auto & sv: svs) {
    T value;
    auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), value);
    if (ec != std::errc())
      return absl::InternalError(absl::StrCat("Can't convert: ", sv));
    res.push_back(value);
  }
  return  res;
}

}  // namespace pierank

#endif //PIERANK_STRING_UTILS_H_
