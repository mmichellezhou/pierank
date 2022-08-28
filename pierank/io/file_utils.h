//
// Created by Michelle Zhou on 8/21/22.
//

#ifndef PIERANK_IO_FILE_UTILS_H_
#define PIERANK_IO_FILE_UTILS_H_

#include <iostream>
#include <limits>
#include <memory>

#include <glog/logging.h>

#include "absl/strings/string_view.h"

namespace pierank {

template<typename SrcType, typename DestType>
inline void ConvertAndWriteInteger(std::ostream &os, SrcType src_val) {
  static_assert(std::is_integral_v<SrcType>);
  static_assert(std::is_integral_v<DestType>);
  DCHECK_LE(src_val, std::numeric_limits<DestType>::max());
  DCHECK_GE(src_val, std::numeric_limits<DestType>::min());
  DestType dest_val = static_cast<DestType>(src_val);
  os.write(reinterpret_cast<char *>(&dest_val), sizeof(dest_val));
}

template<typename SrcType>
inline void ConvertAndWriteUint32(std::ostream &os, SrcType src_val) {
  ConvertAndWriteInteger<SrcType, uint32_t>(os, src_val);
}

template<typename SrcType>
inline void ConvertAndWriteUint64(std::ostream &os, SrcType src_val) {
  ConvertAndWriteInteger<SrcType, uint64_t>(os, src_val);
}

template<typename T>
inline void WriteInteger(std::ostream &os, T val) {
  static_assert(std::is_integral_v<T>);
  os.write(reinterpret_cast<char *>(&val), sizeof(val));
}

template<typename T>
inline void WriteInteger(std::ostream &os, T val, uint32_t encode_size) {
  static_assert(std::is_integral_v<T>);
  DCHECK_LE(encode_size, sizeof(T));
  DCHECK(encode_size == sizeof(T)
         || static_cast<uint64_t>(val) < (1ULL << (encode_size * 8)));
  // Assumes little-endian machine!
  os.write(reinterpret_cast<char *>(&val), encode_size);
}

inline void WriteUint32(std::ostream &os, uint32_t val) {
  WriteInteger<uint32_t>(os, val);
}

inline void WriteUint64(std::ostream &os, uint64_t val) {
  WriteInteger<uint64_t>(os, val);
}

template<typename SrcType, typename DestType>
inline DestType ReadAndConvertInteger(std::istream &is) {
  static_assert(std::is_integral_v<SrcType>);
  static_assert(std::is_integral_v<DestType>);
  SrcType src_val;
  is.read(reinterpret_cast<char *>(&src_val), sizeof(src_val));
  DCHECK_LE(src_val, std::numeric_limits<DestType>::max());
  DCHECK_GE(src_val, std::numeric_limits<DestType>::min());
  return static_cast<DestType>(src_val);
}

template<typename DestType>
inline DestType ReadUint32AndConvert(std::istream &is) {
  return ReadAndConvertInteger<uint32_t, DestType>(is);
}

template<typename DestType>
inline DestType ReadUint64AndConvert(std::istream &is) {
  return ReadAndConvertInteger<uint64_t, DestType>(is);
}

template<typename T>
inline T ReadInteger(std::istream &is) {
  static_assert(std::is_integral_v<T>);
  T val;
  is.read(reinterpret_cast<char *>(&val), sizeof(val));
  return val;
}

inline uint32_t ReadUint32(std::istream &is) {
  return ReadInteger<uint32_t>(is);
}

inline uint64_t ReadUint64(std::istream &is) {
  return ReadInteger<uint64_t>(is);
}

inline bool EatString(std::istream &is, absl::string_view str) {
  auto size = str.size();
  auto buf = std::make_unique<char[]>(size);
  is.read(buf.get(), size);
  return str == absl::string_view(buf.get(), size);
}

}  // namespace pierank

#endif //PIERANK_IO_FILE_UTILS_H_
