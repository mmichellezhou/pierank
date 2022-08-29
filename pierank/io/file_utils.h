//
// Created by Michelle Zhou on 8/21/22.
//

#ifndef PIERANK_IO_FILE_UTILS_H_
#define PIERANK_IO_FILE_UTILS_H_

#include <fstream>
#include <iostream>
#include <limits>
#include <memory>

#include <glog/logging.h>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "pierank/io/mio.h"

namespace pierank {

inline absl::StatusOr<std::ifstream> OpenReadFile(
    const std::string &path,
    std::ios_base::openmode mode = std::ifstream::binary) {
  std::ifstream inf(path, mode);
  if (!inf)
    return absl::InternalError(absl::StrCat("Error open read file: ", path));
  return inf;
}

inline absl::StatusOr<std::ofstream> OpenWriteFile(
    const std::string &path,
    std::ios_base::openmode mode = std::ofstream::binary) {
  std::ofstream outf(path, mode);
  if (!outf)
    return absl::InternalError(absl::StrCat("Error open write file: ", path));
  return outf;
}

inline absl::StatusOr<mio::mmap_source>
MmapReadOnlyFile(const std::string &path, uint64_t offset, uint64_t size) {
  std::error_code error;
  mio::mmap_source res = mio::make_mmap_source(path, offset, size, error);
  if (error) {
    return absl::InternalError(
        absl::StrCat("Error mmap file: ", path, "\n", error.message()));
  }
  return res;
}

template<typename SrcType, typename DestType>
inline bool ConvertAndWriteInteger(std::ostream &os, SrcType src_val) {
  static_assert(std::is_integral_v<SrcType>);
  static_assert(std::is_integral_v<DestType>);
  DCHECK_LE(src_val, std::numeric_limits<DestType>::max());
  DCHECK_GE(src_val, std::numeric_limits<DestType>::min());
  DestType dest_val = static_cast<DestType>(src_val);
  return static_cast<bool>(os.write(reinterpret_cast<char *>(&dest_val),
                                    sizeof(dest_val)));
}

template<typename SrcType>
inline bool ConvertAndWriteUint32(std::ostream &os, SrcType src_val) {
  return ConvertAndWriteInteger<SrcType, uint32_t>(os, src_val);
}

template<typename SrcType>
inline bool ConvertAndWriteUint64(std::ostream &os, SrcType src_val) {
  return ConvertAndWriteInteger<SrcType, uint64_t>(os, src_val);
}

template<typename T>
inline bool WriteInteger(std::ostream &os, T val, uint32_t size = sizeof(T)) {
  static_assert(std::is_integral_v<T>);
  DCHECK(size == sizeof(T) ||
         (size < sizeof(T) && static_cast<uint64_t>(val) < 1ULL << size * 8));
  // Assumes little-endian machine!
  return static_cast<bool>(os.write(reinterpret_cast<char *>(&val), size));
}

inline bool
WriteUint32(std::ostream &os, uint32_t val, uint32_t size = sizeof(uint32_t)) {
  return WriteInteger<uint32_t>(os, val, size);
}

inline bool
WriteUint64(std::ostream &os, uint64_t val, uint32_t size = sizeof(uint64_t)) {
  return WriteInteger<uint64_t>(os, val, size);
}

template<typename SrcType, typename DestType>
inline DestType ReadAndConvertInteger(std::istream &is) {
  static_assert(std::is_integral_v<SrcType>);
  static_assert(std::is_integral_v<DestType>);
  SrcType src_val;
  if (!is.read(reinterpret_cast<char *>(&src_val), sizeof(src_val)))
    return std::numeric_limits<DestType>::max();
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
  if (is.read(reinterpret_cast<char *>(&val), sizeof(val)))
    return val;
  return std::numeric_limits<T>::max();
}

inline uint32_t ReadUint32(std::istream &is) {
  return ReadInteger<uint32_t>(is);
}

inline uint64_t ReadUint64(std::istream &is) {
  return ReadInteger<uint64_t>(is);
}

template<typename T>
inline T ReadIntegerAtOffset(std::istream &is, uint64_t offset) {
  if (is.seekg(offset))
    return ReadInteger<T>(is);
  return std::numeric_limits<T>::max();
}

inline uint32_t ReadUint32AtOffset(std::istream &is, uint64_t offset) {
  return ReadIntegerAtOffset<uint32_t>(is, offset);
}

inline uint64_t ReadUint64AtOffset(std::istream &is, uint64_t offset) {
  return ReadIntegerAtOffset<uint64_t>(is, offset);
}

inline bool EatString(std::istream &is, absl::string_view str) {
  auto size = str.size();
  auto buf = std::make_unique<char[]>(size);
  if (is.read(buf.get(), size))
    return str == absl::string_view(buf.get(), size);
  return false;
}

}  // namespace pierank

#endif //PIERANK_IO_FILE_UTILS_H_
