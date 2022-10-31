//
// Created by Michelle Zhou on 8/21/22.
//

#ifndef PIERANK_IO_FILE_UTILS_H_
#define PIERANK_IO_FILE_UTILS_H_

#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdio.h>
#include <string.h>

#include <glog/logging.h>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "pierank/io/mio.h"

namespace pierank {

#ifdef _WIN32
inline constexpr absl::string_view kPathSeparator = "\\";
#else
inline constexpr absl::string_view kPathSeparator = "/";
#endif

// Get file name from path with or without extension.
inline absl::string_view
FileNameInPath(absl::string_view path, bool with_extension = true) {
  auto sep = path.rfind(kPathSeparator);
  auto dot = path.rfind('.');
  if (with_extension || dot == absl::string_view::npos || dot <= sep + 1)
    return path.substr(sep + 1);
  return path.substr(sep + 1, dot - sep - 1);
}

// Get file name from path with multiple extensions.
// The returned vector `res` has the following structure:
// res[0]: file name without any extensions
// res[1]: first extension (without the dot)
// ...
// res[n]: n-th extension (without the dot)
// Ex: abc.ext1.ext2.ext3 -> res[] = { "abc", "ext1", "ext2", "ext3" }
inline std::vector<absl::string_view>
FileNameAndExtensionsInPath(absl::string_view path) {
  auto sep = path.rfind(kPathSeparator);
  path = path.substr(sep + 1);
  std::vector<absl::string_view> res;
  while (!path.empty()) {
    auto dot = path.find('.');
    res.push_back(path.substr(0, dot));
    if (dot == absl::string_view::npos) break;
    path = path.substr(dot + 1);
  }
  return res;
}

inline absl::string_view DirectoryInPath(absl::string_view path) {
  auto sep = path.rfind(kPathSeparator);
  return sep != absl::string_view::npos ? path.substr(0, sep) : ".";
}

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

inline std::pair<FILE*, std::string>
OpenTmpFile(const std::string &mkstemp_template_prefix) {
  if (mkstemp_template_prefix.empty())
    return std::make_pair(nullptr, "");
  std::string path(mkstemp_template_prefix + ".XXXXXX");
  int fd = mkstemp(path.data());
  FILE *fp = fdopen(fd, "wb");
  return std::make_pair(fp, path);
}

inline std::string MakeTmpDir(const std::string &mkdtemp_template_prefix) {
  if (mkdtemp_template_prefix.empty())
    return "";
  std::string dir(mkdtemp_template_prefix + ".XXXXXX");
  if (mkdtemp(dir.data())) return dir;
  return "";
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
inline bool ConvertAndWriteInteger(std::ostream *os, SrcType src_val) {
  static_assert(std::is_integral_v<SrcType>);
  static_assert(std::is_integral_v<DestType>);
  DCHECK_LE(src_val, std::numeric_limits<DestType>::max());
  DCHECK_GE(src_val, std::numeric_limits<DestType>::min());
  DestType dest_val = static_cast<DestType>(src_val);
  return static_cast<bool>(os->write(reinterpret_cast<char *>(&dest_val),
                                     sizeof(dest_val)));
}

template<typename SrcType>
inline bool ConvertAndWriteUint32(std::ostream *os, SrcType src_val) {
  return ConvertAndWriteInteger<SrcType, uint32_t>(os, src_val);
}

template<typename SrcType>
inline bool ConvertAndWriteUint64(std::ostream *os, SrcType src_val) {
  return ConvertAndWriteInteger<SrcType, uint64_t>(os, src_val);
}

template<typename OutputStreamType>
bool WriteData(OutputStreamType *os, const char *str, uint64_t size);

template<>
inline bool
WriteData<std::ostream>(std::ostream *os, const char *str, uint64_t size) {
  return static_cast<bool>(os->write(str, size));
}

template<>
inline bool
WriteData<std::ofstream>(std::ofstream *ofs, const char *str, uint64_t size) {
  return static_cast<bool>(ofs->write(str, size));
}

template<>
inline bool WriteData<FILE>(FILE *fp, const char *str, uint64_t size) {
  return fwrite(str, 1, size, fp) == size;
}

template<typename ValueType, typename OutputStreamType>
inline bool WriteInteger(OutputStreamType *os, ValueType val,
                         uint64_t size = sizeof(ValueType)) {
  static_assert(std::is_integral_v<ValueType>);
  DCHECK(size == sizeof(ValueType) ||
         (size < sizeof(ValueType) &&
          static_cast<uint64_t>(val) < 1ULL << size * 8));
  // Assumes little-endian machine!
  return WriteData<OutputStreamType>(os, reinterpret_cast<char *>(&val), size);
}

template<typename OutputStreamType>
inline bool WriteUint32(OutputStreamType *os, uint32_t val,
                        uint64_t size = sizeof(uint32_t)) {
  return WriteInteger<uint32_t, OutputStreamType>(os, val, size);
}

template<typename OutputStreamType>
inline bool WriteUint64(OutputStreamType *os, uint64_t val,
                        uint64_t size = sizeof(uint64_t)) {
  return WriteInteger<uint64_t, OutputStreamType>(os, val, size);
}

template<typename InputStreamType>
bool ReadData(InputStreamType *is, char *str, uint64_t size);

template<>
inline bool ReadData<std::istream>(std::istream *is, char *str, uint64_t size) {
  return static_cast<bool>(is->read(str, size));
}

template<>
inline bool
ReadData<std::ifstream>(std::ifstream *ifs, char *str, uint64_t size) {
  return static_cast<bool>(ifs->read(str, size));
}

template<>
inline bool ReadData<FILE>(FILE *fp, char *str, uint64_t size) {
  return fread(str, size, 1, fp) == size;
}

template<typename SrcType, typename DestType, typename InputStreamType>
inline DestType ReadAndConvertInteger(InputStreamType *is,
                                      uint64_t *offset = nullptr) {
  static_assert(std::is_integral_v<SrcType>);
  static_assert(std::is_integral_v<DestType>);
  SrcType src_val;
  if (offset)
    *offset += sizeof(src_val);
  if (!ReadData<InputStreamType>(is, reinterpret_cast<char *>(&src_val),
                                 sizeof(src_val)))
    return std::numeric_limits<DestType>::max();
  DCHECK_LE(src_val, std::numeric_limits<DestType>::max());
  DCHECK_GE(src_val, std::numeric_limits<DestType>::min());
  return static_cast<DestType>(src_val);
}

template<typename DestType, typename InputStreamType>
inline DestType ReadUint32AndConvert(InputStreamType *is,
                                     uint64_t *offset = nullptr) {
  return ReadAndConvertInteger<uint32_t, DestType, InputStreamType>(is, offset);
}

template<typename DestType, typename InputStreamType>
inline DestType ReadUint64AndConvert(InputStreamType *is,
                                     uint64_t *offset = nullptr) {
  return ReadAndConvertInteger<uint64_t, DestType, InputStreamType>(is, offset);
}

template<typename ValueType, typename InputStreamType>
inline ValueType ReadInteger(InputStreamType *is, uint64_t *offset = nullptr) {
  static_assert(std::is_integral_v<ValueType>);
  ValueType val;
  if (offset)
    *offset += sizeof(val);
  if (ReadData<InputStreamType>(is, reinterpret_cast<char *>(&val),
                                sizeof(val)))
    return val;
  return std::numeric_limits<ValueType>::max();
}

template<typename InputStreamType>
inline uint32_t ReadUint32(InputStreamType *is, uint64_t *offset = nullptr) {
  return ReadInteger<uint32_t, InputStreamType>(is, offset);
}

template<typename InputStreamType>
inline uint64_t ReadUint64(InputStreamType *is, uint64_t *offset = nullptr) {
  return ReadInteger<uint64_t, InputStreamType>(is, offset);
}

template<typename InputStreamType>
bool Seek(InputStreamType *is, uint64_t offset);

template<>
inline bool Seek<std::istream>(std::istream *is, uint64_t offset) {
  return static_cast<bool>(is->seekg(offset));
}

template<>
inline bool Seek<std::ifstream>(std::ifstream *ifs, uint64_t offset) {
  return static_cast<bool>(ifs->seekg(offset));
}

template<>
inline bool Seek<FILE>(FILE *fp, uint64_t offset) {
  return fseek(fp, offset, SEEK_SET) == 0;
}

template<typename ValueType, typename InputStreamType>
inline ValueType ReadIntegerAtOffset(InputStreamType *is, uint64_t *offset) {
  DCHECK(offset);
  if (Seek<InputStreamType>(is, *offset))
    return ReadInteger<ValueType, InputStreamType>(is, offset);
  return std::numeric_limits<ValueType>::max();
}

template<typename InputStreamType>
inline uint32_t ReadUint32AtOffset(InputStreamType *is, uint64_t *offset) {
  return ReadIntegerAtOffset<uint32_t, InputStreamType>(is, offset);
}

template<typename InputStreamType>
inline uint64_t ReadUint64AtOffset(InputStreamType *is, uint64_t *offset) {
  return ReadIntegerAtOffset<uint64_t, InputStreamType>(is, offset);
}

template<typename InputStreamType>
inline bool EatString(InputStreamType *is, absl::string_view str,
                      uint64_t *offset = nullptr) {
  auto size = str.size();
  auto buf = std::make_unique<char[]>(size);
  if (ReadData<InputStreamType>(is, buf.get(), size)) {
    if (offset)
      *offset += size;
    return str == absl::string_view(buf.get(), size);
  }
  return false;
}

}  // namespace pierank

#endif //PIERANK_IO_FILE_UTILS_H_
