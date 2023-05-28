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

template<typename OutputStreamType, typename DataType = char>
bool WriteData(OutputStreamType *os, const DataType *data, uint64_t size = 1);

template<>
inline bool
WriteData<std::ostream, char>(std::ostream *os, const char *data,
                              uint64_t size) {
  return static_cast<bool>(os->write(data, size));
}

template<>
inline bool
WriteData<std::ofstream, char>(std::ofstream *ofs, const char *data,
                               uint64_t size) {
  return static_cast<bool>(ofs->write(data, size));
}

template<>
inline bool WriteData<FILE, char>(FILE *fp, const char *data, uint64_t size) {
  return fwrite(data, 1, size, fp) == size;
}

template<>
inline bool
WriteData<std::ostream, uint32_t>(std::ostream *os, const uint32_t *data,
                                  uint64_t size) {
  return static_cast<bool>(os->write(reinterpret_cast<const char *>(data),
                                     size * sizeof(*data)));
}

template<>
inline bool
WriteData<std::ofstream, uint32_t>(std::ofstream *ofs, const uint32_t *data,
                                   uint64_t size) {
  return static_cast<bool>(ofs->write(reinterpret_cast<const char *>(data),
                                      size * sizeof(*data)));
}

template<>
inline bool
WriteData<FILE, uint32_t>(FILE *fp, const uint32_t *data, uint64_t size) {
  return fwrite(data, sizeof(*data), size, fp) == size;
}

template<>
inline bool
WriteData<std::ostream, uint64_t>(std::ostream *os, const uint64_t *data,
                                  uint64_t size) {
  return static_cast<bool>(os->write(reinterpret_cast<const char *>(data),
                                     size * sizeof(*data)));
}

template<>
inline bool
WriteData<std::ofstream, uint64_t>(std::ofstream *ofs, const uint64_t *data,
                                   uint64_t size) {
  return static_cast<bool>(ofs->write(reinterpret_cast<const char *>(data),
                                      size * sizeof(*data)));
}

template<>
inline bool
WriteData<FILE, uint64_t>(FILE *fp, const uint64_t *data, uint64_t size) {
  return fwrite(data, sizeof(*data), size, fp) == size;
}

template<typename ValueType, typename OutputStreamType>
inline bool WriteInteger(OutputStreamType *os, ValueType val,
                         uint64_t size = sizeof(ValueType)) {
  static_assert(std::is_integral_v<ValueType>);
  DCHECK(size == sizeof(ValueType) ||
         (size < sizeof(ValueType) &&
          static_cast<uint64_t>(val) < 1ULL << size * 8));
  // Assumes little-endian machine!
  return WriteData<OutputStreamType, char>(
      os, reinterpret_cast<const char *>(&val), size);
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

template<typename InputStreamType, typename DataType = char>
bool ReadData(InputStreamType *is, DataType *data, uint64_t size = 1);

template<>
inline bool
ReadData<std::istream, char>(std::istream *is, char *data, uint64_t size) {
  return static_cast<bool>(is->read(data, size));
}

template<>
inline bool
ReadData<std::ifstream, char>(std::ifstream *ifs, char *data, uint64_t size) {
  return static_cast<bool>(ifs->read(data, size));
}

template<>
inline bool ReadData<FILE, char>(FILE *fp, char *data, uint64_t size) {
  return fread(data, 1, size, fp) == size;
}

template<>
inline bool
ReadData<std::istream, uint32_t>(std::istream *is, uint32_t *data,
                                 uint64_t size) {
  return static_cast<bool>(is->read(reinterpret_cast<char *>(data),
                                    size * sizeof(*data)));
}

template<>
inline bool
ReadData<std::ifstream, uint32_t>(std::ifstream *ifs, uint32_t *data,
                                  uint64_t size) {
  return static_cast<bool>(ifs->read(reinterpret_cast<char *>(data),
                                     size * sizeof(*data)));
}

template<>
inline bool ReadData<FILE, uint32_t>(FILE *fp, uint32_t *data, uint64_t size) {
  return fread(data, sizeof(*data), size, fp) == size;
}

template<>
inline bool
ReadData<std::istream, uint64_t>(std::istream *is, uint64_t *data,
                                 uint64_t size) {
  return static_cast<bool>(is->read(reinterpret_cast<char *>(data),
                                    size * sizeof(*data)));
}

template<>
inline bool
ReadData<std::ifstream, uint64_t>(std::ifstream *ifs, uint64_t *data,
                                  uint64_t size) {
  return static_cast<bool>(ifs->read(reinterpret_cast<char *>(data),
                                     size * sizeof(*data)));
}

template<>
inline bool ReadData<FILE, uint64_t>(FILE *fp, uint64_t *data, uint64_t size) {
  return fread(data, sizeof(*data), size, fp) == size;
}

template<typename SrcType, typename DestType, typename InputStreamType>
inline DestType ReadAndConvertInteger(InputStreamType *is,
                                      uint64_t *offset = nullptr) {
  static_assert(std::is_integral_v<SrcType>);
  static_assert(std::is_integral_v<DestType>);
  SrcType src_val;
  if (offset)
    *offset += sizeof(src_val);
  if (!ReadData<InputStreamType, SrcType>(is, &src_val, 1))
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
inline ValueType ReadInteger(InputStreamType *is,
                             uint64_t size = sizeof(ValueType),
                             uint64_t *offset = nullptr) {
  static_assert(std::is_integral_v<ValueType>);
  ValueType val = 0;
  if (offset)
    *offset += size;
  if (ReadData<InputStreamType, char>(is, reinterpret_cast<char *>(&val), size))
    return val;
  return std::numeric_limits<ValueType>::max();
}

template<typename InputStreamType>
inline uint32_t ReadUint32(InputStreamType *is, uint64_t *offset = nullptr) {
  return ReadInteger<uint32_t, InputStreamType>(is, sizeof(uint32_t), offset);
}

template<typename InputStreamType>
inline uint64_t ReadUint64(InputStreamType *is, uint64_t *offset = nullptr) {
  return ReadInteger<uint64_t, InputStreamType>(is, sizeof(uint64_t), offset);
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
inline ValueType ReadIntegerAtOffset(InputStreamType *is,
                                     uint64_t *offset,
                                     uint64_t size = sizeof(ValueType)) {
  DCHECK(offset);
  if (Seek<InputStreamType>(is, *offset))
    return ReadInteger<ValueType, InputStreamType>(is, size, offset);
  return std::numeric_limits<ValueType>::max();
}

template<typename InputStreamType>
inline uint32_t ReadUint32AtOffset(InputStreamType *is, uint64_t *offset) {
  return ReadIntegerAtOffset<uint32_t, InputStreamType>(is, offset,
                                                        sizeof(uint32_t));
}

template<typename InputStreamType>
inline uint64_t ReadUint64AtOffset(InputStreamType *is, uint64_t *offset) {
  return ReadIntegerAtOffset<uint64_t, InputStreamType>(is, offset,
                                                        sizeof(uint64_t));
}

template<typename InputStreamType>
inline bool EatString(InputStreamType *is, absl::string_view str,
                      uint64_t *offset = nullptr) {
  auto size = str.size();
  auto buf = std::make_unique<char[]>(size);
  if (ReadData<InputStreamType, char>(is, buf.get(), size)) {
    if (offset)
      *offset += size;
    return str == absl::string_view(buf.get(), size);
  }
  return false;
}

}  // namespace pierank

#endif //PIERANK_IO_FILE_UTILS_H_
