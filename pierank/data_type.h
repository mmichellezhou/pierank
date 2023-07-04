//
// Created by Michelle Zhou on 7/3/23.
//

#ifndef PIERANK_DATA_TYPE_H_
#define PIERANK_DATA_TYPE_H_

namespace pierank {

class DataType {
public:
  constexpr static std::size_t kMaxStringLength = 3;

  enum Enum : uint32_t {
    kUnknown,
    kInt8,
    kUint8,
    kInt16,
    kUint16,
    kInt32,
    kUint32,
    kInt64,
    kUint64,
    kDouble,
    kFloat,
    kComplexDouble,
    kComplexFloat,
    kUserDefined
  };

  static Enum FromString(absl::string_view str) {
    if (str == "unknown") return kUnknown;
    if (str == "i8") return kInt8;
    if (str == "u8") return kUint8;
    if (str == "i16") return kInt16;
    if (str == "u16") return kUint16;
    if (str == "i32") return kInt32;
    if (str == "u32") return kUint32;
    if (str == "i64") return kInt64;
    if (str == "u64") return kUint64;
    if (str == "f64") return kDouble;
    if (str == "f32") return kFloat;
    if (str == "c64") return kComplexDouble;
    if (str == "c32") return kComplexFloat;
    if (str == "usr") return kUserDefined;
    return kUnknown;
  }

  template<typename ValueType>
  static Enum FromValueType() {
    if constexpr (std::is_same_v<ValueType, int8_t>) return kInt8;
    if constexpr (std::is_same_v<ValueType, uint8_t>) return kUint8;
    if constexpr (std::is_same_v<ValueType, int16_t>) return kInt16;
    if constexpr (std::is_same_v<ValueType, uint16_t>) return kUint16;
    if constexpr (std::is_same_v<ValueType, int32_t>) return kInt32;
    if constexpr (std::is_same_v<ValueType, uint32_t>) return kUint32;
    if constexpr (std::is_same_v<ValueType, int64_t>) return kInt64;
    if constexpr (std::is_same_v<ValueType, uint64_t>) return kUint64;
    if constexpr (std::is_same_v<ValueType, double>) return kDouble;
    if constexpr (std::is_same_v<ValueType, float>) return kFloat;
    if constexpr (std::is_same_v<ValueType, std::complex<double>>)
      return kComplexDouble;
    if constexpr (std::is_same_v<ValueType, std::complex<float>>)
      return kComplexFloat;
    return kUserDefined;
  }

  DataType() = default;

  DataType(Enum value) : val_(value) {}

  DataType(absl::string_view str) : val_(FromString(str)) {}

  inline bool IsInteger() const {
    return val_ == kInt8 || val_ == kUint8 ||
           val_ == kInt16 || val_ == kUint16 ||
           val_ == kInt32 || val_ == kUint32 ||
           val_ == kInt64 || val_ == kUint64;
  }

  inline bool IsFloatingPoint() const {
    return val_ == kDouble || val_ == kFloat;
  }

  inline bool IsReal() const {
    return IsInteger() || IsFloatingPoint();
  }

  inline bool IsComplex() const {
    return val_ == kComplexDouble || val_ == kComplexFloat;
  }

  constexpr operator Enum() const { return val_; }

  std::string ToString() const {
    switch (val_) {
      case kUnknown: return "unk";
      case kInt8: return "i8";
      case kUint8: return "u8";
      case kInt16: return "i16";
      case kUint16: return "u16";
      case kInt32: return "i32";
      case kUint32: return "u32";
      case kInt64: return "i64";
      case kUint64: return "u64";
      case kDouble: return "f64";
      case kFloat: return "f32";
      case kComplexDouble: return "c64";
      case kComplexFloat: return "c32";
      case kUserDefined: return "usr";
      default: return "und";
    }
  }

  // Reads a null-terminated "C" string as a MatrixType short string.
  template<typename InputStreamType>
  absl::Status Read(InputStreamType *is, uint64_t *offset = nullptr) {
    char buf[kMaxStringLength + 1];
    if (!ReadData(is, buf, sizeof(buf)))
      return absl::InternalError("Error reading matrix type string");
    if (strlen(buf) > kMaxStringLength)
      return absl::InternalError("Bad data type string: " + std::string(buf));
    if (offset) *offset += kMaxStringLength + 1; // +1 for the null terminator
    val_ = DataType(FromString(buf));
    return absl::OkStatus();
  }

  friend std::istream &operator>>(std::istream &is, DataType &type) {
    auto status = type.Read(&is, nullptr);
    if (!status.ok())
      LOG(ERROR) << status.message();
    return is;
  }

  // Writes a DataType string as a null-terminated "C" string.
  template<typename OutputStreamType>
  absl::Status Write(OutputStreamType *os) const {
    auto str = ToString();
    CHECK_LE(str.size(), kMaxStringLength);
    char buf[kMaxStringLength + 1];
    memset(buf, 0, sizeof(buf));
    memcpy(buf, str.data(), str.size());
    if (!WriteData(os, buf, kMaxStringLength + 1))
      return absl::InternalError("Error writing matrix type string");
    return absl::OkStatus();
  }

private:
  Enum val_;
};

}  // namespace pierank

#endif //PIERANK_DATA_TYPE_H_
