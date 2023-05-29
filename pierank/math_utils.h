//
// Created by Michelle Zhou on 11/5/22.
//

#ifndef PIERANK_MATH_UTILS_H_
#define PIERANK_MATH_UTILS_H_

#include <limits>
#include <type_traits>

#include <glog/logging.h>

namespace pierank {

template<typename T1, typename T2>
inline T1 UnsignedDivideCeil(const T1 numerator, const T2 denominator) {
  static_assert(std::is_unsigned_v<T1>, "Unsigned integer required.");
  static_assert(std::is_unsigned_v<T2>, "Unsigned integer required.");
  DCHECK_GT(denominator, 0);
  if (numerator == 0) return 0;
  if (denominator == 0) return std::numeric_limits<T1>::max();
  return 1 + (numerator - 1) / denominator;;
}

inline uint32_t MinEncodeSize(uint64_t max_value) {
  if (max_value <= 0xFF) return 1;
  else if (max_value <= 0xFFFF) return 2;
  else if (max_value <= 0xFFFFFF) return 3;
  else if (max_value <= 0xFFFFFFFF) return 4;
  else if (max_value <= 0xFFFFFFFFFF) return 5;
  else if (max_value <= 0xFFFFFFFFFFFF) return 6;
  else if (max_value <= 0xFFFFFFFFFFFFFF) return 7;
  else return 8;
}

inline uint32_t Log2Floor32(uint32_t n) {
  if (n == 0) return -1;
  uint32_t log = 0;
  uint32_t value = n;
  for (int i = 4; i >= 0; --i) {
    int shift = (1 << i);
    uint32_t x = value >> shift;
    if (x != 0) {
      value = x;
      log += shift;
    }
  }
  DCHECK_EQ(value, 1);
  return log;
}

inline uint32_t Log2Floor64(uint64_t n) {
  const uint32_t top_bits = static_cast<uint32_t>(n >> 32);
  if (top_bits == 0)
    return Log2Floor32(static_cast<uint32_t>(n));
  else
    return 32 + Log2Floor32(top_bits);
}

}  // namespace pierank

#endif //PIERANK_MATH_UTILS_H_
