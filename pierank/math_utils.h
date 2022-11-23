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

}  // namespace pierank

#endif //PIERANK_MATH_UTILS_H_
