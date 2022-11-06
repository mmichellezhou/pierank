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

}  // namespace pierank

#endif //PIERANK_MATH_UTILS_H_
