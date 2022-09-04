//
// Created by Michelle Zhou on 9/4/22.
//

#ifndef PIERANK_MACROS_H_
#define PIERANK_MACROS_H_

#include <glog/logging.h>

#define CHECK_OK(status)  CHECK(::pierank::ok(status))
#define DCHECK_OK(status) DCHECK(::pierank::ok(status))

namespace pierank {

template<typename T>
inline bool ok(T status) { return status.ok(); }

template<>
inline bool ok(bool status) { return status; }

}  // namespace pierank

#endif //PIERANK_MACROS_H_
