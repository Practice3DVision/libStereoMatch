/**
 * @file typeDef.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-06
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __TYPE_DEF_H_
#define __TYPE_DEF_H_

#include "libStereoMatchConfig.h"

#include <memory>

#define IN
#define OUT

#ifdef BUILD_SHARED
#ifdef WIN32
#define LIBSM_API __declspec(dllexport)
#else
#define LIBSM_API __declspec(dllimport)
#endif
#else
#define LIBSM_API
#endif

#define OCCLUDED_PIXEL -0.111f
#define MISMATCHED_PIXEL 0.111f
#define NONE_PIXEL 0.f
#define IS_OCCLUDED_PIXEL(val) (std::abs(val - OCCLUDED_PIXEL) < 0.0001f)
#define IS_MISMATCHED_PIXEL(val) (std::abs(val - MISMATCHED_PIXEL) < 0.0001f)
#define IS_NONE_PIXEL(val) (std::abs(val - NONE_PIXEL) < 0.0001f)

namespace libSM {
    template <typename T>
    using Ptr = std::shared_ptr<T>;
}

#endif //!__TYPE_DEF_H_