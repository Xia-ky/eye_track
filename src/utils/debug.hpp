/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#ifndef EYE_TRACK_DEBUG_HPP
#define EYE_TRACK_DEBUG_HPP

#include "utils/debug.h"

namespace eye_track {
class Debug final {
public:
    static void init() { debug_init(); }

    template <typename... Args>
    static void print(const char *format, Args... args)
    {
        debug_printf(format, args...);
    }
};
} // namespace eye_track

#endif
