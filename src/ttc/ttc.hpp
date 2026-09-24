/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#ifndef EYE_TRACK_TTC_HPP
#define EYE_TRACK_TTC_HPP

#include "ttc/ttc_timer.h"

namespace eye_track {
class TtcTimer final {
public:
    /* Initializes and starts the underlying Xilinx TTC driver instance. */
    static bool init() { return ttc_timer_init(); }
    static bool is_ready() { return ttc_timer_is_ready(); }
    static uint32_t counter() { return ttc_timer_counter(); }
    static uint32_t elapsed_ms() { return ttc_timer_elapsed_ms(); }
};
} // namespace eye_track

#endif
