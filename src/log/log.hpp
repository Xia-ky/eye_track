/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#ifndef EYE_TRACK_LOG_HPP
#define EYE_TRACK_LOG_HPP

#include "log/log.h"

namespace eye_track {

class Log final {
public:
    /* Initializes the UART backend and bounded log queue. */
    static bool init() noexcept { return eye_log_init(); }
    /* Creates the background task that persists queued records to the SD card. */
    static bool create_task(UBaseType_t priority, uint16_t stack_words) noexcept
    {
        return eye_log_task_create(priority, stack_words);
    }
};

} // namespace eye_track

#endif
