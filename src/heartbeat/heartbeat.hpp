/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#ifndef EYE_TRACK_HEARTBEAT_HPP
#define EYE_TRACK_HEARTBEAT_HPP

#include "heartbeat/heartbeat_task.h"

namespace eye_track {
class HeartbeatTask final {
public:
    /* C++ façade for creating the optional heartbeat worker. */
    static bool create(UBaseType_t priority, uint16_t stack_words)
    { return heartbeat_task_create(priority, stack_words); }
};
} // namespace eye_track

#endif
