/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#include "event/framing/frame_policy.h"
event_frame_policy_result_t event_frame_fixed_window_boundary(uint64_t start, uint64_t timestamp, void *context)
{
    /* This policy uses no external state; keep the callback signature generic. */
    (void)context;
    /* Reject time regression before subtracting unsigned timestamps. */
    if (timestamp < start) return EVENT_FRAME_POLICY_TIMESTAMP_ERROR;
    /* Close the frame once elapsed event time reaches the configured window. */
    return timestamp - start >= EVENT_FRAME_WINDOW_US ? EVENT_FRAME_POLICY_BOUNDARY : EVENT_FRAME_POLICY_OK;
}
