/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#ifndef EYE_TRACK_FRAME_POLICY_H
#define EYE_TRACK_FRAME_POLICY_H
#include <stdint.h>
#define EVENT_FRAME_WINDOW_US UINT64_C(50000)
/* Outcome returned by a frame-boundary strategy for an incoming timestamp. */
typedef enum { EVENT_FRAME_POLICY_OK = 0, EVENT_FRAME_POLICY_BOUNDARY, EVENT_FRAME_POLICY_TIMESTAMP_ERROR } event_frame_policy_result_t;
/* Callback signature shared by fixed-window and future adaptive frame policies. */
typedef event_frame_policy_result_t (*event_frame_boundary_fn)(uint64_t start_us /* First event timestamp in the active frame. */,
                                                               uint64_t timestamp_us /* Timestamp of the incoming event. */,
                                                               void *context /* Policy-specific state supplied by the caller. */);
/* Strategy callback and context used to detect a transition to the next frame. */
typedef struct
{
    event_frame_boundary_fn is_boundary; /* Function that evaluates an incoming event boundary. */
    void *context;                       /* Opaque policy-specific state passed to the callback. */
} event_frame_policy_t;
/* Applies the configured fixed-duration window and detects timestamp regressions. */
event_frame_policy_result_t event_frame_fixed_window_boundary(uint64_t start_us /* First event timestamp in the active frame. */,
                                                              uint64_t timestamp_us /* Timestamp of the incoming event. */,
                                                              void *context /* Unused by the fixed-window implementation. */);
#endif
