/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#ifndef EYE_TRACK_EVENT_SOURCE_H
#define EYE_TRACK_EVENT_SOURCE_H
#include "event/types/event_types.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Describes whether the source returned an event, reached EOF, or failed. */
typedef enum {
    EVENT_SOURCE_EVENT = 0,
    EVENT_SOURCE_EOF,
    EVENT_SOURCE_ERROR
} event_source_result_t;
/* Interface that decouples event consumers from the concrete input device. */
typedef struct {
    event_source_result_t (*next)(void *context, event_t *event); /* Reads the next event or source status. */
    void (*close)(void *context);                                /* Releases resources when ingestion ends. */
    void *context;                                                /* Concrete source state passed to both callbacks. */
} event_source_t;
#ifdef __cplusplus
}
#endif
#endif
