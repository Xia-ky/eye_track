/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#ifndef EYE_TRACK_EVENT_TENSOR_H
#define EYE_TRACK_EVENT_TENSOR_H
#include "event/types/event_types.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#define EVENT_TENSOR_BYTES (EVENT_SENSOR_WIDTH * EVENT_SENSOR_HEIGHT * EVENT_POLARITY_COUNT)
/* Mutable storage and frame metadata owned by the tensor-building stage. */
typedef struct
{
    uint8_t *data;                    /* Pointer to the tensor's pixel storage. */
    size_t capacity;                  /* Number of bytes available in the storage. */
    uint32_t event_count;             /* Number of valid events accumulated in this frame. */
    uint64_t start_timestamp_us;      /* Timestamp of the first event in the frame. */
    uint64_t end_timestamp_us;        /* Timestamp of the most recent event in the frame. */
} event_tensor_t;

/* Clears frame metadata and zeroes the tensor storage before reuse. */
void event_tensor_clear(event_tensor_t *tensor);

/* Adds one event to the tensor and reports whether it was accepted. */
bool event_tensor_add(event_tensor_t *tensor, const event_t *event);
#endif
