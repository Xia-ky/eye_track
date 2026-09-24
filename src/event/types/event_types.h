/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#ifndef EYE_TRACK_EVENT_TYPES_H
#define EYE_TRACK_EVENT_TYPES_H
#include <stdint.h>
#define EVENT_SENSOR_WIDTH 640U
#define EVENT_SENSOR_HEIGHT 480U
#define EVENT_POLARITY_COUNT 2U
/* A single event-camera sample using sensor coordinates and a microsecond timestamp. */
typedef struct
{
    uint64_t timestamp_us; /* Timestamp of the event in microseconds. */
    uint16_t x;            /* Horizontal coordinate of the event pixel. */
    uint16_t y;            /* Vertical coordinate of the event pixel. */
    uint8_t polarity;      /* Event polarity plane selected for this sample. */
} event_t;
#endif
