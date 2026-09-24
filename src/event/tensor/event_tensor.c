/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#include "event/tensor/event_tensor.h"
#include <string.h>
void event_tensor_clear(event_tensor_t *tensor)
{
    if (tensor == NULL || tensor->data == NULL) return;
    /* Reset both pixel contents and temporal/count metadata before buffer reuse. */
    (void)memset(tensor->data, 0, tensor->capacity);
    tensor->event_count = 0U;
    tensor->start_timestamp_us = 0U;
    tensor->end_timestamp_us = 0U;
}
bool event_tensor_add(event_tensor_t *tensor, const event_t *event)
{
    size_t index; /* Linear byte offset for the event's polarity/y/x location. */
    if (tensor == NULL || tensor->data == NULL || event == NULL ||
            tensor->capacity < EVENT_TENSOR_BYTES || event->x >= EVENT_SENSOR_WIDTH ||
            event->y >= EVENT_SENSOR_HEIGHT || event->polarity >= EVENT_POLARITY_COUNT) return false;
    /* Arrange planes contiguously, then rows, then columns within each plane. */
    index = ((size_t)event->polarity * EVENT_SENSOR_HEIGHT + event->y) * EVENT_SENSOR_WIDTH + event->x;
    if (tensor->data[index] != UINT8_MAX) ++tensor->data[index];
    /* Keep frame metadata synchronized with the events represented by the tensor. */
    if (tensor->event_count == 0U) tensor->start_timestamp_us = event->timestamp_us;
    tensor->end_timestamp_us = event->timestamp_us;
    if (tensor->event_count != UINT32_MAX) ++tensor->event_count;
    return true;
}
