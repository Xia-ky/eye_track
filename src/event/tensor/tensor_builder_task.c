/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#include "event/tensor/tensor_builder_task.h"
#include "event/pipeline/event_pipeline.h"
#include "log/log.h"
#include "task.h"
static bool publish_frame(event_pipeline_t *p, uint8_t *active_id, bool acquire_replacement)
{
    uint8_t id = *active_id; /* Tensor buffer currently owned by the builder. */
    ready_tensor_t ready; /* Small descriptor queued without copying tensor pixels. */
    event_tensor_t *t = &p->tensors[id]; /* Metadata for the active ping-pong tensor. */
    if (t->event_count == 0U) return true;
    /* Publish frame metadata only; tensor pixels stay in their owned buffer. */
    ready.buffer_id = id;
    ready.frame_index = p->next_frame_index++;
    ready.event_count = t->event_count;
    ready.start_timestamp_us = t->start_timestamp_us;
    ready.end_timestamp_us = t->end_timestamp_us;
    /* Transfer ownership before enqueueing so producer and consumer never write concurrently. */
    if (!event_pipeline_transfer_owner(id, EVENT_BUFFER_OWNER_TENSOR_BUILDER, EVENT_BUFFER_OWNER_READY_QUEUE)) return false;
    if (xQueueSend(p->ready_queue, &ready, portMAX_DELAY) != pdPASS) return false;
    if (acquire_replacement) {
        uint8_t replacement; /* Free tensor buffer that becomes the next active frame. */
        if (xQueueReceive(p->free_queue, &replacement, portMAX_DELAY) != pdPASS) return false;
        if (!event_pipeline_transfer_owner(replacement, EVENT_BUFFER_OWNER_FREE, EVENT_BUFFER_OWNER_TENSOR_BUILDER)) return false;
        id = replacement;
        event_tensor_clear(&p->tensors[id]);
        *active_id = id;
    }
    return true;
}
static void tensor_builder_task_main(void *argument)
{
    event_pipeline_t *p = (event_pipeline_t *)argument; /* Shared event queue and tensor ownership state. */
    event_queue_item_t item; /* Event or source-termination item received from acquisition. */
    uint8_t active = 0U; /* Tensor buffer currently being filled by this worker. */
    bool have_timestamp = false; /* Whether previous contains a valid event timestamp. */
    uint64_t previous = 0U; /* Last accepted timestamp used to reject time regressions. */
    /* Begin with an empty frame; frame completion is determined by event timestamps below. */
    event_tensor_clear(&p->tensors[active]);
    for (;;) {
        if (xQueueReceive(p->event_queue, &item, portMAX_DELAY) != pdPASS) continue;
        if (item.result != EVENT_SOURCE_EVENT) {
            if (!publish_frame(p, &active, false)) LOG_ERROR("tensor_builder", "failed to publish final frame\r\n");
            if (p->tensors[active].event_count == 0U) {
                if (event_pipeline_transfer_owner(active, EVENT_BUFFER_OWNER_TENSOR_BUILDER, EVENT_BUFFER_OWNER_FREE) &&
                        xQueueSend(p->free_queue, &active, portMAX_DELAY) != pdPASS) {
                    LOG_ERROR("tensor_builder", "failed to return unused tensor buffer\r\n");
                }
            }
            if (item.result == EVENT_SOURCE_ERROR) LOG_ERROR("tensor_builder", "stopping after source error\r\n");
            vTaskDelete(NULL);
        }
        if (item.event.x >= EVENT_SENSOR_WIDTH || item.event.y >= EVENT_SENSOR_HEIGHT || item.event.polarity >= EVENT_POLARITY_COUNT) {
            LOG_ERROR("tensor_builder", "discarding out-of-range event\r\n"); continue;
        }
        if (have_timestamp && item.event.timestamp_us < previous) {
            LOG_ERROR("tensor_builder", "discarding timestamp regression\r\n"); continue;
        }
        if (p->tensors[active].event_count == 0U) p->tensors[active].start_timestamp_us = item.event.timestamp_us;
        else {
            /* Evaluate the incoming timestamp before adding the event to the active tensor. */
            event_frame_policy_result_t boundary = p->frame_policy.is_boundary(p->tensors[active].start_timestamp_us, item.event.timestamp_us, p->frame_policy.context);
            if (boundary == EVENT_FRAME_POLICY_TIMESTAMP_ERROR) { LOG_ERROR("tensor_builder", "frame policy timestamp error\r\n"); continue; }
            if (boundary == EVENT_FRAME_POLICY_BOUNDARY) {
                if (!publish_frame(p, &active, true)) { LOG_ERROR("tensor_builder", "frame ownership transfer failed\r\n"); vTaskDelete(NULL); }
                p->tensors[active].start_timestamp_us = item.event.timestamp_us;
            }
        }
        if (!event_tensor_add(&p->tensors[active], &item.event)) LOG_ERROR("tensor_builder", "tensor rejected event\r\n");
        previous = item.event.timestamp_us;
        have_timestamp = true;
    }
}
bool tensor_builder_task_create(uint16_t stack_words, UBaseType_t priority)
{
    extern event_pipeline_t *event_pipeline_get(void);
    event_pipeline_t *p = event_pipeline_get(); /* Pipeline context passed to the new worker. */
    return p != NULL && xTaskCreate(tensor_builder_task_main, "tns_build", stack_words, p, priority, NULL) == pdPASS;
}
