/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#ifndef EYE_TRACK_EVENT_PIPELINE_H
#define EYE_TRACK_EVENT_PIPELINE_H
#include "event/types/event_types.h"
#include "event/acquisition/event_source.h"
#include "event/tensor/event_tensor.h"
#include "event/framing/frame_policy.h"
#include "FreeRTOS.h"
#include "queue.h"
#include <stdbool.h>
#include <stdint.h>
#define EVENT_QUEUE_DEPTH 128U
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
    EVENT_BUFFER_OWNER_FREE = 0,
    EVENT_BUFFER_OWNER_TENSOR_BUILDER,
    EVENT_BUFFER_OWNER_READY_QUEUE,
    EVENT_BUFFER_OWNER_ACCELERATOR
} event_buffer_owner_t;
/* Queue payload identifying a completed tensor without copying its pixel data. */
typedef struct
{
    uint8_t buffer_id;                /* Index of the completed tensor in the ping-pong pair. */
    uint32_t frame_index;             /* Monotonic frame sequence number assigned by the builder. */
    uint32_t event_count;             /* Number of events accumulated for this frame. */
    uint64_t start_timestamp_us;      /* Timestamp of the first event in this frame. */
    uint64_t end_timestamp_us;        /* Timestamp of the last event in this frame. */
} ready_tensor_t;

/* Queue payload carrying either one parsed event or a source-state notification. */
typedef struct
{
    event_source_result_t result;     /* Result code produced by the event source. */
    event_t event;                    /* Parsed event associated with an EVENT result. */
} event_queue_item_t;

/* Shared handles and state that connect the event-pipeline stages. */
typedef struct {
    QueueHandle_t event_queue;                 /* Queue from acquisition to tensor building. */
    QueueHandle_t ready_queue;                 /* Queue from tensor building to acceleration. */
    QueueHandle_t free_queue;                  /* Queue returning released tensor buffer IDs. */
    event_source_t source;                     /* Active event-source interface and context. */
    event_frame_policy_t frame_policy;         /* Strategy that decides when a frame ends. */
    event_tensor_t tensors[2];                 /* Two alternating buffers for ping-pong ownership. */
    volatile event_buffer_owner_t owners[2];   /* Current owner state of each tensor buffer. */
    uint32_t next_frame_index;                 /* Sequence number assigned to the next completed frame. */
} event_pipeline_t;

/* Creates shared queues/buffers and starts the event-pipeline worker tasks. */
bool event_pipeline_start(const event_source_t *source);

/* Returns the process-wide pipeline instance used by the worker tasks. */
event_pipeline_t *event_pipeline_get(void);

/* Atomically transfers a buffer between stages if its current owner matches expected. */
bool event_pipeline_transfer_owner(uint8_t buffer_id, event_buffer_owner_t expected, event_buffer_owner_t next);
#ifdef __cplusplus
}
#endif
#endif
