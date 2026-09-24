/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#include "event/pipeline/event_pipeline.h"
#include "event/tensor/event_tensor.h"
#include "event/acquisition/event_ingest_task.h"
#include "event/tensor/tensor_builder_task.h"
#include "event/accelerator/accelerator_task.h"
#include "event/framing/frame_policy.h"
#include "log/log.h"
#include <string.h>
#define EVENT_INGEST_TASK_STACK_WORDS 1536U
#define TENSOR_BUILDER_TASK_STACK_WORDS 2048U
#define ACCELERATOR_TASK_STACK_WORDS 2048U
#define EVENT_INGEST_TASK_PRIORITY (tskIDLE_PRIORITY + 3U)
#define TENSOR_BUILDER_TASK_PRIORITY (tskIDLE_PRIORITY + 4U)
#define ACCELERATOR_TASK_PRIORITY (tskIDLE_PRIORITY + 5U)
#if defined(__GNUC__)
static uint8_t tensor_storage[2][EVENT_TENSOR_BYTES] __attribute__((aligned(64))); /* Two cache-line-aligned ping-pong tensors. */
#else
static uint8_t tensor_storage[2][EVENT_TENSOR_BYTES]; /* Two tensor buffers used when alignment attributes are unavailable. */
#endif
static event_pipeline_t runtime; /* Process-wide state shared by all event-pipeline workers. */
/* Returns the singleton used by all task entry points in this pipeline. */
event_pipeline_t *event_pipeline_get(void)
{
    return &runtime;
}
/* Changes tensor ownership inside a short critical section shared by the stages. */
bool event_pipeline_transfer_owner(uint8_t id /* Tensor index in the ping-pong pair. */,
                                   event_buffer_owner_t expected /* Required current owner. */,
                                   event_buffer_owner_t next /* New owner after a successful transfer. */)
{
    taskENTER_CRITICAL();
    if (id > 1U || runtime.owners[id] != expected) { taskEXIT_CRITICAL(); return false; }
    runtime.owners[id] = next;
    taskEXIT_CRITICAL();
    return true;
}
bool event_pipeline_start(const event_source_t *source)
{
    uint8_t free_id = 1U; /* Buffer initially returned to the free-buffer queue. */
    if (source == NULL || source->next == NULL) return false;
    /* Reset shared state before publishing the source and queue handles to tasks. */
    (void)memset(&runtime, 0, sizeof(runtime));
    runtime.source = *source;
    runtime.frame_policy.is_boundary = event_frame_fixed_window_boundary;
    runtime.tensors[0].data = tensor_storage[0];
    runtime.tensors[0].capacity = EVENT_TENSOR_BYTES;
    runtime.tensors[1].data = tensor_storage[1];
    runtime.tensors[1].capacity = EVENT_TENSOR_BYTES;
    runtime.owners[0] = EVENT_BUFFER_OWNER_TENSOR_BUILDER;
    runtime.owners[1] = EVENT_BUFFER_OWNER_FREE;
    /* Create bounded queues so each stage blocks instead of polling or copying tensors. */
    runtime.event_queue = xQueueCreate(EVENT_QUEUE_DEPTH, sizeof(event_queue_item_t));
    runtime.ready_queue = xQueueCreate(2U, sizeof(ready_tensor_t));
    runtime.free_queue = xQueueCreate(1U, sizeof(uint8_t));
    /* Do not start workers unless every queue and initial ownership token is valid. */
    if (runtime.event_queue == NULL || runtime.ready_queue == NULL || runtime.free_queue == NULL ||
            xQueueSend(runtime.free_queue, &free_id, 0U) != pdPASS) return false;
    /* Start the consumer first, then frame construction and finally event acquisition. */
    if (!accelerator_task_create(ACCELERATOR_TASK_STACK_WORDS, ACCELERATOR_TASK_PRIORITY)) return false;
    if (!tensor_builder_task_create(TENSOR_BUILDER_TASK_STACK_WORDS, TENSOR_BUILDER_TASK_PRIORITY)) return false;
    if (!event_ingest_task_create(EVENT_INGEST_TASK_STACK_WORDS, EVENT_INGEST_TASK_PRIORITY)) return false;
    LOG_RUNTIME("event", "pipeline started, 640x480x2 tensor, 50ms windows\r\n");
    return true;
}
