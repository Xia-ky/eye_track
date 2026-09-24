/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#include "event/accelerator/accelerator_task.h"
#include "event/pipeline/event_pipeline.h"
#include "event/accelerator/event_accelerator_port.h"
#include "log/log.h"
#include "xil_cache.h"
#include "task.h"
static void accelerator_task_main(void *argument)
{
    event_pipeline_t *p = (event_pipeline_t *)argument; /* Shared queues, tensors, and ownership state. */
    ready_tensor_t ready; /* Descriptor for the next completed tensor. */
    bool warned_unconfigured = false; /* Suppresses repeated warnings while the PL port is absent. */
    /* Sleep on the ready queue until the tensor builder publishes a completed frame. */
    for (;;) {
        uint8_t id; /* Buffer identifier carried by the ready descriptor. */
        event_accelerator_result_t result; /* Outcome of cache preparation and accelerator submission. */
        if (xQueueReceive(p->ready_queue, &ready, portMAX_DELAY) != pdPASS) continue;
        id = ready.buffer_id;
        if (id > 1U || !event_pipeline_transfer_owner(id, EVENT_BUFFER_OWNER_READY_QUEUE, EVENT_BUFFER_OWNER_ACCELERATOR)) {
            LOG_ERROR("accelerator", "invalid tensor ownership in ready queue\r\n"); continue;
        }
        /* Do not touch PL registers or cache state when no accelerator port is installed. */
        if (!event_accelerator_is_configured()) {
            result = EVENT_ACCELERATOR_NOT_CONFIGURED;
        } else {
            Xil_DCacheFlushRange((INTPTR)p->tensors[id].data,
                                 (uint32_t)p->tensors[id].capacity);
            result = event_accelerator_submit(&p->tensors[id], ready.frame_index);
        }
        if (result == EVENT_ACCELERATOR_NOT_CONFIGURED) {
            if (!warned_unconfigured) LOG_ERROR("accelerator", "PL accelerator port not configured; frames are discarded\r\n");
            warned_unconfigured = true;
        } else if (result != EVENT_ACCELERATOR_OK) LOG_ERROR("accelerator", "PL transfer failed for frame %lu\r\n", (unsigned long)ready.frame_index);
        /* Return ownership only after the port has stopped using the tensor buffer. */
        if (!event_pipeline_transfer_owner(id, EVENT_BUFFER_OWNER_ACCELERATOR, EVENT_BUFFER_OWNER_FREE) ||
                xQueueSend(p->free_queue, &id, portMAX_DELAY) != pdPASS) {
            LOG_ERROR("accelerator", "failed to release tensor buffer\r\n");
            vTaskDelete(NULL);
        }
    }
}
bool accelerator_task_create(uint16_t stack_words, UBaseType_t priority)
{
    extern event_pipeline_t *event_pipeline_get(void);
    event_pipeline_t *p = event_pipeline_get(); /* Pipeline context passed to the created worker. */
    return p != NULL && xTaskCreate(accelerator_task_main, "pl_submit", stack_words, p, priority, NULL) == pdPASS;
}
