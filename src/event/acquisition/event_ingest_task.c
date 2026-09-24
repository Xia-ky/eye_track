/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#include "event/acquisition/event_ingest_task.h"
#include "event/pipeline/event_pipeline.h"
#include "log/log.h"
#include "task.h"
static void event_ingest_task_main(void *argument)
{
    event_pipeline_t *pipeline = (event_pipeline_t *)argument; /* Shared queues and source configured by the pipeline. */
    event_queue_item_t item; /* Queue payload reused for every source result. */
    /* Pull one source item at a time and block when the consumer queue is full. */
    for (;;) {
        item.result = pipeline->source.next(pipeline->source.context, &item.event);
        /* Preserve source status in the queue so downstream stages can stop cleanly. */
        if (item.result == EVENT_SOURCE_ERROR) LOG_ERROR("event_ingest", "event source read/parse failed\r\n");
        if (xQueueSend(pipeline->event_queue, &item, portMAX_DELAY) != pdPASS) {
            LOG_ERROR("event_ingest", "event queue send failed\r\n");
            break;
        }
        if (item.result != EVENT_SOURCE_EVENT) break;
    }
    if (pipeline->source.close != NULL) pipeline->source.close(pipeline->source.context);
    vTaskDelete(NULL);
}
bool event_ingest_task_create(uint16_t stack_words, UBaseType_t priority)
{
    extern event_pipeline_t *event_pipeline_get(void);
    event_pipeline_t *pipeline = event_pipeline_get(); /* Pipeline context passed to the created worker. */
    return pipeline != NULL && xTaskCreate(event_ingest_task_main, "ev_ingest", stack_words, pipeline, priority, NULL) == pdPASS;
}
