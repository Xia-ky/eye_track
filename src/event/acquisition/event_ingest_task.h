#ifndef EYE_TRACK_EVENT_INGEST_TASK_H
#define EYE_TRACK_EVENT_INGEST_TASK_H
#include "FreeRTOS.h"
#include <stdbool.h>
#include <stdint.h>
/* Creates the worker that reads the source and queues event records. */
bool event_ingest_task_create(uint16_t stack_words /* FreeRTOS stack depth in words. */,
                              UBaseType_t priority /* Scheduler priority for the worker. */);
#endif
