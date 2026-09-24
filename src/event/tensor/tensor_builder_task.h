#ifndef EYE_TRACK_TENSOR_BUILDER_TASK_H
#define EYE_TRACK_TENSOR_BUILDER_TASK_H
#include "FreeRTOS.h"
#include <stdbool.h>
#include <stdint.h>
/* Creates the worker that consumes events and builds frame tensors. */
bool tensor_builder_task_create(uint16_t stack_words /* FreeRTOS stack depth in words. */,
                                UBaseType_t priority /* Scheduler priority for the worker. */);
#endif
