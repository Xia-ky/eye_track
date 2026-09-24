#ifndef EYE_TRACK_ACCELERATOR_TASK_H
#define EYE_TRACK_ACCELERATOR_TASK_H
#include "FreeRTOS.h"
#include <stdbool.h>
#include <stdint.h>
/* Creates the worker that submits completed tensors to the accelerator port. */
bool accelerator_task_create(uint16_t stack_words /* FreeRTOS stack depth in words. */,
                             UBaseType_t priority /* Scheduler priority for the worker. */);
#endif
