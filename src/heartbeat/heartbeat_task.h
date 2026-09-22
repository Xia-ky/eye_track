/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#ifndef EYE_TRACK_HEARTBEAT_TASK_H
#define EYE_TRACK_HEARTBEAT_TASK_H

#include "FreeRTOS.h"
#include "task.h"

#include <stdbool.h>
#include <stdint.h>

#define HEARTBEAT_PERIOD_MS 1000U

#ifdef __cplusplus
extern "C" {
#endif

bool heartbeat_task_create(UBaseType_t priority, uint16_t stack_words);

#ifdef __cplusplus
}
#endif

#endif
