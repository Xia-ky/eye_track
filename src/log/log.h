/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#ifndef EYE_TRACK_LOG_H
#define EYE_TRACK_LOG_H

#include "FreeRTOS.h"

#include <stdbool.h>
#include <stdint.h>

#ifndef EYE_LOG_LEVEL
#define EYE_LOG_LEVEL 2
#endif

#if (EYE_LOG_LEVEL < 0) || (EYE_LOG_LEVEL > 3)
#error "EYE_LOG_LEVEL must be in the range 0..3"
#endif

#define EYE_LOG_FILE_PATH "0:/eye_track.log"
#define EYE_LOG_MESSAGE_SIZE 256U
#define EYE_LOG_QUEUE_DEPTH 32U

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    EYE_LOG_DEBUG = 1,
    EYE_LOG_RUNTIME = 2,
    EYE_LOG_ERROR = 3
} eye_log_level_t;

bool eye_log_init(void);
bool eye_log_task_create(UBaseType_t priority, uint16_t stack_words);
void eye_log_write(eye_log_level_t level, const char *module,
                   const char *format, ...);

#ifdef __cplusplus
}
#endif

#if (EYE_LOG_LEVEL > 0) && (EYE_LOG_LEVEL <= 1)
#define LOG_DEBUG(module_, ...) \
    eye_log_write(EYE_LOG_DEBUG, (module_), __VA_ARGS__)
#else
#define LOG_DEBUG(module_, ...) ((void)0)
#endif

#if (EYE_LOG_LEVEL > 0) && (EYE_LOG_LEVEL <= 2)
#define LOG_RUNTIME(module_, ...) \
    eye_log_write(EYE_LOG_RUNTIME, (module_), __VA_ARGS__)
#else
#define LOG_RUNTIME(module_, ...) ((void)0)
#endif

#if (EYE_LOG_LEVEL > 0) && (EYE_LOG_LEVEL <= 3)
#define LOG_ERROR(module_, ...) \
    eye_log_write(EYE_LOG_ERROR, (module_), __VA_ARGS__)
#else
#define LOG_ERROR(module_, ...) ((void)0)
#endif

#endif
