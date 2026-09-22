/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#ifndef EYE_TRACK_CLI_TASK_H
#define EYE_TRACK_CLI_TASK_H

#include "FreeRTOS.h"
#include "task.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define CLI_TASK_INPUT_BUFFER_SIZE 128U
#define CLI_TASK_OUTPUT_BUFFER_SIZE 1024U

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CLI_LINE_PENDING = 0,
    CLI_LINE_READY,
    CLI_LINE_IGNORED,
    CLI_LINE_OVERFLOW
} cli_line_result_t;

cli_line_result_t cli_line_accept_char(char input, char *line,
                                       size_t capacity, size_t *length);
bool cli_task_create(UBaseType_t priority, uint16_t stack_words);

#ifdef __cplusplus
}
#endif

#endif
