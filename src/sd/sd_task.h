/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#ifndef EYE_TRACK_SD_TASK_H
#define EYE_TRACK_SD_TASK_H

#include "sd/sd_card.h"

#include "FreeRTOS.h"
#include "task.h"

#include <stdbool.h>
#include <stdint.h>

#define SD_TASK_PATH_MAX 96U
#define SD_TASK_RESPONSE_MAX 1024U
#define SD_TASK_QUEUE_DEPTH 8U
#define SD_TASK_LOG_TEXT_MAX 256U

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SD_REQUEST_STATUS = 0,
    SD_REQUEST_MOUNT,
    SD_REQUEST_LIST,
    SD_REQUEST_READ_AT,
    SD_REQUEST_FIND_TAIL,
    SD_REQUEST_TOUCH,
    SD_REQUEST_MKDIR,
    SD_REQUEST_REMOVE,
    SD_REQUEST_MOVE,
    SD_REQUEST_COPY,
    SD_REQUEST_APPEND_LOG,
    SD_REQUEST_WRITE_TEST,
    SD_REQUEST_STREAM_OPEN,
    SD_REQUEST_STREAM_READ,
    SD_REQUEST_STREAM_CLOSE
} sd_request_type_t;

/* Caller-owned options copied into a stable request before queue submission. */
typedef struct {
    sd_request_type_t type; /* Filesystem operation requested from the SD service. */
    const char *path; /* Primary path used by the selected operation. */
    const char *destination; /* Destination path for move and copy operations. */
    const char *text; /* Text payload used by append-log requests. */
    uint32_t offset; /* Byte offset for a positioned text read. */
    uint32_t line_count; /* Number of trailing lines requested by a tail operation. */
} sd_request_options_t;

/* Self-contained request payload stored in the FreeRTOS request queue. */
typedef struct {
    sd_request_type_t type; /* Operation dispatched by the SD service task. */
    char path[SD_TASK_PATH_MAX]; /* Copied primary path, safe after caller returns. */
    char destination[SD_TASK_PATH_MAX]; /* Copied destination path for move/copy. */
    char text[SD_TASK_LOG_TEXT_MAX]; /* Copied append payload for asynchronous persistence. */
    uint32_t offset; /* Byte offset associated with positioned reads. */
    uint32_t line_count; /* Tail line count associated with tail-offset queries. */
    TaskHandle_t requester; /* Task that submitted this request. */
    uint32_t request_id; /* Correlation identifier used to match the response. */
} sd_request_t;

/* Service response containing status, metadata, and optional text data. */
typedef struct {
    uint32_t request_id; /* Correlation identifier copied from the matching request. */
    sd_card_result_t result; /* Native SD operation status and byte count. */
    uint32_t offset; /* Next byte offset or computed tail offset, depending on request. */
    bool end_of_file; /* Indicates whether the returned read reached end of file. */
    char text[SD_TASK_RESPONSE_MAX]; /* Bounded text data returned by read/list/stream operations. */
} sd_response_t;

/* Allocates service queues and creates the single task that owns FatFs calls. */
bool sd_task_create(UBaseType_t priority, uint16_t stack_words);
/* Submits one request and optionally copies its correlated response to the caller. */
bool sd_task_submit(const sd_request_options_t *options,
                    sd_response_t *response, TickType_t timeout);
/* Returns a synchronized snapshot of SD service availability. */
bool sd_task_is_available(void);

/* Registers SD commands with the CLI before the scheduler starts. */
bool sd_task_cli_register(void);

#ifdef __cplusplus
}
#endif

#endif
