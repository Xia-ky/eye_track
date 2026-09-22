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
    SD_REQUEST_WRITE_TEST
} sd_request_type_t;

typedef struct {
    sd_request_type_t type;
    const char *path;
    const char *destination;
    const char *text;
    uint32_t offset;
    uint32_t line_count;
} sd_request_options_t;

typedef struct {
    sd_request_type_t type;
    char path[SD_TASK_PATH_MAX];
    char destination[SD_TASK_PATH_MAX];
    char text[SD_TASK_LOG_TEXT_MAX];
    uint32_t offset;
    uint32_t line_count;
    TaskHandle_t requester;
    uint32_t request_id;
} sd_request_t;

typedef struct {
    uint32_t request_id;
    sd_card_result_t result;
    uint32_t offset;
    bool end_of_file;
    char text[SD_TASK_RESPONSE_MAX];
} sd_response_t;

bool sd_task_create(UBaseType_t priority, uint16_t stack_words);
bool sd_task_submit(const sd_request_options_t *options,
                    sd_response_t *response, TickType_t timeout);
bool sd_task_is_available(void);

/* 注册 sd 命令到 CLI。由 app_init 在调度器启动前调用。 */
bool sd_task_cli_register(void);

#ifdef __cplusplus
}
#endif

#endif
