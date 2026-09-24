/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#include "log/log.h"

#include "sd/sd_task.h"
#include "utils/debug.h"

#include "queue.h"
#include "task.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define EYE_LOG_TASK_NAME "log"
#define EYE_LOG_SD_TIMEOUT pdMS_TO_TICKS(500U)

#if EYE_LOG_LEVEL > 0
typedef struct {
    char text[EYE_LOG_MESSAGE_SIZE]; /* Fully formatted record sent to UART and SD storage. */
} eye_log_record_t;

static QueueHandle_t log_queue; /* Bounded queue separating log producers from SD persistence. */
static TaskHandle_t log_task_handle; /* Handle preventing duplicate logger-task creation. */

static char eye_log_level_tag(eye_log_level_t level)
{
    switch (level) {
    case EYE_LOG_DEBUG:
        return 'D';
    case EYE_LOG_RUNTIME:
        return 'R';
    case EYE_LOG_ERROR:
        return 'E';
    default:
        return '?';
    }
}
#endif

bool eye_log_init(void)
{
#if EYE_LOG_LEVEL == 0
    return true;
#else
    debug_init();
    if (log_queue == NULL) {
        log_queue = xQueueCreate(EYE_LOG_QUEUE_DEPTH,
                                 (UBaseType_t)sizeof(eye_log_record_t));
    }
    return log_queue != NULL;
#endif
}

void eye_log_write(eye_log_level_t level, const char *module,
                   const char *format, ...)
{
#if EYE_LOG_LEVEL == 0
    (void)level;
    (void)module;
    (void)format;
#else
    eye_log_record_t record; /* Completed log record copied into the asynchronous queue. */
    char payload[EYE_LOG_MESSAGE_SIZE]; /* Formatted caller message before adding the log prefix. */
    va_list arguments; /* Variadic argument list used to format the caller message. */
    int prefix_length; /* Number of bytes written for level, tick, and module prefix. */
    size_t length; /* Bounded offset where the formatted payload is appended. */

    if (level < EYE_LOG_DEBUG || level > EYE_LOG_ERROR ||
            module == NULL || format == NULL) {
        return;
    }

    /* Format the user payload separately so the prefix and text can be bounded. */
    va_start(arguments, format);
    (void)vsnprintf(payload, sizeof(payload), format, arguments);
    va_end(arguments);
    payload[sizeof(payload) - 1U] = '\0';

    prefix_length = snprintf(record.text, sizeof(record.text),
                             "[%c][%lu][%s] ", eye_log_level_tag(level),
                             (unsigned long)xTaskGetTickCount(), module);
    if (prefix_length < 0) {
        return;
    }
    length = (size_t)prefix_length;
    if (length >= sizeof(record.text)) {
        length = sizeof(record.text) - 1U;
    }
    (void)snprintf(record.text + length, sizeof(record.text) - length,
                   "%s", payload);
    record.text[sizeof(record.text) - 1U] = '\0';

    /* Emit synchronously to UART, then enqueue a copy for best-effort SD backup. */
    debug_printf("%s", record.text);
    if (log_queue != NULL) {
        (void)xQueueSend(log_queue, &record, 0U);
    }
#endif
}

#if EYE_LOG_LEVEL > 0
static void eye_log_task_main(void *argument)
{
    eye_log_record_t record; /* Next queued log line waiting for persistent storage. */
    sd_request_options_t request; /* Reused SD request descriptor for append operations. */
    sd_response_t response; /* SD task response reused for each log append. */
    bool storage_error_reported = false; /* Prevents printing the same SD failure on every line. */

    (void)argument;
    (void)memset(&request, 0, sizeof(request));
    request.type = SD_REQUEST_APPEND_LOG;

    /* Wait for log records and append each one without blocking application callers. */
    for (;;) {
        if (xQueueReceive(log_queue, &record, portMAX_DELAY) != pdTRUE) {
            continue;
        }
        request.text = record.text;
        if (!sd_task_submit(&request, &response, EYE_LOG_SD_TIMEOUT) ||
                response.result.status != SD_CARD_OK) {
            if (!storage_error_reported) {
                debug_printf("[E][log] SD log backup unavailable\r\n");
                storage_error_reported = true;
            }
        } else {
            storage_error_reported = false;
        }
    }
}
#endif

bool eye_log_task_create(UBaseType_t priority, uint16_t stack_words)
{
#if EYE_LOG_LEVEL == 0
    (void)priority;
    (void)stack_words;
    return true;
#else
    /* Require initialized queue and reject a second logger task instance. */
    if (log_queue == NULL || log_task_handle != NULL) {
        return false;
    }
    return xTaskCreate(eye_log_task_main, EYE_LOG_TASK_NAME, stack_words,
                       NULL, priority, &log_task_handle) == pdPASS;
#endif
}
