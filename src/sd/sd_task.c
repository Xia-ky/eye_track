/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#include "sd/sd_task.h"

#include "cli/cli_api.h"
#include "log/log.h"

#include "FreeRTOS.h"
#include "queue.h"
#include "semphr.h"
#include "task.h"

#include <stdio.h>
#include <string.h>

#define SD_TASK_NAME "sd"
#define SD_TASK_RESPONSE_SEND_WAIT_TICKS pdMS_TO_TICKS(100U)
#define SD_TASK_RESPONSE_QUEUE_DEPTH 1U
#define SD_TASK_SNAPSHOT_WAIT_TICKS pdMS_TO_TICKS(10U)

static QueueHandle_t sd_request_queue; /* Bounded queue of copied filesystem requests. */
static QueueHandle_t sd_response_queue; /* Single-slot response mailbox protected by sd_submit_lock. */
static SemaphoreHandle_t sd_submit_lock; /* Serializes request/response transactions from callers. */
static TaskHandle_t sd_service_task; /* Handle used to report whether the service task exists. */
static uint32_t sd_next_request_id; /* Monotonic nonzero identifier for request correlation. */
static bool sd_available; /* Startup-published availability flag read under the submit lock. */
static sd_request_t sd_service_request; /* Service-task-owned storage for the next dequeued request. */
static sd_response_t sd_service_response; /* Service-task-owned response storage reused for each request. */
/* sd_submit_lock serializes access to these large buffers.  Keeping them out
 * of caller stacks is important for CLI commands that also run std::regex. */
static sd_request_t sd_submit_request; /* Static caller-side request copy, kept off task stacks. */
static sd_response_t sd_submit_incoming; /* Static caller-side response copy, kept off task stacks. */

static bool sd_task_path_required(sd_request_type_t type)
{
    return type == SD_REQUEST_READ_AT || type == SD_REQUEST_STREAM_OPEN ||
           type == SD_REQUEST_FIND_TAIL ||
           type == SD_REQUEST_TOUCH || type == SD_REQUEST_MKDIR ||
           type == SD_REQUEST_REMOVE || type == SD_REQUEST_MOVE ||
           type == SD_REQUEST_COPY;
}

static bool sd_task_destination_required(sd_request_type_t type)
{
    return type == SD_REQUEST_MOVE || type == SD_REQUEST_COPY;
}

static const char *sd_task_request_name(sd_request_type_t type)
{
    switch (type) {
    case SD_REQUEST_STATUS: return "status";
    case SD_REQUEST_MOUNT: return "mount";
    case SD_REQUEST_LIST: return "ls";
    case SD_REQUEST_READ_AT: return "cat";
    case SD_REQUEST_FIND_TAIL: return "tail";
    case SD_REQUEST_TOUCH: return "touch";
    case SD_REQUEST_MKDIR: return "mkdir";
    case SD_REQUEST_REMOVE: return "rm";
    case SD_REQUEST_MOVE: return "mv";
    case SD_REQUEST_COPY: return "cp";
    case SD_REQUEST_APPEND_LOG: return "append-log";
    case SD_REQUEST_WRITE_TEST: return "write-test";
    case SD_REQUEST_STREAM_OPEN: return "stream-open";
    case SD_REQUEST_STREAM_READ: return "stream-read";
    case SD_REQUEST_STREAM_CLOSE: return "stream-close";
    default: return "unknown";
    }
}

/* Callers hold sd_submit_lock, so this counter needs no extra guard. */
static uint32_t sd_task_allocate_request_id(void)
{
    /* Reserve zero to distinguish uninitialized identifiers from real requests. */
    ++sd_next_request_id;
    if (sd_next_request_id == 0U) {
        sd_next_request_id = 1U;
    }
    return sd_next_request_id;
}

static void sd_task_dispatch(const sd_request_t *request,
                             sd_response_t *response)
{
    const char *path; /* Resolved path used by operations where an empty path means card root. */

    /* Initialize a safe error response before dispatching the requested operation. */
    (void)memset(response, 0, sizeof(*response));
    response->request_id = request->request_id;
    response->result.status = SD_CARD_INVALID_ARGUMENT;
    response->result.fatfs_result = 0;
    response->result.bytes_used = 0U;

    /* Keep all FatFs calls on this service task to serialize filesystem access. */
    switch (request->type) {
    case SD_REQUEST_STATUS:
        response->result.status =
            sd_card_is_mounted() ? SD_CARD_OK : SD_CARD_NOT_MOUNTED;
        break;
    case SD_REQUEST_MOUNT:
        response->result = sd_card_mount();
        break;
    case SD_REQUEST_LIST:
        path = request->path[0] == '\0' ? "0:/" : request->path;
        response->result =
            sd_card_list(path, response->text, sizeof(response->text));
        break;
    case SD_REQUEST_READ_AT:
        response->result = sd_card_read_text_at(
            request->path, request->offset, response->text,
            sizeof(response->text), &response->end_of_file);
        response->offset = request->offset +
                           (uint32_t)response->result.bytes_used;
        break;
    case SD_REQUEST_STREAM_OPEN:
        response->result = sd_card_stream_open(request->path);
        break;
    case SD_REQUEST_STREAM_READ:
        response->result = sd_card_stream_read(
            response->text, sizeof(response->text), &response->end_of_file);
        break;
    case SD_REQUEST_STREAM_CLOSE:
        response->result = sd_card_stream_close();
        break;
    case SD_REQUEST_FIND_TAIL:
        response->result = sd_card_find_tail_offset(
            request->path, request->line_count, &response->offset);
        break;
    case SD_REQUEST_TOUCH:
        response->result = sd_card_touch(request->path);
        break;
    case SD_REQUEST_MKDIR:
        response->result = sd_card_mkdir(request->path);
        break;
    case SD_REQUEST_REMOVE:
        response->result = sd_card_remove(request->path);
        break;
    case SD_REQUEST_MOVE:
        response->result = sd_card_move(request->path, request->destination);
        break;
    case SD_REQUEST_COPY:
        response->result = sd_card_copy(request->path, request->destination);
        break;
    case SD_REQUEST_APPEND_LOG:
        response->result = sd_card_append_text(EYE_LOG_FILE_PATH,
                                               request->text);
        break;
    case SD_REQUEST_WRITE_TEST:
        response->result =
            sd_card_run_write_test((uint32_t)xTaskGetTickCount());
        break;
    default:
        break;
    }

    /* Centralize operation diagnostics here rather than duplicating them in CLI handlers. */
    if (request->type == SD_REQUEST_STATUS) {
        LOG_RUNTIME("sd_task", "mounted=%s\r\n",
                    sd_card_is_mounted() ? "yes" : "no");
    } else if (request->type != SD_REQUEST_APPEND_LOG &&
            response->result.status != SD_CARD_OK) {
        LOG_ERROR("sd_task", "%s failed: status=%u fresult=%d\r\n",
                  sd_task_request_name(request->type),
                  (unsigned int)response->result.status,
                  response->result.fatfs_result);
    }
}

static void sd_task_main(void *argument)
{
    (void)argument;

    /* Process requests serially and publish one correlated response per operation. */
    for (;;) {
        if (xQueueReceive(sd_request_queue, &sd_service_request,
                          portMAX_DELAY) !=
                pdTRUE) {
            continue;
        }
        sd_task_dispatch(&sd_service_request, &sd_service_response);
        (void)xQueueSend(sd_response_queue, &sd_service_response,
                         SD_TASK_RESPONSE_SEND_WAIT_TICKS);
    }
}

static void sd_task_release_resources(void)
{
    /* Delete partially created resources so task creation can be retried safely. */
    if (sd_request_queue != NULL) {
        vQueueDelete(sd_request_queue);
        sd_request_queue = NULL;
    }
    if (sd_response_queue != NULL) {
        vQueueDelete(sd_response_queue);
        sd_response_queue = NULL;
    }
    if (sd_submit_lock != NULL) {
        vSemaphoreDelete(sd_submit_lock);
        sd_submit_lock = NULL;
    }
}

bool sd_task_create(UBaseType_t priority, uint16_t stack_words)
{
    /* Reject duplicate service instances before allocating queues or task state. */
    if (sd_service_task != NULL) {
        return false;
    }

    sd_request_queue =
        xQueueCreate(SD_TASK_QUEUE_DEPTH, (UBaseType_t)sizeof(sd_request_t));
    sd_response_queue =
        /* sd_submit_lock permits only one outstanding response at a time. */
        xQueueCreate(SD_TASK_RESPONSE_QUEUE_DEPTH,
                     (UBaseType_t)sizeof(sd_response_t));
    sd_submit_lock = xSemaphoreCreateMutex();
    if (sd_request_queue == NULL || sd_response_queue == NULL ||
            sd_submit_lock == NULL) {
        sd_task_release_resources();
        return false;
    }

    sd_next_request_id = 0U;
    sd_available = false;

    /* The service task is the sole owner of all FatFs operations. */
    if (xTaskCreate(sd_task_main, SD_TASK_NAME, stack_words, NULL, priority,
            &sd_service_task) != pdPASS) {
        sd_service_task = NULL;
        sd_task_release_resources();
        return false;
    }

    /* Written before the scheduler starts; runtime readers take a snapshot. */
    sd_available = true;
    return true;
}

#ifdef SD_TASK_TEST_RESET
void sd_task_reset_for_test(void)
{
    sd_task_release_resources();
    sd_service_task = NULL;
    sd_next_request_id = 0U;
    sd_available = false;
}
#endif

bool sd_task_submit(const sd_request_options_t *options,
                    sd_response_t *response, TickType_t timeout)
{
    unsigned int attempts; /* Maximum response-mailbox reads before this transaction gives up. */
    bool completed = false; /* Set only after the response ID matches this request. */

    /* Validate all pointers, lengths, and operation-specific required fields first. */
    if (options == NULL || options->type < SD_REQUEST_STATUS ||
            options->type > SD_REQUEST_STREAM_CLOSE) {
        LOG_ERROR("sd_task", "invalid request options\r\n");
        return false;
    }
    if (sd_task_path_required(options->type) &&
            (options->path == NULL || options->path[0] == '\0')) {
        return false;
    }
    if (sd_task_destination_required(options->type) &&
            (options->destination == NULL ||
             options->destination[0] == '\0')) {
        return false;
    }
    if (options->path != NULL && strlen(options->path) >= SD_TASK_PATH_MAX) {
        return false;
    }
    if (options->destination != NULL &&
            strlen(options->destination) >= SD_TASK_PATH_MAX) {
        return false;
    }
    if (options->type == SD_REQUEST_APPEND_LOG &&
            (options->text == NULL || options->text[0] == '\0' ||
             strlen(options->text) >= SD_TASK_LOG_TEXT_MAX)) {
        return false;
    }
    if (sd_request_queue == NULL || sd_response_queue == NULL ||
            sd_submit_lock == NULL) {
        if (options->type != SD_REQUEST_APPEND_LOG) {
            LOG_ERROR("sd_task", "%s rejected: service unavailable\r\n",
                      sd_task_request_name(options->type));
        }
        return false;
    }

    /* Hold one lock across send and receive because the service uses a shared response slot. */
    if (xSemaphoreTake(sd_submit_lock, timeout) != pdTRUE) {
        if (options->type != SD_REQUEST_APPEND_LOG) {
            LOG_ERROR("sd_task", "%s rejected: submit lock timeout\r\n",
                      sd_task_request_name(options->type));
        }
        return false;
    }

    /* Copy caller-owned strings into static storage before the caller can return. */
    (void)memset(&sd_submit_request, 0, sizeof(sd_submit_request));
    sd_submit_request.type = options->type;
    sd_submit_request.offset = options->offset;
    sd_submit_request.line_count = options->line_count;
    if (options->path != NULL) {
        (void)memcpy(sd_submit_request.path, options->path,
                     strlen(options->path) + 1U);
    }
    if (options->destination != NULL) {
        (void)memcpy(sd_submit_request.destination, options->destination,
                     strlen(options->destination) + 1U);
    }
    if (options->text != NULL) {
        (void)memcpy(sd_submit_request.text, options->text,
                     strlen(options->text) + 1U);
    }
    sd_submit_request.requester = xTaskGetCurrentTaskHandle();
    sd_submit_request.request_id = sd_task_allocate_request_id();

    /* Send the request, then discard stale responses until the matching ID arrives. */
    if (xQueueSend(sd_request_queue, &sd_submit_request, timeout) == pdTRUE) {
        for (attempts = 0U; attempts < SD_TASK_QUEUE_DEPTH; ++attempts) {
            if (xQueueReceive(sd_response_queue, &sd_submit_incoming,
                              timeout) !=
                    pdTRUE) {
                break;
            }
            if (sd_submit_incoming.request_id !=
                    sd_submit_request.request_id) {
                continue;
            }
            if (response != NULL) {
                *response = sd_submit_incoming;
            }
            completed = true;
            break;
        }
    }

    (void)xSemaphoreGive(sd_submit_lock);
    if (!completed && options->type != SD_REQUEST_APPEND_LOG) {
        LOG_ERROR("sd_task", "%s request timed out\r\n",
                  sd_task_request_name(options->type));
    }
    return completed;
}

bool sd_task_is_available(void)
{
    bool available = false; /* Conservative answer if the lock cannot be acquired. */

    if (sd_submit_lock == NULL) {
        return false;
    }
    if (xSemaphoreTake(sd_submit_lock, SD_TASK_SNAPSHOT_WAIT_TICKS) ==
            pdTRUE) {
        available = sd_available && sd_service_task != NULL;
        (void)xSemaphoreGive(sd_submit_lock);
    }
    return available;
}
