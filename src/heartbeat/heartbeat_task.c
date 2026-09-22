/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#include "heartbeat/heartbeat_task.h"

#include "ttc/ttc_timer.h"
#include "sd/sd_task.h"
#include "log/log.h"

#include "FreeRTOS.h"
#include "task.h"

#include <stdint.h>

#define HEARTBEAT_TASK_NAME "heartbeat"
#define HEARTBEAT_PERIOD_TICKS pdMS_TO_TICKS(HEARTBEAT_PERIOD_MS)

static void heartbeat_task_main(void *argument)
{
    TickType_t last_wake = xTaskGetTickCount();
    uint32_t count = 0U;

    (void)argument;

    for (;;) {
        vTaskDelayUntil(&last_wake, HEARTBEAT_PERIOD_TICKS);
        ++count;
        LOG_RUNTIME("heartbeat",
                    "count=%u tick=%u uptime_ms=%u sd=%s\r\n",
                    (unsigned int)count,
                    (unsigned int)xTaskGetTickCount(),
                    (unsigned int)ttc_timer_elapsed_ms(),
                    sd_task_is_available() ? "available" : "unavailable");
    }
}

bool heartbeat_task_create(UBaseType_t priority, uint16_t stack_words)
{
    return xTaskCreate(heartbeat_task_main, HEARTBEAT_TASK_NAME, stack_words,
                       NULL, priority, NULL) == pdPASS;
}
