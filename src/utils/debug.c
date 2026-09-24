/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#include "debug.h"

#include "FreeRTOS.h"
#include "semphr.h"
#include "task.h"
#include "xil_printf.h"

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>

#define DEBUG_BUFFER_SIZE 256U
#define DEBUG_MUTEX_WAIT_TICKS ((TickType_t)1U)

static SemaphoreHandle_t debug_mutex; /* Serializes UART writes after concurrent tasks start. */

static bool debug_scheduler_is_running(void)
{
#if (INCLUDE_xTaskGetSchedulerState == 1)
    /* Protect UART output once concurrent FreeRTOS tasks can log at the same time. */
    return xTaskGetSchedulerState() == taskSCHEDULER_RUNNING;
#else
    return false;
#endif
}

void debug_init(void)
{
    /* Create one shared mutex lazily so early startup logging needs no scheduler. */
    if (debug_mutex == NULL) {
        debug_mutex = xSemaphoreCreateMutex();
    }
}

void debug_printf(const char *format, ...)
{
    char buffer[DEBUG_BUFFER_SIZE]; /* Bounded line buffer passed to the UART driver. */
    va_list arguments; /* Variadic values used to format the caller's message. */

    if (format == NULL) {
        return;
    }

    buffer[sizeof(buffer) - 1U] = '\0';
    va_start(arguments, format);
    (void)vsnprintf(buffer, sizeof(buffer), format, arguments);
    va_end(arguments);
    buffer[sizeof(buffer) - 1U] = '\0';

    /* Before scheduler startup there can be no competing task writers. */
    if (debug_mutex == NULL || !debug_scheduler_is_running()) {
        xil_printf("%s", buffer);
        return;
    }

    if (xSemaphoreTake(debug_mutex, DEBUG_MUTEX_WAIT_TICKS) != pdTRUE) {
        return;
    }

    xil_printf("%s", buffer);
    (void)xSemaphoreGive(debug_mutex);
}
