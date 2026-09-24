/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#include "app/sys_cli.h"

#include "cli/cli_api.h"
#include "log/log.h"
#include "sd/sd.h"
#include "ttc/ttc_timer.h"

#include "FreeRTOS.h"
#include "task.h"

#include <stdio.h>

static cli_process_result_t sys_cli_status_handler(
    char *pcWriteBuffer /* CLI response buffer populated by this handler. */,
    size_t xWriteBufferLen /* Capacity of the CLI response buffer in bytes. */,
    const cli_invocation_t *invocation /* Parsed command invocation, unused by status. */,
    void *context /* Optional command-specific context, unused here. */)
{
    (void)invocation;
    (void)context;
    /* Report service/card/timer state without performing blocking SD operations. */
    (void)snprintf(pcWriteBuffer, xWriteBufferLen,
                   "sd_service=%s sd_mounted=%s ttc=%s\r\n",
                   sd_task_is_available() ? "up" : "down",
                   sd_card_is_mounted() ? "yes" : "no",
                   ttc_timer_is_ready() ? "ready" : "unavailable");
    return CLI_PROCESS_DONE;
}

static cli_process_result_t sys_cli_uptime_handler(
    char *pcWriteBuffer /* CLI response buffer populated by this handler. */,
    size_t xWriteBufferLen /* Capacity of the CLI response buffer in bytes. */,
    const cli_invocation_t *invocation /* Parsed command invocation, unused by uptime. */,
    void *context /* Optional command-specific context, unused here. */)
{
    (void)invocation;
    (void)context;
    (void)snprintf(pcWriteBuffer, xWriteBufferLen,
                   "tick=%u uptime_ms=%u ttc=%s\r\n",
                   (unsigned int)xTaskGetTickCount(),
                   (unsigned int)ttc_timer_elapsed_ms(),
                   ttc_timer_is_ready() ? "ready" : "unavailable");
    return CLI_PROCESS_DONE;
}

static cli_process_result_t sys_cli_tasks_handler(
    char *pcWriteBuffer /* CLI response buffer populated by this handler. */,
    size_t xWriteBufferLen /* Capacity of the CLI response buffer in bytes. */,
    const cli_invocation_t *invocation /* Parsed command invocation, unused by tasks. */,
    void *context /* Optional command-specific context, unused here. */)
{
    (void)invocation;
    (void)context;
#if (configUSE_TRACE_FACILITY == 1) && \
    (configUSE_STATS_FORMATTING_FUNCTIONS == 1)
    vTaskList(pcWriteBuffer);
    (void)xWriteBufferLen;
#else
    (void)snprintf(pcWriteBuffer, xWriteBufferLen,
                   "task statistics unavailable (enable "
                   "configUSE_TRACE_FACILITY and "
                   "configUSE_STATS_FORMATTING_FUNCTIONS)\r\n");
#endif
    return CLI_PROCESS_DONE;
}

static cli_process_result_t sys_cli_reboot_handler(
    char *pcWriteBuffer /* CLI response buffer populated by this handler. */,
    size_t xWriteBufferLen /* Capacity of the CLI response buffer in bytes. */,
    const cli_invocation_t *invocation /* Parsed command invocation, unused by reboot. */,
    void *context /* Optional command-specific context, unused here. */)
{
    (void)invocation;
    (void)context;
    (void)snprintf(pcWriteBuffer, xWriteBufferLen,
                   "reboot is not supported by this platform\r\n");
    return CLI_PROCESS_DONE;
}

static const cli_command_definition_t sys_cli_status_definition = {
    "status", "status - show SD service, card and timer status",
    sys_cli_status_handler, NULL
};

static const cli_command_definition_t sys_cli_uptime_definition = {
    "uptime", "uptime - show tick count and elapsed milliseconds",
    sys_cli_uptime_handler, NULL
};

static const cli_command_definition_t sys_cli_tasks_definition = {
    "tasks", "tasks - list FreeRTOS tasks when statistics are enabled",
    sys_cli_tasks_handler, NULL
};

static const cli_command_definition_t sys_cli_reboot_definition = {
    "reboot", "reboot - report whether platform reset is supported",
    sys_cli_reboot_handler, NULL
};

bool sys_cli_register(void)
{
    bool registered = true; /* Aggregates success across all system command registrations. */

    /* Attempt every registration so the log identifies a partial registry failure. */
    if (!cli_register_command(&sys_cli_status_definition)) {
        registered = false;
    }
    if (!cli_register_command(&sys_cli_uptime_definition)) {
        registered = false;
    }
    if (!cli_register_command(&sys_cli_tasks_definition)) {
        registered = false;
    }
    if (!cli_register_command(&sys_cli_reboot_definition)) {
        registered = false;
    }
    if (!registered) {
        LOG_ERROR("sys_cli", "one or more commands failed to register\r\n");
    }
    return registered;
}
