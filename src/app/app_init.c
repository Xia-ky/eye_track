/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#include "app/app_init.h"

#include "app/sys_cli.h"
#include "cli/cli.h"
#include "event/pipeline/event_pipeline.h"
#include "event/acquisition/file_event_source.h"
#include "log/log.h"
#include "sd/sd.h"
#include "ttc/ttc_timer.h"

#include "FreeRTOS.h"
#include "task.h"

#include <stdbool.h>

#define SD_TASK_PRIORITY ((UBaseType_t)(tskIDLE_PRIORITY + 1U))
#define CLI_TASK_PRIORITY ((UBaseType_t)(tskIDLE_PRIORITY + 2U))
#define LOG_TASK_PRIORITY ((UBaseType_t)(tskIDLE_PRIORITY + 1U))
#define SD_TASK_STACK_WORDS ((uint16_t)1536U)
#define CLI_TASK_STACK_WORDS ((uint16_t)2048U)
#define LOG_TASK_STACK_WORDS ((uint16_t)1280U)

static void app_run_self_test(void)
{
    sd_card_result_t mount_result; /* Outcome of mounting the SD card during startup. */
    sd_card_result_t test_result; /* Outcome of the SD write/read/verify startup check. */
    bool ttc_ok = ttc_timer_init(); /* Whether the TTC timer initialized successfully. */
    bool sd_ok = false; /* Whether SD mount and data verification both succeeded. */

    /* Mount the card before running the bounded write/read verification. */
    mount_result = sd_card_mount();
    if (mount_result.status != SD_CARD_OK) {
        LOG_ERROR("self_test", "SD mount failed (status=%u, fresult=%d)\r\n",
                  (unsigned int)mount_result.status,
                  mount_result.fatfs_result);
    } else {
        test_result = sd_card_run_write_test((uint32_t)xTaskGetTickCount());
        sd_ok = test_result.status == SD_CARD_OK;
        if (!sd_ok) {
            LOG_ERROR("self_test",
                      "SD read/write verification failed (status=%u, fresult=%d)\r\n",
                      (unsigned int)test_result.status,
                      test_result.fatfs_result);
        }
    }

    if (!ttc_ok) LOG_ERROR("self_test", "TTC initialization failed\r\n");
    if (ttc_ok && sd_ok) {
        LOG_RUNTIME("app",
                    "[BOOT] startup checks passed (TTC, SD mount/read/write/verify)\r\n");
    } else {
        LOG_ERROR("app", "[BOOT] startup checks failed (TTC=%s, SD=%s)\r\n",
                  ttc_ok ? "ok" : "failed",
                  sd_ok ? "ok" : "failed");
    }
}

int app_run(void)
{
    static file_event_source_t file_source; /* Persistent storage referenced by the file source callbacks. */
    event_source_t event_source; /* Generic input interface passed into the event pipeline. */

    /* Initialize logging first so subsequent startup failures can be reported. */
    if (!eye_log_init()) {
        LOG_ERROR("app", "failed to initialize log queue\r\n");
        return -1;
    }

    LOG_RUNTIME("app", "[BOOT] eye_track FreeRTOS\r\n");
    app_run_self_test();

    /* Register CLI commands before the scheduler starts, while registration is single-threaded. */
    if (!cli_api_init()) {
        LOG_ERROR("app", "[BOOT] failed to init CLI registry\r\n");
        return -1;
    }
    if (!sys_cli_register()) {
        LOG_ERROR("app", "[BOOT] failed to register system CLI commands\r\n");
        return -1;
    }
    if (!sd_task_cli_register()) {
        LOG_ERROR("app", "[BOOT] failed to register sd CLI command\r\n");
        return -1;
    }

    /* Start the SD service before event ingestion and log persistence can submit requests. */
    if (!sd_task_create(SD_TASK_PRIORITY, SD_TASK_STACK_WORDS)) {
        LOG_ERROR("app", "[BOOT] failed to create SD task\r\n");
        return -1;
    }
    if (!file_event_source_init(&file_source, "0:/test/1_1/1_1.txt")) {
        LOG_ERROR("app", "[BOOT] invalid event file source configuration\r\n");
        return -1;
    }
    /* Bind file-backed input behind the generic event-source interface. */
    event_source = file_event_source_interface(&file_source);
    if (!event_pipeline_start(&event_source)) {
        LOG_ERROR("app", "[BOOT] failed to start event pipeline\r\n");
        return -1;
    }
    if (!eye_log_task_create(LOG_TASK_PRIORITY, LOG_TASK_STACK_WORDS)) {
        LOG_ERROR("app", "[BOOT] failed to create log task\r\n");
        return -1;
    }
    if (!cli_task_create(CLI_TASK_PRIORITY, CLI_TASK_STACK_WORDS)) {
        LOG_ERROR("app", "[BOOT] failed to create CLI task\r\n");
        return -1;
    }
    LOG_RUNTIME("app", "[BOOT] tasks created; starting scheduler\r\n");
    /* Control should not return unless the scheduler failed to start. */
    vTaskStartScheduler();

    LOG_ERROR("app", "[BOOT] scheduler returned unexpectedly\r\n");
    return -2;
}
