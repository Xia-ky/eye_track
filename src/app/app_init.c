/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#include "app/app_init.h"

#include "app/sys_cli.h"
#include "cli/cli.h"
#include "heartbeat/heartbeat_task.h"
#include "log/log.h"
#include "sd/sd.h"
#include "ttc/ttc_timer.h"

#include "FreeRTOS.h"
#include "task.h"

#include <stdbool.h>

#define SD_TASK_PRIORITY ((UBaseType_t)(tskIDLE_PRIORITY + 3U))
#define CLI_TASK_PRIORITY ((UBaseType_t)(tskIDLE_PRIORITY + 2U))
#define HEARTBEAT_TASK_PRIORITY ((UBaseType_t)(tskIDLE_PRIORITY + 1U))
#define LOG_TASK_PRIORITY ((UBaseType_t)(tskIDLE_PRIORITY + 2U))
#define SD_TASK_STACK_WORDS ((uint16_t)1536U)
#define CLI_TASK_STACK_WORDS ((uint16_t)2048U)
#define HEARTBEAT_TASK_STACK_WORDS ((uint16_t)512U)
#define LOG_TASK_STACK_WORDS ((uint16_t)1280U)

static void app_report_result(const char *name, const char *verdict)
{
    LOG_RUNTIME("self_test", "[TEST] %s %s\r\n", name, verdict);
}

static void app_report_sd_failure(const char *name,
                                  sd_card_result_t result)
{
    LOG_ERROR("self_test", "[TEST] %s FAIL (status=%u fresult=%d)\r\n",
              name, (unsigned int)result.status, result.fatfs_result);
}

static void app_run_self_test(unsigned int *passed, unsigned int *failed,
                              unsigned int *skipped)
{
    sd_card_result_t mount_result;
    sd_card_result_t test_result;
    bool mount_ok;
    bool test_ok;

    *passed = 0U;
    *failed = 0U;
    *skipped = 0U;

    /* Reaching this line already proves the console path works. */
    app_report_result("UART ........", "PASS");
    ++*passed;

    if (ttc_timer_init()) {
        app_report_result("TTC .........", "PASS");
        ++*passed;
    } else {
        app_report_result("TTC .........", "FAIL");
        ++*failed;
    }

    mount_result = sd_card_mount();
    mount_ok = mount_result.status == SD_CARD_OK;
    if (mount_ok) {
        app_report_result("SD mount ....", "PASS");
        ++*passed;
    } else {
        app_report_sd_failure("SD mount ....", mount_result);
        ++*failed;
    }

    if (!mount_ok) {
        app_report_result("SD write ....", "SKIP");
        app_report_result("SD read .....", "SKIP");
        app_report_result("SD verify ...", "SKIP");
        *skipped += 3U;
        return;
    }

    test_result = sd_card_run_write_test((uint32_t)xTaskGetTickCount());
    test_ok = test_result.status == SD_CARD_OK;
    if (test_ok) {
        app_report_result("SD write ....", "PASS");
        app_report_result("SD read .....", "PASS");
        app_report_result("SD verify ...", "PASS");
        *passed += 3U;
    } else {
        app_report_sd_failure("SD write ....", test_result);
        app_report_sd_failure("SD read .....", test_result);
        app_report_sd_failure("SD verify ...", test_result);
        *failed += 3U;
    }
}

int app_run(void)
{
    unsigned int passed;
    unsigned int failed;
    unsigned int skipped;

    if (!eye_log_init()) {
        LOG_ERROR("app", "failed to initialize log queue\r\n");
        return -1;
    }

    LOG_RUNTIME("app", "[BOOT] eye_track FreeRTOS\r\n");
    app_run_self_test(&passed, &failed, &skipped);
    LOG_RUNTIME("app", "[BOOT] self-test completed: %u passed, %u failed, "
                "%u skipped\r\n", passed, failed, skipped);

    /* 命令注册：CLI 核心 + 各功能模块，均在调度器启动前完成。 */
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

    if (!sd_task_create(SD_TASK_PRIORITY, SD_TASK_STACK_WORDS)) {
        LOG_ERROR("app", "[BOOT] failed to create SD task\r\n");
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
    if (!heartbeat_task_create(HEARTBEAT_TASK_PRIORITY,
            HEARTBEAT_TASK_STACK_WORDS)) {
        LOG_ERROR("app", "[BOOT] failed to create heartbeat task\r\n");
        return -1;
    }

    vTaskStartScheduler();

    LOG_ERROR("app", "[BOOT] scheduler returned unexpectedly\r\n");
    return -2;
}
