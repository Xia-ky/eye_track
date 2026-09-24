/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#include "ttc_timer.h"
#include "log/log.h"

#include "xparameters.h"
#include "xstatus.h"
#include "xttcps.h"

#include <limits.h>

static XTtcPs ttc_timer_instance; /* Xilinx TTC driver instance used to read the hardware counter. */
static bool ttc_timer_ready; /* Whether the TTC instance has been configured and started. */
static uint32_t ttc_timer_input_clock_hz; /* Input rate used to convert counter ticks to time. */

bool ttc_timer_init(void)
{
    XTtcPs_Config *config = NULL; /* Configuration located from generated platform data. */

    ttc_timer_ready = false;
    ttc_timer_input_clock_hz = 0U;

#if defined(SDT)
#if defined(XPAR_XTTCPS_2_BASEADDR)
    config = XTtcPs_LookupConfig(XPAR_XTTCPS_2_BASEADDR);
#endif
#else
#if defined(XPAR_XTTCPS_2_DEVICE_ID)
    config = XTtcPs_LookupConfig(XPAR_XTTCPS_2_DEVICE_ID);
#endif
#endif

    /* Reject missing or unusable platform configuration before touching hardware. */
    if (config == NULL || config->InputClockHz == 0U) {
        LOG_ERROR("ttc", "configuration unavailable\r\n");
        return false;
    }
    if (XTtcPs_CfgInitialize(&ttc_timer_instance, config,
            config->BaseAddress) != XST_SUCCESS) {
        LOG_ERROR("ttc", "initialization failed\r\n");
        return false;
    }
    if (XTtcPs_SetOptions(&ttc_timer_instance,
            XTTCPS_OPTION_WAVE_DISABLE) != XST_SUCCESS) {
        LOG_ERROR("ttc", "option setup failed\r\n");
        return false;
    }

    /* Store the generated clock before starting the counter and exposing readiness. */
    ttc_timer_input_clock_hz = config->InputClockHz;
    XTtcPs_Start(&ttc_timer_instance);
    ttc_timer_ready = true;
    LOG_DEBUG("ttc", "started at %u Hz\r\n",
              (unsigned int)ttc_timer_input_clock_hz);
    return true;
}

bool ttc_timer_is_ready(void)
{
    return ttc_timer_ready;
}

uint32_t ttc_timer_counter(void)
{
    if (!ttc_timer_ready) {
        return 0U;
    }
    return (uint32_t)XTtcPs_GetCounterValue(&ttc_timer_instance);
}

uint32_t ttc_timer_elapsed_ms(void)
{
    uint64_t elapsed_ms; /* 64-bit intermediate prevents overflow during millisecond scaling. */

    if (!ttc_timer_ready || ttc_timer_input_clock_hz == 0U) {
        return 0U;
    }

    /* Diagnostic snapshot: counter * 1000 / generated InputClockHz. */
    elapsed_ms = ((uint64_t)ttc_timer_counter() * UINT64_C(1000)) /
        (uint64_t)ttc_timer_input_clock_hz;
    if (elapsed_ms > UINT32_MAX) {
        return UINT32_MAX;
    }
    return (uint32_t)elapsed_ms;
}
