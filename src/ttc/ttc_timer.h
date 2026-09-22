/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#ifndef EYE_TRACK_TTC_TIMER_H
#define EYE_TRACK_TTC_TIMER_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool ttc_timer_init(void);
bool ttc_timer_is_ready(void);
uint32_t ttc_timer_counter(void);
/* Diagnostic counter snapshot converted from the generated TTC clock. */
uint32_t ttc_timer_elapsed_ms(void);

#ifdef __cplusplus
}
#endif

#endif
