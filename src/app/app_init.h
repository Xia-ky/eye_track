/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#ifndef EYE_TRACK_APP_INIT_H
#define EYE_TRACK_APP_INIT_H

#ifdef __cplusplus
extern "C" {
#endif

/* Initializes services and tasks, then transfers control to the FreeRTOS scheduler. */
int app_run(void);

#ifdef __cplusplus
}
#endif

#endif
