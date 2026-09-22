/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#ifndef EYE_TRACK_DEBUG_H
#define EYE_TRACK_DEBUG_H

#ifdef __cplusplus
extern "C" {
#endif

void debug_init(void);
void debug_printf(const char *format, ...);

#ifdef __cplusplus
}
#endif

#endif
