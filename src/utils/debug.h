/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#ifndef EYE_TRACK_DEBUG_H
#define EYE_TRACK_DEBUG_H

#ifdef __cplusplus
extern "C" {
#endif

void debug_init(void);
/* Writes a bounded printf-style message to the board's UART console. */
void debug_printf(const char *format /* printf-style format string. */, ...);

#ifdef __cplusplus
}
#endif

#endif
