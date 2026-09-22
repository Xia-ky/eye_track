/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#ifndef EYE_TRACK_SYS_CLI_H
#define EYE_TRACK_SYS_CLI_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 注册系统级 CLI 命令（status / uptime / tasks / reboot）。 */
bool sys_cli_register(void);

#ifdef __cplusplus
}
#endif

#endif
