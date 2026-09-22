/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#ifndef EYE_TRACK_CLI_H
#define EYE_TRACK_CLI_H

/*
 * CLI 模块对外接口（伞形头文件）。
 *
 * 其他模块只 include 本文件；cli_api.h / cli_task.h 是模块内部声明站点。
 *  - cli_api.h ：命令注册接口（各功能模块注册自己的命令）
 *  - cli_task.h：控制台任务（行编辑 / 回显 / 分发）
 */

#include "cli/cli_api.h"
#include "cli/cli_task.h"

#endif
