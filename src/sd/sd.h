/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#ifndef EYE_TRACK_SD_H
#define EYE_TRACK_SD_H

/*
 * SD 卡模块对外接口（伞形头文件）。
 *
 * 其他模块只 include 本文件；sd_task.h / sd_card.h 是模块内部声明站点。
 *  - sd_card.h：驱动层（FATFS 直接操作、结果类型）
 *  - sd_task.h：服务层（请求队列、异步提交、CLI 注册）
 */

#include "sd/sd_card.h"
#include "sd/sd_task.h"

#endif
