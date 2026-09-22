/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#ifndef EYE_TRACK_SD_PATH_H
#define EYE_TRACK_SD_PATH_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

bool sd_path_is_valid(const char *path, bool allow_root);
bool sd_path_normalize_cli(const char *path, char *normalized,
                           size_t normalized_size, bool allow_root);

#ifdef __cplusplus
}
#endif

#endif
