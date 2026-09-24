/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#ifndef EYE_TRACK_SD_PATH_H
#define EYE_TRACK_SD_PATH_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Validates an SD path's syntax and rejects current/parent directory components. */
bool sd_path_is_valid(const char *path /* Path to validate. */,
                      bool allow_root /* Whether drive root is permitted. */);
/* Converts accepted CLI slash paths to FatFs drive notation and validates them. */
bool sd_path_normalize_cli(const char *path /* User-entered absolute path. */,
                           char *normalized /* Destination for normalized drive path. */,
                           size_t normalized_size /* Capacity of normalized. */,
                           bool allow_root /* Whether drive root is permitted. */);

#ifdef __cplusplus
}
#endif

#endif
