/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#ifndef EYE_TRACK_CLI_API_H
#define EYE_TRACK_CLI_API_H

#include "cli/cli_types.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

bool cli_register_parameter(const char *placeholder, const char *expression);
bool cli_register_command(const cli_command_definition_t *command);
cli_process_result_t cli_process_command(const char *input, char *output,
                                         size_t output_size);

size_t cli_command_count(void);
bool cli_api_init(void);

#ifdef __cplusplus
}
#endif

#endif
