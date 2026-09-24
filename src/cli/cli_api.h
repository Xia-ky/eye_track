/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#ifndef EYE_TRACK_CLI_API_H
#define EYE_TRACK_CLI_API_H

#include "cli/cli_types.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Registers a named placeholder and its full-token regular expression. */
bool cli_register_parameter(const char *placeholder /* Placeholder token such as _PATH_. */,
                            const char *expression /* ECMAScript regex that accepts valid values. */);
/* Adds one literal/placeholder command pattern to the global router. */
bool cli_register_command(const cli_command_definition_t *command /* Command pattern, handler, and metadata. */);
/* Normalizes and dispatches one input line to its matching command handler. */
cli_process_result_t cli_process_command(const char *input /* Null-terminated command line. */,
                                         char *output /* Response buffer populated by the handler. */,
                                         size_t output_size /* Capacity of the response buffer in bytes. */);

size_t cli_command_count(void);
bool cli_api_init(void);

#ifdef __cplusplus
}
#endif

#endif
