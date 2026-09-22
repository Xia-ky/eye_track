/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#ifndef EYE_TRACK_CLI_TYPES_H
#define EYE_TRACK_CLI_TYPES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CLI_MAX_ARGUMENTS 8U

typedef struct {
    const char *data;
    size_t length;
} cli_string_view_t;

typedef struct {
    cli_string_view_t input;
    cli_string_view_t arguments[CLI_MAX_ARGUMENTS];
    size_t argument_count;
} cli_invocation_t;

typedef enum {
    CLI_PROCESS_DONE = 0,
    CLI_PROCESS_MORE = 1
} cli_process_result_t;

typedef cli_process_result_t (*cli_command_handler_t)(
    char *output, size_t output_size, const cli_invocation_t *invocation,
    void *context);

typedef struct {
    const char *pattern;
    const char *help;
    cli_command_handler_t handler;
    void *context;
} cli_command_definition_t;

#ifdef __cplusplus
}
#endif

#endif

