/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#ifndef EYE_TRACK_CLI_TYPES_H
#define EYE_TRACK_CLI_TYPES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CLI_MAX_ARGUMENTS 8U

typedef struct {
    const char *data; /* Start address of a non-owning string slice. */
    size_t length; /* Number of bytes in the slice, excluding any terminator. */
} cli_string_view_t;

typedef struct {
    cli_string_view_t input; /* Normalized command text passed to the router. */
    cli_string_view_t arguments[CLI_MAX_ARGUMENTS]; /* Captured placeholder values in pattern order. */
    size_t argument_count; /* Number of valid entries in arguments. */
} cli_invocation_t;

typedef enum {
    CLI_PROCESS_DONE = 0,
    CLI_PROCESS_MORE = 1
} cli_process_result_t;

typedef cli_process_result_t (*cli_command_handler_t)(
    char *output /* Buffer in which the handler writes its response. */,
    size_t output_size /* Capacity of the response buffer in bytes. */,
    const cli_invocation_t *invocation /* Parsed input and captured arguments. */,
    void *context /* Command-specific opaque state. */);

typedef struct {
    const char *pattern; /* Space-separated command pattern with optional placeholders. */
    const char *help; /* Help text displayed by the built-in help command. */
    cli_command_handler_t handler; /* Function invoked when the pattern matches. */
    void *context; /* Opaque state forwarded to the handler. */
} cli_command_definition_t;

#ifdef __cplusplus
}
#endif

#endif
