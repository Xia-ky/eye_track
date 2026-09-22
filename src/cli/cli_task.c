/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#include "cli/cli_task.h"

#include "cli/cli_api.h"
#include "log/log.h"

#include "FreeRTOS.h"
#include "task.h"
#include "xil_printf.h"

#include <stdbool.h>
#include <stddef.h>

#define CLI_TASK_NAME "cli"
#define CLI_PROMPT "eye_track> "
#define CLI_BACKSPACE_CHAR ((char)0x08)
#define CLI_DELETE_CHAR ((char)0x7F)

/* Set after a CR so the LF of a CRLF pair is swallowed exactly once. */
static bool cli_ignore_lf;

cli_line_result_t cli_line_accept_char(char input, char *line,
                                       size_t capacity, size_t *length)
{
    if (line == NULL || length == NULL || capacity == 0U) {
        return CLI_LINE_IGNORED;
    }

    if (input == '\n') {
        if (cli_ignore_lf) {
            cli_ignore_lf = false;
            return CLI_LINE_IGNORED;
        }
        line[*length] = '\0';
        return CLI_LINE_READY;
    }
    cli_ignore_lf = false;

    if (input == '\r') {
        line[*length] = '\0';
        cli_ignore_lf = true;
        return CLI_LINE_READY;
    }
    if (input == CLI_BACKSPACE_CHAR || input == CLI_DELETE_CHAR) {
        if (*length == 0U) {
            return CLI_LINE_IGNORED;
        }
        --*length;
        line[*length] = '\0';
        return CLI_LINE_PENDING;
    }
    if (input < ' ' || input > '~') {
        return CLI_LINE_IGNORED;
    }
    if (*length + 1U >= capacity) {
        return CLI_LINE_OVERFLOW;
    }

    line[*length] = input;
    ++*length;
    line[*length] = '\0';
    return CLI_LINE_PENDING;
}

static void cli_task_echo_pending(char received, size_t before, size_t after)
{
    if (received == CLI_BACKSPACE_CHAR || received == CLI_DELETE_CHAR) {
        if (after < before) {
            xil_printf("\b \b");
        }
        return;
    }
    xil_printf("%c", received);
}

static void cli_task_run_command(const char *line, char *output,
                                 size_t output_size)
{
    cli_process_result_t more;

    do {
        more = cli_process_command(line, output, output_size);
        xil_printf("%s", output);
    } while (more == CLI_PROCESS_MORE);
}

static void cli_task_main(void *argument)
{
    static char input_line[CLI_TASK_INPUT_BUFFER_SIZE];
    static char output_buffer[CLI_TASK_OUTPUT_BUFFER_SIZE];
    size_t length = 0U;

    (void)argument;

    xil_printf("%s", CLI_PROMPT);
    for (;;) {
        char received = inbyte();
        size_t before = length;
        cli_line_result_t result =
            cli_line_accept_char(received, input_line, sizeof(input_line),
                                 &length);

        switch (result) {
        case CLI_LINE_READY:
            LOG_DEBUG("cli", "command received: %s\r\n", input_line);
            xil_printf("\r\n");
            cli_task_run_command(input_line, output_buffer,
                                 sizeof(output_buffer));
            length = 0U;
            input_line[0] = '\0';
            xil_printf("%s", CLI_PROMPT);
            break;
        case CLI_LINE_OVERFLOW:
            LOG_ERROR("cli", "input line exceeded %u bytes\r\n",
                      (unsigned int)sizeof(input_line));
            xil_printf("\r\nerror: line too long\r\n%s", CLI_PROMPT);
            length = 0U;
            input_line[0] = '\0';
            break;
        case CLI_LINE_IGNORED:
            break;
        case CLI_LINE_PENDING:
        default:
            cli_task_echo_pending(received, before, length);
            break;
        }
    }
}

bool cli_task_create(UBaseType_t priority, uint16_t stack_words)
{
    bool created = xTaskCreate(cli_task_main, CLI_TASK_NAME, stack_words,
                              NULL, priority, NULL) == pdPASS;
    if (!created) {
        LOG_ERROR("cli", "task creation failed\r\n");
    }
    return created;
}
