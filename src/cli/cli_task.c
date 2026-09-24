/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#include "cli/cli_task.h"

#include "cli/cli_api.h"
#include "log/log.h"

#include "FreeRTOS.h"
#include "task.h"
#include "bspconfig.h"
#include "xil_printf.h"
#include "xuartps_hw.h"

#include <stdbool.h>
#include <stddef.h>

#define CLI_TASK_NAME "cli"
#define CLI_PROMPT "eye_track> "
#define CLI_BACKSPACE_CHAR ((char)0x08)
#define CLI_DELETE_CHAR ((char)0x7F)

/* Set after a CR so the LF of a CRLF pair is swallowed exactly once. */
static bool cli_ignore_lf; /* Remembers CR so the following LF in a CRLF pair is ignored. */

static bool cli_task_try_read(char *received)
{
    /* Avoid the BSP inbyte helper because it blocks while UART input is idle. */
    if (received == NULL ||
            XUartPs_IsReceiveData(STDIN_BASEADDRESS) == 0U) {
        return false;
    }
    *received = (char)XUartPs_RecvByte(STDIN_BASEADDRESS);
    return true;
}

cli_line_result_t cli_line_accept_char(char input, char *line,
                                       size_t capacity, size_t *length)
{
    /* Keep the line bounded and interpret terminal editing/control characters first. */
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
    /* Erase one visible terminal character when the line editor accepts backspace. */
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
    cli_process_result_t more; /* Indicates whether the command has more output to emit. */

    /* Re-enter handlers that implement paginated output until they finish. */
    do {
        more = cli_process_command(line, output, output_size);
        xil_printf("%s", output);
    } while (more == CLI_PROCESS_MORE);
}

static void cli_task_main(void *argument)
{
    static char input_line[CLI_TASK_INPUT_BUFFER_SIZE]; /* Persistent command buffer kept off the task stack. */
    static char output_buffer[CLI_TASK_OUTPUT_BUFFER_SIZE]; /* Persistent response buffer reused by every command. */
    size_t length = 0U; /* Number of valid command bytes currently stored. */

    (void)argument;

    xil_printf("%s", CLI_PROMPT);
    for (;;) {
        char received; /* UART byte read by this task on the current iteration. */
        size_t before = length; /* Line length before processing received, for terminal echo. */
        if (!cli_task_try_read(&received)) {
            /* Yield CPU while UART RX is idle; inbyte() busy-waits forever. */
            vTaskDelay(1U);
            continue;
        }
        cli_line_result_t result = /* Line-editor outcome controlling dispatch and prompt behavior. */
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
