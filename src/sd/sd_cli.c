/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#include "sd/sd_task.h"

#include "cli/cli_api.h"
#include "log/log.h"
#include "sd/sd_path.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SD_CLI_TIMEOUT_TICKS pdMS_TO_TICKS(2000U)

typedef struct {
    sd_request_type_t type;
    const char *name;
    size_t path_count;
    bool allow_root;
} sd_cli_operation_t;

typedef struct {
    bool active;
    bool tail;
    char path[SD_TASK_PATH_MAX];
    uint32_t offset;
} sd_cli_cat_context_t;

static bool sd_cli_copy_view(cli_string_view_t view, char *destination,
                             size_t destination_size)
{
    if (view.data == NULL || view.length == 0U ||
            view.length >= destination_size) {
        return false;
    }
    (void)memcpy(destination, view.data, view.length);
    destination[view.length] = '\0';
    return true;
}

static bool sd_cli_copy_path(cli_string_view_t view, char *destination,
                             size_t destination_size, bool allow_root)
{
    char input[SD_TASK_PATH_MAX];

    return sd_cli_copy_view(view, input, sizeof(input)) &&
           sd_path_normalize_cli(input, destination, destination_size,
                                 allow_root);
}

static void sd_cli_clear_output(char *output, size_t output_size)
{
    if (output != NULL && output_size > 0U) {
        output[0] = '\0';
    }
}

static cli_process_result_t sd_cli_operation_handler(
    char *output, size_t output_size, const cli_invocation_t *invocation,
    void *context)
{
    const sd_cli_operation_t *operation =
        (const sd_cli_operation_t *)context;
    sd_request_options_t options;
    sd_response_t response;
    char path[SD_TASK_PATH_MAX];
    char destination[SD_TASK_PATH_MAX];

    (void)memset(&options, 0, sizeof(options));
    options.type = operation->type;
    if (operation->path_count >= 1U) {
        if (invocation->argument_count < operation->path_count ||
                !sd_cli_copy_path(invocation->arguments[0], path,
                                  sizeof(path), operation->allow_root)) {
            (void)snprintf(output, output_size, "%s: invalid path\r\n",
                           operation->name);
            return CLI_PROCESS_DONE;
        }
        options.path = path;
    }
    if (operation->path_count == 2U) {
        if (!sd_cli_copy_path(invocation->arguments[1], destination,
                              sizeof(destination), false)) {
            (void)snprintf(output, output_size,
                           "%s: invalid destination\r\n", operation->name);
            return CLI_PROCESS_DONE;
        }
        options.destination = destination;
    }
    if (!sd_task_submit(&options, &response, SD_CLI_TIMEOUT_TICKS)) {
        sd_cli_clear_output(output, output_size);
        return CLI_PROCESS_DONE;
    }
    if (operation->type == SD_REQUEST_LIST &&
            response.result.status == SD_CARD_OK) {
        (void)snprintf(output, output_size, "%s",
                       response.text[0] == '\0' ? "(empty)\r\n"
                                                 : response.text);
        return CLI_PROCESS_DONE;
    }
    sd_cli_clear_output(output, output_size);
    return CLI_PROCESS_DONE;
}

static bool sd_cli_parse_uint(cli_string_view_t view, uint32_t *value)
{
    char text[16];
    char *end;
    unsigned long parsed;

    if (!sd_cli_copy_view(view, text, sizeof(text))) {
        return false;
    }
    parsed = strtoul(text, &end, 10);
    if (*end != '\0' || parsed > UINT32_MAX) {
        return false;
    }
    *value = (uint32_t)parsed;
    return true;
}

static cli_process_result_t sd_cli_cat_handler(
    char *output, size_t output_size, const cli_invocation_t *invocation,
    void *context)
{
    sd_cli_cat_context_t *cat = (sd_cli_cat_context_t *)context;
    sd_request_options_t options;
    sd_response_t response;

    if (!cat->active) {
        uint32_t line_count;
        if (invocation->argument_count < 1U ||
                !sd_cli_copy_path(invocation->arguments[0], cat->path,
                                  sizeof(cat->path), false)) {
            (void)snprintf(output, output_size, "sd cat: invalid path\r\n");
            return CLI_PROCESS_DONE;
        }
        cat->offset = 0U;
        cat->active = true;
        if (cat->tail) {
            if (invocation->argument_count < 2U ||
                    !sd_cli_parse_uint(invocation->arguments[1],
                                       &line_count)) {
                cat->active = false;
                (void)snprintf(output, output_size,
                               "sd cat: invalid line count\r\n");
                return CLI_PROCESS_DONE;
            }
            (void)memset(&options, 0, sizeof(options));
            options.type = SD_REQUEST_FIND_TAIL;
            options.path = cat->path;
            options.line_count = line_count;
            if (!sd_task_submit(&options, &response,
                                SD_CLI_TIMEOUT_TICKS)) {
                cat->active = false;
                sd_cli_clear_output(output, output_size);
                return CLI_PROCESS_DONE;
            }
            if (response.result.status != SD_CARD_OK) {
                cat->active = false;
                sd_cli_clear_output(output, output_size);
                return CLI_PROCESS_DONE;
            }
            cat->offset = response.offset;
        }
    }

    (void)memset(&options, 0, sizeof(options));
    options.type = SD_REQUEST_READ_AT;
    options.path = cat->path;
    options.offset = cat->offset;
    if (!sd_task_submit(&options, &response, SD_CLI_TIMEOUT_TICKS)) {
        cat->active = false;
        sd_cli_clear_output(output, output_size);
        return CLI_PROCESS_DONE;
    }
    if (response.result.status != SD_CARD_OK) {
        cat->active = false;
        sd_cli_clear_output(output, output_size);
        return CLI_PROCESS_DONE;
    }
    (void)snprintf(output, output_size, "%s", response.text);
    cat->offset = response.offset;
    if (response.end_of_file) {
        cat->active = false;
        return CLI_PROCESS_DONE;
    }
    return CLI_PROCESS_MORE;
}

static const sd_cli_operation_t status_operation = {
    SD_REQUEST_STATUS, "sd status", 0U, false
};
static const sd_cli_operation_t mount_operation = {
    SD_REQUEST_MOUNT, "sd mount", 0U, false
};
static const sd_cli_operation_t list_root_operation = {
    SD_REQUEST_LIST, "sd ls", 0U, true
};
static const sd_cli_operation_t list_operation = {
    SD_REQUEST_LIST, "sd ls", 1U, true
};
static const sd_cli_operation_t touch_operation = {
    SD_REQUEST_TOUCH, "sd touch", 1U, false
};
static const sd_cli_operation_t mkdir_operation = {
    SD_REQUEST_MKDIR, "sd mkdir", 1U, false
};
static const sd_cli_operation_t remove_operation = {
    SD_REQUEST_REMOVE, "sd rm", 1U, false
};
static const sd_cli_operation_t move_operation = {
    SD_REQUEST_MOVE, "sd mv", 2U, false
};
static const sd_cli_operation_t copy_operation = {
    SD_REQUEST_COPY, "sd cp", 2U, false
};
static const sd_cli_operation_t write_test_operation = {
    SD_REQUEST_WRITE_TEST, "sd write-test", 0U, false
};
static sd_cli_cat_context_t cat_context;
static sd_cli_cat_context_t tail_context = { false, true, { 0 }, 0U };

#define SD_COMMAND(pattern_, help_, handler_, context_) \
    { pattern_, help_, handler_, (void *)(context_) }

static const cli_command_definition_t commands[] = {
    SD_COMMAND("sd status", "sd status - show SD status",
               sd_cli_operation_handler, &status_operation),
    SD_COMMAND("sd mount", "sd mount - mount the card",
               sd_cli_operation_handler, &mount_operation),
    SD_COMMAND("sd ls", "sd ls - list the root directory",
               sd_cli_operation_handler, &list_root_operation),
    SD_COMMAND("sd ls _PATH_", "sd ls <path> - list a directory",
               sd_cli_operation_handler, &list_operation),
    SD_COMMAND("sd touch _PATH_", "sd touch <path> - create a file",
               sd_cli_operation_handler, &touch_operation),
    SD_COMMAND("sd mkdir _PATH_", "sd mkdir <path> - create a directory",
               sd_cli_operation_handler, &mkdir_operation),
    SD_COMMAND("sd rm _PATH_", "sd rm <path> - remove a file or directory",
               sd_cli_operation_handler, &remove_operation),
    SD_COMMAND("sd mv _PATH_ _PATH_", "sd mv <src> <dst> - move a path",
               sd_cli_operation_handler, &move_operation),
    SD_COMMAND("sd cp _PATH_ _PATH_", "sd cp <src> <dst> - copy a file",
               sd_cli_operation_handler, &copy_operation),
    SD_COMMAND("sd cat _PATH_", "sd cat <path> - print a text file",
               sd_cli_cat_handler, &cat_context),
    SD_COMMAND("sd cat _PATH_ --last _UINT_",
               "sd cat <path> --last <n> - print the last n lines",
               sd_cli_cat_handler, &tail_context),
    SD_COMMAND("sd write-test", "sd write-test - run the SD write test",
               sd_cli_operation_handler, &write_test_operation)
};

bool sd_task_cli_register(void)
{
    size_t index;

    if (!cli_register_parameter("_PATH_",
                                "^(0:)?/[A-Za-z0-9_./]*$")) {
        LOG_ERROR("sd_cli", "failed to register _PATH_ parameter\r\n");
        return false;
    }
    for (index = 0U; index < sizeof(commands) / sizeof(commands[0]); ++index) {
        if (!cli_register_command(&commands[index])) {
            LOG_ERROR("sd_cli", "failed at command %u: %s\r\n",
                      (unsigned int)index, commands[index].pattern);
            return false;
        }
    }
    LOG_RUNTIME("sd_cli", "registered %u commands\r\n",
                (unsigned int)(sizeof(commands) / sizeof(commands[0])));
    return true;
}
