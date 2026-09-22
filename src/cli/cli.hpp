/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#ifndef EYE_TRACK_CLI_HPP
#define EYE_TRACK_CLI_HPP

#include "cli/cli_api.h"
#include "cli/cli_task.h"

#include <cstddef>

namespace eye_track {
class Cli final {
public:
    static bool init() { return cli_api_init(); }
    static bool register_parameter(const char *name, const char *expression)
    { return cli_register_parameter(name, expression); }
    static bool register_command(const cli_command_definition_t &command)
    { return cli_register_command(&command); }
    static cli_process_result_t process(const char *input, char *output,
                                        std::size_t output_size)
    { return cli_process_command(input, output, output_size); }
    static std::size_t command_count() { return cli_command_count(); }
};

class CliTask final {
public:
    static bool create(UBaseType_t priority, uint16_t stack_words)
    { return cli_task_create(priority, stack_words); }
    static cli_line_result_t accept_char(char input, char *line,
                                         std::size_t capacity,
                                         std::size_t &length)
    { return cli_line_accept_char(input, line, capacity, &length); }
};
} // namespace eye_track

#endif
