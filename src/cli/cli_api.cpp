/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#include "cli/cli_api.h"

#include "cli/cli_router.hpp"
#include "log/log.h"

#include <memory>
#include <new>
#include <string_view>

namespace {

std::unique_ptr<eye_track::cli::Router> router;

cli_process_result_t help_handler(char *output, size_t output_size,
                                  const cli_invocation_t *, void *)
{
    if (router == nullptr || !router->write_help(output, output_size)) {
        if (output_size > 0U) {
            output[0] = '\0';
        }
    }
    return CLI_PROCESS_DONE;
}

cli_process_result_t empty_handler(char *output, size_t output_size,
                                   const cli_invocation_t *, void *)
{
    if (output != nullptr && output_size > 0U) {
        output[0] = '\0';
    }
    return CLI_PROCESS_DONE;
}

const cli_command_definition_t help_command = {
    "help", "help - list registered commands", help_handler, nullptr
};

const cli_command_definition_t empty_command = {
    "", nullptr, empty_handler, nullptr
};

} // namespace

extern "C" bool cli_api_init(void)
{
    if (router != nullptr) {
        LOG_ERROR("cli", "registry initialized more than once\r\n");
        return false;
    }
    try {
        router = std::make_unique<eye_track::cli::Router>();
        if (!router->register_command(empty_command) ||
                !router->register_command(help_command)) {
            LOG_ERROR("cli", "failed to register help command\r\n");
            router.reset();
            return false;
        }
        return true;
    } catch (...) {
        LOG_ERROR("cli", "exception while initializing registry\r\n");
        router.reset();
        return false;
    }
}

extern "C" bool cli_register_parameter(const char *placeholder,
                                         const char *expression)
{
    if (router == nullptr || placeholder == nullptr || expression == nullptr) {
        LOG_ERROR("cli", "invalid parameter registration request\r\n");
        return false;
    }
    const bool registered = router->register_parameter(placeholder, expression);
    if (!registered) {
        LOG_ERROR("cli", "failed to register parameter %s\r\n", placeholder);
    }
    return registered;
}

extern "C" bool cli_register_command(
    const cli_command_definition_t *command)
{
    const bool registered = router != nullptr && command != nullptr &&
                            router->register_command(*command);
    if (!registered) {
        LOG_ERROR("cli", "failed to register command %s\r\n",
                  command == nullptr ? "<null>" : command->pattern);
    }
    return registered;
}

extern "C" cli_process_result_t cli_process_command(const char *input,
                                                       char *output,
                                                       size_t output_size)
{
    if (router == nullptr || input == nullptr) {
        if (output != nullptr && output_size > 0U) {
            output[0] = '\0';
        }
        return CLI_PROCESS_DONE;
    }
    try {
        return router->process(std::string_view(input), output, output_size);
    } catch (...) {
        LOG_ERROR("cli", "exception while processing command\r\n");
        if (output != nullptr && output_size > 0U) {
            static constexpr char message[] = "CLI internal error\r\n";
            const size_t count = output_size - 1U < sizeof(message) - 1U
                ? output_size - 1U : sizeof(message) - 1U;
            for (size_t index = 0U; index < count; ++index) {
                output[index] = message[index];
            }
            output[count] = '\0';
        }
        return CLI_PROCESS_DONE;
    }
}

extern "C" size_t cli_command_count(void)
{
    return router == nullptr ? 0U : router->command_count();
}
