/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#ifndef EYE_TRACK_CLI_ROUTER_HPP
#define EYE_TRACK_CLI_ROUTER_HPP

#include "cli/cli_types.h"

#include <cstddef>
#include <memory>
#include <string_view>

namespace eye_track::cli {

class Router final {
public:
    Router();
    ~Router();

    Router(const Router &) = delete;
    Router &operator=(const Router &) = delete;

    /* Registers a regular-expression type used by placeholder command nodes. */
    bool register_parameter(std::string_view placeholder /* Placeholder spelling including underscores. */,
                            std::string_view expression /* Regex used to validate one complete token. */);
    /* Inserts a command definition into the token tree. */
    bool register_command(const cli_command_definition_t &command /* Pattern and handler to register. */);
    /* Normalizes input, resolves a tree path, and dispatches the matched command. */
    cli_process_result_t process(std::string_view input /* Raw command text supplied by the user. */,
                                 char *output /* Handler response buffer. */,
                                 std::size_t output_size /* Response buffer capacity. */);
    /* Writes all registered help strings into a bounded output buffer. */
    bool write_help(char *output /* Destination for concatenated help lines. */,
                    std::size_t output_size /* Capacity of the destination buffer. */) const;
    std::size_t command_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace eye_track::cli

#endif
