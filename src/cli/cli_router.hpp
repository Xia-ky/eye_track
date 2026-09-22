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

    bool register_parameter(std::string_view placeholder,
                            std::string_view expression);
    bool register_command(const cli_command_definition_t &command);
    cli_process_result_t process(std::string_view input, char *output,
                                 std::size_t output_size);
    bool write_help(char *output, std::size_t output_size) const;
    std::size_t command_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace eye_track::cli

#endif
