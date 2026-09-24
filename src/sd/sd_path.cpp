/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#include "sd/sd_path.h"
#include "log/log.h"

#include <regex>
#include <cstring>
#include <string_view>

extern "C" bool sd_path_is_valid(const char *path, bool allow_root)
{
    if (path == nullptr) {
        return false;
    }

    const std::string_view value(path); /* Non-owning view used for syntax checks without allocation. */
    if (value == "0:/") {
        return allow_root;
    }

    try {
        static const std::regex shape( /* Allowed drive-prefixed names; only alphanumerics, underscore, dot, and slash. */
            "^0:/([A-Za-z0-9_.]+/)*[A-Za-z0-9_.]+$");
        if (!std::regex_match(value.begin(), value.end(), shape)) {
            return false;
        }
    } catch (...) {
        return false;
    }

    std::size_t start = 3U; /* First path component after the 0:/ drive prefix. */
    while (start < value.size()) {
        const std::size_t end = value.find('/', start); /* Slash ending the current component, if present. */
        const std::string_view component = value.substr(
            start, end == std::string_view::npos ? value.size() - start
                                                  : end - start);
        if (component == "." || component == "..") {
            return false;
        }
        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1U;
    }
    return true;
}

extern "C" bool sd_path_normalize_cli(const char *path, char *normalized,
                                        size_t normalized_size,
                                        bool allow_root)
{
    if (path == nullptr || normalized == nullptr || normalized_size == 0U) {
        return false;
    }

    const std::string_view value(path); /* View of the original CLI path used for prefix and capacity checks. */
    if (value.size() >= 3U && value.substr(0U, 3U) == "0:/") {
        if (value.size() + 1U > normalized_size) {
            return false;
        }
        std::memmove(normalized, path, value.size() + 1U);
    } else if (!value.empty() && value.front() == '/') {
        if (value.size() + 3U > normalized_size) {
            return false;
        }
        std::memmove(normalized + 2U, path, value.size() + 1U);
        normalized[0] = '0';
        normalized[1] = ':';
    } else {
        return false;
    }
    return sd_path_is_valid(normalized, allow_root);
}
