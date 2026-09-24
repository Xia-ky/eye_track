/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#ifndef EYE_TRACK_APP_HPP
#define EYE_TRACK_APP_HPP

#include "app/app_init.h"
#include "app/sys_cli.h"

namespace eye_track {
class Application final {
public:
    /* C++ entry point that delegates startup to the C application module. */
    static int run() { return app_run(); }
};

class SystemCli final {
public:
    /* Registers system commands through the shared C CLI registry. */
    static bool register_commands() { return sys_cli_register(); }
};
} // namespace eye_track

#endif
