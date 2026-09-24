/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#include "app/app.hpp"
#include "log/log.hpp"

/* Delegate process startup to the application façade. */
int main()
{
    return eye_track::Application::run();
}
