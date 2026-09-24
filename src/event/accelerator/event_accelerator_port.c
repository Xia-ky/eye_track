/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#include "event/accelerator/event_accelerator_port.h"
#include <stdbool.h>
#if defined(__GNUC__)
__attribute__((weak))
#endif
bool event_accelerator_is_configured(void)
{
    return false;
}
#if defined(__GNUC__)
__attribute__((weak))
#endif
event_accelerator_result_t event_accelerator_submit(const event_tensor_t *tensor, uint32_t frame_index)
{
    (void)tensor;
    (void)frame_index;
    return EVENT_ACCELERATOR_NOT_CONFIGURED;
}
