/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#ifndef EYE_TRACK_EVENT_ACCELERATOR_PORT_H
#define EYE_TRACK_EVENT_ACCELERATOR_PORT_H
#include "event/tensor/event_tensor.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum { EVENT_ACCELERATOR_OK = 0, EVENT_ACCELERATOR_NOT_CONFIGURED, EVENT_ACCELERATOR_TRANSFER_ERROR } event_accelerator_result_t;
/* Reports whether a concrete PL/DMA accelerator implementation is available. */
bool event_accelerator_is_configured(void);
/* Implementations must return only after PL/DMA no longer reads tensor->data. */
/* Transfers one complete frame and waits until hardware no longer reads its data. */
event_accelerator_result_t event_accelerator_submit(const event_tensor_t *tensor /* Completed tensor buffer to submit. */,
                                                    uint32_t frame_index /* Sequence number used for diagnostics. */);
#ifdef __cplusplus
}
#endif
#endif
