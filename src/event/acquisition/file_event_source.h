/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#ifndef EYE_TRACK_FILE_EVENT_SOURCE_H
#define EYE_TRACK_FILE_EVENT_SOURCE_H
#include "event/acquisition/event_source.h"
#include "sd/sd_task.h"
#define EVENT_FILE_LINE_MAX 64U
/* Buffered state for reading event records from an SD-card text file. */
typedef struct {
    char path[SD_TASK_PATH_MAX];       /* SD path opened by this file-backed source. */
    sd_response_t response;            /* Reusable response buffer for SD stream operations. */
    char line[EVENT_FILE_LINE_MAX];    /* Partial text record assembled across SD chunks. */
    size_t chunk_offset;               /* Next unread byte offset in the current SD response. */
    size_t chunk_length;               /* Number of valid bytes in the current SD response. */
    size_t line_length;                /* Number of bytes currently stored in line. */
    bool opened;                       /* Whether the SD stream has been opened. */
    bool eof;                          /* Whether the current response indicates end of file. */
    bool pending_final_line;           /* Whether a final non-newline-terminated record remains. */
} file_event_source_t;
/* Initializes source state and copies the SD path into caller-owned storage. */
bool file_event_source_init(file_event_source_t *source /* File-source instance to initialize. */,
                            const char *path /* SD path containing event records. */);
/* Returns the generic callback interface bound to this file-source instance. */
event_source_t file_event_source_interface(file_event_source_t *source /* File-source state bound to the callbacks. */);
#endif
