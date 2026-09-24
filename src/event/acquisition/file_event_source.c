/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#include "event/acquisition/file_event_source.h"
#include "event/parsing/event_parser.h"
#include <string.h>
static event_source_result_t refill(file_event_source_t *source)
{
    sd_request_options_t request; /* SD operation descriptor reused for stream open and read. */
    /* Clear fields so this request contains only the options set below. */
    (void)memset(&request, 0, sizeof(request));
    if (!source->opened) {
        request.type = SD_REQUEST_STREAM_OPEN;
        request.path = source->path;
        if (!sd_task_submit(&request, &source->response, portMAX_DELAY) ||
                source->response.result.status != SD_CARD_OK) {
            return EVENT_SOURCE_ERROR;
        }
        source->opened = true;
    }
    request.type = SD_REQUEST_STREAM_READ;
    if (!sd_task_submit(&request, &source->response, portMAX_DELAY) ||
            source->response.result.status != SD_CARD_OK) return EVENT_SOURCE_ERROR;
    source->chunk_offset = 0U;
    source->chunk_length = source->response.result.bytes_used;
    source->eof = source->response.end_of_file;
    return EVENT_SOURCE_EVENT;
}
bool file_event_source_init(file_event_source_t *source, const char *path)
{
    size_t length; /* Number of path characters that must fit in source storage. */
    if (source == NULL || path == NULL) return false;
    length = strlen(path);
    if (length == 0U || length >= sizeof(source->path)) return false;
    (void)memset(source, 0, sizeof(*source));
    (void)memcpy(source->path, path, length + 1U);
    return true;
}
static event_source_result_t next_event(void *context, event_t *event)
{
    file_event_source_t *source = (file_event_source_t *)context; /* Reader state bound to the generic source interface. */
    if (source == NULL || event == NULL) return EVENT_SOURCE_ERROR;
    for (;;) {
        if (source->chunk_offset >= source->chunk_length) {
            if (source->eof) {
                if (source->line_length != 0U && source->pending_final_line) {
                    event_parse_result_t parsed; /* Parse result for the final unterminated input record. */
                    source->line[source->line_length] = '\0';
                    source->line_length = 0U;
                    source->pending_final_line = false;
                    parsed = event_parse_line(source->line, event);
                    if (parsed == EVENT_PARSE_OK) return EVENT_SOURCE_EVENT;
                    return parsed == EVENT_PARSE_EMPTY ? EVENT_SOURCE_EOF : EVENT_SOURCE_ERROR;
                }
                return EVENT_SOURCE_EOF;
            }
            if (refill(source) != EVENT_SOURCE_EVENT) return EVENT_SOURCE_ERROR;
            if (source->chunk_length == 0U && source->eof) {
                source->pending_final_line = source->line_length != 0U;
                continue;
            }
        }
        /* Consume the response chunk byte-by-byte, assembling complete records. */
        while (source->chunk_offset < source->chunk_length) {
            char value = source->response.text[source->chunk_offset++]; /* Current byte consumed from the SD response. */
            if (value == '\n') {
                event_parse_result_t parsed; /* Parse result for the record completed by this newline. */
                source->line[source->line_length] = '\0';
                source->line_length = 0U;
                parsed = event_parse_line(source->line, event);
                if (parsed == EVENT_PARSE_OK) return EVENT_SOURCE_EVENT;
                if (parsed == EVENT_PARSE_EMPTY) continue;
                return EVENT_SOURCE_ERROR;
            }
            if (source->line_length + 1U >= sizeof(source->line)) return EVENT_SOURCE_ERROR;
            source->line[source->line_length++] = value;
        }
        if (source->eof && source->line_length != 0U) source->pending_final_line = true;
    }
}
static void close_source(void *context)
{
    file_event_source_t *source = (file_event_source_t *)context; /* Reader state bound to the generic source interface. */
    if (source != NULL && source->opened) {
        sd_request_options_t request; /* Request descriptor used to close the SD stream. */
        (void)memset(&request, 0, sizeof(request));
        request.type = SD_REQUEST_STREAM_CLOSE;
        (void)sd_task_submit(&request, &source->response, portMAX_DELAY);
        source->opened = false;
    }
}
event_source_t file_event_source_interface(file_event_source_t *source)
{
    event_source_t interface = { next_event, close_source, source }; /* Callbacks and reader context exposed to consumers. */
    return interface;
}
