/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#ifndef EYE_TRACK_EVENT_PARSER_H
#define EYE_TRACK_EVENT_PARSER_H
#include "event/types/event_types.h"
/* Result of converting one whitespace-separated record into an event. */
typedef enum { EVENT_PARSE_OK = 0, EVENT_PARSE_EMPTY, EVENT_PARSE_INVALID } event_parse_result_t;
/* Parses one t/x/y/p record and validates its coordinates and polarity. */
event_parse_result_t event_parse_line(const char *line /* Null-terminated input record. */,
                                      event_t *event /* Output event populated on success. */);
#endif
