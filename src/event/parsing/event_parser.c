/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#include "event/parsing/event_parser.h"
#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
static bool parse_uint(const char **cursor, uint64_t *value)
{
    char *end; /* First character after the converted unsigned integer. */
    if (!isdigit((unsigned char)**cursor)) return false;
    errno = 0;
    *value = strtoull(*cursor, &end, 10);
    if (errno == ERANGE || end == *cursor) return false;
    *cursor = end;
    return true;
}
event_parse_result_t event_parse_line(const char *line, event_t *event)
{
    const char *cursor = line; /* Current position while consuming record fields. */
    uint64_t timestamp;        /* Parsed event timestamp in microseconds. */
    uint64_t x;                /* Parsed horizontal coordinate before narrowing. */
    uint64_t y;                /* Parsed vertical coordinate before narrowing. */
    uint64_t polarity;         /* Parsed polarity plane before narrowing. */
    if (line == NULL || event == NULL) return EVENT_PARSE_INVALID;
    while (isspace((unsigned char)*cursor)) ++cursor;
    if (*cursor == '\0') return EVENT_PARSE_EMPTY;
    if (!parse_uint(&cursor, &timestamp) || !isspace((unsigned char)*cursor)) return EVENT_PARSE_INVALID;
    while (isspace((unsigned char)*cursor)) ++cursor;
    if (!parse_uint(&cursor, &x) || !isspace((unsigned char)*cursor)) return EVENT_PARSE_INVALID;
    while (isspace((unsigned char)*cursor)) ++cursor;
    if (!parse_uint(&cursor, &y) || !isspace((unsigned char)*cursor)) return EVENT_PARSE_INVALID;
    while (isspace((unsigned char)*cursor)) ++cursor;
    if (!parse_uint(&cursor, &polarity)) return EVENT_PARSE_INVALID;
    while (isspace((unsigned char)*cursor)) ++cursor;
    if (*cursor != '\0' || x >= EVENT_SENSOR_WIDTH || y >= EVENT_SENSOR_HEIGHT ||
            polarity >= EVENT_POLARITY_COUNT) return EVENT_PARSE_INVALID;
    /* Commit output fields only after all input values and ranges are valid. */
    event->timestamp_us = timestamp;
    event->x = (uint16_t)x;
    event->y = (uint16_t)y;
    event->polarity = (uint8_t)polarity;
    return EVENT_PARSE_OK;
}
