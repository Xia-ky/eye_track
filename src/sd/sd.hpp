/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#ifndef EYE_TRACK_SD_HPP
#define EYE_TRACK_SD_HPP

#include "sd/sd.h"

namespace eye_track {
class SdCard final {
public:
    static sd_card_result_t mount() { return sd_card_mount(); }
    static bool is_mounted() { return sd_card_is_mounted(); }
    static sd_card_result_t list(const char *path, char *buffer,
                                 std::size_t size)
    { return sd_card_list(path, buffer, size); }
    static sd_card_result_t read_text(const char *path, char *buffer,
                                      std::size_t size)
    { return sd_card_read_text(path, buffer, size); }
    static sd_card_result_t read_text_at(const char *path, uint32_t offset,
                                         char *buffer, std::size_t size,
                                         bool &end_of_file)
    { return sd_card_read_text_at(path, offset, buffer, size, &end_of_file); }
    static sd_card_result_t find_tail_offset(const char *path, uint32_t lines,
                                             uint32_t &offset)
    { return sd_card_find_tail_offset(path, lines, &offset); }
    static sd_card_result_t touch(const char *path)
    { return sd_card_touch(path); }
    static sd_card_result_t mkdir(const char *path)
    { return sd_card_mkdir(path); }
    static sd_card_result_t remove(const char *path)
    { return sd_card_remove(path); }
    static sd_card_result_t move(const char *source, const char *destination)
    { return sd_card_move(source, destination); }
    static sd_card_result_t copy(const char *source, const char *destination)
    { return sd_card_copy(source, destination); }
    static sd_card_result_t run_write_test(uint32_t tick)
    { return sd_card_run_write_test(tick); }
};

class SdTask final {
public:
    static bool create(UBaseType_t priority, uint16_t stack_words)
    { return sd_task_create(priority, stack_words); }
    static bool submit(const sd_request_options_t &options,
                       sd_response_t &response, TickType_t timeout)
    { return sd_task_submit(&options, &response, timeout); }
    static bool is_available() { return sd_task_is_available(); }
    static bool register_cli() { return sd_task_cli_register(); }
};
} // namespace eye_track

#endif
