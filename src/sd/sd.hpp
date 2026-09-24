/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */
#ifndef EYE_TRACK_SD_HPP
#define EYE_TRACK_SD_HPP

#include "sd/sd.h"

namespace eye_track {
class SdCard final {
public:
    /* The card façade forwards operations to the C FatFs module. */
    static sd_card_result_t mount() { return sd_card_mount(); }
    static bool is_mounted() { return sd_card_is_mounted(); }
    static sd_card_result_t list(const char *path /* Directory path to enumerate. */,
                                 char *buffer /* Destination for listing text. */,
                                 std::size_t size /* Capacity of the listing buffer. */)
    { return sd_card_list(path, buffer, size); }
    static sd_card_result_t read_text(const char *path /* File path to read. */,
                                      char *buffer /* Destination for file text. */,
                                      std::size_t size /* Capacity of the text buffer. */)
    { return sd_card_read_text(path, buffer, size); }
    static sd_card_result_t read_text_at(const char *path /* File path to read. */,
                                         uint32_t offset /* Byte offset where reading starts. */,
                                         char *buffer /* Destination for file text. */,
                                         std::size_t size /* Capacity of the text buffer. */,
                                         bool &end_of_file /* Receives whether this read reaches EOF. */)
    { return sd_card_read_text_at(path, offset, buffer, size, &end_of_file); }
    static sd_card_result_t find_tail_offset(const char *path /* File path to scan. */,
                                             uint32_t lines /* Number of trailing lines to retain. */,
                                             uint32_t &offset /* Receives the starting byte offset. */)
    { return sd_card_find_tail_offset(path, lines, &offset); }
    static sd_card_result_t touch(const char *path)
    { return sd_card_touch(path); }
    static sd_card_result_t mkdir(const char *path)
    { return sd_card_mkdir(path); }
    static sd_card_result_t remove(const char *path)
    { return sd_card_remove(path); }
    static sd_card_result_t move(const char *source /* Existing path to rename. */,
                                 const char *destination /* New path for the source. */)
    { return sd_card_move(source, destination); }
    static sd_card_result_t copy(const char *source /* File path to copy from. */,
                                 const char *destination /* File path to copy to. */)
    { return sd_card_copy(source, destination); }
    static sd_card_result_t run_write_test(uint32_t tick)
    { return sd_card_run_write_test(tick); }
};

class SdTask final {
public:
    /* Starts the serialized SD filesystem service task. */
    static bool create(UBaseType_t priority, uint16_t stack_words)
    { return sd_task_create(priority, stack_words); }
    static bool submit(const sd_request_options_t &options /* Operation and input fields to submit. */,
                       sd_response_t &response /* Receives the matching service response. */,
                       TickType_t timeout /* Maximum wait for submission and response. */)
    { return sd_task_submit(&options, &response, timeout); }
    static bool is_available() { return sd_task_is_available(); }
    static bool register_cli() { return sd_task_cli_register(); }
};
} // namespace eye_track

#endif
