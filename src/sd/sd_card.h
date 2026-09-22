/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#ifndef EYE_TRACK_SD_CARD_H
#define EYE_TRACK_SD_CARD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SD_CARD_OK = 0,
    SD_CARD_NOT_MOUNTED,
    SD_CARD_NO_FILESYSTEM,
    SD_CARD_NOT_READY,
    SD_CARD_WRITE_PROTECTED,
    SD_CARD_NOT_FOUND,
    SD_CARD_INVALID_ARGUMENT,
    SD_CARD_BUFFER_TOO_SMALL,
    SD_CARD_IO_ERROR
} sd_card_status_t;

typedef struct {
    sd_card_status_t status;
    int fatfs_result;
    size_t bytes_used;
} sd_card_result_t;

sd_card_result_t sd_card_mount(void);
sd_card_result_t sd_card_list(const char *path, char *buffer, size_t buffer_size);
sd_card_result_t sd_card_read_text(const char *path, char *buffer, size_t buffer_size);
sd_card_result_t sd_card_read_text_at(const char *path, uint32_t offset,
                                      char *buffer, size_t buffer_size,
                                      bool *end_of_file);
sd_card_result_t sd_card_find_tail_offset(const char *path,
                                          uint32_t line_count,
                                          uint32_t *offset);
sd_card_result_t sd_card_touch(const char *path);
sd_card_result_t sd_card_mkdir(const char *path);
sd_card_result_t sd_card_remove(const char *path);
sd_card_result_t sd_card_move(const char *source, const char *destination);
sd_card_result_t sd_card_copy(const char *source, const char *destination);
sd_card_result_t sd_card_append_text(const char *path, const char *text);
sd_card_result_t sd_card_run_write_test(uint32_t tick);
bool sd_card_is_mounted(void);

#ifdef __cplusplus
}
#endif

#endif
