/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#include "sd_card.h"
#include "log/log.h"

#include "ff.h"

#include <stdio.h>
#include <string.h>

#define SD_CARD_DRIVE_PATH "0:/"
#define SD_CARD_SELFTEST_PATH "0:/eye_track_selftest.txt"
#define SD_CARD_COPY_BUFFER_SIZE 4096U

static FATFS sd_card_fatfs;
static bool sd_card_mounted;
/* FatFs access is serialized by sd_task, so one fixed workspace is enough. */
static unsigned char sd_card_copy_buffer[SD_CARD_COPY_BUFFER_SIZE];

static sd_card_status_t sd_card_status_from_fresult(FRESULT result)
{
    switch (result) {
    case FR_OK:
        return SD_CARD_OK;
    case FR_NO_FILESYSTEM:
        return SD_CARD_NO_FILESYSTEM;
    case FR_NOT_READY:
        return SD_CARD_NOT_READY;
    case FR_WRITE_PROTECTED:
        return SD_CARD_WRITE_PROTECTED;
    case FR_NO_FILE:
    case FR_NO_PATH:
        return SD_CARD_NOT_FOUND;
    case FR_INVALID_NAME:
    case FR_INVALID_DRIVE:
    case FR_INVALID_PARAMETER:
        return SD_CARD_INVALID_ARGUMENT;
    default:
        return SD_CARD_IO_ERROR;
    }
}

static sd_card_result_t sd_card_result(FRESULT fatfs_result, size_t bytes_used)
{
    sd_card_result_t result;

    result.status = sd_card_status_from_fresult(fatfs_result);
    result.fatfs_result = (int)fatfs_result;
    result.bytes_used = bytes_used;
    return result;
}

static sd_card_result_t sd_card_local_result(sd_card_status_t status)
{
    sd_card_result_t result;

    result.status = status;
    result.fatfs_result = (int)FR_OK;
    result.bytes_used = 0U;
    return result;
}

static sd_card_result_t sd_card_close_after(FIL *file, sd_card_result_t result)
{
    FRESULT close_result = f_close(file);

    if (result.status == SD_CARD_OK && close_result != FR_OK) {
        return sd_card_result(close_result, result.bytes_used);
    }
    return result;
}

sd_card_result_t sd_card_mount(void)
{
    FRESULT mount_result = f_mount(&sd_card_fatfs, SD_CARD_DRIVE_PATH, 1U);

    sd_card_mounted = mount_result == FR_OK;
    if (sd_card_mounted) {
        LOG_RUNTIME("sd", "card mounted\r\n");
    } else {
        LOG_ERROR("sd", "mount failed: %d\r\n", (int)mount_result);
    }
    return sd_card_result(mount_result, 0U);
}

bool sd_card_is_mounted(void)
{
    return sd_card_mounted;
}

sd_card_result_t sd_card_touch(const char *path)
{
    FIL file;
    FRESULT result;

    if (path == NULL) {
        return sd_card_local_result(SD_CARD_INVALID_ARGUMENT);
    }
    if (!sd_card_mounted) {
        return sd_card_local_result(SD_CARD_NOT_MOUNTED);
    }
    result = f_open(&file, path, FA_WRITE | FA_OPEN_ALWAYS);
    if (result != FR_OK) {
        return sd_card_result(result, 0U);
    }
    return sd_card_close_after(&file, sd_card_result(FR_OK, 0U));
}

sd_card_result_t sd_card_mkdir(const char *path)
{
    if (path == NULL) {
        return sd_card_local_result(SD_CARD_INVALID_ARGUMENT);
    }
    if (!sd_card_mounted) {
        return sd_card_local_result(SD_CARD_NOT_MOUNTED);
    }
    return sd_card_result(f_mkdir(path), 0U);
}

sd_card_result_t sd_card_remove(const char *path)
{
    if (path == NULL) {
        return sd_card_local_result(SD_CARD_INVALID_ARGUMENT);
    }
    if (!sd_card_mounted) {
        return sd_card_local_result(SD_CARD_NOT_MOUNTED);
    }
    return sd_card_result(f_unlink(path), 0U);
}

sd_card_result_t sd_card_move(const char *source, const char *destination)
{
    if (source == NULL || destination == NULL) {
        return sd_card_local_result(SD_CARD_INVALID_ARGUMENT);
    }
    if (!sd_card_mounted) {
        return sd_card_local_result(SD_CARD_NOT_MOUNTED);
    }
    return sd_card_result(f_rename(source, destination), 0U);
}

sd_card_result_t sd_card_copy(const char *source, const char *destination)
{
    FIL input;
    FIL output;
    FRESULT result;
    UINT bytes_read;
    UINT bytes_written;
    size_t total = 0U;
    sd_card_result_t card_result;

    if (source == NULL || destination == NULL) {
        return sd_card_local_result(SD_CARD_INVALID_ARGUMENT);
    }
    if (!sd_card_mounted) {
        return sd_card_local_result(SD_CARD_NOT_MOUNTED);
    }
    result = f_open(&input, source, FA_READ);
    if (result != FR_OK) {
        return sd_card_result(result, 0U);
    }
    result = f_open(&output, destination, FA_WRITE | FA_CREATE_ALWAYS);
    if (result != FR_OK) {
        (void)f_close(&input);
        return sd_card_result(result, 0U);
    }

    card_result = sd_card_result(FR_OK, 0U);
    do {
        bytes_read = 0U;
        result = f_read(&input, sd_card_copy_buffer,
                        (UINT)sizeof(sd_card_copy_buffer), &bytes_read);
        if (result != FR_OK) {
            card_result = sd_card_result(result, total);
            break;
        }
        if (bytes_read == 0U) {
            break;
        }
        bytes_written = 0U;
        result = f_write(&output, sd_card_copy_buffer, bytes_read,
                         &bytes_written);
        total += (size_t)bytes_written;
        if (result != FR_OK || bytes_written != bytes_read) {
            card_result = result == FR_OK
                ? sd_card_local_result(SD_CARD_IO_ERROR)
                : sd_card_result(result, total);
            card_result.bytes_used = total;
            break;
        }
    } while (bytes_read == (UINT)sizeof(sd_card_copy_buffer));

    (void)f_close(&input);
    card_result.bytes_used = total;
    return sd_card_close_after(&output, card_result);
}

sd_card_result_t sd_card_append_text(const char *path, const char *text)
{
    FIL file;
    FRESULT result;
    UINT bytes_written = 0U;
    size_t text_length;
    sd_card_result_t card_result;

    if (path == NULL || text == NULL) {
        return sd_card_local_result(SD_CARD_INVALID_ARGUMENT);
    }
    if (!sd_card_mounted) {
        return sd_card_local_result(SD_CARD_NOT_MOUNTED);
    }
    text_length = strlen(text);
    if (text_length > (size_t)((UINT)-1)) {
        return sd_card_local_result(SD_CARD_INVALID_ARGUMENT);
    }

    result = f_open(&file, path, FA_WRITE | FA_OPEN_APPEND);
    if (result != FR_OK) {
        return sd_card_result(result, 0U);
    }
    result = f_write(&file, text, (UINT)text_length, &bytes_written);
    if (result == FR_OK && bytes_written != (UINT)text_length) {
        card_result = sd_card_local_result(SD_CARD_IO_ERROR);
        card_result.bytes_used = (size_t)bytes_written;
    } else {
        card_result = sd_card_result(result, (size_t)bytes_written);
    }
    return sd_card_close_after(&file, card_result);
}

sd_card_result_t sd_card_read_text_at(const char *path, uint32_t offset,
                                      char *buffer, size_t buffer_size,
                                      bool *end_of_file)
{
    FIL file;
    FRESULT result;
    UINT bytes_read = 0U;
    FSIZE_t file_size;
    sd_card_result_t card_result;

    if (path == NULL || buffer == NULL || end_of_file == NULL) {
        return sd_card_local_result(SD_CARD_INVALID_ARGUMENT);
    }
    if (buffer_size < 2U) {
        return sd_card_local_result(SD_CARD_BUFFER_TOO_SMALL);
    }
    buffer[0] = '\0';
    *end_of_file = false;
    if (!sd_card_mounted) {
        return sd_card_local_result(SD_CARD_NOT_MOUNTED);
    }
    result = f_open(&file, path, FA_READ);
    if (result != FR_OK) {
        return sd_card_result(result, 0U);
    }
    file_size = f_size(&file);
    if ((FSIZE_t)offset > file_size) {
        return sd_card_close_after(
            &file, sd_card_local_result(SD_CARD_INVALID_ARGUMENT));
    }
    result = f_lseek(&file, (FSIZE_t)offset);
    if (result == FR_OK) {
        result = f_read(&file, buffer, (UINT)(buffer_size - 1U), &bytes_read);
    }
    buffer[bytes_read] = '\0';
    *end_of_file = ((FSIZE_t)offset + bytes_read) >= file_size;
    card_result = sd_card_result(result, (size_t)bytes_read);
    return sd_card_close_after(&file, card_result);
}

sd_card_result_t sd_card_find_tail_offset(const char *path,
                                          uint32_t line_count,
                                          uint32_t *offset)
{
    FIL file;
    FRESULT result;
    FSIZE_t position;
    FSIZE_t file_size;
    uint32_t newlines = 0U;
    unsigned char value;
    UINT bytes_read;
    sd_card_result_t card_result;

    if (path == NULL || offset == NULL) {
        return sd_card_local_result(SD_CARD_INVALID_ARGUMENT);
    }
    if (!sd_card_mounted) {
        return sd_card_local_result(SD_CARD_NOT_MOUNTED);
    }
    result = f_open(&file, path, FA_READ);
    if (result != FR_OK) {
        return sd_card_result(result, 0U);
    }
    file_size = f_size(&file);
    if (line_count == 0U) {
        *offset = (uint32_t)file_size;
        return sd_card_close_after(&file, sd_card_result(FR_OK, 0U));
    }

    position = file_size;
    while (position > 0U) {
        --position;
        result = f_lseek(&file, position);
        if (result != FR_OK) {
            break;
        }
        bytes_read = 0U;
        result = f_read(&file, &value, 1U, &bytes_read);
        if (result != FR_OK || bytes_read != 1U) {
            break;
        }
        if (value == '\n' && position + 1U != file_size) {
            ++newlines;
            if (newlines == line_count) {
                ++position;
                break;
            }
        }
    }
    if (result == FR_OK) {
        *offset = (uint32_t)position;
    }
    card_result = sd_card_result(result, 0U);
    return sd_card_close_after(&file, card_result);
}

sd_card_result_t sd_card_read_text(const char *path, char *buffer, size_t buffer_size)
{
    FIL file;
    FRESULT result;
    UINT bytes_read = 0U;
    sd_card_result_t card_result;

    if (path == NULL || buffer == NULL) {
        return sd_card_local_result(SD_CARD_INVALID_ARGUMENT);
    }
    if (buffer_size < 2U) {
        return sd_card_local_result(SD_CARD_BUFFER_TOO_SMALL);
    }
    buffer[0] = '\0';
    if (!sd_card_mounted) {
        return sd_card_local_result(SD_CARD_NOT_MOUNTED);
    }

    result = f_open(&file, path, FA_READ);
    if (result != FR_OK) {
        return sd_card_result(result, 0U);
    }

    result = f_read(&file, buffer, (UINT)(buffer_size - 1U), &bytes_read);
    buffer[bytes_read] = '\0';
    card_result = sd_card_result(result, (size_t)bytes_read);
    return sd_card_close_after(&file, card_result);
}

sd_card_result_t sd_card_list(const char *path, char *buffer, size_t buffer_size)
{
    DIR directory;
    FILINFO entry;
    FRESULT result;
    size_t used = 0U;

    if (!sd_card_mounted) {
        return sd_card_local_result(SD_CARD_NOT_MOUNTED);
    }
    if (path == NULL || buffer == NULL) {
        return sd_card_local_result(SD_CARD_INVALID_ARGUMENT);
    }
    if (buffer_size == 0U) {
        return sd_card_local_result(SD_CARD_BUFFER_TOO_SMALL);
    }
    buffer[0] = '\0';

    result = f_opendir(&directory, path);
    if (result != FR_OK) {
        return sd_card_result(result, 0U);
    }

    for (;;) {
        size_t name_length;
        size_t entry_length;
        bool is_directory;
        result = f_readdir(&directory, &entry);
        if (result != FR_OK) {
            sd_card_result_t card_result = sd_card_result(result, used);
            (void)f_closedir(&directory);
            return card_result;
        }
        if (entry.fname[0] == '\0') {
            sd_card_result_t card_result = sd_card_result(FR_OK, used);
            FRESULT close_result = f_closedir(&directory);
            if (close_result != FR_OK) {
                return sd_card_result(close_result, used);
            }
            return card_result;
        }

        name_length = strlen(entry.fname);
        is_directory = (entry.fattrib & AM_DIR) != 0U;
        entry_length = 2U + name_length + (is_directory ? 1U : 0U) + 1U;
        if (entry_length > buffer_size - used - 1U) {
            sd_card_result_t card_result;
            (void)f_closedir(&directory);
            buffer[used] = '\0';
            card_result = sd_card_local_result(SD_CARD_BUFFER_TOO_SMALL);
            card_result.bytes_used = used;
            return card_result;
        }
        buffer[used++] = is_directory ? 'd' : '-';
        buffer[used++] = ' ';
        (void)memcpy(&buffer[used], entry.fname, name_length);
        used += name_length;
        if (is_directory) {
            buffer[used++] = '/';
        }
        buffer[used++] = '\n';
        buffer[used] = '\0';
    }
}

sd_card_result_t sd_card_run_write_test(uint32_t tick)
{
    char expected[64];
    char actual[64];
    FIL file;
    FRESULT result;
    UINT bytes_written = 0U;
    UINT bytes_read = 0U;
    int formatted_length;
    size_t expected_length;
    sd_card_result_t card_result;

    if (!sd_card_mounted) {
        return sd_card_local_result(SD_CARD_NOT_MOUNTED);
    }

    formatted_length = snprintf(expected, sizeof(expected),
        "eye_track self-test tick=%u\r\n", (unsigned int)(uint32_t)tick);
    if (formatted_length < 0 || (size_t)formatted_length >= sizeof(expected)) {
        return sd_card_local_result(SD_CARD_IO_ERROR);
    }
    expected_length = (size_t)formatted_length;

    result = f_open(&file, SD_CARD_SELFTEST_PATH, FA_WRITE | FA_CREATE_ALWAYS);
    if (result != FR_OK) {
        return sd_card_result(result, 0U);
    }
    result = f_write(&file, expected, (UINT)expected_length, &bytes_written);
    if (result == FR_OK && bytes_written != (UINT)expected_length) {
        card_result = sd_card_local_result(SD_CARD_IO_ERROR);
    } else {
        card_result = sd_card_result(result, (size_t)bytes_written);
    }
    card_result = sd_card_close_after(&file, card_result);
    if (card_result.status != SD_CARD_OK) {
        return card_result;
    }

    result = f_open(&file, SD_CARD_SELFTEST_PATH, FA_READ);
    if (result != FR_OK) {
        return sd_card_result(result, 0U);
    }
    result = f_read(&file, actual, (UINT)expected_length, &bytes_read);
    if (result == FR_OK && ((size_t)bytes_read != expected_length ||
            memcmp(actual, expected, expected_length) != 0)) {
        card_result = sd_card_local_result(SD_CARD_IO_ERROR);
    } else {
        card_result = sd_card_result(result, (size_t)bytes_read);
    }
    return sd_card_close_after(&file, card_result);
}
