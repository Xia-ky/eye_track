/* Copyright (c) 2026 eye_track contributors. SPDX-License-Identifier: MIT */

#include "sd_card.h"
#include "log/log.h"

#include "ff.h"

#include <stdio.h>
#include <string.h>

#define SD_CARD_DRIVE_PATH "0:/"
#define SD_CARD_SELFTEST_PATH "0:/eye_track_selftest.txt"
#define SD_CARD_COPY_BUFFER_SIZE 4096U

static FATFS sd_card_fatfs; /* FatFs work area registered for drive 0. */
static bool sd_card_mounted; /* Last successful mount state exposed to callers. */
static FIL sd_card_stream_file; /* Persistent file object used by sequential stream requests. */
static bool sd_card_stream_is_open; /* Tracks whether the persistent stream file needs closing. */
/* FatFs access is serialized by sd_task, so one fixed workspace is enough. */
static unsigned char sd_card_copy_buffer[SD_CARD_COPY_BUFFER_SIZE]; /* Fixed copy workspace kept out of task stacks. */

static sd_card_status_t sd_card_status_from_fresult(FRESULT result)
{
    /* Translate common FatFs failures into stable application-level status codes. */
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
    /* Preserve both portable status and native FatFs details in one result value. */
    sd_card_result_t result; /* Fully populated result returned to the API caller. */

    result.status = sd_card_status_from_fresult(fatfs_result);
    result.fatfs_result = (int)fatfs_result;
    result.bytes_used = bytes_used;
    return result;
}

static sd_card_result_t sd_card_local_result(sd_card_status_t status)
{
    /* Represent validation or lifecycle failures that did not originate in FatFs. */
    sd_card_result_t result; /* Result for validation/state errors that have no FatFs code. */

    result.status = status;
    result.fatfs_result = (int)FR_OK;
    result.bytes_used = 0U;
    return result;
}

static sd_card_result_t sd_card_close_after(FIL *file, sd_card_result_t result)
{
    /* Always close temporary files and prefer a close error only when no prior error exists. */
    FRESULT close_result = f_close(file); /* Preserve close failure when the operation itself succeeded. */

    if (result.status == SD_CARD_OK && close_result != FR_OK) {
        return sd_card_result(close_result, result.bytes_used);
    }
    return result;
}

sd_card_result_t sd_card_mount(void)
{
    /* Close any active stream before remounting the filesystem volume. */
    if (sd_card_stream_is_open) {
        (void)f_close(&sd_card_stream_file);
        sd_card_stream_is_open = false;
    }
    FRESULT mount_result = f_mount(&sd_card_fatfs, SD_CARD_DRIVE_PATH, 1U); /* Native FatFs mount outcome. */

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
    /* The SD task serializes mount changes with all filesystem operations. */
    return sd_card_mounted;
}

sd_card_result_t sd_card_touch(const char *path)
{
    /* Open-or-create and immediately close a file without changing its contents. */
    FIL file; /* FatFs file object opened only to create a missing file. */
    FRESULT result; /* Outcome of opening the requested file. */

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
    /* Delegate directory creation to FatFs after validating module state. */
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
    /* Remove a filesystem entry using the native FatFs unlink operation. */
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
    /* Rename a path without copying its contents. */
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
    FIL input; /* Source file handle used for sequential reads. */
    FIL output; /* Destination file handle used for sequential writes. */
    FRESULT result; /* Native result of the current FatFs read/write operation. */
    UINT bytes_read; /* Bytes returned by the most recent source read. */
    UINT bytes_written; /* Bytes accepted by the most recent destination write. */
    size_t total = 0U; /* Total successfully written bytes reported to the caller. */
    sd_card_result_t card_result; /* Preserved failure/status returned after handles close. */

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

    /* Copy in bounded chunks so file size does not affect task stack usage. */
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
    /* Append a bounded text record and report short writes as I/O failures. */
    FIL file; /* FatFs handle opened in append mode for the target file. */
    FRESULT result; /* Outcome of the append write operation. */
    UINT bytes_written = 0U; /* Number of payload bytes accepted by FatFs. */
    size_t text_length; /* Input payload length checked before narrowing to UINT. */
    sd_card_result_t card_result; /* Write result retained while closing the file. */

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

sd_card_result_t sd_card_stream_open(const char *path)
{
    /* Open the single persistent stream handle used by sequential readers. */
    FRESULT result; /* FatFs result from opening the persistent stream file. */

    if (path == NULL || path[0] == '\0') {
        return sd_card_local_result(SD_CARD_INVALID_ARGUMENT);
    }
    if (!sd_card_mounted) {
        return sd_card_local_result(SD_CARD_NOT_MOUNTED);
    }
    if (sd_card_stream_is_open) {
        return sd_card_local_result(SD_CARD_IO_ERROR);
    }
    result = f_open(&sd_card_stream_file, path, FA_READ);
    if (result == FR_OK) {
        sd_card_stream_is_open = true;
    }
    return sd_card_result(result, 0U);
}

sd_card_result_t sd_card_stream_read(char *buffer, size_t buffer_size,
                                     bool *end_of_file)
{
    /* Read one bounded chunk and close the handle automatically after I/O failure. */
    FRESULT result; /* FatFs result from reading the current stream position. */
    UINT bytes_read = 0U; /* Actual number of bytes placed in the caller buffer. */

    if (buffer == NULL || end_of_file == NULL || buffer_size < 2U) {
        return sd_card_local_result(SD_CARD_INVALID_ARGUMENT);
    }
    buffer[0] = '\0';
    *end_of_file = false;
    if (!sd_card_stream_is_open) {
        return sd_card_local_result(SD_CARD_NOT_FOUND);
    }
    result = f_read(&sd_card_stream_file, buffer,
                    (UINT)(buffer_size - 1U), &bytes_read);
    buffer[bytes_read] = '\0';
    *end_of_file = f_eof(&sd_card_stream_file) != 0;
    if (result != FR_OK) {
        (void)f_close(&sd_card_stream_file);
        sd_card_stream_is_open = false;
    }
    return sd_card_result(result, (size_t)bytes_read);
}

sd_card_result_t sd_card_stream_close(void)
{
    /* Close an active stream and make repeated close requests harmless. */
    FRESULT result; /* FatFs result from closing the active stream file. */

    if (!sd_card_stream_is_open) {
        return sd_card_local_result(SD_CARD_OK);
    }
    result = f_close(&sd_card_stream_file);
    sd_card_stream_is_open = false;
    return sd_card_result(result, 0U);
}

sd_card_result_t sd_card_read_text_at(const char *path, uint32_t offset,
                                      char *buffer, size_t buffer_size,
                                      bool *end_of_file)
{
    /* Read a bounded page from a byte offset and return pagination metadata. */
    FIL file; /* Temporary FatFs handle for the positioned read. */
    FRESULT result; /* Outcome of open, seek, or read, whichever fails first. */
    UINT bytes_read = 0U; /* Actual bytes returned from the requested offset. */
    FSIZE_t file_size; /* File length used to validate offset and calculate EOF. */
    sd_card_result_t card_result; /* Read result retained until the file handle closes. */

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
    /* Validate the requested position before seeking into the file. */
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
    /* Locate the beginning of the requested trailing lines without loading the file. */
    FIL file; /* Temporary FatFs handle scanned backward from the end. */
    FRESULT result; /* Outcome of the current seek/read operation. */
    FSIZE_t position; /* Current byte position in the reverse scan. */
    FSIZE_t file_size; /* Initial file length used as the scan starting point. */
    uint32_t newlines = 0U; /* Number of line terminators found from the file end. */
    unsigned char value; /* Single byte examined at the current reverse-scan position. */
    UINT bytes_read; /* Number of bytes returned by each one-byte read. */
    sd_card_result_t card_result; /* Final scan result preserved while closing the file. */

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

    /* Walk backward so a tail request reads only the suffix needed for the result. */
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
    /* Read the first bounded portion of a text file and add a terminator. */
    FIL file; /* Temporary FatFs handle for reading from the start of the file. */
    FRESULT result; /* Outcome of the open or read operation. */
    UINT bytes_read = 0U; /* Actual bytes returned before EOF or buffer capacity. */
    sd_card_result_t card_result; /* Read result retained while closing the file. */

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
    /* Enumerate a directory into a bounded listing while preserving file/directory markers. */
    DIR directory; /* FatFs directory handle being enumerated. */
    FILINFO entry; /* Metadata for the current directory entry. */
    FRESULT result; /* Result from opening or reading the directory. */
    size_t used = 0U; /* Number of bytes already written to the output listing. */

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

    /* Read one entry at a time and stop before the bounded output buffer overflows. */
    for (;;) {
        size_t name_length; /* Number of filename bytes emitted for the current entry. */
        size_t entry_length; /* Required output capacity including type, slash, and newline. */
        bool is_directory; /* Whether FatFs marks this entry as a directory. */
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
            sd_card_result_t card_result; /* Listing failure and bytes already written before closing the directory. */
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
    char expected[64]; /* Deterministic self-test payload written to the SD card. */
    char actual[64]; /* Bytes read back for comparison with the expected payload. */
    FIL file; /* FatFs handle reused for the write pass and verification read. */
    FRESULT result; /* Outcome of the current open, write, or read operation. */
    UINT bytes_written = 0U; /* Actual payload bytes persisted by FatFs. */
    UINT bytes_read = 0U; /* Actual payload bytes returned during verification. */
    int formatted_length; /* snprintf result used to detect payload formatting failure. */
    size_t expected_length; /* Valid payload size used for write, read, and comparison. */
    sd_card_result_t card_result; /* Preserved operation result returned after closing the file. */

    if (!sd_card_mounted) {
        return sd_card_local_result(SD_CARD_NOT_MOUNTED);
    }

    /* Include the boot tick so each startup test writes a recognizable payload. */
    formatted_length = snprintf(expected, sizeof(expected),
        "eye_track self-test tick=%u\r\n", (unsigned int)(uint32_t)tick);
    if (formatted_length < 0 || (size_t)formatted_length >= sizeof(expected)) {
        return sd_card_local_result(SD_CARD_IO_ERROR);
    }
    expected_length = (size_t)formatted_length;

    /* Write a fresh test file, close it, then reopen and compare the exact bytes. */
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
