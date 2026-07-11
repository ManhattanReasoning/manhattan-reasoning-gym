/*
 * test_network_stack.c — bare-metal Wishbone TCP framing tests
 *
 * Tests the request-parsing and response-building functions from
 * wishbone_tcp.c directly on the host. The functions under test are
 * the pure-C logic that runs bare-metal on the RISC-V: they parse raw
 * TCP payload bytes, execute Wishbone bus transactions against the user
 * memory region, and pack results back into response buffers.
 *
 * No lwIP, no FPGA, no cross-compiler required. These functions are
 * copied verbatim from firmware/sw/wishbone_tcp.c -- if that file
 * changes, update the matching block here.
 *
 * Full TCP stack confirmation (lwIP, chunked sends, MSS behaviour)
 * requires the live board; see firmware/HARDWARE.md.
 *
 * Build:
 *   gcc -std=c11 -Wall -Wextra -g \
 *       test_network_stack.c -o test_network_stack
 *   ./test_network_stack
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Protocol constants and firmware framing functions
 *
 * Copied verbatim from firmware/sw/wishbone_tcp.c.
 * ---------------------------------------------------------------------- */

#define OP_WRITE     0x01
#define OP_READ      0x02
#define STATUS_OK    0x00
#define STATUS_ERROR 0x01
#define HEADER_LEN   8

#define USER_REGION_WORDS 512
#define REQ_BUF_LEN  (HEADER_LEN + USER_REGION_WORDS * 4)
#define RESP_BUF_LEN (4 + USER_REGION_WORDS * 4)

static uint8_t  req_buf[REQ_BUF_LEN];
static uint32_t req_fill;
static uint8_t  resp_buf[RESP_BUF_LEN];
static uint32_t resp_len;
static uint32_t resp_queued;

/* On hardware this is (volatile uint32_t *)USER_BASE (0x90000000).
 * Here we back it with a plain array so the functions run on the host. */
static uint32_t test_user_mem[USER_REGION_WORDS];
static volatile uint32_t *user_region = (volatile uint32_t *)test_user_mem;

static uint32_t be32_load(const uint8_t *p)
{
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  | (uint32_t)p[3];
}

static void be32_store(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)v;
}

static uint32_t be24_load(const uint8_t *p)
{
    return ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | (uint32_t)p[2];
}

static void be24_store(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v >> 16);
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)v;
}

static uint32_t make_error(void)
{
    resp_buf[0] = STATUS_ERROR;
    be24_store(resp_buf + 1, 0);
    return 4;
}

static uint32_t handle_request(const uint8_t *req, uint32_t req_len)
{
    uint8_t  op     = req[0];
    uint32_t length = be24_load(req + 1);
    uint32_t addr   = be32_load(req + 4);

    if (addr & 3)
        return make_error();
    uint32_t word_addr = addr / 4;

    if (op == OP_WRITE) {
        if (req_len != HEADER_LEN + length * 4)
            return make_error();
        if (length == 0 || word_addr + length > USER_REGION_WORDS)
            return make_error();

        for (uint32_t i = 0; i < length; i++)
            user_region[word_addr + i] = be32_load(req + HEADER_LEN + i * 4);

        resp_buf[0] = STATUS_OK;
        be24_store(resp_buf + 1, 0);
        return 4;
    }

    if (op == OP_READ) {
        if (req_len != HEADER_LEN + 4 || length != 1)
            return make_error();
        uint32_t count = be32_load(req + HEADER_LEN);
        if (count == 0 || word_addr + count > USER_REGION_WORDS)
            return make_error();

        resp_buf[0] = STATUS_OK;
        be24_store(resp_buf + 1, count);
        for (uint32_t i = 0; i < count; i++)
            be32_store(resp_buf + 4 + i * 4, user_region[word_addr + i]);
        return 4 + count * 4;
    }

    return make_error();
}

static uint32_t expected_request_len(void)
{
    if (req_fill < HEADER_LEN)
        return 0;
    uint8_t  op     = req_buf[0];
    uint32_t length = be24_load(req_buf + 1);
    if (op == OP_READ)
        return HEADER_LEN + 4;
    return HEADER_LEN + length * 4;
}

/* -------------------------------------------------------------------------
 * Request buffer builders
 * ---------------------------------------------------------------------- */

static void fill_write_req(uint8_t *buf, uint32_t addr,
                            const uint32_t *data, uint32_t n_words)
{
    buf[0] = OP_WRITE;
    be24_store(buf + 1, n_words);
    be32_store(buf + 4, addr);
    for (uint32_t i = 0; i < n_words; i++)
        be32_store(buf + HEADER_LEN + i * 4, data[i]);
}

static void fill_read_req(uint8_t *buf, uint32_t addr, uint32_t count)
{
    buf[0] = OP_READ;
    be24_store(buf + 1, 1);        /* length field is always 1 for reads */
    be32_store(buf + 4, addr);
    be32_store(buf + HEADER_LEN, count);
}

/* -------------------------------------------------------------------------
 * Test framework — minimal TAP output
 * ---------------------------------------------------------------------- */

static int g_run = 0, g_pass = 0;

#define CHECK(cond, name)                                              \
    do {                                                               \
        g_run++;                                                       \
        if (cond) {                                                    \
            printf("ok %d - %s\n", g_run, name);                      \
            g_pass++;                                                  \
        } else {                                                       \
            printf("not ok %d - %s  (line %d)\n",                     \
                   g_run, name, __LINE__);                             \
        }                                                              \
    } while (0)

static void reset_state(void)
{
    memset(req_buf,       0, sizeof req_buf);
    memset(resp_buf,      0, sizeof resp_buf);
    memset(test_user_mem, 0, sizeof test_user_mem);
    req_fill = resp_len = resp_queued = 0;
}

/* =========================================================================
 * Group 1: Big-endian encode / decode helpers
 * ====================================================================== */

static void test_be32_round_trip(void)
{
    uint8_t buf[4];
    be32_store(buf, 0xDEADBEEF);
    CHECK(be32_load(buf) == 0xDEADBEEF, "T1: be32 round-trip");
}

static void test_be32_byte_order(void)
{
    uint8_t buf[4];
    be32_store(buf, 0x01020304);
    CHECK(buf[0] == 0x01 && buf[1] == 0x02 && buf[2] == 0x03 && buf[3] == 0x04,
          "T2: be32_store writes most-significant byte first");
}

static void test_be24_round_trip(void)
{
    uint8_t buf[3];
    be24_store(buf, 0x123456);
    CHECK(be24_load(buf) == 0x123456, "T3: be24 round-trip");
}

static void test_be24_byte_order(void)
{
    uint8_t buf[3];
    be24_store(buf, 0x010203);
    CHECK(buf[0] == 0x01 && buf[1] == 0x02 && buf[2] == 0x03,
          "T4: be24_store writes most-significant byte first");
}

static void test_be32_max_value(void)
{
    uint8_t buf[4];
    be32_store(buf, 0xFFFFFFFF);
    CHECK(be32_load(buf) == 0xFFFFFFFF, "T5: be32 handles max value 0xFFFFFFFF");
}

static void test_be32_zero(void)
{
    uint8_t buf[4];
    be32_store(buf, 0);
    CHECK(be32_load(buf) == 0, "T6: be32 handles zero");
}

static void test_be24_max_value(void)
{
    uint8_t buf[3];
    be24_store(buf, 0xFFFFFF);
    CHECK(be24_load(buf) == 0xFFFFFF, "T7: be24 handles max 24-bit value 0xFFFFFF");
}

/* =========================================================================
 * Group 2: handle_request — write operations
 * ====================================================================== */

static void test_write_single_word_ok(void)
{
    reset_state();
    uint32_t val = 0xCAFEBABE;
    uint8_t req[HEADER_LEN + 4];
    fill_write_req(req, 0, &val, 1);
    uint32_t rlen = handle_request(req, HEADER_LEN + 4);
    CHECK(resp_buf[0] == STATUS_OK, "T8: single-word WRITE returns STATUS_OK");
    CHECK(rlen == 4,                "T9: WRITE response is exactly 4 bytes");
}

static void test_write_stores_correct_value(void)
{
    reset_state();
    uint32_t val = 0xABCD1234;
    uint8_t req[HEADER_LEN + 4];
    fill_write_req(req, 0, &val, 1);
    handle_request(req, HEADER_LEN + 4);
    CHECK(test_user_mem[0] == 0xABCD1234,
          "T10: WRITE stores the value at the correct word in user memory");
}

static void test_write_to_last_valid_word(void)
{
    reset_state();
    uint32_t val = 0x55AA55AA;
    uint8_t req[HEADER_LEN + 4];
    fill_write_req(req, (USER_REGION_WORDS - 1) * 4, &val, 1);
    uint32_t rlen = handle_request(req, HEADER_LEN + 4);
    CHECK(resp_buf[0] == STATUS_OK &&
          test_user_mem[USER_REGION_WORDS - 1] == 0x55AA55AA,
          "T11: WRITE to last valid word offset (word 511) succeeds");
    CHECK(rlen == 4, "T12: WRITE to last word response is 4 bytes");
}

static void test_write_burst_stores_all_words(void)
{
    reset_state();
    uint32_t vals[4] = {0x11, 0x22, 0x33, 0x44};
    uint8_t req[HEADER_LEN + 16];
    fill_write_req(req, 8, vals, 4);   /* byte offset 8 = word offset 2 */
    handle_request(req, HEADER_LEN + 16);
    CHECK(test_user_mem[2] == 0x11 && test_user_mem[3] == 0x22 &&
          test_user_mem[4] == 0x33 && test_user_mem[5] == 0x44,
          "T13: burst WRITE stores all words at consecutive offsets");
}

static void test_write_zero_length_rejected(void)
{
    reset_state();
    /* length field = 0 in header, no payload */
    uint8_t req[HEADER_LEN] = {OP_WRITE, 0, 0, 0,  0, 0, 0, 0};
    uint32_t rlen = handle_request(req, HEADER_LEN);
    CHECK(resp_buf[0] == STATUS_ERROR, "T14: WRITE with length=0 returns STATUS_ERROR");
    CHECK(rlen == 4,                   "T15: error response is 4 bytes");
}

static void test_write_unaligned_address_rejected(void)
{
    reset_state();
    uint32_t val = 1;
    uint8_t req[HEADER_LEN + 4];
    fill_write_req(req, 1, &val, 1);   /* byte address 1 is not word-aligned */
    handle_request(req, HEADER_LEN + 4);
    CHECK(resp_buf[0] == STATUS_ERROR,
          "T16: WRITE to unaligned address (not multiple of 4) returns STATUS_ERROR");
}

static void test_write_past_end_rejected(void)
{
    reset_state();
    uint32_t val = 1;
    uint8_t req[HEADER_LEN + 4];
    fill_write_req(req, USER_REGION_WORDS * 4, &val, 1);   /* one past end */
    handle_request(req, HEADER_LEN + 4);
    CHECK(resp_buf[0] == STATUS_ERROR,
          "T17: WRITE past end of user region returns STATUS_ERROR");
}

static void test_write_spanning_end_rejected(void)
{
    reset_state();
    uint32_t vals[2] = {1, 2};
    uint8_t req[HEADER_LEN + 8];
    /* Start at word 511, try to write 2 words — overflows by 1 */
    fill_write_req(req, (USER_REGION_WORDS - 1) * 4, vals, 2);
    handle_request(req, HEADER_LEN + 8);
    CHECK(resp_buf[0] == STATUS_ERROR,
          "T18: WRITE spanning past end of user region returns STATUS_ERROR");
}

/* =========================================================================
 * Group 3: handle_request — read operations
 * ====================================================================== */

static void test_read_single_word_ok(void)
{
    reset_state();
    test_user_mem[0] = 0xBEEFCAFE;
    uint8_t req[HEADER_LEN + 4];
    fill_read_req(req, 0, 1);
    uint32_t rlen = handle_request(req, HEADER_LEN + 4);
    CHECK(resp_buf[0] == STATUS_OK, "T19: single-word READ returns STATUS_OK");
    CHECK(rlen == 8, "T20: single-word READ response is 8 bytes (4 hdr + 4 data)");
}

static void test_read_returns_correct_value(void)
{
    reset_state();
    test_user_mem[3] = 0x12345678;
    uint8_t req[HEADER_LEN + 4];
    fill_read_req(req, 12, 1);   /* byte offset 12 = word offset 3 */
    handle_request(req, HEADER_LEN + 4);
    uint32_t val = be32_load(resp_buf + 4);
    CHECK(val == 0x12345678,
          "T21: READ returns the value previously written at that address");
}

static void test_read_response_length_field(void)
{
    reset_state();
    uint8_t req[HEADER_LEN + 4];
    fill_read_req(req, 0, 5);
    handle_request(req, HEADER_LEN + 4);
    uint32_t count = be24_load(resp_buf + 1);
    CHECK(count == 5, "T22: READ response length field (bytes 1-3) equals requested word count");
}

static void test_read_full_region(void)
{
    reset_state();
    for (uint32_t i = 0; i < USER_REGION_WORDS; i++)
        test_user_mem[i] = i;
    uint8_t req[HEADER_LEN + 4];
    fill_read_req(req, 0, USER_REGION_WORDS);
    uint32_t rlen = handle_request(req, HEADER_LEN + 4);
    CHECK(resp_buf[0] == STATUS_OK,
          "T23: READ of full 512-word user region returns STATUS_OK");
    CHECK(rlen == 4 + USER_REGION_WORDS * 4,
          "T24: full-region READ response is 2052 bytes (4 hdr + 2048 data)");
}

static void test_read_response_big_endian(void)
{
    reset_state();
    test_user_mem[0] = 0xAABBCCDD;
    uint8_t req[HEADER_LEN + 4];
    fill_read_req(req, 0, 1);
    handle_request(req, HEADER_LEN + 4);
    CHECK(resp_buf[4] == 0xAA && resp_buf[5] == 0xBB &&
          resp_buf[6] == 0xCC && resp_buf[7] == 0xDD,
          "T25: READ response data word is serialised big-endian");
}

static void test_read_zero_count_rejected(void)
{
    reset_state();
    uint8_t req[HEADER_LEN + 4];
    fill_read_req(req, 0, 0);
    handle_request(req, HEADER_LEN + 4);
    CHECK(resp_buf[0] == STATUS_ERROR, "T26: READ with count=0 returns STATUS_ERROR");
}

static void test_read_past_end_rejected(void)
{
    reset_state();
    uint8_t req[HEADER_LEN + 4];
    fill_read_req(req, USER_REGION_WORDS * 4, 1);
    handle_request(req, HEADER_LEN + 4);
    CHECK(resp_buf[0] == STATUS_ERROR,
          "T27: READ past end of user region returns STATUS_ERROR");
}

/* =========================================================================
 * Group 4: handle_request — unknown opcode and error response format
 * ====================================================================== */

static void test_unknown_opcode_rejected(void)
{
    reset_state();
    uint8_t req[HEADER_LEN] = {0xFF, 0, 0, 0,  0, 0, 0, 0};
    handle_request(req, HEADER_LEN);
    CHECK(resp_buf[0] == STATUS_ERROR, "T28: unknown opcode returns STATUS_ERROR");
}

static void test_error_response_format(void)
{
    reset_state();
    uint8_t req[HEADER_LEN] = {0xFF, 0, 0, 0,  0, 0, 0, 0};
    uint32_t rlen = handle_request(req, HEADER_LEN);
    uint32_t n_words = be24_load(resp_buf + 1);
    CHECK(n_words == 0, "T29: error response contains zero data words");
    CHECK(rlen == 4,    "T30: error response is exactly 4 bytes");
}

/* =========================================================================
 * Group 5: expected_request_len — TCP stream framing parser
 *
 * The firmware accumulates bytes from multiple TCP segments into req_buf.
 * expected_request_len() tells the accumulator when a complete request has
 * arrived, handling the case where the header arrives in one segment and
 * the payload in another.
 * ====================================================================== */

static void test_framing_incomplete_header(void)
{
    reset_state();
    req_fill = 4;   /* only 4 of the 8 header bytes received */
    CHECK(expected_request_len() == 0,
          "T31: returns 0 (wait for more data) when header is not yet complete");
}

static void test_framing_write_request_length(void)
{
    reset_state();
    req_buf[0] = OP_WRITE;
    be24_store(req_buf + 1, 10);   /* 10 data words */
    req_fill = HEADER_LEN;
    CHECK(expected_request_len() == HEADER_LEN + 40,
          "T32: WRITE expected length = HEADER_LEN + n_words*4");
}

static void test_framing_read_request_length(void)
{
    reset_state();
    req_buf[0] = OP_READ;
    be24_store(req_buf + 1, 1);    /* reads always carry length=1 */
    req_fill = HEADER_LEN;
    CHECK(expected_request_len() == HEADER_LEN + 4,
          "T33: READ expected length = HEADER_LEN + 4 (one word-count word)");
}

static void test_framing_zero_word_write(void)
{
    reset_state();
    req_buf[0] = OP_WRITE;
    be24_store(req_buf + 1, 0);    /* length=0 in header */
    req_fill = HEADER_LEN;
    CHECK(expected_request_len() == HEADER_LEN,
          "T34: zero-word WRITE expected length = HEADER_LEN (header only)");
}

/* =========================================================================
 * main
 * ====================================================================== */

int main(void)
{
    printf("TAP version 13\n");
    printf("1..34\n");

    /* Big-endian helpers */
    test_be32_round_trip();
    test_be32_byte_order();
    test_be24_round_trip();
    test_be24_byte_order();
    test_be32_max_value();
    test_be32_zero();
    test_be24_max_value();

    /* handle_request — writes */
    test_write_single_word_ok();
    test_write_stores_correct_value();
    test_write_to_last_valid_word();
    test_write_burst_stores_all_words();
    test_write_zero_length_rejected();
    test_write_unaligned_address_rejected();
    test_write_past_end_rejected();
    test_write_spanning_end_rejected();

    /* handle_request — reads */
    test_read_single_word_ok();
    test_read_returns_correct_value();
    test_read_response_length_field();
    test_read_full_region();
    test_read_response_big_endian();
    test_read_zero_count_rejected();
    test_read_past_end_rejected();

    /* Unknown opcode */
    test_unknown_opcode_rejected();
    test_error_response_format();

    /* TCP stream framing parser */
    test_framing_incomplete_header();
    test_framing_write_request_length();
    test_framing_read_request_length();
    test_framing_zero_word_write();

    printf("\n# Results: %d/%d passed\n", g_pass, g_run);
    return (g_pass == g_run) ? 0 : 1;
}
