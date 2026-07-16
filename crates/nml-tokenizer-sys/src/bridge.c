#include "nml_tokenizer.h"

#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/vocab/vocab.h"

_Static_assert(sizeof(iree_tokenizer_token_id_t) == sizeof(uint32_t),
               "IREE and NML token IDs must have the same width");
_Static_assert(_Alignof(iree_tokenizer_token_id_t) == _Alignof(uint32_t),
               "IREE and NML token IDs must have the same alignment");

struct nml_iree_tokenizer_t {
  iree_tokenizer_t* value;
};

struct nml_iree_encoder_t {
  iree_tokenizer_encode_state_t* state;
  uint8_t* state_storage;
  uint8_t* transform_buffer;
};

struct nml_iree_decoder_t {
  const iree_tokenizer_t* tokenizer;
  iree_tokenizer_decode_state_t* state;
  uint8_t* state_storage;
  size_t state_size;
};

// IREE statuses are owning handles. Capture a bounded user-facing diagnostic
// before consuming every non-OK handle; no status is leaked across the ABI.
static _Thread_local char nml_last_error[2048];

static int32_t nml_fail(iree_status_code_t code, const char* message) {
  strncpy(nml_last_error, message, sizeof(nml_last_error) - 1);
  nml_last_error[sizeof(nml_last_error) - 1] = '\0';
  return (int32_t)code;
}

static int32_t nml_consume_status(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    nml_last_error[0] = '\0';
    return IREE_STATUS_OK;
  }
  const iree_status_code_t code = iree_status_code(status);
  const char* code_text = iree_status_code_string(code);
  const iree_string_view_t message = iree_status_message(status);

  // Full IREE status formatting may walk stack-trace and source-location
  // payloads. The bridge only needs a stable API diagnostic, and must never
  // expose those process-specific details across the FFI boundary. Copy the
  // canonical code and primary annotation directly while the owning status is
  // alive, then consume it below.
  size_t used = 0;
  const size_t capacity = sizeof(nml_last_error) - 1;
  const size_t code_length = strlen(code_text);
  const size_t copied_code = code_length < capacity ? code_length : capacity;
  memcpy(nml_last_error, code_text, copied_code);
  used = copied_code;
  if (message.size != 0 && used < capacity) {
    const char separator[] = ": ";
    size_t separator_length = sizeof(separator) - 1;
    if (separator_length > capacity - used) {
      separator_length = capacity - used;
    }
    memcpy(nml_last_error + used, separator, separator_length);
    used += separator_length;
    size_t message_length = message.size;
    if (message_length > capacity - used) {
      message_length = capacity - used;
    }
    memcpy(nml_last_error + used, message.data, message_length);
    used += message_length;
  }
  nml_last_error[used] = '\0';
  iree_status_ignore(status);
  return (int32_t)code;
}

const char* nml_iree_tokenizer_last_error(void) { return nml_last_error; }

int32_t nml_iree_tokenizer_from_huggingface_json(
    const uint8_t* json, size_t json_length,
    nml_iree_tokenizer_t** out_tokenizer) {
  if (!out_tokenizer || (!json && json_length != 0)) {
    return nml_fail(IREE_STATUS_INVALID_ARGUMENT,
                    "invalid tokenizer JSON input or output pointer");
  }
  *out_tokenizer = NULL;
  iree_tokenizer_t* value = NULL;
  iree_string_view_t json_view = {(const char*)json, json_length};
  int32_t code = nml_consume_status(iree_tokenizer_from_huggingface_json(
      json_view, iree_allocator_system(), &value));
  if (code != IREE_STATUS_OK) return code;

  nml_iree_tokenizer_t* tokenizer = malloc(sizeof(*tokenizer));
  if (!tokenizer) {
    iree_tokenizer_free(value);
    return nml_fail(IREE_STATUS_RESOURCE_EXHAUSTED,
                    "unable to allocate tokenizer owner");
  }
  tokenizer->value = value;
  *out_tokenizer = tokenizer;
  return IREE_STATUS_OK;
}

void nml_iree_tokenizer_free(nml_iree_tokenizer_t* tokenizer) {
  if (!tokenizer) return;
  iree_tokenizer_free(tokenizer->value);
  free(tokenizer);
}

bool nml_iree_tokenizer_lookup(const nml_iree_tokenizer_t* tokenizer,
                               const uint8_t* text, size_t text_length,
                               uint32_t* out_token_id) {
  if (!tokenizer || !out_token_id || (!text && text_length != 0)) return false;
  const iree_tokenizer_vocab_t* vocab =
      iree_tokenizer_vocab(tokenizer->value);
  if (!vocab) return false;
  const iree_string_view_t text_view = {(const char*)text, text_length};
  const int32_t token_id = iree_tokenizer_vocab_lookup(vocab, text_view);
  if (token_id < 0) return false;
  *out_token_id = (uint32_t)token_id;
  return true;
}

int32_t nml_iree_encoder_create(const nml_iree_tokenizer_t* tokenizer,
                                size_t input_size,
                                nml_iree_encoder_t** out_encoder) {
  if (!tokenizer || !out_encoder) {
    return nml_fail(IREE_STATUS_INVALID_ARGUMENT,
                    "invalid tokenizer or encoder output pointer");
  }
  *out_encoder = NULL;

  size_t state_size = 0;
  int32_t code = nml_consume_status(
      iree_tokenizer_encode_state_calculate_size(tokenizer->value, &state_size));
  if (code != IREE_STATUS_OK) return code;

  nml_iree_encoder_t* encoder = calloc(1, sizeof(*encoder));
  if (!encoder) {
    return nml_fail(IREE_STATUS_RESOURCE_EXHAUSTED,
                    "unable to allocate encoder owner");
  }
  const size_t transform_size =
      iree_tokenizer_transform_buffer_oneshot_size(input_size);
  encoder->state_storage = malloc(state_size);
  encoder->transform_buffer = malloc(transform_size);
  if (!encoder->state_storage || !encoder->transform_buffer) {
    nml_iree_encoder_free(encoder);
    return nml_fail(IREE_STATUS_RESOURCE_EXHAUSTED,
                    "unable to allocate encoder storage");
  }

  iree_byte_span_t state_storage = {encoder->state_storage, state_size};
  iree_byte_span_t transform_buffer = {encoder->transform_buffer,
                                       transform_size};
  code = nml_consume_status(iree_tokenizer_encode_state_initialize(
      tokenizer->value, state_storage, transform_buffer,
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &encoder->state));
  if (code != IREE_STATUS_OK) {
    nml_iree_encoder_free(encoder);
    return code;
  }
  *out_encoder = encoder;
  return IREE_STATUS_OK;
}

void nml_iree_encoder_free(nml_iree_encoder_t* encoder) {
  if (!encoder) return;
  if (encoder->state) iree_tokenizer_encode_state_deinitialize(encoder->state);
  free(encoder->transform_buffer);
  free(encoder->state_storage);
  free(encoder);
}

int32_t nml_iree_encoder_feed(nml_iree_encoder_t* encoder,
                              const uint8_t* text, size_t text_length,
                              uint32_t* token_ids, size_t token_capacity,
                              size_t* out_bytes_consumed,
                              size_t* out_token_count) {
  if (!encoder || !out_bytes_consumed || !out_token_count ||
      (!text && text_length != 0) || (!token_ids && token_capacity != 0)) {
    return nml_fail(IREE_STATUS_INVALID_ARGUMENT,
                    "invalid encoder input or output buffer");
  }
  iree_string_view_t text_view = {(const char*)text, text_length};
  iree_tokenizer_token_output_t output = {
      token_capacity, (iree_tokenizer_token_id_t*)token_ids, NULL, NULL};
  return nml_consume_status(iree_tokenizer_encode_state_feed(
      encoder->state, text_view, output, out_bytes_consumed,
      out_token_count));
}

size_t nml_iree_encoder_pending_token_bound(
    const nml_iree_encoder_t* encoder) {
  return encoder && encoder->state
             ? iree_tokenizer_encode_state_pending_token_bound(encoder->state)
             : 0;
}

int32_t nml_iree_encoder_finalize(nml_iree_encoder_t* encoder,
                                  uint32_t* token_ids,
                                  size_t token_capacity,
                                  size_t* out_token_count) {
  if (!encoder || !out_token_count || (!token_ids && token_capacity != 0)) {
    return nml_fail(IREE_STATUS_INVALID_ARGUMENT,
                    "invalid encoder finalization buffer");
  }
  iree_tokenizer_token_output_t output = {
      token_capacity, (iree_tokenizer_token_id_t*)token_ids, NULL, NULL};
  return nml_consume_status(iree_tokenizer_encode_state_finalize(
      encoder->state, output, out_token_count));
}

static int32_t nml_decoder_initialize(nml_iree_decoder_t* decoder) {
  iree_byte_span_t storage = {decoder->state_storage, decoder->state_size};
  return nml_consume_status(iree_tokenizer_decode_state_initialize(
      decoder->tokenizer, IREE_TOKENIZER_DECODE_FLAG_NONE, storage,
      &decoder->state));
}

int32_t nml_iree_decoder_create(const nml_iree_tokenizer_t* tokenizer,
                                nml_iree_decoder_t** out_decoder) {
  if (!tokenizer || !out_decoder) {
    return nml_fail(IREE_STATUS_INVALID_ARGUMENT,
                    "invalid tokenizer or decoder output pointer");
  }
  *out_decoder = NULL;

  nml_iree_decoder_t* decoder = calloc(1, sizeof(*decoder));
  if (!decoder) {
    return nml_fail(IREE_STATUS_RESOURCE_EXHAUSTED,
                    "unable to allocate decoder owner");
  }
  decoder->tokenizer = tokenizer->value;
  int32_t code = nml_consume_status(iree_tokenizer_decode_state_calculate_size(
      decoder->tokenizer, &decoder->state_size));
  if (code != IREE_STATUS_OK) {
    nml_iree_decoder_free(decoder);
    return code;
  }
  decoder->state_storage = malloc(decoder->state_size);
  if (!decoder->state_storage) {
    nml_iree_decoder_free(decoder);
    return nml_fail(IREE_STATUS_RESOURCE_EXHAUSTED,
                    "unable to allocate decoder storage");
  }
  code = nml_decoder_initialize(decoder);
  if (code != IREE_STATUS_OK) {
    nml_iree_decoder_free(decoder);
    return code;
  }
  *out_decoder = decoder;
  return IREE_STATUS_OK;
}

void nml_iree_decoder_free(nml_iree_decoder_t* decoder) {
  if (!decoder) return;
  if (decoder->state) iree_tokenizer_decode_state_deinitialize(decoder->state);
  free(decoder->state_storage);
  free(decoder);
}

int32_t nml_iree_decoder_reset(nml_iree_decoder_t* decoder) {
  if (!decoder) {
    return nml_fail(IREE_STATUS_INVALID_ARGUMENT, "invalid decoder owner");
  }
  if (decoder->state) {
    iree_tokenizer_decode_state_deinitialize(decoder->state);
  }
  decoder->state = NULL;
  return nml_decoder_initialize(decoder);
}

int32_t nml_iree_decoder_feed(nml_iree_decoder_t* decoder,
                              const uint32_t* token_ids, size_t token_count,
                              uint8_t* text, size_t text_capacity,
                              size_t* out_tokens_consumed,
                              size_t* out_text_length) {
  if (!decoder || !out_tokens_consumed || !out_text_length ||
      (!token_ids && token_count != 0) || (!text && text_capacity != 0)) {
    return nml_fail(IREE_STATUS_INVALID_ARGUMENT,
                    "invalid decoder input or output buffer");
  }
  iree_tokenizer_token_id_list_t tokens = {
      token_count, (const iree_tokenizer_token_id_t*)token_ids};
  iree_mutable_string_view_t output = {(char*)text, text_capacity};
  return nml_consume_status(iree_tokenizer_decode_state_feed(
      decoder->state, tokens, output, out_tokens_consumed, out_text_length));
}

int32_t nml_iree_decoder_finalize(nml_iree_decoder_t* decoder,
                                  uint8_t* text, size_t text_capacity,
                                  size_t* out_text_length) {
  if (!decoder || !out_text_length || (!text && text_capacity != 0)) {
    return nml_fail(IREE_STATUS_INVALID_ARGUMENT,
                    "invalid decoder finalization buffer");
  }
  iree_mutable_string_view_t output = {(char*)text, text_capacity};
  return nml_consume_status(iree_tokenizer_decode_state_finalize(
      decoder->state, output, out_text_length));
}
