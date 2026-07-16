#ifndef NML_TOKENIZER_H_
#define NML_TOKENIZER_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque bridge-owned objects keep IREE declarations out of Rust's ABI. The
// safe Rust crate owns every pointer returned through this header.
typedef struct nml_iree_tokenizer_t nml_iree_tokenizer_t;
typedef struct nml_iree_encoder_t nml_iree_encoder_t;
typedef struct nml_iree_decoder_t nml_iree_decoder_t;

// Every function returning an integer uses IREE's stable status-code values;
// zero means success. A failure also stores a diagnostic in thread-local bridge
// state until the next bridge call on that thread.
const char* nml_iree_tokenizer_last_error(void);

int32_t nml_iree_tokenizer_from_huggingface_json(
    const uint8_t* json, size_t json_length,
    nml_iree_tokenizer_t** out_tokenizer);
void nml_iree_tokenizer_free(nml_iree_tokenizer_t* tokenizer);
bool nml_iree_tokenizer_lookup(const nml_iree_tokenizer_t* tokenizer,
                               const uint8_t* text, size_t text_length,
                               uint32_t* out_token_id);

// `input_size` selects a one-shot transform buffer large enough for the whole
// input. Encoders are one-shot owners; construct another encoder for another
// complete input.
int32_t nml_iree_encoder_create(const nml_iree_tokenizer_t* tokenizer,
                                size_t input_size,
                                nml_iree_encoder_t** out_encoder);
void nml_iree_encoder_free(nml_iree_encoder_t* encoder);
int32_t nml_iree_encoder_feed(nml_iree_encoder_t* encoder,
                              const uint8_t* text, size_t text_length,
                              uint32_t* token_ids, size_t token_capacity,
                              size_t* out_bytes_consumed,
                              size_t* out_token_count);
size_t nml_iree_encoder_pending_token_bound(
    const nml_iree_encoder_t* encoder);
int32_t nml_iree_encoder_finalize(nml_iree_encoder_t* encoder,
                                  uint32_t* token_ids,
                                  size_t token_capacity,
                                  size_t* out_token_count);

int32_t nml_iree_decoder_create(const nml_iree_tokenizer_t* tokenizer,
                                nml_iree_decoder_t** out_decoder);
void nml_iree_decoder_free(nml_iree_decoder_t* decoder);
int32_t nml_iree_decoder_reset(nml_iree_decoder_t* decoder);
int32_t nml_iree_decoder_feed(nml_iree_decoder_t* decoder,
                              const uint32_t* token_ids, size_t token_count,
                              uint8_t* text, size_t text_capacity,
                              size_t* out_tokens_consumed,
                              size_t* out_text_length);
int32_t nml_iree_decoder_finalize(nml_iree_decoder_t* decoder,
                                  uint8_t* text, size_t text_capacity,
                                  size_t* out_text_length);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // NML_TOKENIZER_H_
