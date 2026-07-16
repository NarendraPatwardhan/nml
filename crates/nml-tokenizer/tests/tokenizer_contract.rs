use nml_tokenizer::Tokenizer;

const WORDPIECE: &str = r###"{
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {"[UNK]": 0, "hello": 1, "world": 2}
  },
  "pre_tokenizer": {"type": "Whitespace"},
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
}"###;

#[test]
fn huggingface_json_owns_complete_and_incremental_tokenization() {
    let tokenizer = Tokenizer::from_bytes(WORDPIECE.as_bytes()).unwrap();
    assert_eq!(tokenizer.token_id("hello"), Some(1));
    assert_eq!(tokenizer.token_id("missing"), None);
    assert_eq!(tokenizer.encode("hello world").unwrap(), [1, 2]);
    assert_eq!(tokenizer.decode(&[1, 2]).unwrap(), "hello world");

    let mut decoder = tokenizer.decoder().unwrap();
    let mut streamed = decoder.push(1).unwrap();
    streamed.extend(decoder.push(2).unwrap());
    streamed.extend(decoder.finish().unwrap());
    assert_eq!(String::from_utf8(streamed).unwrap(), "hello world");
    assert!(decoder.push(1).is_err());
    assert!(decoder.finish().is_err());

    decoder.reset().unwrap();
    let mut reset = decoder.push(2).unwrap();
    reset.extend(decoder.finish().unwrap());
    assert_eq!(String::from_utf8(reset).unwrap(), "world");
}

#[test]
fn large_inputs_make_bounded_progress() {
    let tokenizer = Tokenizer::from_bytes(WORDPIECE.as_bytes()).unwrap();
    let text = "hello ".repeat(10_000);
    let tokens = tokenizer.encode(&text).unwrap();
    assert_eq!(tokens.len(), 10_000);
    assert!(tokens.iter().all(|token| *token == 1));
}

#[test]
fn malformed_json_is_diagnostic() {
    let error = match Tokenizer::from_bytes(br#"{"model":null}"#) {
        Ok(_) => panic!("malformed tokenizer unexpectedly loaded"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("IREE tokenizer status"), "{error}");
}
