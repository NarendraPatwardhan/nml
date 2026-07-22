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

#[test]
fn decoder_keeps_the_shared_tokenizer_owner_alive() {
    fn require_static<T: 'static>() {}
    require_static::<nml_tokenizer::Decoder>();

    let mut decoder = {
        let tokenizer = Tokenizer::from_bytes(WORDPIECE.as_bytes()).unwrap();
        let last_visible_clone = tokenizer.clone();
        let decoder = last_visible_clone.decoder().unwrap();
        drop(last_visible_clone);
        drop(tokenizer);
        decoder
    };

    let mut bytes = decoder.push(1).unwrap();
    bytes.extend(decoder.push(2).unwrap());
    bytes.extend(decoder.finish().unwrap());
    assert_eq!(String::from_utf8(bytes).unwrap(), "hello world");
}

#[test]
fn independent_sessions_are_serialized_across_threads() {
    fn require_send_sync<T: Send + Sync>() {}
    fn require_send<T: Send>() {}
    require_send_sync::<Tokenizer>();
    require_send::<nml_tokenizer::Decoder>();

    let tokenizer = Tokenizer::from_bytes(WORDPIECE.as_bytes()).unwrap();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(33));
    std::thread::scope(|scope| {
        let mut threads = Vec::new();
        for index in 0..32 {
            let tokenizer = tokenizer.clone();
            let barrier = barrier.clone();
            threads.push(scope.spawn(move || {
                let expected = if index % 2 == 0 { 1 } else { 2 };
                let text = if index % 2 == 0 { "hello" } else { "world" };
                barrier.wait();
                assert_eq!(tokenizer.encode(text).unwrap(), [expected]);
                let mut decoder = tokenizer.decoder().unwrap();
                let mut bytes = decoder.push(expected).unwrap();
                bytes.extend(decoder.finish().unwrap());
                assert_eq!(String::from_utf8(bytes).unwrap(), text);
            }));
        }
        barrier.wait();
        for thread in threads {
            thread.join().unwrap();
        }
    });
}

#[test]
fn decoder_owner_can_move_between_threads() {
    let tokenizer = Tokenizer::from_bytes(WORDPIECE.as_bytes()).unwrap();
    let mut decoder = tokenizer.decoder().unwrap();
    let bytes = std::thread::spawn(move || {
        let mut bytes = decoder.push(1).unwrap();
        bytes.extend(decoder.push(2).unwrap());
        bytes.extend(decoder.finish().unwrap());
        bytes
    })
    .join()
    .unwrap();
    assert_eq!(String::from_utf8(bytes).unwrap(), "hello world");
}
