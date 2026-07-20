//! Versioned GPT-OSS Harmony rendering and incremental response parsing.

#![forbid(unsafe_code)]

use nml::tokenizer::{Decoder, Tokenizer};
use serde_json::Value;
use std::collections::HashSet;
use std::error::Error as StdError;
use std::fmt;
use std::path::Path;

pub const TOKENIZER_IDENTITY: &str = "o200k_harmony";

const FIRST_SPECIAL_TOKEN: u32 = 199_998;
const START_OF_TEXT: u32 = 199_998;
const END_OF_TEXT: u32 = 199_999;
const RETURN: u32 = 200_002;
const CONSTRAIN: u32 = 200_003;
const CHANNEL: u32 = 200_005;
const START: u32 = 200_006;
const END: u32 = 200_007;
const MESSAGE: u32 = 200_008;
const CALL: u32 = 200_012;
const END_OF_PROMPT: u32 = 200_018;

const SPECIAL_TOKENS: [(&str, u32); 10] = [
    ("<|startoftext|>", START_OF_TEXT),
    ("<|endoftext|>", END_OF_TEXT),
    ("<|return|>", RETURN),
    ("<|constrain|>", CONSTRAIN),
    ("<|channel|>", CHANNEL),
    ("<|start|>", START),
    ("<|end|>", END),
    ("<|message|>", MESSAGE),
    ("<|call|>", CALL),
    ("<|endofprompt|>", END_OF_PROMPT),
];

/// Exact product-owned Harmony protocol over the already selected tokenizer.
pub struct HarmonyProtocol {
    tokenizer: Tokenizer,
}

impl HarmonyProtocol {
    pub fn load(model_directory: &Path) -> Result<Self> {
        Self::from_tokenizer_file(model_directory.join("tokenizer.json"))
    }

    fn from_tokenizer_file(path: impl AsRef<Path>) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(Error::tokenizer)?;
        for (text, expected) in SPECIAL_TOKENS {
            let actual = tokenizer.token_id(text);
            if actual != Some(expected) {
                return Err(Error::contract(format!(
                    "{TOKENIZER_IDENTITY} maps {text:?} to {actual:?}, expected {expected}"
                )));
            }
        }
        Ok(Self { tokenizer })
    }

    pub fn render_for_completion(&self, conversation: &Conversation) -> Result<Vec<u32>> {
        validate_conversation(conversation)?;
        let has_function_tools = conversation.messages.iter().any(|message| {
            matches!(message, Message::Developer(content) if !content.functions.is_empty())
        });
        let mut output = TokenOutput::new(&self.tokenizer);
        let first_final = conversation.messages.iter().position(|message| {
            matches!(
                message,
                Message::Assistant {
                    channel: Channel::Final,
                    ..
                }
            )
        });
        let last_assistant_is_final =
            conversation
                .messages
                .iter()
                .rev()
                .find_map(|message| match message {
                    Message::Assistant { channel, .. } => Some(*channel == Channel::Final),
                    Message::ToolCall { .. } => Some(false),
                    _ => None,
                })
                == Some(true);

        for (index, message) in conversation.messages.iter().enumerate() {
            if last_assistant_is_final
                && first_final.is_some_and(|first| index < first)
                && matches!(
                    message,
                    Message::Assistant {
                        channel: Channel::Analysis,
                        ..
                    }
                )
            {
                continue;
            }
            render_message(&mut output, message, has_function_tools)?;
        }
        output.special(START);
        output.text("assistant")?;
        Ok(output.tokens)
    }

    pub fn parser(&self) -> HarmonyParser<'_> {
        HarmonyParser::new(&self.tokenizer)
    }
}

pub const fn is_stop_token(token: u32) -> bool {
    matches!(token, RETURN | CALL)
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Conversation {
    pub messages: Vec<Message>,
}

impl Conversation {
    pub fn new(messages: impl IntoIterator<Item = Message>) -> Self {
        Self {
            messages: messages.into_iter().collect(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
// Rendering owns the complete message grammar even though the first public
// request constructor currently supplies only system and user turns. These
// variants are exercised by the protocol fixtures and will be consumed by the
// conversation-serving boundary without widening NML's public facade.
#[allow(dead_code)]
pub enum Message {
    System(SystemContent),
    Developer(DeveloperContent),
    User {
        name: Option<String>,
        content: String,
    },
    Assistant {
        channel: Channel,
        content: String,
    },
    ToolCall {
        recipient: String,
        arguments: String,
    },
    ToolResult {
        name: String,
        content: String,
    },
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            name: None,
            content: content.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
// The exact Harmony wire contract admits all three values even though the
// current product request uses the medium default. Keep the closed domain here
// rather than accepting an arbitrary string at the renderer boundary.
#[allow(dead_code)]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SystemContent {
    pub model_identity: String,
    pub knowledge_cutoff: String,
    pub current_date: Option<String>,
    pub reasoning_effort: ReasoningEffort,
}

impl Default for SystemContent {
    fn default() -> Self {
        Self {
            model_identity: "You are ChatGPT, a large language model trained by OpenAI.".to_owned(),
            knowledge_cutoff: "2024-06".to_owned(),
            current_date: None,
            reasoning_effort: ReasoningEffort::Medium,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DeveloperContent {
    pub instructions: Option<String>,
    pub functions: Vec<FunctionTool>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FunctionTool {
    pub name: String,
    pub description: String,
    pub parameters: Option<Value>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Channel {
    Analysis,
    Commentary,
    Final,
}

impl Channel {
    fn as_str(self) -> &'static str {
        match self {
            Self::Analysis => "analysis",
            Self::Commentary => "commentary",
            Self::Final => "final",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "analysis" => Ok(Self::Analysis),
            "commentary" => Ok(Self::Commentary),
            "final" => Ok(Self::Final),
            _ => Err(Error::contract(format!(
                "assistant header uses unsupported channel {value:?}"
            ))),
        }
    }
}

struct TokenOutput<'tokenizer> {
    tokenizer: &'tokenizer Tokenizer,
    tokens: Vec<u32>,
}

impl<'tokenizer> TokenOutput<'tokenizer> {
    fn new(tokenizer: &'tokenizer Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
        }
    }

    fn special(&mut self, token: u32) {
        self.tokens.push(token);
    }

    fn text(&mut self, text: &str) -> Result<()> {
        self.tokens.extend(
            self.tokenizer
                .encode_ordinary(text)
                .map_err(Error::tokenizer)?,
        );
        Ok(())
    }
}

fn render_message(
    output: &mut TokenOutput<'_>,
    message: &Message,
    has_function_tools: bool,
) -> Result<()> {
    output.special(START);
    match message {
        Message::System(content) => {
            output.text("system")?;
            output.special(MESSAGE);
            output.text(&render_system(content, has_function_tools))?;
            output.special(END);
        }
        Message::Developer(content) => {
            output.text("developer")?;
            output.special(MESSAGE);
            output.text(&render_developer(content))?;
            output.special(END);
        }
        Message::User { name, content } => {
            output.text("user")?;
            if let Some(name) = name {
                output.text(":")?;
                output.text(name)?;
            }
            output.special(MESSAGE);
            output.text(content)?;
            output.special(END);
        }
        Message::Assistant { channel, content } => {
            output.text("assistant")?;
            output.special(CHANNEL);
            output.text(channel.as_str())?;
            output.special(MESSAGE);
            output.text(content)?;
            output.special(END);
        }
        Message::ToolCall {
            recipient,
            arguments,
        } => {
            serde_json::from_str::<Value>(arguments).map_err(|error| {
                Error::contract(format!("tool call arguments are not JSON: {error}"))
            })?;
            output.text("assistant to=")?;
            output.text(recipient)?;
            output.special(CHANNEL);
            output.text("commentary ")?;
            output.special(CONSTRAIN);
            output.text("json")?;
            output.special(MESSAGE);
            output.text(arguments)?;
            output.special(CALL);
        }
        Message::ToolResult { name, content } => {
            output.text(name)?;
            output.special(MESSAGE);
            output.text(content)?;
            output.special(END);
        }
    }
    Ok(())
}

fn render_system(content: &SystemContent, has_function_tools: bool) -> String {
    let mut top = vec![
        content.model_identity.clone(),
        format!("Knowledge cutoff: {}", content.knowledge_cutoff),
    ];
    if let Some(date) = &content.current_date {
        top.push(format!("Current date: {date}"));
    }
    let mut sections = vec![
        top.join("\n"),
        format!("Reasoning: {}", content.reasoning_effort.as_str()),
    ];
    let mut channels =
        "# Valid channels: analysis, commentary, final. Channel must be included for every message."
            .to_owned();
    if has_function_tools {
        channels.push_str("\nCalls to these tools must go to the commentary channel: 'functions'.");
    }
    sections.push(channels);
    sections.join("\n\n")
}

fn render_developer(content: &DeveloperContent) -> String {
    let mut sections = Vec::new();
    if let Some(instructions) = &content.instructions {
        sections.push("# Instructions".to_owned());
        sections.push(instructions.clone());
    }
    if !content.functions.is_empty() {
        sections.push(render_function_tools(&content.functions));
    }
    sections.join("\n\n")
}

fn render_function_tools(functions: &[FunctionTool]) -> String {
    let mut lines = vec![
        "# Tools".to_owned(),
        "".to_owned(),
        "## functions\n".to_owned(),
        "namespace functions {\n".to_owned(),
    ];
    for function in functions {
        for line in function.description.lines() {
            lines.push(format!("// {line}"));
        }
        match &function.parameters {
            Some(parameters) => lines.push(format!(
                "type {} = (_: {}) => any;\n",
                function.name,
                json_schema_to_typescript(parameters, "")
            )),
            None => lines.push(format!("type {} = () => any;\n", function.name)),
        }
    }
    lines.push("} // namespace functions".to_owned());
    lines.join("\n")
}

// This rendering is kept byte-compatible with the Apache-2.0
// `openai/harmony` schema template at the pinned reference commit. NML owns
// the narrow implementation so the product keeps one IREE tokenizer and no
// runtime network/cache dependency.
fn json_schema_to_typescript(schema: &Value, indent: &str) -> String {
    fn is_enum(schema: &Value) -> bool {
        schema
            .get("enum")
            .and_then(Value::as_array)
            .is_some_and(|values| !values.is_empty())
    }

    if let Some(variants) = schema.get("oneOf").and_then(Value::as_array) {
        let mut output = String::new();
        for variant in variants {
            output.push_str(&format!("\n{indent} | "));
            let mut kind = json_schema_to_typescript(variant, &format!("{indent}   "));
            if variant
                .get("nullable")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                && !kind.contains("null")
            {
                kind.push_str(" | null");
            }
            output.push_str(&kind);
            let mut comments = Vec::new();
            if let Some(description) = variant.get("description").and_then(Value::as_str) {
                comments.push(description.to_owned());
            }
            if let Some(default) = variant.get("default") {
                if let Some(default) = default.as_str().filter(|_| !is_enum(variant)) {
                    comments.push(format!("default: \"{default}\""));
                } else {
                    comments.push(format!("default: {default}"));
                }
            }
            if !comments.is_empty() {
                output.push_str(&format!(" // {}", comments.join(" ")));
            }
        }
        return output;
    }

    if let Some(types) = schema.get("type").and_then(Value::as_array) {
        let kinds: Vec<_> = types
            .iter()
            .filter_map(Value::as_str)
            .map(|kind| match kind {
                "integer" => "number",
                other => other,
            })
            .collect();
        if !kinds.is_empty() {
            return kinds.join(" | ");
        }
    }

    match schema.get("type").and_then(Value::as_str) {
        Some("object") => render_object_schema(schema, indent),
        Some("string") => schema
            .get("enum")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(|value| format!("\"{value}\""))
                    .collect::<Vec<_>>()
                    .join(" | ")
            })
            .filter(|values| !values.is_empty())
            .unwrap_or_else(|| "string".to_owned()),
        Some("number" | "integer") => "number".to_owned(),
        Some("boolean") => "boolean".to_owned(),
        Some("array") => schema
            .get("items")
            .map(|items| format!("{}[]", json_schema_to_typescript(items, indent)))
            .unwrap_or_else(|| "Array<any>".to_owned()),
        _ => "any".to_owned(),
    }
}

fn render_object_schema(schema: &Value, indent: &str) -> String {
    let mut output = String::new();
    if let Some(description) = schema.get("description").and_then(Value::as_str) {
        output.push_str(&format!("{indent}// {description}\n"));
    }
    output.push_str("{\n");
    let required: HashSet<&str> = schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for (name, property) in properties {
            if let Some(title) = property.get("title").and_then(Value::as_str) {
                output.push_str(&format!("{indent}// {title}\n{indent}//\n"));
            }
            if property.get("oneOf").is_none() {
                if let Some(description) = property.get("description").and_then(Value::as_str) {
                    output.push_str(&format!("{indent}// {description}\n"));
                }
            }
            if let Some(examples) = property.get("examples").and_then(Value::as_array) {
                if !examples.is_empty() {
                    output.push_str(&format!("{indent}// Examples:\n"));
                    for example in examples.iter().filter_map(Value::as_str) {
                        output.push_str(&format!("{indent}// - \"{example}\"\n"));
                    }
                }
            }
            if render_one_of_property(
                &mut output,
                indent,
                name,
                property,
                required.contains(name.as_str()),
            ) {
                continue;
            }
            output.push_str(&format!(
                "{indent}{name}{}: ",
                if required.contains(name.as_str()) {
                    ""
                } else {
                    "?"
                }
            ));
            let mut kind = json_schema_to_typescript(property, &format!("{indent}    "));
            if property
                .get("nullable")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                && !kind.contains("null")
            {
                kind.push_str(" | null");
            }
            output.push_str(&kind);
            output.push(',');
            if let Some(default) = property.get("default") {
                output.push_str(" // ");
                output.push_str(&render_default_comment(default, property));
            }
            output.push('\n');
        }
    }
    output.push_str(&format!("{indent}}}"));
    output
}

fn render_one_of_property(
    output: &mut String,
    indent: &str,
    name: &str,
    property: &Value,
    required: bool,
) -> bool {
    let Some(variants) = property.get("oneOf").and_then(Value::as_array) else {
        return false;
    };
    let description = property.get("description").and_then(Value::as_str);
    let duplicate_first_description = description.is_some_and(|description| {
        variants
            .first()
            .and_then(|variant| variant.get("description"))
            .and_then(Value::as_str)
            == Some(description)
    });
    let rendered_description = if duplicate_first_description {
        false
    } else if let Some(description) = description {
        output.push_str(&format!("{indent}// {description}\n"));
        true
    } else {
        false
    };
    if let Some(default) = property.get("default") {
        output.push_str(&format!(
            "{indent}// {}\n",
            render_default_comment(default, property)
        ));
    }
    output.push_str(&format!(
        "{indent}{name}{}:\n",
        if required { "" } else { "?" }
    ));
    for (index, variant) in variants.iter().enumerate() {
        output.push_str(&format!("{indent} | "));
        let mut kind = json_schema_to_typescript(variant, &format!("{indent}   "));
        if variant
            .get("nullable")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            && !kind.contains("null")
        {
            kind.push_str(" | null");
        }
        output.push_str(&kind);
        let mut comments = Vec::new();
        if !(index == 0 && rendered_description) {
            if let Some(variant_description) = variant.get("description").and_then(Value::as_str) {
                if Some(variant_description) != description {
                    comments.push(variant_description.to_owned());
                }
            }
        }
        if let Some(default) = variant.get("default") {
            comments.push(render_default_comment(default, variant));
        }
        if !comments.is_empty() {
            output.push_str(&format!(" // {}", comments.join(" ")));
        }
        output.push('\n');
    }
    output.push_str(&format!("{indent},\n"));
    true
}

fn render_default_comment(default: &Value, schema: &Value) -> String {
    if let Some(default) = default.as_str() {
        if schema
            .get("enum")
            .and_then(Value::as_array)
            .is_some_and(|values| !values.is_empty())
        {
            format!("default: {default}")
        } else {
            format!("default: \"{default}\"")
        }
    } else {
        format!("default: {default}")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StopReason {
    Return,
    ToolCall,
    Length,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParsedMessage {
    pub channel: Channel,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParsedToolCall {
    pub recipient: String,
    pub arguments: Value,
    pub raw_arguments: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    ContentDelta { channel: Channel, text: String },
    Message(ParsedMessage),
    ToolCall(ParsedToolCall),
    Done(StopReason),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ParsedHeader {
    channel: Channel,
    recipient: Option<String>,
    content_type: Option<String>,
}

enum ParserState<'tokenizer> {
    Header {
        implicit_assistant: bool,
        tokens: Vec<u32>,
    },
    Content {
        header: ParsedHeader,
        decoder: Decoder<'tokenizer>,
        content: String,
        pending_utf8: Vec<u8>,
    },
    ExpectStart,
    Terminal(StopReason),
    Truncated,
    Failed,
}

/// Strict token-at-a-time parser for one assistant completion.
pub struct HarmonyParser<'tokenizer> {
    tokenizer: &'tokenizer Tokenizer,
    state: ParserState<'tokenizer>,
}

impl<'tokenizer> HarmonyParser<'tokenizer> {
    fn new(tokenizer: &'tokenizer Tokenizer) -> Self {
        Self {
            tokenizer,
            state: ParserState::Header {
                implicit_assistant: true,
                tokens: Vec::new(),
            },
        }
    }

    pub fn process(&mut self, token: u32) -> Result<Vec<Event>> {
        // Enter a permanent failure state first. Any error below therefore
        // poisons the parser instead of leaving partially consumed decoder
        // state available to a caller that accidentally continues.
        let state = std::mem::replace(&mut self.state, ParserState::Failed);
        let mut events = Vec::new();
        self.state = match state {
            ParserState::Header {
                implicit_assistant,
                mut tokens,
            } => {
                if token == MESSAGE {
                    let header = parse_header(self.tokenizer, &tokens, implicit_assistant)?;
                    ParserState::Content {
                        header,
                        decoder: self.tokenizer.decoder().map_err(Error::tokenizer)?,
                        content: String::new(),
                        pending_utf8: Vec::new(),
                    }
                } else if token == END || token == RETURN || token == CALL || token == START {
                    return Err(Error::contract(format!(
                        "unexpected structural token {token} before <|message|>"
                    )));
                } else {
                    tokens.push(token);
                    ParserState::Header {
                        implicit_assistant,
                        tokens,
                    }
                }
            }
            ParserState::Content {
                header,
                mut decoder,
                mut content,
                mut pending_utf8,
            } => {
                if matches!(token, END | RETURN | CALL) {
                    pending_utf8.extend(decoder.finish().map_err(Error::tokenizer)?);
                    let tail = std::str::from_utf8(&pending_utf8).map_err(|error| {
                        Error::contract(format!(
                            "assistant content ended with invalid UTF-8: {error}"
                        ))
                    })?;
                    if !tail.is_empty() {
                        content.push_str(tail);
                        if header.recipient.is_none() {
                            events.push(Event::ContentDelta {
                                channel: header.channel,
                                text: tail.to_owned(),
                            });
                        }
                    }
                    close_message(token, header, content, &mut events)?
                } else if token >= FIRST_SPECIAL_TOKEN {
                    return Err(Error::contract(format!(
                        "unexpected special token {token} inside assistant content"
                    )));
                } else {
                    pending_utf8.extend(decoder.push(token).map_err(Error::tokenizer)?);
                    if let Some(delta) = take_valid_utf8_prefix(&mut pending_utf8)? {
                        content.push_str(&delta);
                        if header.recipient.is_none() {
                            events.push(Event::ContentDelta {
                                channel: header.channel,
                                text: delta,
                            });
                        }
                    }
                    ParserState::Content {
                        header,
                        decoder,
                        content,
                        pending_utf8,
                    }
                }
            }
            ParserState::ExpectStart => {
                if token != START {
                    return Err(Error::contract(format!(
                        "expected <|start|> after a completed assistant message, received {token}"
                    )));
                }
                ParserState::Header {
                    implicit_assistant: false,
                    tokens: Vec::new(),
                }
            }
            ParserState::Terminal(reason) => {
                self.state = ParserState::Terminal(reason);
                return Err(Error::contract(
                    "received a token after the terminal Harmony stop token",
                ));
            }
            ParserState::Truncated => {
                self.state = ParserState::Truncated;
                return Err(Error::contract(
                    "received a token after bounded Harmony generation ended",
                ));
            }
            ParserState::Failed => {
                return Err(Error::contract(
                    "Harmony parser cannot continue after a previous failure",
                ));
            }
        };
        Ok(events)
    }

    pub fn finish(&mut self) -> Result<StopReason> {
        match self.state {
            ParserState::Terminal(reason) => Ok(reason),
            ParserState::Truncated => Err(Error::contract(
                "bounded Harmony generation ended without <|return|> or <|call|>",
            )),
            ParserState::Failed => Err(Error::contract(
                "Harmony parser cannot finish after a previous failure",
            )),
            _ => Err(Error::contract(
                "Harmony completion ended before <|return|> or <|call|>",
            )),
        }
    }

    /// Ends a generation because its caller-provided token budget was reached.
    ///
    /// This is deliberately distinct from [`Self::finish`]: a token bound is
    /// not a Harmony delimiter and therefore cannot complete a message or a
    /// tool call. Decoder bytes that form complete UTF-8 are still safe to
    /// publish; an incomplete code point at the arbitrary token boundary is
    /// retained only long enough to be discarded here.
    pub fn truncate(&mut self) -> Result<Vec<Event>> {
        let state = std::mem::replace(&mut self.state, ParserState::Failed);
        let mut events = Vec::new();
        match state {
            ParserState::Content {
                header,
                mut pending_utf8,
                ..
            } => {
                if header.recipient.is_none() {
                    if let Some(delta) = take_valid_utf8_prefix(&mut pending_utf8)? {
                        events.push(Event::ContentDelta {
                            channel: header.channel,
                            text: delta,
                        });
                    }
                }
                // A recipient-bearing body is unpublished until valid JSON is
                // closed by <|call|>, so truncation discards it wholesale.
                // The decoder must not be finalized at an arbitrary length
                // boundary: finalization may materialize a replacement
                // character for its incomplete byte state. For ordinary text,
                // every complete delta has already been emitted; any decoder
                // or UTF-8 suffix therefore dies with the parser state.
            }
            ParserState::Header { .. } | ParserState::ExpectStart => {}
            ParserState::Terminal(reason) => {
                self.state = ParserState::Terminal(reason);
                return Err(Error::contract(
                    "cannot truncate Harmony generation after a terminal stop token",
                ));
            }
            ParserState::Truncated => {
                self.state = ParserState::Truncated;
                return Err(Error::contract(
                    "bounded Harmony generation was already ended",
                ));
            }
            ParserState::Failed => {
                return Err(Error::contract(
                    "Harmony parser cannot truncate after a previous failure",
                ));
            }
        }
        events.push(Event::Done(StopReason::Length));
        self.state = ParserState::Truncated;
        Ok(events)
    }
}

fn close_message<'tokenizer>(
    delimiter: u32,
    header: ParsedHeader,
    content: String,
    events: &mut Vec<Event>,
) -> Result<ParserState<'tokenizer>> {
    match delimiter {
        END => {
            if header.channel == Channel::Final {
                return Err(Error::contract(
                    "generated final message must end with <|return|>",
                ));
            }
            if header.recipient.is_some() {
                return Err(Error::contract(
                    "assistant tool call must end with <|call|>",
                ));
            }
            events.push(Event::Message(ParsedMessage {
                channel: header.channel,
                content,
            }));
            Ok(ParserState::ExpectStart)
        }
        RETURN => {
            if header.channel != Channel::Final
                || header.recipient.is_some()
                || header.content_type.is_some()
            {
                return Err(Error::contract(
                    "<|return|> requires one recipient-free final text message",
                ));
            }
            events.push(Event::Message(ParsedMessage {
                channel: header.channel,
                content,
            }));
            events.push(Event::Done(StopReason::Return));
            Ok(ParserState::Terminal(StopReason::Return))
        }
        CALL => {
            if header.channel != Channel::Commentary {
                return Err(Error::contract(
                    "tool calls must use the commentary channel",
                ));
            }
            let recipient = header
                .recipient
                .ok_or_else(|| Error::contract("<|call|> requires a tool recipient"))?;
            if header.content_type.as_deref() != Some("json") {
                return Err(Error::contract(
                    "tool calls must use the <|constrain|> json content type",
                ));
            }
            let arguments = serde_json::from_str(&content).map_err(|error| {
                Error::contract(format!("tool call arguments are not valid JSON: {error}"))
            })?;
            events.push(Event::ToolCall(ParsedToolCall {
                recipient,
                arguments,
                raw_arguments: content,
            }));
            events.push(Event::Done(StopReason::ToolCall));
            Ok(ParserState::Terminal(StopReason::ToolCall))
        }
        _ => unreachable!("caller passes only Harmony message delimiters"),
    }
}

fn parse_header(
    tokenizer: &Tokenizer,
    tokens: &[u32],
    implicit_assistant: bool,
) -> Result<ParsedHeader> {
    let header = tokenizer.decode(tokens).map_err(Error::tokenizer)?;
    let (author, metadata) = header
        .split_once("<|channel|>")
        .ok_or_else(|| Error::contract("assistant message header is missing <|channel|>"))?;
    if metadata.contains("<|channel|>") {
        return Err(Error::contract(
            "assistant message header contains more than one channel marker",
        ));
    }
    let author_recipient = if implicit_assistant {
        if !author.is_empty() {
            return Err(Error::contract(format!(
                "implicit assistant header contains unexpected prefix {author:?}"
            )));
        }
        None
    } else if author == "assistant" {
        None
    } else if let Some(recipient) = author.strip_prefix("assistant to=") {
        validate_header_atom("tool recipient", recipient)?;
        Some(recipient.to_owned())
    } else {
        return Err(Error::contract(format!(
            "generated message author is not assistant: {author:?}"
        )));
    };

    let (metadata, content_type) =
        if let Some((metadata, content_type)) = metadata.split_once(" <|constrain|>") {
            let content_type = content_type.trim();
            validate_header_atom("content type", content_type)?;
            (metadata, Some(content_type.to_owned()))
        } else {
            (metadata, None)
        };
    let mut metadata = metadata.split_ascii_whitespace();
    let channel = metadata
        .next()
        .ok_or_else(|| Error::contract("assistant channel is empty"))?;
    validate_header_atom("channel", channel)?;
    let channel = Channel::parse(channel)?;
    let metadata_recipient = metadata
        .next()
        .map(|value| {
            let recipient = value.strip_prefix("to=").ok_or_else(|| {
                Error::contract(format!("unexpected assistant header field {value:?}"))
            })?;
            validate_header_atom("tool recipient", recipient)?;
            Ok::<_, Error>(recipient.to_owned())
        })
        .transpose()?;
    if let Some(field) = metadata.next() {
        return Err(Error::contract(format!(
            "unexpected assistant header field {field:?}"
        )));
    }
    if author_recipient.is_some() && metadata_recipient.is_some() {
        return Err(Error::contract(
            "assistant header repeats the tool recipient",
        ));
    }
    let recipient = author_recipient.or(metadata_recipient);
    if recipient.is_some() != content_type.is_some() {
        return Err(Error::contract(
            "tool recipient and constrained content type must appear together",
        ));
    }
    if recipient.is_some() && channel != Channel::Commentary {
        return Err(Error::contract(
            "tool recipients are valid only on the commentary channel",
        ));
    }
    Ok(ParsedHeader {
        channel,
        recipient,
        content_type,
    })
}

fn take_valid_utf8_prefix(pending: &mut Vec<u8>) -> Result<Option<String>> {
    match std::str::from_utf8(pending) {
        Ok(text) => {
            let text = (!text.is_empty()).then(|| text.to_owned());
            pending.clear();
            Ok(text)
        }
        Err(error) if error.error_len().is_none() => {
            let valid = error.valid_up_to();
            if valid == 0 {
                Ok(None)
            } else {
                let prefix = std::str::from_utf8(&pending[..valid])
                    .expect("valid_up_to prefix is UTF-8")
                    .to_owned();
                pending.drain(..valid);
                Ok(Some(prefix))
            }
        }
        Err(error) => Err(Error::contract(format!(
            "assistant content contains invalid UTF-8: {error}"
        ))),
    }
}

fn validate_conversation(conversation: &Conversation) -> Result<()> {
    if conversation.messages.is_empty() {
        return Err(Error::contract("Harmony conversation is empty"));
    }
    for (index, message) in conversation.messages.iter().enumerate() {
        match message {
            Message::System(content) => {
                if index != 0 {
                    return Err(Error::contract("system message must be first"));
                }
                validate_nonempty("model identity", &content.model_identity)?;
                validate_nonempty("knowledge cutoff", &content.knowledge_cutoff)?;
                if let Some(date) = &content.current_date {
                    validate_header_atom("current date", date)?;
                }
            }
            Message::Developer(content) => {
                for function in &content.functions {
                    validate_header_atom("function name", &function.name)?;
                    validate_nonempty("function description", &function.description)?;
                    if let Some(parameters) = &function.parameters {
                        if !parameters.is_object() {
                            return Err(Error::contract(format!(
                                "function {} parameters must be a JSON schema object",
                                function.name
                            )));
                        }
                    }
                }
            }
            Message::User { name, .. } => {
                if let Some(name) = name {
                    validate_header_atom("user name", name)?;
                }
            }
            Message::ToolCall { recipient, .. } => {
                validate_header_atom("tool recipient", recipient)?;
            }
            Message::ToolResult { name, .. } => {
                validate_header_atom("tool result name", name)?;
            }
            Message::Assistant { .. } => {}
        }
    }
    Ok(())
}

fn validate_nonempty(name: &str, value: &str) -> Result<()> {
    if value.is_empty() {
        Err(Error::contract(format!("{name} must not be empty")))
    } else {
        Ok(())
    }
}

fn validate_header_atom(name: &str, value: &str) -> Result<()> {
    validate_nonempty(name, value)?;
    if value.chars().any(char::is_whitespace) || value.contains("<|") {
        return Err(Error::contract(format!(
            "{name} contains whitespace or a Harmony delimiter"
        )));
    }
    Ok(())
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Error {
    message: String,
    source: Option<Box<dyn StdError + Send + Sync>>,
}

impl Error {
    fn contract(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source: None,
        }
    }

    fn tokenizer(error: impl StdError + Send + Sync + 'static) -> Self {
        Self {
            message: "Harmony tokenizer operation failed".to_owned(),
            source: Some(Box::new(error)),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(source) = &self.source {
            write!(formatter, "{}: {source}", self.message)
        } else {
            formatter.write_str(&self.message)
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn StdError + 'static))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn protocol() -> HarmonyProtocol {
        let runfiles = std::env::var_os("RUNFILES_DIR").expect("Bazel provides RUNFILES_DIR");
        let rlocation = std::env::var_os("NML_GPT_OSS_TOKENIZER_RLOCATION")
            .expect("Bazel provides the tokenizer rlocation");
        let tokenizer = Path::new(&runfiles).join(rlocation);
        HarmonyProtocol::from_tokenizer_file(tokenizer).unwrap()
    }

    fn decode(protocol: &HarmonyProtocol, tokens: &[u32]) -> String {
        protocol.tokenizer.decode(tokens).unwrap()
    }

    #[test]
    fn official_simple_conversation_fixture_matches() {
        let protocol = protocol();
        let tokens = protocol
            .render_for_completion(&Conversation::new([
                Message::System(SystemContent::default()),
                Message::user("What is 2 + 2?"),
            ]))
            .unwrap();
        assert_eq!(
            decode(&protocol, &tokens),
            "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>What is 2 + 2?<|end|><|start|>assistant"
        );
        assert!(is_stop_token(RETURN));
        assert!(is_stop_token(CALL));
        assert!(!is_stop_token(END));
    }

    #[test]
    fn official_function_schema_fixture_matches() {
        let protocol = protocol();
        let tokens = protocol
            .render_for_completion(&Conversation::new([
                Message::System(SystemContent {
                    current_date: Some("2025-06-28".to_owned()),
                    reasoning_effort: ReasoningEffort::High,
                    ..SystemContent::default()
                }),
                Message::Developer(DeveloperContent {
                    instructions: Some("Always respond in riddles".to_owned()),
                    functions: vec![
                        FunctionTool {
                            name: "get_location".to_owned(),
                            description: "Gets the location of the user.".to_owned(),
                            parameters: None,
                        },
                        FunctionTool {
                            name: "get_current_weather".to_owned(),
                            description: "Gets the current weather in the provided location."
                                .to_owned(),
                            parameters: Some(json!({
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "The city and state, e.g. San Francisco, CA"
                                    },
                                    "format": {
                                        "type": "string",
                                        "enum": ["celsius", "fahrenheit"],
                                        "default": "celsius"
                                    }
                                },
                                "required": ["location"]
                            })),
                        },
                    ],
                }),
                Message::user("What is the weather like in SF?"),
            ]))
            .unwrap();
        assert_eq!(
            decode(&protocol, &tokens),
            "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-06-28\n\nReasoning: high\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.\nCalls to these tools must go to the commentary channel: 'functions'.<|end|><|start|>developer<|message|># Instructions\n\nAlways respond in riddles\n\n# Tools\n\n## functions\n\nnamespace functions {\n\n// Gets the location of the user.\ntype get_location = () => any;\n\n// Gets the current weather in the provided location.\ntype get_current_weather = (_: {\n// The city and state, e.g. San Francisco, CA\nlocation: string,\nformat?: \"celsius\" | \"fahrenheit\", // default: celsius\n}) => any;\n\n} // namespace functions<|end|><|start|>user<|message|>What is the weather like in SF?<|end|><|start|>assistant"
        );
    }

    #[test]
    fn history_drops_completed_analysis_but_preserves_tool_round_trip() {
        let protocol = protocol();
        let completed = protocol
            .render_for_completion(&Conversation::new([
                Message::user("What is 2 + 2?"),
                Message::Assistant {
                    channel: Channel::Analysis,
                    content: "private reasoning".to_owned(),
                },
                Message::Assistant {
                    channel: Channel::Final,
                    content: "2 + 2 equals 4.".to_owned(),
                },
                Message::user("What about 9 / 2?"),
            ]))
            .unwrap();
        let completed = decode(&protocol, &completed);
        assert!(!completed.contains("private reasoning"));
        assert!(completed.contains("<|channel|>final<|message|>2 + 2 equals 4.<|end|>"));

        let tool = protocol
            .render_for_completion(&Conversation::new([
                Message::user("Weather?"),
                Message::Assistant {
                    channel: Channel::Analysis,
                    content: "Use the weather tool.".to_owned(),
                },
                Message::ToolCall {
                    recipient: "functions.lookup_weather".to_owned(),
                    arguments: "{\"location\": \"San Francisco\"}".to_owned(),
                },
                Message::ToolResult {
                    name: "functions.lookup_weather".to_owned(),
                    content: "{\"temperature\": 20}".to_owned(),
                },
            ]))
            .unwrap();
        assert_eq!(
            decode(&protocol, &tool),
            "<|start|>user<|message|>Weather?<|end|><|start|>assistant<|channel|>analysis<|message|>Use the weather tool.<|end|><|start|>assistant to=functions.lookup_weather<|channel|>commentary <|constrain|>json<|message|>{\"location\": \"San Francisco\"}<|call|><|start|>functions.lookup_weather<|message|>{\"temperature\": 20}<|end|><|start|>assistant"
        );
    }

    #[test]
    fn ordinary_content_cannot_inject_control_tokens() {
        let protocol = protocol();
        let tokens = protocol
            .render_for_completion(&Conversation::new([Message::user(
                "literal <|return|> and <|call|>",
            )]))
            .unwrap();
        assert!(!tokens.contains(&RETURN));
        assert!(!tokens.contains(&CALL));
        assert!(decode(&protocol, &tokens).contains("literal <|return|> and <|call|>"));
    }

    #[test]
    fn official_streaming_fixture_parses_channels_and_utf8() {
        let protocol = protocol();
        let fixture = [
            200_005, 35_644, 200_008, 1_844, 31_064, 25, 392, 4_827, 382, 220, 17, 659, 220, 17,
            16_842, 12_295, 81_645, 13, 51_441, 6_052, 13, 200_007, 200_006, 173_781, 200_005,
            17_196, 200_008, 17, 659, 220, 17, 314, 220, 19, 13, 200_002,
        ];
        let mut parser = protocol.parser();
        let mut events = Vec::new();
        for token in fixture {
            events.extend(parser.process(token).unwrap());
        }
        assert_eq!(parser.finish().unwrap(), StopReason::Return);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Message(ParsedMessage {
                channel: Channel::Analysis,
                ..
            })
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Message(ParsedMessage { channel: Channel::Final, content })
                if content == "2 + 2 = 4."
        )));
        assert_eq!(events.last(), Some(&Event::Done(StopReason::Return)));
    }

    #[test]
    fn tool_calls_parse_only_with_commentary_json_contract() {
        let protocol = protocol();
        let tokens = [CHANNEL]
            .into_iter()
            .chain(
                protocol
                    .tokenizer
                    .encode_ordinary("commentary to=functions.lookup_weather ")
                    .unwrap(),
            )
            .chain([CONSTRAIN])
            .chain(protocol.tokenizer.encode_ordinary("json").unwrap())
            .chain([MESSAGE])
            .chain(
                protocol
                    .tokenizer
                    .encode_ordinary("{\"location\":\"東京\"}")
                    .unwrap(),
            );

        let mut parser = protocol.parser();
        let mut events = Vec::new();
        for token in tokens {
            events.extend(parser.process(token).unwrap());
        }
        assert!(
            events.is_empty(),
            "recipient-bearing JSON must remain private until <|call|>"
        );
        events.extend(parser.process(CALL).unwrap());
        assert_eq!(parser.finish().unwrap(), StopReason::ToolCall);
        assert_eq!(
            events,
            vec![
                Event::ToolCall(ParsedToolCall {
                    recipient: "functions.lookup_weather".to_owned(),
                    arguments: json!({"location": "東京"}),
                    raw_arguments: "{\"location\":\"東京\"}".to_owned(),
                }),
                Event::Done(StopReason::ToolCall),
            ]
        );
    }

    #[test]
    fn ordinary_assistant_content_streams_incremental_utf8() {
        let protocol = protocol();
        let content = "Hello, 東京";
        let tokens = [CHANNEL]
            .into_iter()
            .chain(protocol.tokenizer.encode_ordinary("final").unwrap())
            .chain([MESSAGE])
            .chain(protocol.tokenizer.encode_ordinary(content).unwrap());
        let mut parser = protocol.parser();
        let mut events = Vec::new();
        for token in tokens {
            events.extend(parser.process(token).unwrap());
        }

        let streamed_before_stop = events
            .iter()
            .filter_map(|event| match event {
                Event::ContentDelta {
                    channel: Channel::Final,
                    text,
                } => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert!(!streamed_before_stop.is_empty());
        assert!(content.starts_with(&streamed_before_stop));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Event::Message(_))),
            "a message is complete only after its Harmony delimiter"
        );

        events.extend(parser.process(RETURN).unwrap());
        let streamed = events
            .iter()
            .filter_map(|event| match event {
                Event::ContentDelta {
                    channel: Channel::Final,
                    text,
                } => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(streamed, content);
        assert_eq!(
            &events[events.len() - 2..],
            [
                Event::Message(ParsedMessage {
                    channel: Channel::Final,
                    content: content.to_owned(),
                }),
                Event::Done(StopReason::Return),
            ]
        );
    }

    #[test]
    fn malformed_transitions_fail_deterministically() {
        let protocol = protocol();
        let final_header = [CHANNEL]
            .into_iter()
            .chain(protocol.tokenizer.encode_ordinary("final").unwrap())
            .chain([MESSAGE])
            .chain(protocol.tokenizer.encode_ordinary("answer").unwrap())
            .chain([END]);
        let mut parser = protocol.parser();
        let error = final_header
            .map(|token| parser.process(token))
            .find_map(Result::err)
            .unwrap();
        assert!(error.to_string().contains("must end with <|return|>"));
        assert!(parser
            .process(RETURN)
            .unwrap_err()
            .to_string()
            .contains("cannot continue after a previous failure"));

        let mut parser = protocol.parser();
        assert!(parser.process(RETURN).is_err());
        assert!(protocol
            .render_for_completion(&Conversation::new([Message::ToolCall {
                recipient: "functions.bad tool".to_owned(),
                arguments: "{}".to_owned(),
            }]))
            .is_err());
    }

    #[test]
    fn bounded_generation_flushes_text_without_completing_a_message() {
        let protocol = protocol();
        let mut parser = protocol.parser();
        let tokens = [CHANNEL]
            .into_iter()
            .chain(protocol.tokenizer.encode_ordinary("final").unwrap())
            .chain([MESSAGE])
            .chain(
                protocol
                    .tokenizer
                    .encode_ordinary("bounded answer")
                    .unwrap(),
            );
        let mut events = Vec::new();
        for token in tokens {
            events.extend(parser.process(token).unwrap());
        }
        events.extend(parser.truncate().unwrap());

        let text = events
            .iter()
            .filter_map(|event| match event {
                Event::ContentDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(text, "bounded answer");
        assert!(!events
            .iter()
            .any(|event| matches!(event, Event::Message(_) | Event::ToolCall(_))));
        assert_eq!(events.last(), Some(&Event::Done(StopReason::Length)));
        assert!(parser
            .finish()
            .unwrap_err()
            .to_string()
            .contains("without <|return|> or <|call|>"));
    }

    #[test]
    fn bounded_generation_discards_an_incomplete_utf8_suffix() {
        let protocol = protocol();
        let content = protocol.tokenizer.encode_ordinary("\u{10ffff}").unwrap();
        assert!(content.len() > 1, "fixture must span tokenizer tokens");
        let mut parser = protocol.parser();
        let tokens = [CHANNEL]
            .into_iter()
            .chain(protocol.tokenizer.encode_ordinary("final").unwrap())
            .chain([MESSAGE])
            .chain(content.iter().copied().take(content.len() - 1));
        let mut events = Vec::new();
        for token in tokens {
            events.extend(parser.process(token).unwrap());
        }
        events.extend(parser.truncate().unwrap());

        let text = events
            .iter()
            .filter_map(|event| match event {
                Event::ContentDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert!(text.is_empty());
        assert_eq!(events, vec![Event::Done(StopReason::Length)]);
    }

    #[test]
    fn bounded_generation_never_streams_partial_tool_arguments() {
        let protocol = protocol();
        let tokens = [CHANNEL]
            .into_iter()
            .chain(
                protocol
                    .tokenizer
                    .encode_ordinary("commentary to=functions.lookup_weather ")
                    .unwrap(),
            )
            .chain([CONSTRAIN])
            .chain(protocol.tokenizer.encode_ordinary("json").unwrap())
            .chain([MESSAGE])
            .chain(
                protocol
                    .tokenizer
                    .encode_ordinary("{\"location\":\"San")
                    .unwrap(),
            );
        let mut parser = protocol.parser();
        let mut events = Vec::new();
        for token in tokens {
            events.extend(parser.process(token).unwrap());
        }
        events.extend(parser.truncate().unwrap());
        assert_eq!(events, vec![Event::Done(StopReason::Length)]);
    }
}
