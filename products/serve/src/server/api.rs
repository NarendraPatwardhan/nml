//! Strict OpenAI-compatible request and response schemas.

use crate::gpt_oss::protocol::{
    Channel, Conversation, DeveloperContent, FunctionTool, Message, SystemContent,
};
use crate::server::contracts::ServerLimits;
use crate::SamplingOptions;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashSet, VecDeque};
use std::time::Duration;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChatCompletionRequest {
    pub(crate) model: String,
    pub(crate) messages: Vec<ChatMessage>,
    #[serde(default)]
    pub(crate) tools: Vec<Tool>,
    #[serde(default)]
    pub(crate) tool_choice: ToolChoice,
    #[serde(default)]
    pub(crate) stream: bool,
    pub(crate) stream_options: Option<StreamOptions>,
    #[serde(default = "one")]
    pub(crate) n: usize,
    #[serde(alias = "max_tokens")]
    pub(crate) max_completion_tokens: Option<usize>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) seed: Option<u64>,
    pub(crate) top_k: Option<usize>,
    pub(crate) min_p: Option<f32>,
    pub(crate) deadline_ms: Option<u64>,
    pub(crate) cache_salt: Option<String>,
}

const fn one() -> usize {
    1
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StreamOptions {
    #[serde(default)]
    pub(crate) include_usage: bool,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ToolChoice {
    None,
    #[default]
    Auto,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChatMessage {
    pub(crate) role: Role,
    pub(crate) content: Option<String>,
    pub(crate) name: Option<String>,
    #[serde(default)]
    pub(crate) tool_calls: Vec<HistoryToolCall>,
    pub(crate) tool_call_id: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HistoryToolCall {
    pub(crate) id: String,
    #[serde(rename = "type")]
    pub(crate) kind: ToolKind,
    pub(crate) function: FunctionCall,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ToolKind {
    Function,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FunctionCall {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Tool {
    #[serde(rename = "type")]
    pub(crate) kind: ToolKind,
    pub(crate) function: ToolFunction,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ToolFunction {
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) description: String,
    pub(crate) parameters: Option<Value>,
}

pub(crate) struct ValidatedChat {
    pub(crate) conversation: Conversation,
    pub(crate) max_new_tokens: usize,
    pub(crate) sampling: SamplingOptions,
    pub(crate) stream: bool,
    pub(crate) include_usage: bool,
    pub(crate) deadline: Option<Duration>,
    pub(crate) cache_salt: Option<String>,
}

impl ChatCompletionRequest {
    pub(crate) fn validate(
        self,
        served_model: &str,
        limits: &ServerLimits,
    ) -> Result<ValidatedChat, String> {
        if self.model != served_model {
            return Err(format!("model {:?} is not served", self.model));
        }
        if self.n != 1 {
            return Err("only n=1 is supported".to_owned());
        }
        if self.messages.is_empty() {
            return Err("messages must not be empty".to_owned());
        }
        if self.stream_options.is_some() && !self.stream {
            return Err("stream_options requires stream=true".to_owned());
        }
        let max_new_tokens = self.max_completion_tokens.unwrap_or(256);
        if max_new_tokens > limits.max_completion_tokens {
            return Err("max_completion_tokens exceeds the server limit".to_owned());
        }
        let mut tool_names = HashSet::new();
        if self.tools.len() > limits.max_tools {
            return Err("tools exceeds the server count limit".to_owned());
        }
        for tool in &self.tools {
            let ToolKind::Function = tool.kind;
            validate_identifier(
                &tool.function.name,
                limits.max_tool_name_bytes,
                "tool function name",
            )?;
            if !tool_names.insert(tool.function.name.as_str()) {
                return Err("tool function names must be unique".to_owned());
            }
            if tool.function.description.len() > limits.max_tool_description_bytes {
                return Err("tool function description exceeds the server byte limit".to_owned());
            }
            if tool
                .function
                .parameters
                .as_ref()
                .is_some_and(|schema| !schema.is_object())
            {
                return Err("tool parameters must be a JSON Schema object".to_owned());
            }
            if let Some(schema) = &tool.function.parameters {
                if serde_json::to_vec(schema)
                    .map_err(|_| "tool parameters are not serializable JSON".to_owned())?
                    .len()
                    > limits.max_tool_schema_bytes
                {
                    return Err("tool parameters exceed the server byte limit".to_owned());
                }
                validate_object_schema(schema)?;
            }
        }
        if self
            .cache_salt
            .as_ref()
            .is_some_and(|salt| salt.len() > limits.max_prefix_cache_salt_bytes)
        {
            return Err("cache_salt exceeds the server byte limit".to_owned());
        }
        let deadline = self.deadline_ms.map(Duration::from_millis);
        if deadline.is_some_and(|duration| duration.is_zero()) {
            return Err("deadline_ms must be positive".to_owned());
        }

        let mut sampling = SamplingOptions::default();
        if let Some(seed) = self.seed {
            sampling.seed = [seed, seed ^ 0x9e37_79b9_7f4a_7c15];
        }
        if let Some(temperature) = self.temperature {
            sampling.temperature = temperature;
        }
        if let Some(top_p) = self.top_p {
            sampling.top_p = top_p;
        }
        if let Some(top_k) = self.top_k {
            sampling.top_k = top_k;
        }
        if let Some(min_p) = self.min_p {
            sampling.min_p = min_p;
        }
        validate_sampling(sampling)?;

        let mut system = SystemContent::default();
        let mut saw_system = false;
        let mut rendered = Vec::new();
        let mut developer_instructions = Vec::new();
        let mut history_tools = VecDeque::<(String, String)>::new();
        let mut history_ids = HashSet::<String>::new();
        let mut conversation_started = false;
        for message in self.messages {
            if !history_tools.is_empty() && message.role != Role::Tool {
                return Err("assistant tool calls must be followed by their tool result".to_owned());
            }
            match message.role {
                Role::System => {
                    if conversation_started {
                        return Err("system/developer messages must precede conversation turns"
                            .to_owned());
                    }
                    ensure_no_tool_fields(&message)?;
                    if saw_system {
                        return Err("only one system message is supported".to_owned());
                    }
                    // Harmony's system message owns model identity and runtime
                    // facts. Developer instructions remain a distinct role.
                    system.model_identity = required_content(message.content, "system message")?;
                    saw_system = true;
                }
                Role::Developer => {
                    if conversation_started {
                        return Err("system/developer messages must precede conversation turns"
                            .to_owned());
                    }
                    ensure_no_tool_fields(&message)?;
                    developer_instructions
                        .push(required_content(message.content, "developer instruction")?);
                }
                Role::User => {
                    conversation_started = true;
                    if !message.tool_calls.is_empty() || message.tool_call_id.is_some() {
                        return Err("user messages cannot contain tool fields".to_owned());
                    }
                    rendered.push(Message::User {
                        name: message.name,
                        content: required_content(message.content, "user message")?,
                    });
                }
                Role::Assistant => {
                    conversation_started = true;
                    if message.name.is_some() || message.tool_call_id.is_some() {
                        return Err("assistant name/tool_call_id is unsupported".to_owned());
                    }
                    if message.tool_calls.len() > 1 {
                        return Err("only one assistant tool call per turn is supported".to_owned());
                    }
                    if let Some(call) = message.tool_calls.into_iter().next() {
                        let ToolKind::Function = call.kind;
                        if message.content.as_deref().is_some_and(|content| !content.is_empty()) {
                            return Err("assistant tool-call history cannot also contain content"
                                .to_owned());
                        }
                        validate_identifier(
                            &call.id,
                            limits.max_tool_call_id_bytes,
                            "tool call ID",
                        )?;
                        validate_identifier(
                            &call.function.name,
                            limits.max_tool_name_bytes,
                            "tool function name",
                        )?;
                        if call.function.arguments.len() > limits.max_tool_arguments_bytes {
                            return Err(
                                "assistant tool-call arguments exceed the server byte limit"
                                    .to_owned(),
                            );
                        }
                        let arguments: Value = serde_json::from_str(&call.function.arguments)
                            .map_err(|_| {
                                "assistant tool-call arguments must contain valid JSON".to_owned()
                            })?;
                        if !arguments.is_object() {
                            return Err(
                                "assistant tool-call arguments must contain a JSON object"
                                    .to_owned(),
                            );
                        }
                        let recipient = format!("functions.{}", call.function.name);
                        if !history_ids.insert(call.id.clone()) {
                            return Err("tool call IDs must be unique".to_owned());
                        }
                        history_tools.push_back((call.id, recipient.clone()));
                        rendered.push(Message::ToolCall {
                            recipient,
                            arguments: call.function.arguments,
                        });
                    } else {
                        rendered.push(Message::Assistant {
                            channel: Channel::Final,
                            content: required_content(message.content, "assistant message")?,
                        });
                    }
                }
                Role::Tool => {
                    conversation_started = true;
                    if !message.tool_calls.is_empty() {
                        return Err("tool messages cannot contain tool_calls".to_owned());
                    }
                    let call_id = message
                        .tool_call_id
                        .ok_or_else(|| "tool messages require tool_call_id".to_owned())?;
                    let (expected_id, recipient) = history_tools.pop_front().ok_or_else(|| {
                        "tool_call_id does not reference prior history".to_owned()
                    })?;
                    if call_id != expected_id {
                        return Err("tool results must follow tool calls in order".to_owned());
                    }
                    if let Some(name) = message.name {
                        validate_identifier(
                            &name,
                            limits.max_tool_name_bytes,
                            "tool result name",
                        )?;
                        if recipient.strip_prefix("functions.") != Some(name.as_str()) {
                            return Err("tool result name does not match its tool call".to_owned());
                        }
                    }
                    let content = required_content(message.content, "tool result")?;
                    if content.len() > limits.max_tool_result_bytes {
                        return Err("tool result exceeds the server byte limit".to_owned());
                    }
                    rendered.push(Message::ToolResult {
                        name: recipient,
                        content,
                    });
                }
            }
        }
        if !conversation_started {
            return Err("conversation requires at least one user/assistant/tool turn".to_owned());
        }
        if !history_tools.is_empty() {
            return Err("assistant tool calls require exactly one matching tool result".to_owned());
        }
        let functions = if self.tool_choice == ToolChoice::Auto {
            self.tools
                .into_iter()
                .map(|tool| FunctionTool {
                    name: tool.function.name,
                    description: tool.function.description,
                    parameters: tool.function.parameters,
                })
                .collect()
        } else {
            Vec::new()
        };
        rendered.insert(0, Message::System(system));
        if !developer_instructions.is_empty() || !functions.is_empty() {
            rendered.insert(1, Message::Developer(DeveloperContent {
                instructions: (!developer_instructions.is_empty())
                    .then(|| developer_instructions.join("\n\n")),
                functions,
            }));
        }
        Ok(ValidatedChat {
            conversation: Conversation::new(rendered),
            max_new_tokens,
            sampling,
            stream: self.stream,
            include_usage: self
                .stream_options
                .is_some_and(|options| options.include_usage),
            deadline,
            cache_salt: self.cache_salt,
        })
    }
}

fn validate_identifier(value: &str, maximum_bytes: usize, kind: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err(format!("{kind} must not be empty"));
    }
    if value.len() > maximum_bytes {
        return Err(format!("{kind} exceeds the server byte limit"));
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(format!(
            "{kind} must contain only ASCII letters, digits, underscores, or hyphens"
        ));
    }
    Ok(())
}

fn validate_object_schema(schema: &Value) -> Result<(), String> {
    let object = schema
        .as_object()
        .ok_or_else(|| "tool parameters must be a JSON Schema object".to_owned())?;
    if object
        .get("type")
        .is_some_and(|kind| kind.as_str() != Some("object"))
    {
        return Err("tool parameter root type must be object when specified".to_owned());
    }
    let properties = object.get("properties");
    if properties.is_some_and(|properties| !properties.is_object()) {
        return Err("tool parameter properties must be an object".to_owned());
    }
    if let Some(required) = object.get("required") {
        let required = required
            .as_array()
            .ok_or_else(|| "tool parameter required must be an array".to_owned())?;
        let mut names = HashSet::new();
        for name in required {
            let name = name
                .as_str()
                .ok_or_else(|| "tool parameter required entries must be strings".to_owned())?;
            if !names.insert(name) {
                return Err("tool parameter required entries must be unique".to_owned());
            }
            if properties
                .and_then(Value::as_object)
                .is_none_or(|properties| !properties.contains_key(name))
            {
                return Err(
                    "tool parameter required entries must name declared properties".to_owned(),
                );
            }
        }
    }
    Ok(())
}

fn ensure_no_tool_fields(message: &ChatMessage) -> Result<(), String> {
    if message.name.is_some() || !message.tool_calls.is_empty() || message.tool_call_id.is_some() {
        return Err("system/developer messages cannot contain tool fields".to_owned());
    }
    Ok(())
}

fn required_content(content: Option<String>, kind: &str) -> Result<String, String> {
    content
        .filter(|content| !content.is_empty())
        .ok_or_else(|| format!("{kind} content must not be empty"))
}

fn validate_sampling(sampling: SamplingOptions) -> Result<(), String> {
    if !sampling.temperature.is_finite() || sampling.temperature <= 0.0 {
        return Err("temperature must be finite and positive".to_owned());
    }
    if !sampling.top_p.is_finite() || sampling.top_p <= 0.0 || sampling.top_p > 1.0 {
        return Err("top_p must be in (0, 1]".to_owned());
    }
    if sampling.top_k == 0 || sampling.top_k > 64 {
        return Err("top_k must be between 1 and 64".to_owned());
    }
    if !sampling.min_p.is_finite() || !(0.0..=1.0).contains(&sampling.min_p) {
        return Err("min_p must be in [0, 1]".to_owned());
    }
    Ok(())
}

#[derive(Serialize)]
pub(crate) struct ErrorEnvelope {
    pub(crate) error: ErrorBody,
}

#[derive(Serialize)]
pub(crate) struct ErrorBody {
    pub(crate) message: String,
    #[serde(rename = "type")]
    pub(crate) kind: &'static str,
    pub(crate) code: &'static str,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct Usage {
    pub(crate) prompt_tokens: usize,
    pub(crate) completion_tokens: usize,
    pub(crate) total_tokens: usize,
}

#[derive(Serialize)]
pub(crate) struct ChatCompletionResponse {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created: u64,
    pub(crate) model: &'static str,
    pub(crate) choices: Vec<ChatChoice>,
    pub(crate) usage: Usage,
}

#[derive(Serialize)]
pub(crate) struct ChatChoice {
    pub(crate) index: usize,
    pub(crate) message: ResponseMessage,
    pub(crate) finish_reason: &'static str,
}

#[derive(Serialize)]
pub(crate) struct ResponseMessage {
    pub(crate) role: &'static str,
    pub(crate) content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) tool_calls: Vec<ResponseToolCall>,
}

#[derive(Clone, Serialize)]
pub(crate) struct ResponseToolCall {
    pub(crate) id: String,
    #[serde(rename = "type")]
    pub(crate) kind: ToolKind,
    pub(crate) function: ResponseFunctionCall,
}

#[derive(Clone, Serialize)]
pub(crate) struct ResponseFunctionCall {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[derive(Serialize)]
pub(crate) struct ChatChunk {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created: u64,
    pub(crate) model: &'static str,
    pub(crate) choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) usage: Option<Usage>,
}

#[derive(Serialize)]
pub(crate) struct ChunkChoice {
    pub(crate) index: usize,
    pub(crate) delta: ChunkDelta,
    pub(crate) finish_reason: Option<&'static str>,
}

#[derive(Default, Serialize)]
pub(crate) struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) tool_calls: Vec<ResponseToolCall>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(messages: Vec<ChatMessage>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: crate::gpt_oss::MODEL_NAME.to_owned(),
            messages,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            stream: false,
            stream_options: None,
            n: 1,
            max_completion_tokens: Some(10),
            temperature: None,
            top_p: None,
            seed: None,
            top_k: None,
            min_p: None,
            deadline_ms: None,
            cache_salt: None,
        }
    }

    #[test]
    fn tool_results_must_reference_prior_calls() {
        let error = request(vec![ChatMessage {
            role: Role::Tool,
            content: Some("{}".to_owned()),
            name: None,
            tool_calls: Vec::new(),
            tool_call_id: Some("missing".to_owned()),
        }])
        .validate(crate::gpt_oss::MODEL_NAME, &ServerLimits::default())
        .err()
        .unwrap();
        assert!(error.contains("does not reference prior history"));
    }

    #[test]
    fn system_and_developer_roles_keep_distinct_harmony_ownership() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": crate::gpt_oss::MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are the deployed support model."},
                {"role": "developer", "content": "Answer in one sentence."},
                {"role": "user", "content": "Hello"}
            ]
        }))
        .unwrap();
        let validated = request
            .validate(crate::gpt_oss::MODEL_NAME, &ServerLimits::default())
            .unwrap();
        let [Message::System(system), Message::Developer(developer), Message::User { .. }] =
            validated.conversation.messages.as_slice()
        else {
            panic!("validated roles were reordered or collapsed");
        };
        assert_eq!(system.model_identity, "You are the deployed support model.");
        assert_eq!(
            developer.instructions.as_deref(),
            Some("Answer in one sentence."),
        );
    }

    #[test]
    fn schemas_reject_unknown_fields_and_non_json_tool_history() {
        assert!(serde_json::from_value::<ChatCompletionRequest>(serde_json::json!({
            "model": crate::gpt_oss::MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello"}],
            "best_of": 2
        }))
        .is_err());
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": crate::gpt_oss::MODEL_NAME,
            "messages": [
                {"role": "user", "content": "Use a tool"},
                {"role": "assistant", "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "not-json"}
                }]}
            ]
        }))
        .unwrap();
        assert!(request
            .validate(crate::gpt_oss::MODEL_NAME, &ServerLimits::default())
            .err()
            .unwrap()
            .contains("valid JSON"));
    }

    #[test]
    fn a_tool_result_identifier_is_consumed_exactly_once() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": crate::gpt_oss::MODEL_NAME,
            "messages": [
                {"role": "user", "content": "Use a tool"},
                {"role": "assistant", "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"}
                }]},
                {"role": "tool", "tool_call_id": "call-1", "content": "{}"},
                {"role": "tool", "tool_call_id": "call-1", "content": "{}"}
            ]
        }))
        .unwrap();
        assert!(request
            .validate(crate::gpt_oss::MODEL_NAME, &ServerLimits::default())
            .err()
            .unwrap()
            .contains("does not reference prior history"));
    }

    #[test]
    fn tool_choice_controls_exact_harmony_definition_ownership() {
        for (choice, expected_functions) in [("auto", 1), ("none", 0)] {
            let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "model": crate::gpt_oss::MODEL_NAME,
                "messages": [{"role": "user", "content": "Check Rome"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "weather_lookup",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"]
                        }
                    }
                }],
                "tool_choice": choice
            }))
            .unwrap();
            let validated = request
                .validate(crate::gpt_oss::MODEL_NAME, &ServerLimits::default())
                .unwrap();
            let functions = validated
                .conversation
                .messages
                .iter()
                .find_map(|message| match message {
                    Message::Developer(developer) => Some(developer.functions.len()),
                    _ => None,
                })
                .unwrap_or(0);
            assert_eq!(functions, expected_functions);
        }
    }

    #[test]
    fn tool_history_enforces_order_name_and_schema_bounds() {
        let mismatched: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": crate::gpt_oss::MODEL_NAME,
            "messages": [
                {"role": "user", "content": "Use a tool"},
                {"role": "assistant", "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "name": "other", "content": "{}"}
            ]
        }))
        .unwrap();
        assert!(mismatched
            .validate(crate::gpt_oss::MODEL_NAME, &ServerLimits::default())
            .err()
            .unwrap()
            .contains("does not match"));

        let invalid_schema: ChatCompletionRequest =
            serde_json::from_value(serde_json::json!({
                "model": crate::gpt_oss::MODEL_NAME,
                "messages": [{"role": "user", "content": "Use a tool"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["missing"]
                        }
                    }
                }]
            }))
            .unwrap();
        assert!(invalid_schema
            .validate(crate::gpt_oss::MODEL_NAME, &ServerLimits::default())
            .err()
            .unwrap()
            .contains("declared properties"));
    }
}
