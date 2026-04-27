use crate::is_default::is_default;
use anyhow::{Result, anyhow, bail};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize)]
pub struct FunctionTool {
    /// Name of the function
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// JSON Schema describing the parameters
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    /// The type of tool (e.g. "function")
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function definition if this is a function tool
    pub function: FunctionTool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "role", rename = "system")]
pub struct SystemMessage {
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename = "text")]
pub struct TextContent {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(TextContent),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "role", rename = "user")]
pub struct UserMessage {
    pub content: Vec<MessageContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "role", rename = "tool")]
pub struct ToolMessage {
    pub content: String,
    pub tool_call_id: Arc<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct FunctionCall {
    /// The name of the function to call.
    #[serde(default)]
    pub name: Arc<String>,
    /// The arguments to pass to the function, typically serialized as a JSON string.
    #[serde(default)]
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ToolCall {
    /// The ID of the tool call.
    #[serde(default)]
    pub id: Arc<String>,
    /// The type of the tool call (defaults to "function" if not provided).
    #[serde(rename = "type", default)]
    pub call_type: String,
    /// The function to call.
    #[serde(default)]
    pub function: FunctionCall,
}

/// Breakdown of completion tokens.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompletionTokensDetails {
    /// Tokens used for reasoning (for reasoning models)
    #[serde(default, skip_serializing_if = "is_default")]
    pub reasoning_tokens: Option<u32>,
}

/// Breakdown of prompt tokens.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PromptTokensDetails {
    /// Tokens used for cached content
    #[serde(default, skip_serializing_if = "is_default")]
    pub cached_tokens: Option<u32>,
}

/// Usage metadata for a chat response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Usage {
    /// Number of tokens in the prompt
    #[serde(alias = "input_tokens", default, skip_serializing_if = "is_default")]
    pub prompt_tokens: u32,
    /// Number of tokens in the completion
    #[serde(alias = "output_tokens", default, skip_serializing_if = "is_default")]
    pub completion_tokens: u32,
    /// Total number of tokens used
    #[serde(default, skip_serializing_if = "is_default")]
    pub total_tokens: u32,
    /// Breakdown of completion tokens, if available
    #[serde(
        alias = "output_tokens_details",
        default,
        skip_serializing_if = "is_default"
    )]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    /// Breakdown of prompt tokens, if available
    #[serde(
        alias = "input_tokens_details",
        default,
        skip_serializing_if = "is_default"
    )]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(tag = "role", rename = "assistant")]
pub struct AssistantMessage {
    #[serde(default, skip_serializing_if = "is_default")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "is_default")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "is_default")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "is_default")]
    pub usage: Option<Usage>,
}

/// Generic -compatible chat message
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Message {
    System(SystemMessage),
    User(UserMessage),
    Tool(ToolMessage),
    Assistant(AssistantMessage),
}

impl Message {
    pub fn system_message(text: &str) -> Message {
        Message::System(SystemMessage {
            content: text.into(),
        })
    }

    pub fn user_text_message(text: &str) -> Message {
        Message::User(UserMessage {
            content: vec![MessageContent::Text(TextContent { text: text.into() })],
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_schema")]
    JsonSchema,
    #[serde(rename = "json_object")]
    JsonObject,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub response_type: ResponseType,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// API key for authentication.
    pub api_key: Arc<String>,
    /// Base URL for API requests.
    pub base_url: Arc<reqwest::Url>,
    /// Model identifier.
    pub model: Arc<String>,
    /// Maximum tokens to generate in responses.
    pub max_tokens: Option<u32>,
    /// Sampling temperature for response randomness.
    pub temperature: Option<f32>,
    /// Top-p (nucleus) sampling parameter.
    pub top_p: Option<f32>,
    /// Top-k sampling parameter.
    pub top_k: Option<u32>,
    /// Available tools for the model to use.
    pub tools: Option<Arc<Vec<Tool>>>,
    /// Whether to enable parallel tool calls.
    pub parallel_tool_calls: Option<bool>,
    /// Reasoning effort level for supported models.
    pub reasoning_effort: Option<Arc<String>>,
    /// JSON schema for structured output.
    pub response_format: Option<Arc<ResponseFormat>>,
    /// Extra body parameters for custom fields.
    pub extra_body: Option<Arc<serde_json::Value>>,
}

#[derive(Debug, Clone)]
pub struct Model {
    /// Shared configuration wrapped in Arc for cheap cloning.
    pub config: ModelConfig,
    /// HTTP client for making requests.
    pub client: reqwest::Client,
}

#[derive(Debug, Serialize)]
struct StreamOptions {
    pub include_usage: bool,
}

/// Generic -compatible chat request
#[derive(Debug, Serialize)]
struct Request {
    pub model: Arc<String>,
    pub messages: Arc<Vec<Message>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Arc<Vec<Tool>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<Arc<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Arc<ResponseFormat>>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<Arc<serde_json::Value>>,
}

impl Model {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_key: Arc<String>,
        base_url: Arc<String>,
        model: Arc<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        tools: Option<Arc<Vec<Tool>>>,
        reasoning_effort: Option<Arc<String>>,
        response_format: Option<Arc<ResponseFormat>>,
        parallel_tool_calls: Option<bool>,
        extra_body: Option<Arc<serde_json::Value>>,
    ) -> Result<Self> {
        if api_key.is_empty() {
            bail!("Empty API Key");
        }
        if model.is_empty() {
            bail!("Empty model name");
        }
        Ok(Self {
            config: ModelConfig {
                api_key,
                base_url: reqwest::Url::parse(&format!("{}/", base_url.trim_end_matches("/")))?
                    .into(),
                model,
                max_tokens,
                temperature,
                top_p,
                top_k,
                tools,
                reasoning_effort,
                response_format,
                parallel_tool_calls,
                extra_body,
            },
            client: reqwest::Client::new(),
        })
    }

    pub async fn bytes_stream(
        &self,
        stack: Arc<Vec<Message>>,
    ) -> Result<std::pin::Pin<Box<dyn futures::Stream<Item = Result<bytes::Bytes>> + Send>>> {
        let body = Request {
            model: self.config.model.clone(),
            messages: stack,
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            stream: true,
            top_p: self.config.top_p,
            top_k: self.config.top_k,
            tools: self.config.tools.clone(),
            reasoning_effort: self.config.reasoning_effort.clone(),
            response_format: self.config.response_format.clone(),
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
            parallel_tool_calls: self.config.parallel_tool_calls,
            extra_body: self.config.extra_body.clone(),
        };
        let url = self.config.base_url.join("chat/completions")?;
        let request = self
            .client
            .post(url)
            .bearer_auth(&self.config.api_key)
            .json(&body);
        let response = request.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            bail!("API returned error status: status={status} message={error_text}");
        }
        Ok(Box::pin(
            response
                .bytes_stream()
                .map(|bytes| bytes.map_err(|e| anyhow!(e))),
        ))
    }
}

#[derive(Debug, Deserialize)]
struct StreamChoiceDelta {
    pub delta: Option<AssistantMessage>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamChunk {
    #[serde(default, skip_serializing_if = "is_default")]
    pub choices: Vec<StreamChoiceDelta>,
    #[serde(default, skip_serializing_if = "is_default")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Default)]
struct ParseState {
    buffer: String,
    /// Only parse the first choice
    skip_choices: bool,
    message: AssistantMessage,
}

impl ParseState {
    fn parse_tool_call_delta(&mut self, tool_call_delta: ToolCall) {
        if !tool_call_delta.id.is_empty() {
            self.message.tool_calls.push(ToolCall {
                id: tool_call_delta.id,
                ..Default::default()
            });
        }
        let current_tool_call = self.message.tool_calls.last_mut().unwrap();
        if !tool_call_delta.call_type.is_empty() {
            current_tool_call.call_type = tool_call_delta.call_type;
        }
        if !tool_call_delta.function.name.is_empty() {
            current_tool_call.function.name = tool_call_delta.function.name;
        }
        if !tool_call_delta.function.arguments.is_empty() {
            current_tool_call
                .function
                .arguments
                .push_str(&tool_call_delta.function.arguments)
        }
    }

    fn parse_choice_delta(&mut self, choice_delta: StreamChoiceDelta) {
        let Some(assistant_message_delta) = choice_delta.delta else {
            return;
        };
        if let Some(reasoning_content) = assistant_message_delta.reasoning_content {
            self.message
                .reasoning_content
                .get_or_insert_default()
                .push_str(&reasoning_content);
        }
        if let Some(content) = assistant_message_delta.content {
            self.message
                .content
                .get_or_insert_default()
                .push_str(&content);
        }
        for tool_call in assistant_message_delta.tool_calls {
            self.parse_tool_call_delta(tool_call);
        }
        if choice_delta.finish_reason.is_some() {
            self.skip_choices = true;
        }
    }

    fn parse_bytes(
        &mut self,
        bytes: Result<bytes::Bytes>,
    ) -> Option<Option<Result<AssistantMessage>>> {
        let Ok(bytes) = bytes else {
            return Some(Some(Err(bytes.unwrap_err())));
        };
        let text = String::from_utf8_lossy(&bytes);
        self.buffer.push_str(&text);
        let Some(offset) = self.buffer.find("\n\n") else {
            return Some(None); // Skip and not emitting new messages
        };
        let rest = self.buffer.split_off(offset + 2);
        let buffer = std::mem::replace(&mut self.buffer, rest);
        let Some(data) = buffer.strip_prefix("data: ") else {
            return Some(Some(Err(anyhow!("Invalid event data: {:?}", buffer))));
        };
        if data.trim_end() == "[DONE]" {
            return None; // Terminate the message stream
        }
        match serde_json::from_str::<StreamChunk>(data) {
            Ok(chunk) => {
                if !self.skip_choices {
                    for choice_delta in chunk.choices {
                        self.parse_choice_delta(choice_delta);
                    }
                }
                if let Some(usage) = chunk.usage {
                    self.message.usage = Some(usage);
                }
                Some(Some(Ok(self.message.clone())))
            }
            Err(e) => {
                let error = anyhow!("Invalid event JSON: data={:?} error={:?}", data, e);
                Some(Some(Err(error)))
            }
        }
    }
}

pub fn partial_assistant_message_stream(
    bytes_stream: std::pin::Pin<Box<dyn futures::Stream<Item = Result<bytes::Bytes>> + Send>>,
) -> std::pin::Pin<Box<dyn futures::Stream<Item = Result<AssistantMessage>> + Send>> {
    Box::pin(
        bytes_stream
            .scan(ParseState::default(), |parse_state, bytes| {
                let maybe_results = parse_state.parse_bytes(bytes);
                async move { maybe_results }
            })
            .flat_map(futures::stream::iter),
    )
}
