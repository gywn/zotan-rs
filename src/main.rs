mod config;
mod is_default;
mod merge;
mod openai;
mod work_mode;

use anyhow::Result;
use futures::TryStreamExt;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let config = config::Config::from_workspace(Some(std::env::current_dir()?), None)?;
    let llm_config = config.get_llm_config("");

    println!("{:#?}", llm_config);

    let model = openai::Model::new(
        llm_config.api_key.clone(),
        llm_config.base_url.clone(),
        llm_config.model_name.clone(),
        None, // max_tokens
        None, // temperature
        None, // top_p
        None, // top_k
        Some(vec![openai::Tool {
            tool_type: "function".into(),
            function: openai::FunctionTool {
                name: "write_file".into(),
                description: "Write content to a file. Creates a new file or overwrites an existing file with the given content.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file. Can be absolute or relative to workspace."
                        },
                        "content": {
                            "type": "string",
                            "description": "The text content to write to the file.",
                        }
                    },
                    "required": ["path", "content"]
                }),
            },
        }].into()), // tools
        None, // reasoning_effort
        None, // response_format
        Some(true),
        Some(json!({  // extra_body
            // Default to `true` for both Qwen3.5 and Qwen3.6 series of models
            "enable_thinking": true,
            // Only applicable to Qwen3.6 series of models; Default: `false`
            // Whether to append the reasoning content of assistant messages in the dialogue history to the model input
            // Must be enabled, otherwise the same things will be constantly rethought
            "preserve_thinking": true,
        }).into()),
    )?;

    let stack = vec![
        openai::Message::system_message("You are a helpful agent."),
        openai::Message::user_text_message(
            "Write at least 100 words to file /workspace/randome_text.txt",
        ),
    ]
    .into();
    let bytes_stream = model.bytes_stream(stack).await?;
    let mut partial_assistant_message_stream =
        openai::partial_assistant_message_stream(bytes_stream);
    while let Some(message) = partial_assistant_message_stream.try_next().await? {
        println!("{:#?}", message);
    }
    Ok(())
}
