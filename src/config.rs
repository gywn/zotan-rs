//! Configuration data structures and loading logic

use crate::is_default::is_default;
use crate::merge;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LLMConfig {
    #[serde(default, skip_serializing_if = "is_default")]
    pub model_name: Arc<String>,
    #[serde(default, skip_serializing_if = "is_default")]
    pub api_key: Arc<String>,
    #[serde(default, skip_serializing_if = "is_default")]
    pub base_url: Arc<String>,
}

/// Single LLM provider configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum LLMConfigRef {
    Owned(LLMConfig),
    Ref(String),
}

/// Main configuration structure  
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Config {
    #[serde(default, skip_serializing_if = "is_default")]
    pub llm_configs: BTreeMap<String, LLMConfigRef>,
}

impl Config {
    fn from_config_path<P: AsRef<Path>>(config_path: P) -> Result<Config> {
        let content = std::fs::read_to_string(&config_path)?;
        Ok(toml_edit::de::from_str(&content)?)
    }

    /// Load and merge configurations with priority:
    /// 1. Override config (if provided)
    /// 2. Project configs (from workspace_dir and parent directories)
    /// 3. User config (~/.config/zotan.toml)
    /// 4. Default empty config
    pub fn from_workspace<P: AsRef<Path>>(
        workspace_dir: Option<P>,
        override_config: Option<Config>,
    ) -> Result<Config> {
        // Priority 1: Override config (if provided)
        let mut configs: Vec<Config> = vec![];
        if let Some(cfg) = override_config {
            configs.push(cfg);
        }

        // Priority 2: Project configs - search from workspace_dir up to home directory
        if let Some(workspace_dir) = workspace_dir
            && let Some(home_dir) = dirs::home_dir()
        {
            let workspace_dir = PathBuf::from(workspace_dir.as_ref()).canonicalize()?;
            // Collect all directories from workspace up to (but not including) root
            let mut project_dirs: Vec<PathBuf> = vec![workspace_dir.clone()];
            let mut current_project_dir = workspace_dir.clone();
            while let Some(parent_dir) = current_project_dir.parent() {
                let parent_dir = PathBuf::from(parent_dir);
                if parent_dir == home_dir
                    || parent_dir.metadata()?.uid() != users::get_current_uid()
                {
                    break;
                }
                project_dirs.push(parent_dir.clone());
                current_project_dir = parent_dir;
            }
            for project_dir in project_dirs.iter() {
                let project_config_path = project_dir.join(".zotan").join("config.toml");
                if project_config_path.exists() {
                    let config = Self::from_config_path(&project_config_path)?;
                    configs.push(config);
                }
            }
        }

        // Priority 3: User configuration directory (~/.config/zotan.toml)
        if let Some(config_dir) = dirs::config_dir() {
            let user_config_path = config_dir.join("zotan.toml");
            if user_config_path.exists() {
                let config = Self::from_config_path(&user_config_path)?;
                configs.push(config);
            }
        }

        let config = configs
            .into_iter()
            .rev() // Items listed earlier have higher priority
            .try_fold(Config::default(), merge::merge)?; // Priority 4: Default configuration

        // Validate that at least one LLM is configured
        if config
            .llm_configs
            .values()
            .filter(|llm_config_ref| matches!(llm_config_ref, LLMConfigRef::Owned(_)))
            .count()
            <= 0
        {
            anyhow::bail!("At least one LLM must be configured");
        }

        Ok(config)
    }

    /// Get an LLM configuration by name, with fallback behavior:
    /// 1. First tries to find a config by the provided `name`
    /// 2. Falls back to trying `"reasoning"` or `"text_processing"` names
    /// 3. Finally returns the first owned config
    pub fn get_llm_config(&self, name: &str) -> &LLMConfig {
        // If name exists in llm_configs
        if let Some(llm_config_ref) = self.llm_configs.get(name) {
            return match llm_config_ref {
                LLMConfigRef::Owned(llm_config) => llm_config,
                LLMConfigRef::Ref(name) => self.get_llm_config(name),
            };
        }

        // Try "reasoning" or "text_processing"
        for semantic_name in ["reasoning", "text_processing"] {
            if self.llm_configs.contains_key(semantic_name) {
                return self.get_llm_config(semantic_name);
            }
        }

        // Return first owned config
        for llm_config_ref in self.llm_configs.values() {
            if let LLMConfigRef::Owned(llm_config) = llm_config_ref {
                return llm_config;
            }
        }

        unreachable!("At least one LLM must be configured")
    }
}
