use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Working mode for bash execution environment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum WorkingMode {
    /// Normal POSIX environment - execute shell commands in Podman containers
    Normal,
    /// Windows Subsystem for Linux - path remapping required; execute shell commands in Podman containers
    Wsl,
    /// Running inside containers - execute shell commands on host directly
    Container,
    /// Podman images are not available - shell command execution is disabled
    #[default]
    NoShell,
}

/// Lazy-initialized global working mode instance using std::sync::LazyLock
pub static WORKING_MODE: std::sync::LazyLock<WorkingMode> =
    std::sync::LazyLock::new(detect_working_mode);

/// Check if required Podman images exist
fn assert_podman_image(tag: &str) -> Result<()> {
    let output = std::process::Command::new("podman")
        .args(&[
            "images",
            "--quiet",
            "--filter",
            &format!("reference=zotan:{tag}"),
        ])
        .output()?;
    if String::from_utf8_lossy(&output.stdout).trim().is_empty() {
        anyhow::bail!("Required Podman image 'zotan:{tag}' is not installed")
    }
    Ok(())
}

/// Determine working mode based on system detection
fn detect_working_mode() -> WorkingMode {
    // Check for WSL
    if let Ok(osrelease) = std::fs::read_to_string("/proc/sys/kernel/osrelease") {
        let content = osrelease.to_lowercase();
        if content.contains("microsoft") || content.contains("wsl") {
            return WorkingMode::Wsl;
        }
    }

    // Check if running in container
    if let Ok(mountinfo) = std::fs::read_to_string("/proc/self/mountinfo") {
        for line in mountinfo.lines() {
            let fields: Vec<&str> = line.split(' ').collect();
            if fields.len() >= 5 && fields[4] == "/" && fields[5].contains("overlay") {
                return WorkingMode::Container;
            }
        }
    }

    // Check if Podman images are available
    match assert_podman_image("base") {
        Ok(_) => WorkingMode::Normal,
        Err(_) => WorkingMode::NoShell,
    }
}
