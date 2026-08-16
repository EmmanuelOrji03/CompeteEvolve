use anyhow::{Context, Result};
use std::fs;
use std::io::{self, Write};
use std::path::Path;

/// Reads a single markdown file and returns its contents.
pub fn load_markdown_file(path: &str) -> Result<String> {
    fs::read_to_string(path).with_context(|| format!("failed to read {path}"))
}

/// Lists the .md files in `dir` and lets the user pick one interactively.
/// Returns the contents of the chosen file.
pub fn choose_markdown_file(dir: &str) -> Result<String> {
    let path = Path::new(dir);

    let mut files: Vec<_> = fs::read_dir(path)
        .with_context(|| format!("failed to read directory: {dir}"))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|ext| ext == "md").unwrap_or(false))
        .collect();

    files.sort_by_key(|e| e.file_name());

    if files.is_empty() {
        anyhow::bail!("no .md files found in {dir}");
    }

    println!("Select a context file:");
    for (i, entry) in files.iter().enumerate() {
        println!("  {}) {}", i + 1, entry.file_name().to_string_lossy());
    }
    print!("> ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let choice: usize = input.trim().parse().context("invalid selection")?;

    let entry = files
        .get(choice.saturating_sub(1))
        .context("selection out of range")?;

    load_markdown_file(entry.path().to_str().unwrap())
}