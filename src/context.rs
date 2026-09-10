use anyhow::{Context, Result};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Reads a single markdown file and returns its contents.
pub fn load_markdown_file(path: &str) -> Result<String> {
    fs::read_to_string(path).with_context(|| format!("failed to read {path}"))
}

/// Recursively collects every `.md` file under `dir`, however deeply
/// nested (e.g. `context/scikitlearn ML algorithms/svm.md`). A plain
/// `fs::read_dir` only sees `dir`'s immediate children, so files grouped
/// into subfolders would otherwise never be found — that's what was
/// causing "no .md files found in context" when the files existed but
/// were one level deeper than `dir` itself.
fn collect_markdown_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read directory: {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_markdown_files(&path, files)?;
        } else if path.extension().map(|ext| ext == "md").unwrap_or(false) {
            files.push(path);
        }
    }
    Ok(())
}

/// Lists every `.md` file under `dir`, at any nesting depth, and lets the
/// user pick one interactively. Returns `(algorithm_name, content)`,
/// where `algorithm_name` is the file's stem (e.g. `"svm.md"` -> `"svm"`)
/// — used to tag the training/test data and evolved code for that
/// algorithm in the database, so agents optimizing different algorithms
/// don't collide.
pub fn choose_markdown_file(dir: &str) -> Result<(String, String)> {
    let root = Path::new(dir);

    let mut files = Vec::new();
    collect_markdown_files(root, &mut files)?;
    files.sort();

    if files.is_empty() {
        anyhow::bail!("no .md files found under {dir} (including subfolders)");
    }

    println!("Select a context file:");
    for (i, path) in files.iter().enumerate() {
        // Show the path relative to `dir` so files with the same name in
        // different subfolders stay distinguishable, and nested
        // organization (like grouping by category) is visible at a glance.
        let label = path.strip_prefix(root).unwrap_or(path).display();
        println!("  {}) {}", i + 1, label);
    }
    print!("> ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let choice: usize = input.trim().parse().context("invalid selection")?;

    let path = files.get(choice.saturating_sub(1)).context("selection out of range")?;

    let algorithm = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .context("couldn't determine algorithm name from file name")?;

    let content = load_markdown_file(path.to_str().unwrap())?;

    Ok((algorithm, content))
}
