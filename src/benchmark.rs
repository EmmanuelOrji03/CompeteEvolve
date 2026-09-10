//! Benchmark
//!
//! Produces a performance report covering:
//! 1. The performance metrics of the evolved code of each agent.
//! 2. The performance of the top-ranking evolved code overall.
//! 3. The top-ranking evolved code's performance versus the original
//!    (pre-evolution) algorithm code's performance.
//!
//! Output is both a table (Markdown, printed to the console and saved
//! alongside the chart in a `.md` report) and a graph (a PNG bar chart).
//!
//! ## Where the numbers come from
//! Per-agent accuracy/time/error-count/performance/novelty are read from
//! `SharedTools::agent_evaluations`, populated by `evaluator::run` — so
//! this reports whatever the most recent `evaluator` call actually
//! observed in the sandbox for each agent, not a fresh re-run. Rank
//! positions come from `evaluator::compute_rank_positions`, the same
//! source `reinforcement::run` uses. The **original** code's numbers,
//! however, aren't cached anywhere, so this re-runs it through the
//! sandbox fresh, against the same training/test data, so the comparison
//! in requirement 3 is apples-to-apples.
//!
//! Both the original's and each agent's "performance" figure use the
//! same `accuracy / time` convention `evaluator.rs` uses (its `a0`/`a1`),
//! not `evolution.rs`'s differing `time / accuracy` convention — see
//! `evaluator.rs`'s module docs for why those two differ in the specs
//! this project follows. Reusing one convention here keeps requirement 3
//! a fair comparison rather than mixing two inverse scales.

use anyhow::{anyhow, Result};
use plotters::prelude::*;
use plotters::style::register_font;
use plotters::style::text_anchor::{HPos, Pos, VPos};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Once;

/// Plain grey for the "original" bar — avoids depending on plotters'
/// `full_palette` feature just for one named color.
const GREY: RGBColor = RGBColor(128, 128, 128);

use crate::agent::{database, evaluator, sandbox, SharedTools};
use crate::agent::evolution::squash;

/// Embedded so chart text renders identically on every machine this runs
/// on, regardless of what fonts (if any) happen to be installed —
/// relying on the OS to have a "sans-serif" font at a well-known path is
/// what breaks most often, especially across Windows/macOS/Linux.
/// DejaVu Sans is redistributed under a permissive, attribution-free
/// license, and is bundled in `assets/DejaVuSans.ttf`.
const FONT_BYTES: &[u8] = include_bytes!("assets/DejaVuSans.ttf");
const FONT_NAME: &str = "benchmark-embedded-sans";
static REGISTER_FONT_ONCE: Once = Once::new();

fn ensure_font_registered() {
    REGISTER_FONT_ONCE.call_once(|| {
        if register_font(FONT_NAME, FontStyle::Normal, FONT_BYTES).is_err() {
            eprintln!(
                "warning: failed to register the embedded benchmark chart font; \
                 chart text may not render"
            );
        }
    });
}

/// Requirement 1: one agent's evolved-code performance metrics.
#[derive(Clone, Debug)]
pub struct AgentMetrics {
    pub agent: String,
    pub accuracy: f64,
    pub training_time_seconds: f64,
    pub syntax_errors: u32,
    pub runtime_errors: u32,
    /// `evaluator`'s `a1` (or `a0` for the baseline) — `accuracy / time`.
    pub performance: f64,
    pub novelty: f64,
    pub rank_position: usize,
    pub reward: u8,
}

/// Requirement 3: the original (pre-evolution) code's performance,
/// freshly measured for a fair comparison.
#[derive(Clone, Debug)]
pub struct OriginalMetrics {
    pub accuracy: f64,
    pub training_time_seconds: f64,
    pub performance: f64,
    /// `false` if the baseline errored in the sandbox — the numbers
    /// above are then not meaningful measurements (just the zeroed
    /// fallback `sandbox::run` reports for a broken candidate). The
    /// system must not halt over this (an error here is counted and
    /// reported, not fatal), but the report needs to say "not measured"
    /// rather than display a bare `0.000` that reads as a real result.
    pub measurement_ok: bool,
    pub measurement_note: Option<String>,
}

#[derive(Clone, Debug)]
pub struct BenchmarkReport {
    pub algorithm: String,
    /// Sorted by rank position ascending — `per_agent[0]` is requirement 2's answer.
    pub per_agent: Vec<AgentMetrics>,
    // Public API surface for callers who want the raw baseline numbers
    // (tests do; `main.rs` currently only consumes the pre-rendered
    // table strings below) — kept even though this binary doesn't read
    // it directly.
    #[allow(dead_code)]
    pub original: OriginalMetrics,
    pub per_agent_table_md: String,
    pub top_vs_original_table_md: String,
    pub chart_path: PathBuf,
    pub report_path: PathBuf,
}

impl BenchmarkReport {
    /// Requirement 2's answer: the top-ranking evolved code overall.
    pub fn top(&self) -> &AgentMetrics {
        &self.per_agent[0]
    }
}

/// Builds the full report: fetches/measures everything, renders the
/// chart, writes the combined Markdown report to disk, and returns the
/// structured result. `original_code_override` skips the database fetch
/// for the baseline (useful for tests, or if the caller already has it).
pub async fn run(shared: &SharedTools, algorithm: &str, original_code_override: Option<&str>) -> Result<BenchmarkReport> {
    ensure_font_registered();

    // ---- requirement 1: per-agent metrics ----
    let snapshot: HashMap<String, evaluator::AgentEvaluation> = shared.agent_evaluations.lock().await.clone();
    if snapshot.is_empty() {
        return Err(anyhow!(
            "no agent has been evaluated yet; run the 'evaluator' tool for at least one agent first"
        ));
    }

    let positions = evaluator::compute_rank_positions(shared).await?;
    let agent_count = positions.len();
    let half = agent_count as f64 / 2.0;

    let mut per_agent: Vec<AgentMetrics> = positions
        .into_iter()
        .map(|(name, rank_position)| {
            let eval = snapshot.get(&name).cloned();
            // Same reward rule as `reinforcement.rs` (Reinforcement.md
            // steps 2-7): reward the better half (lower rank position is
            // better). Duplicated here rather than shared, since it's a
            // one-line formula; keep the two in sync if it ever changes.
            let reward: u8 = if (rank_position as f64) < half { 1 } else { 0 };

            AgentMetrics {
                agent: name,
                accuracy: eval.as_ref().map_or(0.0, |e| e.accuracy),
                training_time_seconds: eval.as_ref().map_or(0.0, |e| e.training_time_seconds),
                syntax_errors: eval.as_ref().map_or(0, |e| e.syntax_errors),
                runtime_errors: eval.as_ref().map_or(0, |e| e.runtime_errors),
                performance: eval.as_ref().map_or(0.0, |e| e.performance),
                novelty: eval.as_ref().map_or(0.0, |e| e.novelty),
                rank_position,
                reward,
            }
        })
        .collect();
    per_agent.sort_by_key(|a| a.rank_position);

    if per_agent.is_empty() {
        return Err(anyhow!("no agents to benchmark"));
    }
    let top = per_agent[0].clone();

    // ---- requirement 3: original vs top evolved, measured fresh and identically ----
    let original_code = match original_code_override {
        Some(code) => code.to_string(),
        None => fetch_original_code(shared, algorithm).await?,
    };
    let training_data = fetch_by_category(shared, "training_data", algorithm).await?;
    let test_data = fetch_by_category(shared, "test_data", algorithm).await?;

    let time_limit = sandbox::default_time_limit();
    let baseline = sandbox::run(&original_code, &training_data, &test_data, time_limit).await?;

    // The system must not halt just because the baseline errored — count
    // it and report it, the same way a per-agent sample failure already
    // does, rather than aborting the whole report. What changes is that
    // the resulting numbers are explicitly marked "not measured" rather
    // than displayed as a bare 0.000 that reads as a real result.
    let measurement_ok = baseline.syntax_errors == 0 && baseline.runtime_errors == 0;
    let measurement_note = if measurement_ok {
        None
    } else {
        eprintln!(
            "[benchmark:{algorithm}] WARNING: original code failed in the sandbox \
             ({} syntax error(s), {} runtime error(s)) — reporting as unmeasured, not halting.",
            baseline.syntax_errors, baseline.runtime_errors
        );
        Some(format!(
            "original code failed in the sandbox ({} syntax error(s), {} runtime error(s)); \
             make sure it defines `def run(train_path, test_path) -> float`. Sandbox output: {}",
            baseline.syntax_errors, baseline.runtime_errors, baseline.stderr
        ))
    };

    let original = OriginalMetrics {
        accuracy: baseline.accuracy,
        training_time_seconds: baseline.training_time_seconds,
        performance: baseline.accuracy / baseline.training_time_seconds.max(1e-6),
        measurement_ok,
        measurement_note,
    };

    // ---- tables ----
    let per_agent_table_md = build_per_agent_table(&per_agent);
    let top_vs_original_table_md = build_comparison_table(&top, &original);

    // ---- chart ----
    let chart_path = PathBuf::from(format!("benchmark_{algorithm}.png"));
    render_chart(&chart_path, &per_agent, &original, algorithm)?;

    // ---- combined report file ----
    let top_code = snapshot.get(&top.agent).map(|e| e.code.as_str()).unwrap_or("");
    let report_path = PathBuf::from(format!("benchmark_{algorithm}.md"));
    let baseline_warning = match &original.measurement_note {
        Some(note) => format!("\n> **Note:** {note}\n"),
        None => String::new(),
    };
    let report_md = format!(
        "# Benchmark report: {algorithm}\n\n\
         ## 1. Per-agent performance\n\n{per_agent_table_md}\n\
         ## 2. Top ranking evolved code overall\n\n\
         Agent **{top_agent}** (rank 1 of {agent_count}) — accuracy {top_acc:.3}, \
         {top_time:.3}s, performance {top_perf:.3}, novelty {top_nov:.3}.\n\n\
         ```python\n{top_code}\n```\n\n\
         ## 3. Top evolved vs original\n\n{top_vs_original_table_md}\n\
         {baseline_warning}\n\
         ![Benchmark chart]({chart_file})\n",
        top_agent = top.agent,
        top_acc = top.accuracy,
        top_time = top.training_time_seconds,
        top_perf = top.performance,
        top_nov = top.novelty,
        chart_file = chart_path.file_name().unwrap_or_default().to_string_lossy(),
    );
    std::fs::write(&report_path, &report_md)
        .map_err(|e| anyhow!("failed to write benchmark report to {}: {e}", report_path.display()))?;

    Ok(BenchmarkReport {
        algorithm: algorithm.to_string(),
        per_agent,
        original,
        per_agent_table_md,
        top_vs_original_table_md,
        chart_path,
        report_path,
    })
}

/// Convenience wrapper: runs the benchmark and prints it to stdout.
pub async fn run_and_print(shared: &SharedTools, algorithm: &str) -> Result<BenchmarkReport> {
    let report = run(shared, algorithm, None).await?;
    print_report(&report);
    Ok(report)
}

/// Prints the report's tables to the console; the chart and full report
/// are already saved to disk by `run`.
pub fn print_report(report: &BenchmarkReport) {
    let top = report.top();
    println!("\n=== Benchmark: {} ===\n", report.algorithm);
    println!("-- 1. Per-agent performance --\n");
    println!("{}", report.per_agent_table_md);
    println!("-- 2. Top ranking evolved code overall --\n");
    println!(
        "Agent: {} (rank 1 of {})  accuracy={:.3}  time={:.3}s  performance={:.3}  novelty={:.3}\n",
        top.agent,
        report.per_agent.len(),
        top.accuracy,
        top.training_time_seconds,
        top.performance,
        top.novelty
    );
    println!("-- 3. Top evolved vs original --\n");
    println!("{}", report.top_vs_original_table_md);
    println!("Chart saved to:  {}", report.chart_path.display());
    println!("Report saved to: {}", report.report_path.display());
}

fn build_per_agent_table(per_agent: &[AgentMetrics]) -> String {
    let headers = [
        "Rank", "Agent", "Accuracy", "Time (s)", "Syntax Err", "Runtime Err", "Performance", "Novelty", "Reward",
    ];
    let rows: Vec<Vec<String>> = per_agent
        .iter()
        .map(|a| {
            vec![
                a.rank_position.to_string(),
                a.agent.clone(),
                format!("{:.3}", a.accuracy),
                format!("{:.3}", a.training_time_seconds),
                a.syntax_errors.to_string(),
                a.runtime_errors.to_string(),
                format!("{:.3}", a.performance),
                format!("{:.3}", a.novelty),
                a.reward.to_string(),
            ]
        })
        .collect();
    markdown_table(&headers, &rows)
}

fn build_comparison_table(top: &AgentMetrics, original: &OriginalMetrics) -> String {
    let headers = ["", "Accuracy", "Time (s)", "Performance"];

    let original_row = if original.measurement_ok {
        vec![
            "Original".to_string(),
            format!("{:.3}", original.accuracy),
            format!("{:.3}", original.training_time_seconds),
            format!("{:.3}", original.performance),
        ]
    } else {
        // Not "0.000" — that would read as a real measurement. The
        // error is still counted (see `measurement_note`, included in
        // the full report) but doesn't block this table from rendering.
        vec!["Original".to_string(), "N/A*".to_string(), "N/A*".to_string(), "N/A*".to_string()]
    };

    let top_accuracy_cell = if original.measurement_ok && original.accuracy > 1e-9 {
        let accuracy_delta_pct = (top.accuracy - original.accuracy) / original.accuracy * 100.0;
        format!("{:.3} ({accuracy_delta_pct:+.1}% vs original)", top.accuracy)
    } else {
        format!("{:.3}", top.accuracy)
    };

    let rows = vec![
        original_row,
        vec![
            format!("Top evolved ({})", top.agent),
            top_accuracy_cell,
            format!("{:.3}", top.training_time_seconds),
            format!("{:.3}", top.performance),
        ],
    ];

    let mut table = markdown_table(&headers, &rows);
    if !original.measurement_ok {
        table.push_str("\n\\* the original code failed in the sandbox — see the note below.\n");
    }
    table
}

fn markdown_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut out = String::new();
    out.push_str("| ");
    out.push_str(&headers.join(" | "));
    out.push_str(" |\n|");
    for _ in headers {
        out.push_str(" --- |");
    }
    out.push('\n');
    for row in rows {
        out.push_str("| ");
        out.push_str(&row.join(" | "));
        out.push_str(" |\n");
    }
    out
}

/// Fetches the baseline "code" entry for `algorithm` — the one written
/// with no `agent` tag, distinguishing it from every agent's evolved
/// submissions (which are always tagged with an agent by `evolution.rs`).
async fn fetch_original_code(shared: &SharedTools, algorithm: &str) -> Result<String> {
    let mut conn = shared.db.lock().await;
    let raw = database::run(
        &mut conn,
        &json!({ "action": "read", "top_n": 100, "category": "code", "algorithm": algorithm }),
    )?;
    drop(conn);

    let parsed: Value = serde_json::from_str(&raw)?;
    let results = parsed["results"].as_array().cloned().unwrap_or_default();

    results
        .into_iter()
        .find(|r| r.get("agent").map_or(true, |v| v.is_null()))
        .and_then(|r| r["code"].as_str().map(String::from))
        .ok_or_else(|| {
            anyhow!(
                "no baseline (agent-less) 'code' entry found for algorithm '{algorithm}'; \
                 write the original implementation first (database write with category: 'code', \
                 algorithm: '{algorithm}', and no 'agent' field), or pass it directly to benchmark::run"
            )
        })
}

async fn fetch_by_category(shared: &SharedTools, category: &str, algorithm: &str) -> Result<String> {
    let mut conn = shared.db.lock().await;
    let raw = database::run(
        &mut conn,
        &json!({ "action": "read", "top_n": 1, "category": category, "algorithm": algorithm }),
    )?;
    drop(conn);

    let parsed: Value = serde_json::from_str(&raw)?;
    parsed["results"][0]["code"]
        .as_str()
        .map(String::from)
        .ok_or_else(|| anyhow!("no '{category}' entry found for algorithm '{algorithm}'"))
}

/// Renders a two-panel PNG: per-agent **performance and novelty** (left,
/// grouped bars — not accuracy; Cθ and Nθ are the pair of metrics the
/// paper's evaluator actually outputs per agent) and
/// original-vs-top-evolved accuracy (right).
fn render_chart(path: &Path, per_agent: &[AgentMetrics], original: &OriginalMetrics, algorithm: &str) -> Result<()> {
    let root = BitMapBackend::new(path, (1000, 500)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| anyhow!("chart error: {e}"))?;
    let (left, right) = root.split_horizontally(560);

    render_per_agent_panel(&left, per_agent, algorithm)?;
    render_comparison_panel(&right, per_agent.first(), original)?;

    root.present().map_err(|e| anyhow!("failed to save chart to {}: {e}", path.display()))?;
    Ok(())
}

type Panel<'a> = DrawingArea<BitMapBackend<'a>, plotters::coord::Shift>;

const PERFORMANCE_COLOR: RGBColor = RGBColor(30, 120, 220); // blue
const NOVELTY_COLOR: RGBColor = RGBColor(255, 140, 0); // orange

/// Left panel: one pair of bars per agent — performance (Cθ) and novelty
/// (Nθ), the two metrics the evaluator actually produces per agent.
/// `performance` is unbounded (it's `accuracy / time`, see
/// `evaluator.rs`), so it's passed through the same `squash` used for
/// ranking to bring it onto novelty's [0,1] scale for a fair visual
/// comparison — the exact, un-squashed number is still in the table.
fn render_per_agent_panel(panel: &Panel, per_agent: &[AgentMetrics], algorithm: &str) -> Result<()> {
    let n = per_agent.len().max(1);
    let y_top = 1.2;
    let y_bottom = -0.15; // headroom below zero for agent-name labels

    let mut chart = ChartBuilder::on(panel)
        .caption(format!("{algorithm}: performance & novelty by agent"), (FONT_NAME, 18))
        .margin(15)
        .x_label_area_size(10)
        .y_label_area_size(45)
        .build_cartesian_2d(0.0..n as f64, y_bottom..y_top)
        .map_err(|e| anyhow!("chart error: {e}"))?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .y_desc("Score (0-1)")
        .label_style((FONT_NAME, 12))
        .axis_desc_style((FONT_NAME, 14))
        .draw()
        .map_err(|e| anyhow!("chart error: {e}"))?;

    chart
        .draw_series(per_agent.iter().enumerate().map(|(i, a)| {
            let base = i as f64;
            let perf_score = squash(a.performance);
            Rectangle::new([(base + 0.08, 0.0), (base + 0.46, perf_score)], PERFORMANCE_COLOR.filled())
        }))
        .map_err(|e| anyhow!("chart error: {e}"))?
        .label("Performance (squashed)")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 14, y + 5)], PERFORMANCE_COLOR.filled()));

    chart
        .draw_series(per_agent.iter().enumerate().map(|(i, a)| {
            let base = i as f64;
            Rectangle::new([(base + 0.54, 0.0), (base + 0.92, a.novelty)], NOVELTY_COLOR.filled())
        }))
        .map_err(|e| anyhow!("chart error: {e}"))?
        .label("Novelty")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 14, y + 5)], NOVELTY_COLOR.filled()));

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font((FONT_NAME, 12))
        .draw()
        .map_err(|e| anyhow!("chart error: {e}"))?;

    // value labels above each bar
    chart
        .draw_series(per_agent.iter().enumerate().map(|(i, a)| {
            let perf_score = squash(a.performance);
            Text::new(
                format!("{perf_score:.2}"),
                (i as f64 + 0.27, perf_score + 0.03),
                (FONT_NAME, 11).into_text_style(panel).pos(Pos::new(HPos::Center, VPos::Bottom)),
            )
        }))
        .map_err(|e| anyhow!("chart error: {e}"))?;

    chart
        .draw_series(per_agent.iter().enumerate().map(|(i, a)| {
            Text::new(
                format!("{:.2}", a.novelty),
                (i as f64 + 0.73, a.novelty + 0.03),
                (FONT_NAME, 11).into_text_style(panel).pos(Pos::new(HPos::Center, VPos::Bottom)),
            )
        }))
        .map_err(|e| anyhow!("chart error: {e}"))?;

    // agent-name labels below zero
    chart
        .draw_series(per_agent.iter().enumerate().map(|(i, a)| {
            Text::new(
                a.agent.clone(),
                (i as f64 + 0.5, y_bottom * 0.5),
                (FONT_NAME, 12).into_text_style(panel).pos(Pos::new(HPos::Center, VPos::Center)),
            )
        }))
        .map_err(|e| anyhow!("chart error: {e}"))?;

    Ok(())
}

fn render_comparison_panel(panel: &Panel, top: Option<&AgentMetrics>, original: &OriginalMetrics) -> Result<()> {
    let top_acc = top.map_or(0.0, |a| a.accuracy);
    let max_acc = original.accuracy.max(top_acc).max(0.1);
    let y_top = max_acc * 1.25;
    let y_bottom = -max_acc * 0.12;

    let mut chart = ChartBuilder::on(panel)
        .caption("Original vs top evolved", (FONT_NAME, 20))
        .margin(15)
        .x_label_area_size(10)
        .y_label_area_size(45)
        .build_cartesian_2d(0.0..2.0, y_bottom..y_top)
        .map_err(|e| anyhow!("chart error: {e}"))?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .y_desc("Accuracy")
        .label_style((FONT_NAME, 12))
        .axis_desc_style((FONT_NAME, 14))
        .draw()
        .map_err(|e| anyhow!("chart error: {e}"))?;

    let bars = [("Original", original.accuracy, GREY.filled()), ("Top evolved", top_acc, GREEN.filled())];

    chart
        .draw_series(bars.iter().enumerate().map(|(i, (_, value, color))| {
            Rectangle::new([(i as f64 + 0.15, 0.0), (i as f64 + 0.85, *value)], color.clone())
        }))
        .map_err(|e| anyhow!("chart error: {e}"))?;

    chart
        .draw_series(bars.iter().enumerate().map(|(i, (label, value, _))| {
            let text = if *label == "Original" && !original.measurement_ok {
                "N/A".to_string()
            } else {
                format!("{value:.2}")
            };
            Text::new(
                text,
                (i as f64 + 0.5, value + max_acc * 0.03),
                (FONT_NAME, 12).into_text_style(panel).pos(Pos::new(HPos::Center, VPos::Bottom)),
            )
        }))
        .map_err(|e| anyhow!("chart error: {e}"))?;

    chart
        .draw_series(bars.iter().enumerate().map(|(i, (label, _, _))| {
            Text::new(
                label.to_string(),
                (i as f64 + 0.5, y_bottom * 0.5),
                (FONT_NAME, 12).into_text_style(panel).pos(Pos::new(HPos::Center, VPos::Center)),
            )
        }))
        .map_err(|e| anyhow!("chart error: {e}"))?;

    Ok(())
}
