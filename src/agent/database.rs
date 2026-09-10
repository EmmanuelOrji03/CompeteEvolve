//! Hybrid Database
//!
//! Implements the tool described in `Database.md`. It is made of two parts:
//!
//! 1. **Vector Database** — stores embeddings of generated code, computes the
//!    cosine similarity of a newly WRITTEN file against everything already
//!    stored, forwards the file (plus that similarity score) on to the
//!    Relational Database, and returns the similarity score to the caller.
//!
//! 2. **Relational Database** — stores code files keyed by their cosine
//!    similarity score and functional accuracy, ranks them by those two
//!    numbers, accepts WRITEs coming from the Vector Database, and answers
//!    READs from agents with the highest ranked code.
//!
//! `Connection` is the single handle agents share (see `SharedTools` in
//! `mod.rs`); `run` is the entry point the `database` tool dispatches to.
//!
//! Entries also carry tags that Database.md itself doesn't mention but
//! that `evolution`/`evaluator`/the sandbox need to function: `category`
//! ("code" / "training_data" / "test_data", default "code"), `agent`
//! (which agent produced this entry), and `algorithm` (which ML
//! algorithm — e.g. "kmeans", "svm" — this entry belongs to). None of
//! these change the ranking rules above — they only let a READ narrow
//! down *which* entries to rank and return, and let training/test data
//! for one algorithm stay separate from another's.
//!
//! Training/test data is expected as CSV text (validated with the `csv`
//! crate on write — malformed CSV is rejected up front rather than
//! silently stored). It can be supplied inline (`"code": "<csv text>"`)
//! or, more conveniently for real files, via `"path": "<file path>"`,
//! which is read from disk at write time — this is how a user uploads a
//! CSV file into the database.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Dimensionality of the (hashing-trick) embedding space used by the
/// vector database. No external embedding model is available in this
/// environment, so tokens are hashed into a fixed-size bag-of-words
/// vector, which is enough to produce a meaningful cosine similarity
/// between two pieces of code.
const EMBEDDING_DIM: usize = 256;

// ---------------------------------------------------------------------
// Vector Database
// ---------------------------------------------------------------------

#[derive(Clone, Debug)]
struct VectorEntry {
    // Kept for traceability (which write produced this embedding), even
    // though similarity scoring only reads `embedding`.
    #[allow(dead_code)]
    id: String,
    embedding: Vec<f32>,
}

#[derive(Default)]
struct VectorDatabase {
    entries: Vec<VectorEntry>,
}

impl VectorDatabase {
    fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Role 1 & 3: embed the written code, compute its cosine similarity
    /// against every file already stored, and return the highest score
    /// found (0.0 if the store is empty, i.e. nothing to compare against).
    /// The new embedding is then kept so future writes can be compared
    /// against it too.
    fn write_and_score(&mut self, id: String, code: &str) -> f32 {
        let embedding = embed(code);

        let similarity = self
            .entries
            .iter()
            .map(|e| cosine_similarity(&e.embedding, &embedding))
            .fold(0.0_f32, f32::max);

        self.entries.push(VectorEntry { id, embedding });
        similarity
    }
}

/// Turns source code into a normalized bag-of-words vector via the
/// hashing trick, so cosine similarity can be computed without needing
/// a real embedding model.
///
/// `pub(crate)` so other tools (e.g. `evolution`, which needs to score
/// "novelty" as the cosine similarity between a generated sample and the
/// original code) reuse the exact same similarity metric instead of
/// re-implementing it.
pub(crate) fn embed(code: &str) -> Vec<f32> {
    let mut vector = vec![0f32; EMBEDDING_DIM];

    for token in code.split_whitespace() {
        let mut hasher = DefaultHasher::new();
        token.hash(&mut hasher);
        let idx = (hasher.finish() as usize) % EMBEDDING_DIM;
        vector[idx] += 1.0;
    }

    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vector.iter_mut() {
            *v /= norm;
        }
    }
    vector
}

/// Both inputs are already normalized (see `embed`), so cosine
/// similarity is just the dot product. `pub(crate)` for the same reason
/// as `embed` above.
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------
// Relational Database
// ---------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RelationalEntry {
    id: String,
    code: String,
    cosine_similarity: f32,
    functional_accuracy: f32,
    /// The evolutionary rank (R = performance * novelty) this entry was
    /// written with, if any (Evolution.md step 19: "write the sample to
    /// the vector database and its rank"). Purely informational — it
    /// does not feed `rank_score` below, which stays true to the
    /// (unchanged) Database.md contract of ranking by cosine similarity
    /// and functional accuracy only.
    rank: Option<f32>,
    /// What kind of content this is: "code" (default), "training_data",
    /// or "test_data". Lets `evaluator` fetch training/test fixtures
    /// separately from generated code (Evaluator.md steps 1-3).
    category: String,
    /// Which agent produced this entry, if known. Lets `evaluator` fetch
    /// "the sample from a particular agent" (Evaluator.md step 1) and
    /// find other agents' samples for novelty comparison (step 25).
    agent: Option<String>,
    /// Which ML algorithm (e.g. "kmeans", "svm") this entry belongs to.
    /// Lets `evolution`/`evaluator` fetch the code and training/test
    /// data for the specific algorithm currently being optimized,
    /// instead of whatever "code"/"training_data" happens to rank
    /// highest across every algorithm ever stored.
    algorithm: Option<String>,
}

impl RelationalEntry {
    /// Combined ranking score used to order stored files. Cosine
    /// similarity and functional accuracy are weighted equally, per the
    /// spec ("Store ... according to their Cosine Similarity Score and
    /// Functional Accuracy").
    fn rank_score(&self) -> f32 {
        (self.cosine_similarity + self.functional_accuracy) / 2.0
    }
}

#[derive(Default)]
struct RelationalDatabase {
    entries: Vec<RelationalEntry>,
}

impl RelationalDatabase {
    fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Role 3: accept a WRITE coming from the vector database and
    /// role 1: store it together with its cosine similarity score and
    /// functional accuracy (plus the optional rank/category/agent/algorithm
    /// tags described above).
    #[allow(clippy::too_many_arguments)]
    fn write(
        &mut self,
        id: String,
        code: String,
        cosine_similarity: f32,
        functional_accuracy: f32,
        rank: Option<f32>,
        category: String,
        agent: Option<String>,
        algorithm: Option<String>,
    ) {
        self.entries.push(RelationalEntry {
            id,
            code,
            cosine_similarity,
            functional_accuracy,
            rank,
            category,
            agent,
            algorithm,
        });
    }

    /// Role 2 & 4: rank stored files by cosine similarity + functional
    /// accuracy and return the top `n` for a READ, optionally restricted
    /// to a given `category`, `agent`, and/or `algorithm`.
    fn read_top(&self, n: usize, category: &str, agent: Option<&str>, algorithm: Option<&str>) -> Vec<Value> {
        let mut ranked: Vec<&RelationalEntry> = self
            .entries
            .iter()
            .filter(|e| e.category == category)
            .filter(|e| agent.map_or(true, |a| e.agent.as_deref() == Some(a)))
            .filter(|e| algorithm.map_or(true, |alg| e.algorithm.as_deref() == Some(alg)))
            .collect();

        ranked.sort_by(|a, b| {
            b.rank_score()
                .partial_cmp(&a.rank_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        ranked
            .into_iter()
            .take(n.max(1))
            .map(|e| {
                json!({
                    "id": e.id,
                    "code": e.code,
                    "cosine_similarity": e.cosine_similarity,
                    "functional_accuracy": e.functional_accuracy,
                    "rank": e.rank,
                    "category": e.category,
                    "agent": e.agent,
                    "algorithm": e.algorithm,
                    "rank_score": e.rank_score(),
                })
            })
            .collect()
    }
}

// ---------------------------------------------------------------------
// Connection — the handle shared across agents
// ---------------------------------------------------------------------

pub struct Connection {
    vector_db: VectorDatabase,
    relational_db: RelationalDatabase,
    next_id: usize,
}

impl Connection {
    pub fn new() -> Self {
        Self {
            vector_db: VectorDatabase::new(),
            relational_db: RelationalDatabase::new(),
            next_id: 0,
        }
    }

    fn fresh_id(&mut self) -> String {
        self.next_id += 1;
        format!("code_{}", self.next_id)
    }
}

impl Default for Connection {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience wrapper around `run` for uploading a CSV file from disk —
/// what a user-facing "upload your training/test data" prompt in
/// `main.rs` should call. `category` should be `"training_data"` or
/// `"test_data"`; `algorithm` ties the data to the ML algorithm it
/// belongs to (e.g. `"kmeans"`), so `evolution`/`evaluator` can fetch the
/// right dataset later. Returns the same JSON string `run` would.
pub fn upload_csv_file(conn: &mut Connection, path: &str, category: &str, algorithm: &str) -> Result<String> {
    run(
        conn,
        &json!({
            "action": "write",
            "path": path,
            "category": category,
            "algorithm": algorithm,
        }),
    )
}

// ---------------------------------------------------------------------
// Tool entry point (called from mod.rs as `database::run`)
// ---------------------------------------------------------------------

/// Validates CSV text (consistent column count per row, at least one data
/// row) using the `csv` crate rather than hand-rolling a parser — real
/// CSV has quoting/escaping edge cases that are easy to get wrong.
/// Returns `(column_count, row_count)` on success.
fn validate_csv(content: &str) -> Result<(usize, usize)> {
    let mut reader = csv::ReaderBuilder::new().flexible(false).from_reader(content.as_bytes());

    let column_count = reader
        .headers()
        .map_err(|e| anyhow!("couldn't read a CSV header row: {e}"))?
        .len();

    if column_count == 0 {
        return Err(anyhow!("CSV header row is empty"));
    }

    let mut row_count = 0usize;
    for record in reader.records() {
        // `flexible(false)` already rejects rows with a different column
        // count than the header, surfacing that mismatch here.
        record.map_err(|e| anyhow!("malformed CSV row {}: {e}", row_count + 1))?;
        row_count += 1;
    }

    if row_count == 0 {
        return Err(anyhow!("CSV has a header row but no data rows"));
    }

    Ok((column_count, row_count))
}

/// Dispatches a `database` tool call. Accepts either a structured call:
///
/// WRITE: `{ "action": "write", "code": "..." (or "path": "<file path>" to
///           read from disk instead), "functional_accuracy": 0.8, "id": "optional",
///           "rank": optional, "category": "code" | "training_data" | "test_data" (default "code"),
///           "agent": "optional agent name", "algorithm": "optional algorithm name" }`
/// READ:  `{ "action": "read", "top_n": 1, "category": "code" (default),
///           "agent": "optional filter", "algorithm": "optional filter" }`
///
/// When `category` is `"training_data"` or `"test_data"`, the content is
/// validated as CSV and the write is rejected with a clear error if it
/// isn't well-formed.
///
/// or, since the tool schema declared in `mod.rs` only exposes a single
/// `query` string field, a plain-text fallback:
///   - `query == "read"` (case-insensitive) triggers a READ
///   - `query` containing a JSON object is parsed and dispatched as above
///   - anything else is treated as the raw code to WRITE
pub fn run(conn: &mut Connection, args: &Value) -> Result<String> {
    if let Some(action) = args.get("action").and_then(|v| v.as_str()) {
        return match action.to_lowercase().as_str() {
            "write" => handle_write(conn, args),
            "read" => handle_read(conn, args),
            other => Err(anyhow!("unknown database action: {other}")),
        };
    }

    if let Some(query) = args.get("query").and_then(|v| v.as_str()) {
        let trimmed = query.trim();

        if trimmed.eq_ignore_ascii_case("read") {
            return handle_read(conn, args);
        }

        if let Ok(parsed) = serde_json::from_str::<Value>(trimmed) {
            if parsed.get("action").is_some() {
                return run(conn, &parsed);
            }
        }

        // Fall back to treating the raw query text as code to store.
        let write_args = json!({ "code": trimmed });
        return handle_write(conn, &write_args);
    }

    Err(anyhow!("database call requires an 'action' field or a 'query' field"))
}

fn handle_write(conn: &mut Connection, args: &Value) -> Result<String> {
    // Content can come inline ("code") or, more conveniently for real
    // files (especially CSV uploads), from disk ("path").
    let code = match args.get("code").and_then(|v| v.as_str()) {
        Some(c) => c.to_string(),
        None => match args.get("path").and_then(|v| v.as_str()) {
            Some(path) => std::fs::read_to_string(path)
                .map_err(|e| anyhow!("failed to read file '{path}': {e}"))?,
            None => return Err(anyhow!("write requires either a 'code' field or a 'path' field")),
        },
    };

    let category = args
        .get("category")
        .and_then(|v| v.as_str())
        .unwrap_or("code")
        .to_string();

    // Training/test data must actually be valid CSV — reject bad uploads
    // up front instead of letting a malformed file surface as a confusing
    // failure later, inside the sandbox.
    let csv_shape = if category == "training_data" || category == "test_data" {
        Some(validate_csv(&code).map_err(|e| anyhow!("invalid CSV for category '{category}': {e}"))?)
    } else {
        None
    };

    let functional_accuracy = args
        .get("functional_accuracy")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;

    let rank = args.get("rank").and_then(|v| v.as_f64()).map(|v| v as f32);

    let agent = args.get("agent").and_then(|v| v.as_str()).map(String::from);
    let algorithm = args.get("algorithm").and_then(|v| v.as_str()).map(String::from);

    let id = args
        .get("id")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_else(|| conn.fresh_id());

    // Vector DB: score against what's already stored, return the score.
    let cosine_similarity = conn.vector_db.write_and_score(id.clone(), &code);

    // Vector DB writes through to the Relational DB, which stores the
    // file keyed by its cosine similarity and functional accuracy (plus
    // the rank/category/agent/algorithm tags, if provided).
    conn.relational_db.write(
        id.clone(),
        code,
        cosine_similarity,
        functional_accuracy,
        rank,
        category,
        agent,
        algorithm,
    );

    let mut response = json!({
        "status": "ok",
        "id": id,
        "cosine_similarity": cosine_similarity
    });
    if let Some((columns, rows)) = csv_shape {
        response["csv_columns"] = json!(columns);
        response["csv_rows"] = json!(rows);
    }

    Ok(response.to_string())
}

fn handle_read(conn: &mut Connection, args: &Value) -> Result<String> {
    let top_n = args
        .get("top_n")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;

    let category = args
        .get("category")
        .and_then(|v| v.as_str())
        .unwrap_or("code");

    let agent = args.get("agent").and_then(|v| v.as_str());
    let algorithm = args.get("algorithm").and_then(|v| v.as_str());

    let results = conn.relational_db.read_top(top_n, category, agent, algorithm);

    if results.is_empty() {
        return Ok(json!({
            "status": "empty",
            "message": format!("no '{category}' entries stored yet")
        })
        .to_string());
    }

    Ok(json!({
        "status": "ok",
        "results": results
    })
    .to_string())
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read_roundtrip() {
        let mut conn = Connection::new();

        let r1 = run(
            &mut conn,
            &json!({ "action": "write", "code": "fn add(a: i32, b: i32) -> i32 { a + b }", "functional_accuracy": 0.9 }),
        )
        .unwrap();
        let v1: Value = serde_json::from_str(&r1).unwrap();
        assert_eq!(v1["status"], "ok");
        assert_eq!(v1["cosine_similarity"], 0.0); // nothing to compare against yet

        let r2 = run(
            &mut conn,
            &json!({ "action": "write", "code": "fn add(a: i32, b: i32) -> i32 { a + b }", "functional_accuracy": 0.95 }),
        )
        .unwrap();
        let v2: Value = serde_json::from_str(&r2).unwrap();
        // Identical code should be maximally similar to the first entry.
        assert!(v2["cosine_similarity"].as_f64().unwrap() > 0.99);

        let read = run(&mut conn, &json!({ "action": "read", "top_n": 1 })).unwrap();
        let rv: Value = serde_json::from_str(&read).unwrap();
        assert_eq!(rv["status"], "ok");
        assert_eq!(rv["results"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn read_on_empty_db() {
        let mut conn = Connection::new();
        let read = run(&mut conn, &json!({ "action": "read" })).unwrap();
        let rv: Value = serde_json::from_str(&read).unwrap();
        assert_eq!(rv["status"], "empty");
    }

    #[test]
    fn query_string_fallback() {
        let mut conn = Connection::new();
        let r = run(&mut conn, &json!({ "query": "print('hello world')" })).unwrap();
        let v: Value = serde_json::from_str(&r).unwrap();
        assert_eq!(v["status"], "ok");

        let read = run(&mut conn, &json!({ "query": "read" })).unwrap();
        let rv: Value = serde_json::from_str(&read).unwrap();
        assert_eq!(rv["status"], "ok");
    }

    #[test]
    fn category_and_agent_filtering() {
        let mut conn = Connection::new();

        run(&mut conn, &json!({ "action": "write", "code": "x\n1\n2", "category": "training_data" })).unwrap();
        run(&mut conn, &json!({ "action": "write", "code": "fn a() {}", "category": "code", "agent": "agent_a", "functional_accuracy": 0.9, "rank": 2.5 })).unwrap();
        run(&mut conn, &json!({ "action": "write", "code": "fn b() {}", "category": "code", "agent": "agent_b", "functional_accuracy": 0.4 })).unwrap();

        // default read (category "code") should not surface training data
        let read_code = run(&mut conn, &json!({ "action": "read", "top_n": 10 })).unwrap();
        let v: Value = serde_json::from_str(&read_code).unwrap();
        let results = v["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r["category"] == "code"));

        // filtering by agent narrows to just that agent's entry, with rank preserved
        let read_agent = run(&mut conn, &json!({ "action": "read", "top_n": 10, "agent": "agent_a" })).unwrap();
        let va: Value = serde_json::from_str(&read_agent).unwrap();
        let results_a = va["results"].as_array().unwrap();
        assert_eq!(results_a.len(), 1);
        assert_eq!(results_a[0]["agent"], "agent_a");
        assert_eq!(results_a[0]["rank"], 2.5);

        // training_data category is retrievable on its own
        let read_training = run(&mut conn, &json!({ "action": "read", "category": "training_data" })).unwrap();
        let vt: Value = serde_json::from_str(&read_training).unwrap();
        assert_eq!(vt["results"][0]["category"], "training_data");
    }

    #[test]
    fn algorithm_filtering() {
        let mut conn = Connection::new();

        run(&mut conn, &json!({ "action": "write", "code": "fn kmeans() {}", "category": "code", "algorithm": "kmeans" })).unwrap();
        run(&mut conn, &json!({ "action": "write", "code": "fn svm() {}", "category": "code", "algorithm": "svm" })).unwrap();

        let read_kmeans = run(&mut conn, &json!({ "action": "read", "algorithm": "kmeans" })).unwrap();
        let v: Value = serde_json::from_str(&read_kmeans).unwrap();
        assert_eq!(v["results"][0]["algorithm"], "kmeans");
        assert!(v["results"][0]["code"].as_str().unwrap().contains("kmeans"));

        let read_svm = run(&mut conn, &json!({ "action": "read", "algorithm": "svm" })).unwrap();
        let v2: Value = serde_json::from_str(&read_svm).unwrap();
        assert_eq!(v2["results"][0]["algorithm"], "svm");
    }

    #[test]
    fn csv_upload_validates_shape() {
        let mut conn = Connection::new();

        // well-formed CSV: accepted, shape reported back
        let good = run(
            &mut conn,
            &json!({
                "action": "write",
                "code": "x1,x2,label\n1,2,0\n3,4,1\n5,6,1",
                "category": "training_data",
                "algorithm": "kmeans"
            }),
        )
        .unwrap();
        let gv: Value = serde_json::from_str(&good).unwrap();
        assert_eq!(gv["status"], "ok");
        assert_eq!(gv["csv_columns"], 3);
        assert_eq!(gv["csv_rows"], 3);

        // malformed CSV (ragged row): rejected with a clear error, not silently stored
        let bad = run(
            &mut conn,
            &json!({
                "action": "write",
                "code": "x1,x2,label\n1,2,0\n3,4", // missing a column
                "category": "training_data",
                "algorithm": "kmeans"
            }),
        );
        assert!(bad.is_err());

        // non-CSV categories are unaffected by CSV validation
        let code_write = run(&mut conn, &json!({ "action": "write", "code": "not,csv,but,fine,here" })).unwrap();
        let cv: Value = serde_json::from_str(&code_write).unwrap();
        assert_eq!(cv["status"], "ok");
        assert!(cv.get("csv_columns").is_none());
    }

    #[test]
    fn upload_csv_file_reads_from_disk() {
        let mut conn = Connection::new();
        let path = std::env::temp_dir().join("database_rs_test_upload.csv");
        std::fs::write(&path, "a,b\n1,2\n3,4").unwrap();

        let result = upload_csv_file(&mut conn, path.to_str().unwrap(), "test_data", "kmeans").unwrap();
        let v: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["status"], "ok");
        assert_eq!(v["csv_columns"], 2);
        assert_eq!(v["csv_rows"], 2);

        let read = run(&mut conn, &json!({ "action": "read", "category": "test_data", "algorithm": "kmeans" })).unwrap();
        let rv: Value = serde_json::from_str(&read).unwrap();
        assert!(rv["results"][0]["code"].as_str().unwrap().contains("1,2"));

        std::fs::remove_file(&path).ok();
    }
}
