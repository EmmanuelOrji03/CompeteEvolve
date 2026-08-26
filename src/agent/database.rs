use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};


const EMBEDDING_DIM: usize = 256;


#[derive(Clone, Debug)]
struct VectorEntry {
   
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
   
   
    rank: Option<f32>,
   
    category: String,
    
    agent: Option<String>,
}

impl RelationalEntry {

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
    ) {
        self.entries.push(RelationalEntry {
            id,
            code,
            cosine_similarity,
            functional_accuracy,
            rank,
            category,
            agent,
        });
    }

   
    fn read_top(&self, n: usize, category: &str, agent: Option<&str>) -> Vec<Value> {
        let mut ranked: Vec<&RelationalEntry> = self
            .entries
            .iter()
            .filter(|e| e.category == category)
            .filter(|e| agent.map_or(true, |a| e.agent.as_deref() == Some(a)))
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
                    "rank_score": e.rank_score(),
                })
            })
            .collect()
    }
}



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
    let code = args
        .get("code")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("write requires a 'code' field"))?;

    let functional_accuracy = args
        .get("functional_accuracy")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;

    let rank = args.get("rank").and_then(|v| v.as_f64()).map(|v| v as f32);

    let category = args
        .get("category")
        .and_then(|v| v.as_str())
        .unwrap_or("code")
        .to_string();

    let agent = args.get("agent").and_then(|v| v.as_str()).map(String::from);

    let id = args
        .get("id")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_else(|| conn.fresh_id());

    
    let cosine_similarity = conn.vector_db.write_and_score(id.clone(), code);

   
    conn.relational_db.write(
        id.clone(),
        code.to_string(),
        cosine_similarity,
        functional_accuracy,
        rank,
        category,
        agent,
    );

    Ok(json!({
        "status": "ok",
        "id": id,
        "cosine_similarity": cosine_similarity
    })
    .to_string())
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

    let results = conn.relational_db.read_top(top_n, category, agent);

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

        run(&mut conn, &json!({ "action": "write", "code": "training set A", "category": "training_data" })).unwrap();
        run(&mut conn, &json!({ "action": "write", "code": "fn a() {}", "category": "code", "agent": "agent_a", "functional_accuracy": 0.9, "rank": 2.5 })).unwrap();
        run(&mut conn, &json!({ "action": "write", "code": "fn b() {}", "category": "code", "agent": "agent_b", "functional_accuracy": 0.4 })).unwrap();


        let read_code = run(&mut conn, &json!({ "action": "read", "top_n": 10 })).unwrap();
        let v: Value = serde_json::from_str(&read_code).unwrap();
        let results = v["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r["category"] == "code"));

        
        let read_agent = run(&mut conn, &json!({ "action": "read", "top_n": 10, "agent": "agent_a" })).unwrap();
        let va: Value = serde_json::from_str(&read_agent).unwrap();
        let results_a = va["results"].as_array().unwrap();
        assert_eq!(results_a.len(), 1);
        assert_eq!(results_a[0]["agent"], "agent_a");
        assert_eq!(results_a[0]["rank"], 2.5);

        
        let read_training = run(&mut conn, &json!({ "action": "read", "category": "training_data" })).unwrap();
        let vt: Value = serde_json::from_str(&read_training).unwrap();
        assert_eq!(vt["results"][0]["category"], "training_data");
    }
}
