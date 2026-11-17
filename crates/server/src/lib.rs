//! HTTP API over a directory of collections.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Path as UrlPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use parking_lot::RwLock;
use serde::Deserialize;
use serde_json::{json, Value};

use vecengine::Metric;
use vecengine_storage::{Collection, Config, Filter, FilterStrategy, Record};

type Shared = Arc<RwLock<Collection>>;

pub struct AppState {
    dir: PathBuf,
    collections: RwLock<HashMap<String, Shared>>,
}

impl AppState {
    /// Opens every collection under `dir` (one sub-directory each).
    pub fn open(dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        let mut collections = HashMap::new();
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            if !path.join("manifest.json").exists() {
                continue;
            }
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let config = Collection::stored_config(&path)?;
            collections.insert(name, Arc::new(RwLock::new(Collection::open(&path, config)?)));
        }
        Ok(Self {
            dir,
            collections: RwLock::new(collections),
        })
    }

    /// Flushes every collection so the next start doesn't replay WALs.
    pub fn flush_all(&self) -> io::Result<()> {
        for c in self.collections.read().values() {
            c.write().flush()?;
        }
        Ok(())
    }

    fn get(&self, name: &str) -> Result<Shared, ApiError> {
        self.collections
            .read()
            .get(name)
            .cloned()
            .ok_or_else(|| ApiError::not_found(format!("no collection named {name:?}")))
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/collections", get(list_collections))
        .route(
            "/collections/{name}",
            put(create_collection)
                .get(collection_info)
                .delete(drop_collection),
        )
        .route("/collections/{name}/points", put(upsert_points))
        .route("/collections/{name}/points/delete", post(delete_points))
        .route("/collections/{name}/points/{id}", get(get_point))
        .route("/collections/{name}/search", post(search))
        .route("/collections/{name}/flush", post(flush))
        .route("/collections/{name}/compact", post(compact))
        .with_state(state)
}

pub struct ApiError(StatusCode, String);

impl ApiError {
    fn bad_request(msg: impl Into<String>) -> Self {
        Self(StatusCode::BAD_REQUEST, msg.into())
    }

    fn not_found(msg: impl Into<String>) -> Self {
        Self(StatusCode::NOT_FOUND, msg.into())
    }
}

impl From<io::Error> for ApiError {
    fn from(e: io::Error) -> Self {
        let status = match e.kind() {
            io::ErrorKind::InvalidInput => StatusCode::BAD_REQUEST,
            io::ErrorKind::NotFound => StatusCode::NOT_FOUND,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self(status, e.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error": self.1}))).into_response()
    }
}

type ApiResult = Result<Json<Value>, ApiError>;

/// Runs CPU-bound or blocking work off the async executor.
async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> Result<T, ApiError> + Send + 'static,
) -> Result<T, ApiError> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
}

fn valid_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

fn info(name: &str, c: &Collection) -> Value {
    let cfg = c.config();
    let s = c.stats();
    json!({
        "name": name,
        "dim": cfg.dim,
        "metric": cfg.metric.name(),
        "m": cfg.m,
        "ef_construction": cfg.ef_construction,
        "text_field": cfg.text_field,
        "points": s.live,
        "segments": s.segments,
        "memtable": s.memtable,
        "deleted": s.deleted,
    })
}

async fn list_collections(State(state): State<Arc<AppState>>) -> ApiResult {
    let collections = state.collections.read();
    let mut names: Vec<&String> = collections.keys().collect();
    names.sort();
    let list: Vec<Value> = names
        .into_iter()
        .map(|n| info(n, &collections[n].read()))
        .collect();
    Ok(Json(json!({ "collections": list })))
}

#[derive(Deserialize)]
struct CreateCollection {
    dim: usize,
    #[serde(default = "default_metric")]
    metric: String,
    m: Option<usize>,
    ef_construction: Option<usize>,
    text_field: Option<String>,
}

fn default_metric() -> String {
    "cosine".into()
}

async fn create_collection(
    State(state): State<Arc<AppState>>,
    UrlPath(name): UrlPath<String>,
    Json(req): Json<CreateCollection>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    if !valid_name(&name) {
        return Err(ApiError::bad_request(
            "collection names are 1-64 characters of letters, digits, _ and -",
        ));
    }
    let metric = Metric::from_name(&req.metric).ok_or_else(|| {
        ApiError::bad_request(format!("unknown metric {:?}, use l2, ip or cosine", req.metric))
    })?;
    let mut config = Config::new(req.dim, metric);
    config.m = req.m.unwrap_or(config.m);
    config.ef_construction = req.ef_construction.unwrap_or(config.ef_construction);
    config.text_field = req.text_field;

    blocking(move || {
        // Held across the open so two creates of one name can't race.
        let mut collections = state.collections.write();
        if collections.contains_key(&name) {
            return Err(ApiError(
                StatusCode::CONFLICT,
                format!("collection {name:?} already exists"),
            ));
        }
        let c = Collection::open(state.dir.join(&name), config)?;
        let body = info(&name, &c);
        collections.insert(name, Arc::new(RwLock::new(c)));
        Ok((StatusCode::CREATED, Json(body)))
    })
    .await
}

async fn collection_info(State(state): State<Arc<AppState>>, UrlPath(name): UrlPath<String>) -> ApiResult {
    let c = state.get(&name)?;
    let body = info(&name, &c.read());
    Ok(Json(body))
}

async fn drop_collection(State(state): State<Arc<AppState>>, UrlPath(name): UrlPath<String>) -> ApiResult {
    blocking(move || {
        let removed = state.collections.write().remove(&name);
        if removed.is_none() {
            return Err(ApiError::not_found(format!("no collection named {name:?}")));
        }
        std::fs::remove_dir_all(state.dir.join(&name))?;
        Ok(Json(json!({ "dropped": name })))
    })
    .await
}

#[derive(Deserialize)]
struct Point {
    id: u64,
    vector: Vec<f32>,
    payload: Option<Value>,
}

#[derive(Deserialize)]
struct UpsertPoints {
    points: Vec<Point>,
}

async fn upsert_points(
    State(state): State<Arc<AppState>>,
    UrlPath(name): UrlPath<String>,
    Json(req): Json<UpsertPoints>,
) -> ApiResult {
    let c = state.get(&name)?;
    let records: Vec<Record> = req
        .points
        .into_iter()
        .map(|p| Record::Upsert {
            id: p.id,
            vector: p.vector,
            payload: p.payload.map(|v| v.to_string().into_bytes()).unwrap_or_default(),
        })
        .collect();
    let n = records.len();
    blocking(move || {
        c.write().write(records)?;
        Ok(Json(json!({ "upserted": n })))
    })
    .await
}

#[derive(Deserialize)]
struct DeletePoints {
    ids: Vec<u64>,
}

async fn delete_points(
    State(state): State<Arc<AppState>>,
    UrlPath(name): UrlPath<String>,
    Json(req): Json<DeletePoints>,
) -> ApiResult {
    let c = state.get(&name)?;
    let n = req.ids.len();
    let records = req.ids.into_iter().map(|id| Record::Delete { id }).collect();
    blocking(move || {
        c.write().write(records)?;
        Ok(Json(json!({ "deleted": n })))
    })
    .await
}

fn payload_json(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).unwrap_or(Value::Null)
}

async fn get_point(
    State(state): State<Arc<AppState>>,
    UrlPath((name, id)): UrlPath<(String, u64)>,
) -> ApiResult {
    let c = state.get(&name)?;
    let c = c.read();
    let (vector, payload) = c
        .get(id)
        .ok_or_else(|| ApiError::not_found(format!("no point with id {id}")))?;
    Ok(Json(
        json!({ "id": id, "vector": vector, "payload": payload_json(payload) }),
    ))
}

#[derive(Deserialize)]
struct SearchRequest {
    vector: Option<Vec<f32>>,
    text: Option<String>,
    #[serde(default = "default_k")]
    k: usize,
    ef: Option<usize>,
    filter: Option<Value>,
    #[serde(default = "default_true")]
    with_payload: bool,
}

fn default_k() -> usize {
    10
}

fn default_true() -> bool {
    true
}

/// Vector search returns `distance` (lower is closer); keyword and hybrid search
/// return `score` (higher is better).
async fn search(
    State(state): State<Arc<AppState>>,
    UrlPath(name): UrlPath<String>,
    Json(req): Json<SearchRequest>,
) -> ApiResult {
    let c = state.get(&name)?;
    if req.k == 0 || req.k > 1000 {
        return Err(ApiError::bad_request("k must be between 1 and 1000"));
    }
    let filter = req
        .filter
        .as_ref()
        .map(Filter::from_json)
        .transpose()
        .map_err(|e| ApiError::bad_request(format!("bad filter: {e}")))?;

    blocking(move || {
        let c = c.read();
        let ef = req.ef.unwrap_or(100).max(req.k);
        if let Some(v) = &req.vector {
            if v.len() != c.config().dim {
                return Err(ApiError::bad_request(format!(
                    "vector has dim {}, collection has {}",
                    v.len(),
                    c.config().dim
                )));
            }
        }
        if req.text.is_some() && c.config().text_field.is_none() {
            return Err(ApiError::bad_request(
                "text search needs a collection created with text_field",
            ));
        }

        let started = Instant::now();
        let (hits, key, sign) = match (&req.vector, &req.text) {
            (Some(v), Some(t)) => (c.search_hybrid(v, t, req.k, ef, filter.as_ref()), "score", -1.0),
            (Some(v), None) => (
                c.search_filtered(v, req.k, ef, filter.as_ref(), FilterStrategy::Auto),
                "distance",
                1.0,
            ),
            (None, Some(t)) => (c.search_text(t, req.k, filter.as_ref()), "score", -1.0),
            (None, None) => return Err(ApiError::bad_request("search needs a vector, text, or both")),
        };
        let took = started.elapsed().as_secs_f64() * 1000.0;

        let hits: Vec<Value> = hits
            .iter()
            .map(|h| {
                let mut hit = json!({ "id": h.id, key: sign * h.distance });
                if req.with_payload {
                    if let Some((_, p)) = c.get(h.id) {
                        hit["payload"] = payload_json(p);
                    }
                }
                hit
            })
            .collect();
        Ok(Json(json!({ "hits": hits, "took_ms": took })))
    })
    .await
}

async fn flush(State(state): State<Arc<AppState>>, UrlPath(name): UrlPath<String>) -> ApiResult {
    let c = state.get(&name)?;
    blocking(move || {
        let mut c = c.write();
        c.flush()?;
        Ok(Json(info(&name, &c)))
    })
    .await
}

async fn compact(State(state): State<Arc<AppState>>, UrlPath(name): UrlPath<String>) -> ApiResult {
    let c = state.get(&name)?;
    blocking(move || {
        let mut c = c.write();
        c.compact()?;
        Ok(Json(info(&name, &c)))
    })
    .await
}
