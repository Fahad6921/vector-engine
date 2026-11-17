use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

use vecengine_server::{router, AppState};

struct Api {
    app: Router,
}

impl Api {
    async fn call(&self, method: Method, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
        let req = Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(body.map_or_else(Body::empty, |b| Body::from(b.to_string())))
            .unwrap();
        let resp = self.app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }
}

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("vecengine-api-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn api(dir: &std::path::Path) -> Api {
    Api {
        app: router(Arc::new(AppState::open(dir).unwrap())),
    }
}

fn ids(v: &Value) -> Vec<u64> {
    v["hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["id"].as_u64().unwrap())
        .collect()
}

#[tokio::test]
async fn full_lifecycle() {
    let dir = temp_dir("lifecycle");
    let api = api(&dir);
    use Method as M;

    let (s, _) = api.call(M::GET, "/health", None).await;
    assert_eq!(s, StatusCode::OK);

    let (s, body) = api
        .call(
            M::PUT,
            "/collections/docs",
            Some(json!({"dim": 3, "metric": "l2", "text_field": "title"})),
        )
        .await;
    assert_eq!(s, StatusCode::CREATED, "{body}");
    let (s, _) = api
        .call(M::PUT, "/collections/docs", Some(json!({"dim": 3})))
        .await;
    assert_eq!(s, StatusCode::CONFLICT);

    let points = json!({"points": [
        {"id": 1, "vector": [0.0, 0.0, 0.0], "payload": {"title": "rust vector search", "year": 2024}},
        {"id": 2, "vector": [1.0, 0.0, 0.0], "payload": {"title": "python notebooks", "year": 2020}},
        {"id": 3, "vector": [5.0, 5.0, 5.0], "payload": {"title": "rust compilers", "year": 2022}},
    ]});
    let (s, body) = api.call(M::PUT, "/collections/docs/points", Some(points)).await;
    assert_eq!((s, body["upserted"].as_u64()), (StatusCode::OK, Some(3)));

    let (_, body) = api
        .call(
            M::POST,
            "/collections/docs/search",
            Some(json!({"vector": [0.9, 0.0, 0.0], "k": 2})),
        )
        .await;
    assert_eq!(ids(&body), vec![2, 1]);
    assert_eq!(body["hits"][0]["payload"]["title"], "python notebooks");
    assert!(body["hits"][0]["distance"].as_f64().unwrap() < 0.02);

    let filtered = json!({"vector": [0.9, 0.0, 0.0], "k": 5,
                          "filter": {"field": "year", "range": {"gte": 2022}}});
    let (_, body) = api
        .call(M::POST, "/collections/docs/search", Some(filtered))
        .await;
    assert_eq!(ids(&body), vec![1, 3]);

    let (_, body) = api
        .call(
            M::POST,
            "/collections/docs/search",
            Some(json!({"text": "rust", "with_payload": false})),
        )
        .await;
    assert_eq!(ids(&body).len(), 2);
    assert!(body["hits"][0]["score"].as_f64().unwrap() > 0.0);
    assert!(body["hits"][0].get("payload").is_none());

    let (_, body) = api
        .call(
            M::POST,
            "/collections/docs/search",
            Some(json!({"vector": [5.0, 5.0, 5.0], "text": "rust vector", "k": 2})),
        )
        .await;
    assert_eq!(ids(&body).len(), 2);

    let (s, body) = api.call(M::GET, "/collections/docs/points/3", None).await;
    assert_eq!(
        (s, body["payload"]["year"].as_u64()),
        (StatusCode::OK, Some(2022))
    );

    let (s, _) = api
        .call(
            M::POST,
            "/collections/docs/points/delete",
            Some(json!({"ids": [3]})),
        )
        .await;
    assert_eq!(s, StatusCode::OK);
    let (s, _) = api.call(M::GET, "/collections/docs/points/3", None).await;
    assert_eq!(s, StatusCode::NOT_FOUND);

    let (s, body) = api.call(M::POST, "/collections/docs/compact", None).await;
    assert_eq!(
        (s, body["points"].as_u64(), body["segments"].as_u64()),
        (StatusCode::OK, Some(2), Some(1))
    );

    // Restart: a new server over the same directory sees the same data.
    drop(api);
    let api = self::api(&dir);
    let (_, body) = api.call(M::GET, "/collections", None).await;
    assert_eq!(body["collections"][0]["name"], "docs");
    assert_eq!(body["collections"][0]["points"], 2);
    let (_, body) = api
        .call(
            M::POST,
            "/collections/docs/search",
            Some(json!({"text": "python"})),
        )
        .await;
    assert_eq!(ids(&body), vec![2]);

    let (s, _) = api.call(M::DELETE, "/collections/docs", None).await;
    assert_eq!(s, StatusCode::OK);
    let (s, _) = api.call(M::GET, "/collections/docs", None).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert!(!dir.join("docs").exists());
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn rejects_bad_requests() {
    let dir = temp_dir("errors");
    let api = api(&dir);
    use Method as M;

    let bad = [
        (M::PUT, "/collections/bad%20name", Some(json!({"dim": 3}))),
        (
            M::PUT,
            "/collections/x",
            Some(json!({"dim": 3, "metric": "manhattan"})),
        ),
        (M::PUT, "/collections/y", Some(json!({"dim": 0}))),
    ];
    for (m, uri, body) in bad {
        let (s, resp) = api.call(m, uri, body).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{uri}: {resp}");
        assert!(resp["error"].is_string());
    }

    api.call(M::PUT, "/collections/v", Some(json!({"dim": 2}))).await;
    let bad = [
        json!({"points": [{"id": 1, "vector": [1.0]}]}),
        json!({"points": [{"id": 1, "vector": [1.0, 2.0], "payload": [1, 2]}]}),
    ];
    for body in bad {
        let (s, _) = api.call(M::PUT, "/collections/v/points", Some(body)).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
    }
    let bad = [
        json!({"k": 3}),
        json!({"vector": [1.0]}),
        json!({"vector": [1.0, 2.0], "k": 0}),
        json!({"text": "hello"}),
        json!({"vector": [1.0, 2.0], "filter": {"field": "x"}}),
    ];
    for body in bad {
        let (s, resp) = api
            .call(M::POST, "/collections/v/search", Some(body.clone()))
            .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{body}: {resp}");
    }
    let (s, _) = api
        .call(
            M::POST,
            "/collections/missing/search",
            Some(json!({"vector": [1.0]})),
        )
        .await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    std::fs::remove_dir_all(dir).unwrap();
}
