//! vecengine-server: serves the collections in a data directory over HTTP.
//!
//!     vecengine-server --data ./collections --addr 127.0.0.1:8080
//!
//! Both can also be set with VECENGINE_DATA and VECENGINE_ADDR.

use std::sync::Arc;

use vecengine_server::{router, AppState};

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let mut data = std::env::var("VECENGINE_DATA").unwrap_or_else(|_| "./collections".into());
    let mut addr = std::env::var("VECENGINE_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data" => data = args.next().expect("--data DIR"),
            "--addr" => addr = args.next().expect("--addr HOST:PORT"),
            other => {
                eprintln!(
                    "unknown argument {other:?}; usage: vecengine-server [--data DIR] [--addr HOST:PORT]"
                );
                std::process::exit(2);
            }
        }
    }

    let state = Arc::new(AppState::open(&data)?);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("vecengine-server listening on http://{addr}, data in {data}");

    axum::serve(listener, router(state.clone()))
        .with_graceful_shutdown(async {
            let ctrl_c = tokio::signal::ctrl_c();
            let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("install SIGTERM handler");
            tokio::select! {
                _ = ctrl_c => {}
                _ = term.recv() => {}
            }
        })
        .await?;

    println!("shutting down, flushing collections");
    state.flush_all()
}
