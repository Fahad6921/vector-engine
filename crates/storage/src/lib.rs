//! Durable collections on top of the core index: write-ahead log, immutable
//! on-disk segments, deletes and compaction.

pub mod collection;
mod segment;
pub mod wal;

pub use collection::{Collection, Config, Hit, Stats};
pub use wal::Record;
