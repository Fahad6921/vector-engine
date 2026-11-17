//! Durable collections on top of the core index: write-ahead log, immutable
//! on-disk segments, deletes, compaction and metadata filters.

mod bitset;
pub mod collection;
pub mod filter;
mod segment;
pub mod wal;

pub use collection::{Collection, Config, FilterStrategy, Hit, Stats};
pub use filter::Filter;
pub use wal::Record;
