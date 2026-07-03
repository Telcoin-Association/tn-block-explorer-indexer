//! Shared indexer health, read by `/health`.
//!
//! The API task and the ExEx loop are separate tokio tasks: when the loop
//! panics or errors, the API stays up and degrades HONESTLY — [`LiveGuard`]'s
//! `Drop` flips `live` to false, `/health` keeps answering 200, and `lag`
//! grows. The explorer renders a banner instead of erroring.

use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};

/// Lock-free indexer health shared between the ExEx loop and the API.
#[derive(Debug, Default)]
pub struct IndexerStatus {
    /// Highest indexed block, stored as `cursor + 1` (0 = nothing indexed yet)
    /// so "no block indexed" needs no separate flag.
    last_indexed_plus_one: AtomicU64,
    /// Node canonical tip as last observed by the indexer.
    node_tip: AtomicU64,
    /// True while the ExEx indexing loop is running.
    live: AtomicBool,
}

impl IndexerStatus {
    /// Record a newly indexed block (monotonic max).
    pub fn set_last_indexed(&self, block: u64) {
        self.last_indexed_plus_one
            .fetch_max(block.saturating_add(1), Ordering::Relaxed);
    }

    /// Highest indexed block, if any.
    pub fn last_indexed(&self) -> Option<u64> {
        match self.last_indexed_plus_one.load(Ordering::Relaxed) {
            0 => None,
            stored => Some(stored - 1),
        }
    }

    /// Record the node tip (monotonic max — TN heights never rewind).
    pub fn set_node_tip(&self, tip: u64) {
        self.node_tip.fetch_max(tip, Ordering::Relaxed);
    }

    /// Node canonical tip as last observed.
    pub fn node_tip(&self) -> u64 {
        self.node_tip.load(Ordering::Relaxed)
    }

    /// Whether the indexing loop is running.
    pub fn live(&self) -> bool {
        self.live.load(Ordering::Relaxed)
    }

    /// Blocks between the observed tip and the cursor.
    pub fn lag(&self) -> u64 {
        self.node_tip()
            .saturating_sub(self.last_indexed().unwrap_or(0))
    }

    fn set_live(&self, live: bool) {
        self.live.store(live, Ordering::Relaxed);
    }
}

/// RAII guard around the indexing loop's liveness bit: constructing it sets
/// `live = true`; ANY exit from the loop — return, error, or panic unwind —
/// drops it and sets `live = false`.
#[derive(Debug)]
pub struct LiveGuard(Arc<IndexerStatus>);

impl LiveGuard {
    /// Mark the loop live until the guard drops.
    pub fn new(status: Arc<IndexerStatus>) -> Self {
        status.set_live(true);
        Self(status)
    }
}

impl Drop for LiveGuard {
    fn drop(&mut self) {
        self.0.set_live(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_indexed_distinguishes_none_from_block_zero() {
        let status = IndexerStatus::default();
        assert_eq!(status.last_indexed(), None);
        status.set_last_indexed(0);
        assert_eq!(status.last_indexed(), Some(0));
        // monotonic: an older report never rewinds the cursor view
        status.set_last_indexed(5);
        status.set_last_indexed(3);
        assert_eq!(status.last_indexed(), Some(5));
    }

    #[test]
    fn lag_and_live_guard() {
        let status = Arc::new(IndexerStatus::default());
        status.set_node_tip(10);
        status.set_last_indexed(7);
        assert_eq!(status.lag(), 3);
        assert!(!status.live());
        {
            let _guard = LiveGuard::new(status.clone());
            assert!(status.live());
        }
        // dropped (as it would be on loop error/panic): honest degradation
        assert!(!status.live());
    }
}
