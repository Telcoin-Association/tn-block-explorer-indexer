//! Bounded, blocking-safe reads over the node's own databases — the
//! direct-read seam that makes the indexer zero-RPC.
//!
//! Every `RethEnv` touch in the binary goes through [`NodeReader::read`]:
//! synchronous MDBX/EVM work is run on the protocol task spawner's blocking
//! pool behind a 64-permit semaphore, mirroring the pattern (and the constant)
//! of tn-3's own `tn_` RPC namespace (`crates/execution/tn-rpc/src/rpc_ext.rs`).
//! One permit covers one request-level operation: a 25-block hydration loop is
//! ONE closure under ONE permit, never one permit per block.
//!
//! `ConsensusChain`/`EpochRecordDb` reads are async message-passing to a
//! background file-owner thread — they are awaited directly on the async
//! runtime and need no permit.

use crate::{
    extract::{SEL_DECIMALS, SEL_NAME, SEL_SYMBOL, SEL_TOTAL_SUPPLY},
    storage::TxPointer,
};
use eyre::eyre;
use std::{collections::BTreeMap, sync::Arc};
use tn_reth::{
    error::EvmReadError,
    system_calls::{ConsensusRegistry, EpochState},
    RethEnv,
};
use tn_storage::consensus::ConsensusChain;
use tn_types::{
    Address, Block, BlockHeader as _, Bytes, Receipt, RecoveredBlock, SealedBlock, SolValue,
    TransactionSigned, TxHash,
};
use tokio::sync::{oneshot, Semaphore};

/// Maximum number of concurrent blocking node reads.
///
/// Mirrors tn-rpc's `MAX_CONCURRENT_REGISTRY_READS` (rpc_ext.rs) and its
/// rationale: tokio's blocking pool defaults to 512 threads, and 64 leaves
/// headroom for BLS signing and engine blocking tasks that share the SAME
/// pool. The observer serves no public JSON-RPC, but the pool is shared
/// regardless. A permit is held for the full lifetime of the blocking work, so
/// this bounds true pool occupancy; overload queues on the semaphore, it never
/// rejects.
pub const MAX_CONCURRENT_NODE_READS: usize = 64;

/// The `EvmReadError::Internal` message prefix tn-reth produces when a contract
/// call HALTs (invalid opcode, out of gas). Used to keep a halting non-token
/// from being treated as a node fault during metadata fetches — see
/// [`read_metadata_bytes`].
const HALT_MESSAGE_PREFIX: &str = "contract call halted";

/// Read-only access to the node's own databases, shared by the API and the
/// ExEx loop's token-metadata step.
#[derive(Debug, Clone)]
pub struct NodeReader {
    /// EVM/MDBX read handle (`Clone` over `Arc`; `Send + Sync`).
    reth: RethEnv,
    /// Consensus DB read handle (async message-passing; needs no wrapper).
    consensus: ConsensusChain,
    /// Bounds concurrent blocking reads — see [`MAX_CONCURRENT_NODE_READS`].
    permits: Arc<Semaphore>,
}

/// Everything needed to render one mined transaction, gathered inside a single
/// blocking read. `types::build_api_transaction` maps it to the wire shape.
#[derive(Debug, Clone)]
pub struct TxData {
    /// Recovered sender.
    pub sender: Address,
    /// The signed transaction.
    pub tx: TransactionSigned,
    /// Receipt status.
    pub success: bool,
    /// Gas used by this transaction alone (cumulative delta).
    pub gas_used: u64,
    /// Containing block number.
    pub block_number: u64,
    /// Zero-based index within the block.
    pub tx_index: u64,
    /// Containing block's base fee (for `effective_gas_price`).
    pub base_fee: Option<u64>,
    /// Containing block's timestamp.
    pub timestamp: u64,
}

/// Result of a token-metadata fetch: per-field `None` for fields that reverted
/// or failed to decode. "Ok" means at least one field decoded.
#[derive(Debug, Clone, Default)]
pub struct TokenMetadata {
    /// Decoded `name()`.
    pub name: Option<String>,
    /// Decoded `symbol()`.
    pub symbol: Option<String>,
    /// Decoded `decimals()`.
    pub decimals: Option<u8>,
}

impl TokenMetadata {
    /// Whether the fetch counts as a success (at least one field decoded).
    pub fn any_decoded(&self) -> bool {
        self.name.is_some() || self.symbol.is_some() || self.decimals.is_some()
    }
}

/// Outcome of a read-only contract call for `POST /call`: an on-chain revert is
/// a structured response, not an error (internal node faults stay errors).
#[derive(Debug, Clone)]
pub enum CallOutcome {
    /// The call succeeded; raw ABI-encoded output bytes.
    Success(Bytes),
    /// The call reverted on-chain.
    Revert {
        /// Decoded human-readable reason, if available.
        reason: Option<String>,
        /// Raw ABI-encoded revert bytes.
        output: Bytes,
    },
}

/// One `/stats` snapshot gathered inside a single blocking read.
#[derive(Debug, Clone)]
pub struct StatsSnapshot {
    /// Canonical tip block number.
    pub latest_block: u64,
    /// Next-block base fee in wei.
    pub gas_price_wei: u128,
    /// The node's chain id.
    pub chain_id: u64,
    /// Total transactions ever mined.
    pub total_txs: u64,
}

impl NodeReader {
    /// Create a reader over the ExEx context's handles with a fresh
    /// [`MAX_CONCURRENT_NODE_READS`]-permit semaphore.
    pub fn new(reth: RethEnv, consensus: ConsensusChain) -> Self {
        Self {
            reth,
            consensus,
            permits: Arc::new(Semaphore::new(MAX_CONCURRENT_NODE_READS)),
        }
    }

    /// The consensus DB handle (async reads; no permit needed).
    pub fn consensus(&self) -> &ConsensusChain {
        &self.consensus
    }

    /// Run one request-level read closure against `RethEnv` on the blocking
    /// pool, bounded by the semaphore.
    ///
    /// The permit is acquired BEFORE the spawn and moved into the closure, so
    /// it is held for the full blocking lifetime — a client disconnect cannot
    /// leak pool occupancy. Overload queues on the semaphore; it never rejects.
    pub async fn read<T, F>(&self, name: &'static str, f: F) -> eyre::Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&RethEnv) -> eyre::Result<T> + Send + 'static,
    {
        let permit = self.permits.clone().acquire_owned().await?;
        let env = self.reth.clone();
        let (tx, rx) = oneshot::channel();
        self.reth
            .get_task_spawner()
            .spawn_blocking_task(name, move || {
                let _permit = permit; // held until the blocking read completes
                let _ = tx.send(f(&env)); // receiver may have dropped; ignore
                Ok(())
            });
        rx.await?
    }

    /// One newest-first page of the chain-wide transaction feed (serves
    /// `/txs`): arithmetic pagination over `TxNumber` — no OFFSET scan.
    ///
    /// Returns the page rows and the feed total.
    pub async fn latest_txs_page(
        &self,
        page: u64,
        per_page: u64,
    ) -> eyre::Result<(Vec<TxData>, u64)> {
        self.read("indexer-latest-txs", move |env| {
            let total = env.total_transactions()?;
            let Some((low, high)) = desc_page_bounds(total, page, per_page) else {
                return Ok((Vec::new(), total));
            };
            let entries = env.transactions_by_tx_range_with_meta(low..=high)?;

            // receipts + base fee once per distinct block
            let mut blocks: BTreeMap<u64, (Vec<Receipt>, Option<u64>)> = BTreeMap::new();
            for entry in &entries {
                if let std::collections::btree_map::Entry::Vacant(vacant) =
                    blocks.entry(entry.block_number)
                {
                    let receipts = env
                        .receipts_by_block(entry.block_number.into())?
                        .ok_or_else(|| {
                            eyre!("receipts missing for block {}", entry.block_number)
                        })?;
                    let base_fee = env
                        .sealed_header_by_number(entry.block_number)?
                        .ok_or_else(|| eyre!("header missing for block {}", entry.block_number))?
                        .base_fee_per_gas();
                    vacant.insert((receipts, base_fee));
                }
            }

            let mut out = Vec::with_capacity(entries.len());
            // range is ascending; serve newest-first
            for entry in entries.into_iter().rev() {
                let (receipts, base_fee) = blocks
                    .get(&entry.block_number)
                    .ok_or_else(|| eyre!("block context missing for {}", entry.block_number))?;
                let index = usize::try_from(entry.index)?;
                let receipt = receipts.get(index).ok_or_else(|| {
                    eyre!(
                        "receipt {} missing in block {}",
                        entry.index,
                        entry.block_number
                    )
                })?;
                let (tx, sender) = entry.transaction.into_parts();
                out.push(TxData {
                    sender,
                    tx,
                    success: receipt.success,
                    gas_used: own_gas_used(receipts, index),
                    block_number: entry.block_number,
                    tx_index: entry.index,
                    base_fee: *base_fee,
                    timestamp: entry.timestamp,
                });
            }
            Ok((out, total))
        })
        .await
    }

    /// Hydrate SQLite pointer rows into full transaction data (serves
    /// `/address/{addr}/txs` and the transfer feeds' tx_hash/timestamp).
    ///
    /// Replays each DISTINCT block once via `replay_block_as_chain` (senders
    /// recovered, receipts included) inside ONE closure holding ONE permit.
    /// Output order matches the pointer order.
    ///
    /// Failure to find a pointed-at block or transaction is the R11
    /// dangling-pointer invariant violated — a bug by construction, surfaced as
    /// an error (the API returns 500 and logs at `error!`).
    pub async fn hydrate_pointers(&self, pointers: Vec<TxPointer>) -> eyre::Result<Vec<TxData>> {
        if pointers.is_empty() {
            return Ok(Vec::new());
        }
        self.read("indexer-hydrate", move |env| {
            let mut blocks: BTreeMap<u64, (RecoveredBlock<Block>, Vec<Receipt>)> = BTreeMap::new();
            for number in distinct_blocks(&pointers) {
                let chain = env.replay_block_as_chain(number)?.ok_or_else(|| {
                    eyre!("dangling index pointer: block {number} not in node DB")
                })?;
                let (block, receipts) = chain
                    .blocks_and_receipts()
                    .next()
                    .ok_or_else(|| eyre!("replayed chain for block {number} is empty"))?;
                blocks.insert(number, (block.clone(), receipts.clone()));
            }

            let mut out = Vec::with_capacity(pointers.len());
            for pointer in &pointers {
                let (block, receipts) = blocks
                    .get(&pointer.block_number)
                    .ok_or_else(|| eyre!("block context missing for {}", pointer.block_number))?;
                let index = usize::try_from(pointer.tx_index)?;
                let (sender, tx) =
                    block.transactions_with_sender().nth(index).ok_or_else(|| {
                        eyre!(
                            "dangling index pointer: tx {} missing in block {}",
                            pointer.tx_index,
                            pointer.block_number
                        )
                    })?;
                let receipt = receipts.get(index).ok_or_else(|| {
                    eyre!(
                        "dangling index pointer: receipt {} missing in block {}",
                        pointer.tx_index,
                        pointer.block_number
                    )
                })?;
                out.push(TxData {
                    sender: *sender,
                    tx: tx.clone(),
                    success: receipt.success,
                    gas_used: own_gas_used(receipts, index),
                    block_number: pointer.block_number,
                    tx_index: pointer.tx_index,
                    base_fee: block.header().base_fee_per_gas(),
                    timestamp: block.header().timestamp(),
                });
            }
            Ok(out)
        })
        .await
    }

    /// A mined transaction by hash (serves `/txs/{hash}`); `None` if unknown.
    pub async fn tx_by_hash(&self, hash: TxHash) -> eyre::Result<Option<TxData>> {
        self.read("indexer-tx-by-hash", move |env| {
            let Some((recovered, meta)) = env.transaction_by_hash_with_meta(hash)? else {
                return Ok(None);
            };
            let Some((receipt, gas_used)) = env.receipt_by_hash_with_gas_used(hash)? else {
                return Ok(None);
            };
            let (tx, sender) = recovered.into_parts();
            Ok(Some(TxData {
                sender,
                tx,
                success: receipt.success,
                gas_used,
                block_number: meta.block_number,
                tx_index: meta.index,
                base_fee: meta.base_fee,
                timestamp: meta.timestamp,
            }))
        })
        .await
    }

    /// Account state + deployed bytecode at the latest canonical state (serves
    /// `/address/{addr}`), in one closure.
    pub async fn account_summary(
        &self,
        address: Address,
    ) -> eyre::Result<(Option<tn_types::Account>, Option<Bytes>)> {
        self.read("indexer-account", move |env| {
            let account = env.retrieve_account(&address)?;
            let code = env.account_code(&address)?;
            Ok((account, code))
        })
        .await
    }

    /// Fetch ERC-20 metadata (`name`/`symbol`/`decimals`) at the canonical tip
    /// in one closure. Used by the ExEx metadata step and `/tokens/{addr}`
    /// cache misses.
    pub async fn token_metadata(&self, token: Address) -> eyre::Result<TokenMetadata> {
        self.read("indexer-token-metadata", move |env| {
            fetch_token_metadata(env, token)
        })
        .await
    }

    /// Metadata AND live `totalSupply` in one closure (a `/tokens/{addr}` cache
    /// miss).
    pub async fn token_metadata_and_supply(
        &self,
        token: Address,
    ) -> eyre::Result<(TokenMetadata, Option<String>)> {
        self.read("indexer-token-full", move |env| {
            let metadata = fetch_token_metadata(env, token)?;
            let supply = fetch_total_supply(env, token)?;
            Ok((metadata, supply))
        })
        .await
    }

    /// Live `totalSupply` only (a `/tokens/{addr}` cache hit; supply is never
    /// cached because it is mutable).
    pub async fn token_supply(&self, token: Address) -> eyre::Result<Option<String>> {
        self.read("indexer-token-supply", move |env| {
            fetch_total_supply(env, token)
        })
        .await
    }

    /// Generic read-only contract call (serves `POST /call`). Reverts come back
    /// as [`CallOutcome::Revert`]; internal node faults are errors.
    pub async fn call(&self, to: Address, data: Bytes) -> eyre::Result<CallOutcome> {
        self.read("indexer-call", move |env| {
            match env.read_contract(to, data) {
                Ok(result) => Ok(CallOutcome::Success(result)),
                Err(EvmReadError::Revert { output, reason }) => {
                    Ok(CallOutcome::Revert { reason, output })
                }
                Err(EvmReadError::Internal(message)) => {
                    Err(eyre!("contract read failed: {message}"))
                }
            }
        })
        .await
    }

    /// Count blocks per beneficiary over the trailing `window` blocks (serves
    /// `/validators/leaders`). Returns `(from_block, to_block, counts)` with
    /// counts sorted by blocks descending, address ascending.
    pub async fn leader_counts(
        &self,
        window: u64,
    ) -> eyre::Result<(u64, u64, Vec<(Address, u64)>)> {
        self.read("indexer-leaders", move |env| {
            let tip = env.last_block_number()?;
            let from = tip.saturating_sub(window.saturating_sub(1));
            let headers = env.blocks_for_range(from..=tip)?;
            let mut counts: BTreeMap<Address, u64> = BTreeMap::new();
            for header in &headers {
                *counts.entry(header.beneficiary()).or_default() += 1;
            }
            let mut counts: Vec<(Address, u64)> = counts.into_iter().collect();
            counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            Ok((from, tip, counts))
        })
        .await
    }

    /// Current epoch state from the consensus registry plus the canonical tip,
    /// in one closure (serves `/epochs/current` and the `/stats` committee
    /// fallback).
    pub async fn current_epoch_with_tip(&self) -> eyre::Result<(EpochState, u64)> {
        self.read("indexer-epoch-state", move |env| {
            let state = env.epoch_state_from_canonical_tip()?;
            let tip = env.last_block_number()?;
            Ok((state, tip))
        })
        .await
    }

    /// Committee `ValidatorInfo`s for an epoch from the on-chain registry
    /// (serves `/validators`).
    pub async fn validators_for_epoch(
        &self,
        epoch: u32,
    ) -> eyre::Result<Vec<ConsensusRegistry::ValidatorInfo>> {
        self.read("indexer-validators", move |env| {
            env.validators_for_epoch(epoch)
        })
        .await
    }

    /// Best-effort committee ADDRESSES for an epoch: `None` outside the
    /// registry's ring buffer (where `getCommitteeValidators` reverts) or on
    /// any read failure. BLS keys are always available from the epoch record
    /// instead — documented limitation.
    pub async fn committee_addresses(&self, epoch: u32) -> Option<Vec<Address>> {
        self.read("indexer-epoch-committee", move |env| {
            Ok(env
                .validators_for_epoch(epoch)
                .ok()
                .map(|infos| infos.iter().map(|info| info.validatorAddress).collect()))
        })
        .await
        .ok()
        .flatten()
    }

    /// Timestamps for a set of block numbers in one closure (epoch boundary
    /// times for `/epochs`).
    pub async fn header_timestamps(&self, numbers: Vec<u64>) -> eyre::Result<BTreeMap<u64, u64>> {
        self.read("indexer-header-times", move |env| {
            let mut out = BTreeMap::new();
            for number in numbers {
                if let Some(header) = env.header_by_number(number)? {
                    out.insert(number, header.timestamp);
                }
            }
            Ok(out)
        })
        .await
    }

    /// One newest-first page of blocks via dense-height arithmetic (serves
    /// `/blocks`). Returns the page and `total = tip + 1`.
    pub async fn blocks_page(
        &self,
        page: u64,
        per_page: u64,
    ) -> eyre::Result<(Vec<SealedBlock>, u64)> {
        self.read("indexer-blocks", move |env| {
            let tip = env.last_block_number()?;
            let total = tip.saturating_add(1);
            let numbers = desc_page_items(total, page, per_page);
            let mut blocks = Vec::with_capacity(numbers.len());
            for number in numbers {
                // heights are dense: a missing block at or below the tip is a bug
                blocks.push(
                    env.sealed_block_by_number(number)?
                        .ok_or_else(|| eyre!("block {number} missing at or below tip {tip}"))?,
                );
            }
            Ok((blocks, total))
        })
        .await
    }

    /// One block by number (serves `/blocks/{number}`); `None` if unknown.
    pub async fn block_by_number(&self, number: u64) -> eyre::Result<Option<SealedBlock>> {
        self.read("indexer-block", move |env| {
            Ok(env.sealed_block_by_number(number)?)
        })
        .await
    }

    /// The `/stats` direct-read snapshot in one closure.
    pub async fn stats_snapshot(&self) -> eyre::Result<StatsSnapshot> {
        self.read("indexer-stats", move |env| {
            Ok(StatsSnapshot {
                latest_block: env.last_block_number()?,
                gas_price_wei: env.get_gas_price()?,
                chain_id: env.chainspec().chain_id(),
                total_txs: env.total_transactions()?,
            })
        })
        .await
    }
}

/// Fetch and decode the three ERC-20 metadata fields against one `RethEnv`.
fn fetch_token_metadata(env: &RethEnv, token: Address) -> eyre::Result<TokenMetadata> {
    let name = read_metadata_bytes(env, token, SEL_NAME)?.and_then(|b| decode_string_return(&b));
    let symbol =
        read_metadata_bytes(env, token, SEL_SYMBOL)?.and_then(|b| decode_string_return(&b));
    let decimals =
        read_metadata_bytes(env, token, SEL_DECIMALS)?.and_then(|b| decode_decimals_return(&b));
    Ok(TokenMetadata {
        name,
        symbol,
        decimals,
    })
}

/// Live `totalSupply()`, decoded to its exact decimal string.
fn fetch_total_supply(env: &RethEnv, token: Address) -> eyre::Result<Option<String>> {
    Ok(read_metadata_bytes(env, token, SEL_TOTAL_SUPPLY)?
        .and_then(|b| tn_types::U256::abi_decode(&b).ok())
        .map(|v| v.to_string()))
}

/// One metadata selector call: `Ok(None)` for an on-chain revert (a
/// non-ERC-20 answering honestly), `Err` for internal node faults.
///
/// Deviation from a literal Revert/Internal split: tn-reth maps a HALTed call
/// (invalid opcode, out-of-gas) into `EvmReadError::Internal` with a
/// distinguishable message. A halting non-token would otherwise convert one
/// weird contract sighting into a permanent indexing-loop error, so
/// halt-shaped Internals are treated as fetch failures (field `None`) too.
fn read_metadata_bytes(
    env: &RethEnv,
    token: Address,
    selector: [u8; 4],
) -> eyre::Result<Option<Bytes>> {
    match env.read_contract(token, Bytes::copy_from_slice(&selector)) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(EvmReadError::Revert { .. }) => Ok(None),
        Err(EvmReadError::Internal(message)) if message.starts_with(HALT_MESSAGE_PREFIX) => {
            Ok(None)
        }
        Err(EvmReadError::Internal(message)) => Err(eyre!("token metadata read failed: {message}")),
    }
}

/// Decode an ABI `string` return, with a lenient fallback for bytes32-string
/// tokens (MKR-style): the trim-NUL utf8 of the raw 32-byte word.
///
/// Empty strings count as not-decoded on both paths: they are useless to the
/// explorer, and the non-validating ABI decode would otherwise accept an
/// all-zero word (offset 0 → length 0 → `""`) from any non-string return.
pub fn decode_string_return(bytes: &[u8]) -> Option<String> {
    if let Ok(decoded) = String::abi_decode(bytes) {
        return Some(decoded).filter(|s| !s.is_empty());
    }
    if bytes.len() == 32 {
        let trimmed: Vec<u8> = bytes
            .iter()
            .copied()
            .take_while(|byte| *byte != 0)
            .collect();
        return String::from_utf8(trimmed).ok().filter(|s| !s.is_empty());
    }
    None
}

/// Decode an ABI `uint8` return. alloy has no `SolValue` impl for `u8`, so the
/// word is decoded WIDE as `u16` and narrowed (the same workaround tn-3 uses in
/// rpc_ext.rs); any value above `u8::MAX` is a decode failure.
pub fn decode_decimals_return(bytes: &[u8]) -> Option<u8> {
    let wide = u16::abi_decode(bytes).ok()?;
    u8::try_from(wide).ok()
}

/// Gas used by the receipt at `index` alone: the cumulative delta vs. the
/// previous receipt in the block.
fn own_gas_used(receipts: &[Receipt], index: usize) -> u64 {
    let cumulative = receipts
        .get(index)
        .map(|r| r.cumulative_gas_used)
        .unwrap_or_default();
    match index.checked_sub(1).and_then(|prev| receipts.get(prev)) {
        Some(prev) => cumulative.saturating_sub(prev.cumulative_gas_used),
        None => cumulative,
    }
}

/// The distinct block numbers referenced by a pointer list, first-seen order
/// preserved (pure; drives the one-replay-per-block hydration grouping).
pub fn distinct_blocks(pointers: &[TxPointer]) -> Vec<u64> {
    let mut out = Vec::new();
    for pointer in pointers {
        if !out.contains(&pointer.block_number) {
            out.push(pointer.block_number);
        }
    }
    out
}

/// Inclusive `(low, high)` item-index bounds for 0-based descending page
/// `page` of `total` items, or `None` when the page is past the end.
///
/// Shared by `/txs` (item = `TxNumber`) and `/blocks` (item = block number,
/// `total = tip + 1`) — both paginate by pure arithmetic, no OFFSET scan.
pub fn desc_page_bounds(total: u64, page: u64, per_page: u64) -> Option<(u64, u64)> {
    let skip = page.checked_mul(per_page)?;
    if skip >= total {
        return None;
    }
    let high = total - 1 - skip;
    let low = high.saturating_sub(per_page.saturating_sub(1));
    Some((low, high))
}

/// The item indices of a descending page, newest first (see
/// [`desc_page_bounds`]).
pub fn desc_page_items(total: u64, page: u64, per_page: u64) -> Vec<u64> {
    match desc_page_bounds(total, page, per_page) {
        Some((low, high)) => (low..=high).rev().collect(),
        None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole-block gas vector derived through [`own_gas_used`].
    fn gas_used_from_cumulative(cumulative: &[u64]) -> Vec<u64> {
        let receipts: Vec<Receipt> = cumulative
            .iter()
            .map(|&cumulative_gas_used| Receipt {
                success: true,
                cumulative_gas_used,
                ..Default::default()
            })
            .collect();
        (0..receipts.len())
            .map(|index| own_gas_used(&receipts, index))
            .collect()
    }

    #[test]
    fn gas_cumulative_diff_vectors() {
        assert_eq!(
            gas_used_from_cumulative(&[21_000, 74_000, 74_100]),
            vec![21_000, 53_000, 100]
        );
        assert_eq!(gas_used_from_cumulative(&[]), Vec::<u64>::new());
        assert_eq!(gas_used_from_cumulative(&[42_000]), vec![42_000]);
    }

    #[test]
    fn distinct_blocks_preserves_first_seen_order() {
        let pointers = [
            TxPointer {
                block_number: 9,
                tx_index: 1,
            },
            TxPointer {
                block_number: 9,
                tx_index: 0,
            },
            TxPointer {
                block_number: 4,
                tx_index: 2,
            },
            TxPointer {
                block_number: 9,
                tx_index: 3,
            },
            TxPointer {
                block_number: 2,
                tx_index: 0,
            },
            TxPointer {
                block_number: 4,
                tx_index: 0,
            },
        ];
        assert_eq!(distinct_blocks(&pointers), vec![9, 4, 2]);
        assert_eq!(distinct_blocks(&[]), Vec::<u64>::new());
    }

    #[test]
    fn desc_page_bounds_arithmetic() {
        // first page of 100 items
        assert_eq!(desc_page_bounds(100, 0, 25), Some((75, 99)));
        // second page
        assert_eq!(desc_page_bounds(100, 1, 25), Some((50, 74)));
        // final partial page
        assert_eq!(desc_page_bounds(10, 0, 25), Some((0, 9)));
        // page past the end => None (handler serves empty items, correct total)
        assert_eq!(desc_page_bounds(10, 1, 25), None);
        assert_eq!(desc_page_bounds(0, 0, 25), None);
        // /blocks edges: tip < per_page (total = tip + 1)
        assert_eq!(desc_page_items(3, 0, 25), vec![2, 1, 0]);
        // page past genesis => empty
        assert_eq!(desc_page_items(3, 1, 25), Vec::<u64>::new());
        // exact boundary
        assert_eq!(
            desc_page_items(50, 1, 25),
            (0..=24).rev().collect::<Vec<_>>()
        );
    }

    #[test]
    fn decode_string_return_abi_and_bytes32() {
        // proper ABI-encoded string
        let encoded = "Telcoin".to_string().abi_encode();
        assert_eq!(decode_string_return(&encoded).as_deref(), Some("Telcoin"));
        // empty strings count as not-decoded (also covers the all-zero word,
        // which the lenient ABI decode reads as offset 0 -> length 0 -> "")
        assert_eq!(decode_string_return(&String::new().abi_encode()), None);
        // bytes32-string token (MKR-style): raw right-padded word
        let mut word = [0u8; 32];
        word[..3].copy_from_slice(b"MKR");
        assert_eq!(decode_string_return(&word).as_deref(), Some("MKR"));
        // all-zero word: not a string
        assert_eq!(decode_string_return(&[0u8; 32]), None);
        // garbage / empty
        assert_eq!(decode_string_return(&[]), None);
        assert_eq!(decode_string_return(&[1, 2, 3]), None);
    }

    #[test]
    fn decode_decimals_narrows_u16_to_u8() {
        assert_eq!(decode_decimals_return(&18u16.abi_encode()), Some(18));
        assert_eq!(decode_decimals_return(&0u16.abi_encode()), Some(0));
        assert_eq!(decode_decimals_return(&255u16.abi_encode()), Some(255));
        // decodes as u16 but exceeds u8::MAX => decode failure
        assert_eq!(decode_decimals_return(&300u16.abi_encode()), None);
        // word too large for u16 / malformed
        assert_eq!(
            decode_decimals_return(&tn_types::U256::MAX.abi_encode()),
            None
        );
        assert_eq!(decode_decimals_return(&[]), None);
    }

    #[test]
    fn token_metadata_any_decoded() {
        assert!(!TokenMetadata::default().any_decoded());
        assert!(TokenMetadata {
            symbol: Some("TOK".into()),
            ..Default::default()
        }
        .any_decoded());
        assert!(TokenMetadata {
            decimals: Some(0),
            ..Default::default()
        }
        .any_decoded());
    }
}
