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
//! runtime and need no permit. That single actor thread serializes them, so a
//! `/consensus/blocks` page of `per_page` header reads is `per_page` sequential
//! round-trips, bounded per request only by `per_page <= 100`.
//!
//! Consensus numbers are guarded to `1..=latest_consensus_number()` BEFORE the
//! pack is touched: number 0 is the pre-genesis anchor (never stored) and a
//! current-epoch miss surfaces as `Err`, not `None`. The in-memory latest
//! counter is updated only after the pack write is acknowledged (TN
//! `crates/storage/src/consensus.rs`: `save_consensus_output(..).await?` and
//! only then `latest_consensus.update(..)`), so any number `<= latest` is
//! readable by construction and a current-epoch `Err` is a real read failure,
//! never a tip race. For a SEALED epoch, TN's by-number reads answer an absent
//! pack and a pack that cannot be opened (truncated, corrupt) identically with
//! `Ok(None)`; only the by-digest read preserves the error. So a by-number
//! miss below the tip is "absent locally or unreadable" — a normal observer
//! state rendered as 404 / `null`, with the node log and the by-digest
//! `/blocks/{n}` path as the diagnostics. Per-epoch pack helpers likewise
//! degrade to `None` + `warn!` when the pack is not held locally. The one
//! CPU-bound consensus operation, BLS certificate verification, runs under the
//! blocking permit like a `RethEnv` read — see
//! [`NodeReader::verify_epoch_certificate`].
//!
//! Per-epoch committee reads must name their pin: tn-reth has no unpinned
//! per-epoch committee reader, because the registry mutates the CURRENT
//! epoch's committee arrays mid-epoch on a governance `burn`. The pin comes
//! from the consensus DB's epoch records (the previous epoch's closing block)
//! and is resolved OUTSIDE the blocking closure — see [`NodeReader::committee_addresses`].

use crate::{
    extract::{SEL_DECIMALS, SEL_NAME, SEL_SYMBOL, SEL_TOTAL_SUPPLY},
    storage::TxPointer,
};
use eyre::eyre;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
use tn_reth::{
    error::EvmReadError,
    system_calls::{ConsensusRegistry, EpochState},
    RethEnv,
};
use tn_storage::consensus::{ConsensusChain, ConsensusChainError};
use tn_types::{
    Address, AuthorityIdentifier, Block, BlockHeader as _, BlsPublicKey, Bytes, ConsensusHeader,
    ConsensusHeaderDigest, ConsensusOutput, EpochCertificate, EpochRecord, Receipt, RecoveredBlock,
    ReputationScores, Round, SealedBlock, SealedHeader, SolValue, TransactionSigned, TxHash, B256,
};
use tokio::sync::{oneshot, Semaphore};
use tracing::warn;

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

    /// The CURRENT committee's registry `ValidatorInfo`s at the canonical tip
    /// (serves `/validators`).
    ///
    /// A tip read, matching the removed `RethEnv::validators_for_epoch` (which
    /// was also unpinned): it reflects a mid-epoch governance `burn`
    /// immediately, which is the live state an explorer should show. The epoch
    /// is the registry's own `getCurrentEpoch` at the tip, so no epoch argument
    /// is needed — this also drops the old cross-source coupling where the
    /// epoch came from the consensus DB but the committee came from the EVM,
    /// which could disagree across a boundary.
    pub async fn current_committee_validators(
        &self,
    ) -> eyre::Result<Vec<ConsensusRegistry::ValidatorInfo>> {
        self.read("indexer-validators", move |env| {
            Ok(env.epoch_state_from_canonical_tip()?.validators)
        })
        .await
    }

    /// Committee ADDRESSES for `epoch`, pinned to the block that seated that
    /// committee (serves `/epochs/{n}`).
    ///
    /// The pin is the previous epoch's closing block — `record(epoch - 1)
    /// .final_state` from the consensus DB's epoch records — or genesis for
    /// epoch 0. Pinning there means `getCommitteeValidators` runs at a state
    /// where `epoch` IS the registry's current epoch, so the registry's
    /// retained-epoch window (`[current - 3, current + 2]`) never applies and
    /// every past epoch answers. This is the ONE place a consensus-DB read
    /// feeds a `RethEnv` read.
    ///
    /// `None` when the predecessor record has not been written yet, when the
    /// pin block is missing from the node DB, or on any read failure — the
    /// route degrades to the epoch record's BLS keys rather than failing.
    ///
    /// For the IN-PROGRESS epoch this returns the committee as SEATED AT THE
    /// BOUNDARY, while `/epochs/current` and `/validators` read the mutable
    /// tip; the two disagree after a mid-epoch governance `burn`. That split is
    /// intended: this route describes the epoch, those describe the registry
    /// right now.
    pub async fn committee_addresses(&self, epoch: u32) -> Option<Vec<Address>> {
        // genesis pin for epoch 0; otherwise the previous epoch's closing block
        let pin: Option<B256> = match epoch.checked_sub(1) {
            None => None,
            Some(previous) => Some(
                self.consensus()
                    .epochs()
                    .record_by_epoch(previous)
                    .await?
                    .final_state
                    .hash,
            ),
        };
        self.read("indexer-epoch-committee", move |env| {
            let Some(header) = (match pin {
                Some(hash) => env.sealed_header_by_hash(hash)?,
                None => env.sealed_header_by_number(0)?,
            }) else {
                return Ok(None);
            };
            let state = env.epoch_state_at_header(&header)?;
            // tripwire: the pin must report the epoch we asked for. Serving a
            // neighbouring epoch's committee silently is worse than `null`.
            if state.epoch != epoch {
                return Ok(None);
            }
            Ok(Some(
                state
                    .validators
                    .iter()
                    .map(|info| info.validatorAddress)
                    .collect(),
            ))
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

    // ----- consensus chain: async actor reads, no permit -----

    /// The newest consensus output number the node has processed. Sync: an
    /// in-memory atomic, no actor round-trip. It advances only after the
    /// output's pack write is acknowledged, so every number in `1..=this` is
    /// readable by construction (the pack tail can be one output AHEAD of it
    /// while a write is in flight, never behind).
    pub fn latest_consensus_number(&self) -> u64 {
        self.consensus.latest_consensus_number()
    }

    /// The epoch of the newest processed consensus output (sync, in-memory).
    pub fn latest_consensus_epoch(&self) -> u32 {
        self.consensus.latest_consensus_epoch()
    }

    /// One FULL consensus output by number — header plus every batch's raw
    /// transactions (serves `/consensus/blocks/{n}` detail and `/batches`).
    /// One actor round-trip decoding roughly a `/blocks` page of tx bytes;
    /// never call it from a list.
    ///
    /// `Ok(None)` outside `1..=latest_consensus_number()` with no pack touch
    /// (see [`consensus_number_stored`]) and for a sealed epoch whose pack this
    /// observer does not hold OR cannot open (absent, truncated and corrupt
    /// all collapse to `Ok(None)` in TN's `consensus_output_by_number`, so a
    /// 404 alone does not prove the pack is absent: check the node log and
    /// `/blocks/{n}.consensus.consensus_number`, which resolves by digest and
    /// surfaces the real error as a `warn!`). `Err` for a current-epoch read
    /// failure or a sealed pack that opens but cannot be read — always a real
    /// fault, because the latest counter advances only after the pack write
    /// is acknowledged (there is no tip race to tolerate).
    pub async fn consensus_output(&self, number: u64) -> eyre::Result<Option<ConsensusOutput>> {
        if !consensus_number_stored(number, self.latest_consensus_number()) {
            return Ok(None);
        }
        self.consensus
            .consensus_output_by_number(number)
            .await
            .map_err(chain_err)
    }

    /// The newest consensus header (serves `/consensus/latest`): one actor
    /// round-trip for the header numbered `latest_consensus_number()`, NOT
    /// the pack's own tail — the tail can be one output ahead of the counter
    /// while a write is in flight, and reading by the counter keeps `number`
    /// consistent with the bounds guard on `/consensus/blocks/{n}` and with
    /// `/consensus/blocks.total`. `Ok(None)` before the first output.
    pub async fn consensus_latest_header(&self) -> eyre::Result<Option<ConsensusHeader>> {
        let latest = self.latest_consensus_number();
        if latest == 0 {
            return Ok(None);
        }
        self.consensus
            .consensus_header_by_number(latest)
            .await
            .map_err(chain_err)
    }

    /// One newest-first page of consensus headers (serves `/consensus/blocks`).
    /// Returns `(headers, total)` with `total = latest_consensus_number()`.
    ///
    /// Numbers come from [`consensus_page_numbers`]; each is one sequential
    /// actor round-trip (the actor serializes them regardless), so a page
    /// costs `per_page` header decodes. Rows this observer lacks (`Ok(None)`:
    /// a sealed epoch's pack absent locally or unreadable — TN's by-number
    /// read does not distinguish the two) are SKIPPED with one summarizing
    /// `warn!` — the page shrinks rather than failing, and `total` still
    /// reports the chain's count. An `Err` fails the page.
    pub async fn consensus_headers_page(
        &self,
        page: u64,
        per_page: u64,
    ) -> eyre::Result<(Vec<ConsensusHeader>, u64)> {
        let total = self.latest_consensus_number();
        let numbers = consensus_page_numbers(total, page, per_page);
        let mut headers = Vec::with_capacity(numbers.len());
        let mut skipped: Vec<u64> = Vec::new();
        for number in numbers {
            match self
                .consensus
                .consensus_header_by_number(number)
                .await
                .map_err(chain_err)?
            {
                Some(header) => headers.push(header),
                None => skipped.push(number),
            }
        }
        if let (Some(newest), Some(oldest)) = (skipped.first(), skipped.last()) {
            warn!(
                count = skipped.len(),
                newest, oldest, "consensus headers absent locally or unreadable; page shrunk"
            );
        }
        Ok((headers, total))
    }

    /// The consensus output number whose header hashes to `digest`, looked up
    /// in `epoch`'s pack (serves `consensus.consensus_number` on
    /// `/blocks/{n}`): one actor round-trip that decodes the header to read
    /// its number. `Ok(None)` for an epoch past the tip (no pack touch), for a
    /// pack this observer does not hold, or an unknown digest.
    pub async fn consensus_number_by_digest(
        &self,
        epoch: u32,
        digest: B256,
    ) -> eyre::Result<Option<u64>> {
        if epoch > self.latest_consensus_epoch() {
            return Ok(None);
        }
        Ok(self
            .consensus
            .consensus_header_by_digest(epoch, ConsensusHeaderDigest::from(digest))
            .await
            .map_err(chain_err)?
            .map(|header| header.number))
    }

    // ----- epoch records: async actor reads, no permit -----

    /// An epoch's record + certificate and its predecessor's record, as
    /// `(this, previous)` (serves `/consensus/epochs` rows and `/epochs/{n}`).
    ///
    /// `this` is `get_epoch_by_number` — record and certificate read
    /// atomically; the certificate is aggregated at the NEXT epoch's start, so
    /// the newest sealed epoch is normally `Some((record, None))`, and the
    /// in-progress epoch has no record yet (`None`). `previous` is
    /// `record_by_epoch(epoch - 1)` (`None` for epoch 0) and yields the epoch's
    /// first numbers: `previous.final_consensus.number + 1` and
    /// `previous.final_state.number + 1`. Two to three actor round-trips.
    pub async fn epoch_record_pair(
        &self,
        epoch: u32,
    ) -> (
        Option<(EpochRecord, Option<EpochCertificate>)>,
        Option<EpochRecord>,
    ) {
        let db = self.consensus.epochs();
        let this = db.get_epoch_by_number(epoch).await;
        let previous = match epoch.checked_sub(1) {
            Some(prev) => db.record_by_epoch(prev).await,
            None => None,
        };
        (this, previous)
    }

    /// Whether this observer holds `record`'s epoch pack through its final
    /// output (serves `pack_complete` on `/consensus/epochs/{n}`): one header
    /// read of `record.final_consensus.number`. `false` when the pack is
    /// absent, truncated or corrupt as well as when it is genuinely
    /// incomplete — TN's `is_epoch_complete` collapses an `Ok(None)` and a
    /// read error (logged at `error!`) alike.
    pub async fn epoch_pack_complete(&self, record: &EpochRecord) -> bool {
        self.consensus.is_epoch_complete(record).await
    }

    /// Each authority's last committed round in `epoch`'s pack (serves
    /// `last_committed_rounds` on `/consensus/epochs/{n}`), sorted by round
    /// DESCENDING then authority ascending — "furthest ahead" first, and
    /// deterministic for equal rounds. One actor round-trip.
    ///
    /// `None` past the tip epoch (no pack touch) and, with a `warn!`, when the
    /// pack is not held locally or cannot be read — a normal observer state the
    /// route renders as `null`.
    pub async fn epoch_last_committed(
        &self,
        epoch: u32,
    ) -> Option<Vec<(AuthorityIdentifier, Round)>> {
        if epoch > self.latest_consensus_epoch() {
            return None;
        }
        match self.consensus.read_last_committed(epoch).await {
            Ok(map) => {
                let mut rounds: Vec<(AuthorityIdentifier, Round)> = map.into_iter().collect();
                rounds.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                Some(rounds)
            }
            Err(e) => {
                warn!(epoch, error = %e, "last committed rounds unavailable");
                None
            }
        }
    }

    /// The final reputation scores of `epoch` (serves `final_reputation_scores`
    /// on `/consensus/epochs/{n}`): the pack's last commit flagged
    /// `final_of_schedule`, decoded as one sub-DAG in one actor round-trip.
    /// `None` past the tip epoch, when the pack lacks such a commit or is not
    /// held locally, and (with a `warn!`) on a read error.
    pub async fn epoch_final_reputation(&self, epoch: u32) -> Option<ReputationScores> {
        if epoch > self.latest_consensus_epoch() {
            return None;
        }
        match self
            .consensus
            .read_latest_commit_with_final_reputation_scores(epoch)
            .await
        {
            Ok(sub_dag) => sub_dag.map(|dag| dag.reputation_scores().clone()),
            Err(e) => {
                warn!(epoch, error = %e, "final reputation scores unavailable");
                None
            }
        }
    }

    /// The BLS keys seated for `epoch` (leader/author → BLS on detail routes):
    /// `record(epoch).committee`, else `record(epoch - 1).next_committee` (the
    /// in-progress epoch), else `None`. One to two actor round-trips.
    pub async fn committee_keys(&self, epoch: u32) -> Option<BTreeSet<BlsPublicKey>> {
        self.consensus.epochs().get_committee_keys(epoch).await
    }

    /// Verify `cert` against `record` (`verified` on `/epochs/{n}` and
    /// `/consensus/epochs/{n}`): a digest match plus an aggregate BLS
    /// signature check over the bitmap's signers. That pairing is CPU work, so
    /// it runs on the blocking pool under a permit exactly like a `RethEnv`
    /// read — never inline on the async runtime. Detail routes only.
    pub async fn verify_epoch_certificate(
        &self,
        record: EpochRecord,
        cert: EpochCertificate,
    ) -> eyre::Result<bool> {
        self.read("indexer-verify-epoch-cert", move |_env| {
            Ok(record.verify_with_cert(&cert))
        })
        .await
    }

    /// The canonical tip's sealed header (serves `exec_tip` /
    /// `exec_tip_consensus` on `/consensus/latest`), via the blocking permit
    /// path. `canonical_tip` reads the canonical-in-memory state, so it can
    /// LEAD `last_block_number()` — the committed-DB view the other reads use —
    /// by an in-flight output.
    pub async fn tip_header(&self) -> eyre::Result<SealedHeader> {
        self.read("indexer-tip-header", |env| Ok(env.canonical_tip()))
            .await
    }
}

/// Lift a `ConsensusChain` read failure into `eyre`.
///
/// `ConsensusChainError` implements `std::error::Error`, and every payload it
/// carries (`PackError`, `EpochDbError`, their `Arc<io::Error>` /
/// `Arc<OpenError>` members) is `Send + Sync + 'static`, so the typed error is
/// preserved for downcasting instead of being flattened to its message.
fn chain_err(e: ConsensusChainError) -> eyre::Report {
    eyre::Report::new(e)
}

/// Whether `number` can name a stored consensus output given the tip
/// `latest`: outputs are numbered densely `1..=latest`. Number 0 is the
/// pre-genesis anchor (never written to a pack) and anything past the tip is
/// not there yet — both are answered WITHOUT a pack read, because a
/// current-epoch miss surfaces as `Err`, not `None`
/// (TN `crates/storage/src/consensus.rs:842-865`).
pub fn consensus_number_stored(number: u64, latest: u64) -> bool {
    number != 0 && number <= latest
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

/// The consensus output numbers of descending page `page`, newest first.
/// Outputs are numbered densely `1..=total` (`total = latest_consensus_number()`;
/// 0 is the pre-genesis anchor, never stored), so this is [`desc_page_items`]'
/// 0-based indices shifted up by one: `total 5, per_page 2` gives page 0
/// `[5, 4]`, page 1 `[3, 2]`, page 2 `[1]`, page 3 `[]`.
pub fn consensus_page_numbers(total: u64, page: u64, per_page: u64) -> Vec<u64> {
    desc_page_items(total, page, per_page)
        .into_iter()
        .map(|index| index + 1)
        .collect()
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
    fn consensus_page_numbers_are_one_based_newest_first() {
        assert_eq!(consensus_page_numbers(5, 0, 2), vec![5, 4]);
        assert_eq!(consensus_page_numbers(5, 1, 2), vec![3, 2]);
        assert_eq!(consensus_page_numbers(5, 2, 2), vec![1]);
        assert_eq!(consensus_page_numbers(5, 3, 2), Vec::<u64>::new());
        // no outputs yet: nothing to page
        assert_eq!(consensus_page_numbers(0, 0, 25), Vec::<u64>::new());
        // number 0 (the pre-genesis anchor) is never yielded
        assert_eq!(consensus_page_numbers(1, 0, 25), vec![1]);
        // the first page always starts at `total`
        assert_eq!(consensus_page_numbers(100, 0, 25)[0], 100);
        assert_eq!(consensus_page_numbers(100, 0, 25).len(), 25);
    }

    #[test]
    fn consensus_number_bounds_exclude_anchor_and_future() {
        assert!(!consensus_number_stored(0, 10));
        assert!(consensus_number_stored(1, 10));
        assert!(consensus_number_stored(10, 10));
        assert!(!consensus_number_stored(11, 10));
        // nothing processed yet: nothing stored
        assert!(!consensus_number_stored(1, 0));
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
