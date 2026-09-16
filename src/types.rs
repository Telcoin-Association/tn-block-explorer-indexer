//! Wire types and pure builders for the HTTP API.
//!
//! [`ApiTransaction`], [`ApiBlock`], [`ApiTokenTransfer`] and [`ApiEpoch`] are
//! field-for-field deserializable into the explorer's existing structs
//! (`tn-block-explorer/src/services/rpc.rs`) — asserted against verbatim
//! copies in this module's tests. Every field added since the explorer was
//! written is ADDITIVE: the explorer's structs do not use
//! `deny_unknown_fields`, so unknown keys are ignored.
//!
//! # Encoding
//!
//! - 32-byte digests (execution hashes, consensus/epoch/header digests, batch
//!   digests) are lowercase `0x` hex ([`hex_digest`]).
//! - Consensus-header and epoch digests additionally carry a `*_bs58`
//!   companion: the FULL base58 rendering of the inner
//!   `tn_types::Digest<32>` ([`bs58_digest`]). The first 16 characters equal
//!   the truncated `Display` form the node writes in its logs, so operators
//!   can grep either way.
//! - BLS public keys / signatures and `AuthorityIdentifier`s are full base58
//!   via their `Display` impls (matches the existing `committee_bls`).
//!
//! TN types are never passed through serde directly: their JSON forms emit
//! bs58 digests, `Vec<u8>` integer arrays and opaque bitmaps.
//!
//! # The u128 wire hazard
//!
//! `ApiTransaction::value` and `ApiTokenTransfer::value` are `u128` JSON
//! NUMBERS. `serde_json` round-trips `u128` natively when deserializing
//! **directly from text**, but routing the same JSON through a
//! `serde_json::Value` intermediate (as the explorer's legacy `rpc_call` does)
//! destroys integers above `u64::MAX` — that is ≈18.4 TEL in wei. Token
//! transfers additionally carry `value_exact`, the exact decimal string.
//! Clients MUST parse responses directly (`serde_json::from_str::<T>`), never
//! via `Value`. Both modes are unit-tested below.

use crate::extract::{decode_consensus_fields, tx_type_name, HeaderConsensusFields};
use crate::node_reads::TxData;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use tn_types::{
    hex, keccak256, Address, AuthorityIdentifier, Batch, BlockHeader as _, BlsPublicKey,
    ConsensusHeader, ConsensusOutput, Digest, EpochCertificate, EpochRecord, Header,
    ReputationScores, SealedBlock, TransactionTrait as _, B256, DIGEST_LENGTH, U256,
};
use tracing::warn;

/// The inner 32-byte digest every TN digest newtype wraps.
type Digest32 = Digest<{ DIGEST_LENGTH }>;

/// Default page size for list endpoints.
pub const DEFAULT_PER_PAGE: u64 = 25;
/// Maximum page size for list endpoints (requests above this clamp down).
pub const MAX_PER_PAGE: u64 = 100;

/// The list envelope every paged endpoint returns; `page` is 0-based and items
/// are newest-first, with one exception: `/txs/{hash}/transfers` is in
/// ascending `log_index` (emission) order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope<T> {
    /// The page of items, newest first (ascending `log_index` on
    /// `/txs/{hash}/transfers`).
    pub items: Vec<T>,
    /// Total number of items across all pages.
    pub total: u64,
    /// The 0-based page these items belong to.
    pub page: u64,
    /// The (clamped) page size used.
    pub per_page: u64,
}

/// One mined transaction — explorer `Transaction`-compatible.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiTransaction {
    /// Transaction hash (lowercase 0x hex).
    pub hash: String,
    /// Recovered sender (lowercase 0x hex).
    pub from: String,
    /// Recipient; `null` = contract creation.
    pub to: Option<String>,
    /// Value in wei as a JSON NUMBER — see the module-level u128 wire hazard.
    /// Values above `u128::MAX` (impossible for TEL supply) would clamp.
    pub value: u128,
    /// `value as f64 / 1e18` — the explorer's own display formula.
    pub value_tel: f64,
    /// Gas limit.
    pub gas: u64,
    /// Effective gas price (saturated to u64).
    pub gas_price: u64,
    /// Gas used by this transaction alone.
    pub gas_used: u64,
    /// Receipt status; the indexer always sends `Some(_)` (mined-only).
    pub status: Option<bool>,
    /// Input policy: lists serve `"0x"` or the 4-byte selector; the detail
    /// route serves the full calldata hex.
    pub input: String,
    /// Always `null` from the indexer (decoding is client-side).
    pub decoded_input: Option<serde_json::Value>,
    /// Containing block; always `Some(_)` (mined-only).
    pub block_number: Option<u64>,
    /// Index within the block; always `Some(_)`.
    pub transaction_index: Option<u64>,
    /// Sender nonce.
    pub nonce: u64,
    /// EIP-2718 transaction type byte (`Typed2718::ty`): 0 legacy, 1 EIP-2930,
    /// 2 EIP-1559, 3 EIP-4844, 4 EIP-7702. Filterable via `?type=`.
    pub tx_type: u8,
    /// Human name for `tx_type` (`"legacy"`, `"eip2930"`, `"eip1559"`,
    /// `"eip4844"`, `"eip7702"`, or `"unknown"`).
    pub tx_type_name: String,
    /// Detail route only (`/txs/{hash}`): the first [`MAX_PER_PAGE`] ERC-20
    /// transfers this transaction emitted, ordered by `log_index`. Absent on
    /// list rows; the paginated feed is `/txs/{hash}/transfers`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_transfers: Option<Vec<ApiTokenTransfer>>,
    /// Detail route only: total ERC-20 transfers emitted by this transaction
    /// (may exceed `token_transfers.len()`). Absent on list rows.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_transfer_count: Option<u64>,
}

/// One block — explorer `Block`-compatible.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiBlock {
    /// Block number.
    pub number: u64,
    /// Block hash (lowercase 0x hex).
    pub hash: String,
    /// Parent block hash.
    pub parent_hash: String,
    /// Block timestamp (unix seconds).
    pub timestamp: u64,
    /// Transaction hashes in the block.
    pub transactions: Vec<String>,
    /// Number of transactions.
    pub transaction_count: usize,
    /// Total gas used.
    pub gas_used: u64,
    /// Block gas limit.
    pub gas_limit: u64,
    /// Block beneficiary.
    pub miner: String,
    /// Same as `miner` (the explorer displays both).
    pub validator: String,
    /// Header extra data (hex).
    pub extra_data: String,
    /// Base fee per gas, if set.
    pub base_fee: Option<u64>,
    /// RLP-encoded block size in bytes (matches RPC's `size`).
    pub size: u64,
    /// Consensus provenance decoded purely from the execution header (list
    /// AND detail). `null` for genesis, which no consensus output produced.
    /// `consensus_number` inside is resolved on `/blocks/{number}` only.
    pub consensus: Option<ApiBlockConsensus>,
}

/// One ERC-20 transfer row — explorer `ApiTokenTransfer`-compatible.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiTokenTransfer {
    /// Hash of the transaction that emitted the transfer.
    pub tx_hash: String,
    /// Token sender (lowercase 0x hex).
    pub from: String,
    /// Token recipient (lowercase 0x hex).
    pub to: String,
    /// Transferred amount as a u128 JSON number, SATURATING above `u128::MAX`
    /// — see the module-level wire hazard; `value_exact` is lossless.
    pub value: u128,
    /// The exact U256 decimal string.
    pub value_exact: String,
    /// Human amount: `value / 10^decimals` from the cached token decimals
    /// (0-decimals fallback when unknown).
    pub amount: f64,
    /// Containing block.
    pub block_number: u64,
    /// Containing block's timestamp.
    pub timestamp: u64,
    /// The emitting token contract (lowercase 0x hex).
    pub token_address: String,
    /// Cached token symbol; `null` for non-ok tokens.
    pub token_symbol: Option<String>,
    /// Position of the `Transfer` log within the transaction's receipt
    /// (`StoredTransfer::log_index`); orders transfers inside one tx.
    pub log_index: u64,
}

/// Token metadata — explorer `TokenInfo`-compatible shape plus
/// `metadata_status` (the cache status machine value).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiTokenInfo {
    /// Token contract (lowercase 0x hex).
    pub address: String,
    /// Cached or live-fetched `name()`.
    pub name: Option<String>,
    /// Cached or live-fetched `symbol()`.
    pub symbol: Option<String>,
    /// Cached or live-fetched `decimals()`.
    pub decimals: Option<u8>,
    /// LIVE `totalSupply()` as a decimal string (never cached — mutable).
    pub total_supply: Option<String>,
    /// 0 = ok, 1 = retry pending, 2 = failed (terminal).
    pub metadata_status: u8,
}

/// Network stats — explorer `NetworkStats`-compatible plus `total_txs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiStats {
    /// Canonical tip block number.
    pub latest_block: u64,
    /// Next-block base fee in gwei.
    pub gas_price_gwei: f64,
    /// Chain id.
    pub chain_id: u64,
    /// Current consensus epoch.
    pub epoch_number: Option<u64>,
    /// Current committee size.
    pub validator_count: usize,
    /// Total transactions ever mined.
    pub total_txs: u64,
}

/// Address summary for `/address/{addr}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiAddress {
    /// The address (lowercase 0x hex).
    pub address: String,
    /// Balance in wei as an exact decimal string.
    pub balance_wei: String,
    /// Balance in TEL (`wei / 1e18`, lossy display value).
    pub balance_tel: f64,
    /// Account nonce.
    pub nonce: u64,
    /// Whether deployed bytecode exists at the address.
    pub is_contract: bool,
    /// Full deployed bytecode hex (the explorer sniffs interface selectors
    /// client-side); `null` for EOAs.
    pub code: Option<String>,
    /// Number of indexed transaction pointers for this address.
    pub indexed_tx_count: u64,
}

/// One epoch row for `/epochs` and `/epochs/{n}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiEpoch {
    /// Epoch number.
    pub epoch: u64,
    /// First execution block of the epoch (`record(N-1).final_state + 1`;
    /// epoch 0 starts at 0).
    pub start_block: u64,
    /// Last execution block; `null` for the current epoch.
    pub end_block: Option<u64>,
    /// Timestamp of the last block; `null` for the current epoch.
    pub end_time: Option<u64>,
    /// Committee size.
    pub committee_size: usize,
    /// Committee BLS public keys (always available from the epoch record).
    pub committee_bls: Vec<String>,
    /// Whether an epoch certificate is stored.
    pub certified: bool,
    /// Whether this is the in-progress epoch (synthesized; no record yet).
    pub is_current: bool,
    /// Committee ADDRESSES, read from the registry pinned to the block that
    /// seated this epoch's committee — available for any past epoch, not just
    /// recent ones. `null` when the pin cannot be resolved locally (missing
    /// predecessor epoch record or missing pin block; BLS keys above are the
    /// always-available identity). Populated on the detail route only.
    pub committee_addresses: Option<Vec<String>>,
    /// The full stored `EpochRecord`; `null` for the current epoch (its record
    /// is only written when the epoch closes).
    pub record: Option<ApiEpochRecord>,
    /// The stored `EpochCertificate` over `record`; `null` while uncertified
    /// (epoch N's certificate is aggregated at the start of epoch N+1).
    /// `verified` inside is populated on `/epochs/{n}` only.
    pub certificate: Option<ApiEpochCertificate>,
}

/// Current-epoch summary — explorer `EpochData`-compatible.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiEpochData {
    /// Epoch number.
    pub epoch: u64,
    /// Epoch duration in seconds (read from chain).
    pub epoch_duration: u64,
    /// Committee size.
    pub validator_count: usize,
    /// Committee validator addresses.
    pub validators: Vec<String>,
    /// First execution block of the epoch.
    pub start_block: u64,
    /// Canonical tip at response time.
    pub latest_block: u64,
}

/// One validator row — explorer `ValidatorInfo`-compatible.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiValidator {
    /// Validator address (lowercase 0x hex).
    pub address: String,
    /// Epoch the validator became active.
    pub activation_epoch: u32,
    /// Epoch the validator exited (0 = none).
    pub exit_epoch: u32,
    /// Registry `ValidatorStatus` as its u8 discriminant.
    pub status: u8,
    /// Permanently disqualified from consensus.
    pub is_retired: bool,
    /// Stake config version.
    pub stake_version: u8,
    /// GSMA region identifier (0 = unspecified).
    pub region: u8,
}

/// `/validators/leaders` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeadersResponse {
    /// The (clamped) window that was scanned.
    pub window: u64,
    /// First block of the scanned range.
    pub from_block: u64,
    /// Last block of the scanned range (the tip).
    pub to_block: u64,
    /// Per-leader block counts, descending.
    pub leaders: Vec<LeaderEntry>,
}

/// One leader row in [`LeadersResponse`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderEntry {
    /// Leader (beneficiary) address.
    pub address: String,
    /// Blocks produced inside the window.
    pub blocks: u64,
}

/// `/health` response — always 200; degradation is expressed in the body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// `"ok"` while the indexing loop is live, `"degraded"` otherwise.
    pub status: String,
    /// Whether the ExEx indexing loop is running.
    pub indexing_live: bool,
    /// Highest indexed block (0 until the first block is indexed).
    pub last_indexed: u64,
    /// Node canonical tip as last observed by the indexer.
    pub node_tip: u64,
    /// `node_tip - last_indexed`.
    pub lag: u64,
}

/// `POST /call` request body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallRequest {
    /// Contract address (0x hex).
    pub to: String,
    /// ABI-encoded calldata (0x hex, max 128 KiB decoded).
    pub data: String,
}

/// `POST /call` response: 200 for both success and on-chain revert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallResponse {
    /// Whether the call succeeded.
    pub ok: bool,
    /// ABI-encoded return data hex (success only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    /// Human-readable revert reason (revert only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Raw ABI-encoded revert bytes hex (revert only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revert_data: Option<String>,
}

// ---------------------------------------------------------------------------
// Consensus wire structs (`/consensus/...`, `ApiBlock.consensus`,
// `ApiEpoch.record` / `.certificate`)
// ---------------------------------------------------------------------------

/// An inclusive range of execution block numbers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiExecRange {
    /// First execution block number (inclusive).
    pub first: u64,
    /// Last execution block number (inclusive).
    pub last: u64,
}

/// An execution block reference (`BlockNumHash`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiNumHash {
    /// Execution block number.
    pub number: u64,
    /// Execution block hash (0x hex).
    pub hash: String,
}

/// A consensus block reference (`ConsensusNumHash`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConsensusNumHash {
    /// Consensus block number (`ConsensusHeader::number`).
    pub number: u64,
    /// `ConsensusHeaderDigest` as 0x hex.
    pub hash: String,
    /// The same digest as full base58 (first 16 chars = node-log form).
    pub hash_bs58: String,
}

/// A possibly still-open inclusive range (`last` is `null` while the epoch is
/// in progress).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiRange {
    /// First number (inclusive): epoch 0 starts at 0 (execution) / 1
    /// (consensus); epoch N > 0 starts one past epoch N-1's record. `null`
    /// when that predecessor record could not be read — `EpochRecordDb` is
    /// dense, so on a healthy node this never happens.
    pub first: Option<u64>,
    /// Last number (inclusive); `null` for the current epoch.
    pub last: Option<u64>,
}

/// One `(batch digest, worker id)` entry of a primary header's payload
/// (`Header::payload()`, an `IndexMap<BlockHash, WorkerId>`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiPayloadEntry {
    /// Batch digest (`BlockHash`) as 0x hex.
    pub batch_digest: String,
    /// The worker that produced the batch.
    pub worker_id: u16,
}

/// One authority's reputation score (`ReputationScores::scores_per_authority`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiAuthorityScore {
    /// `AuthorityIdentifier` as base58.
    pub authority: String,
    /// Accumulated score within the current schedule.
    pub score: u64,
}

/// One authority's last committed round (`ConsensusChain::read_last_committed`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiAuthorityRound {
    /// `AuthorityIdentifier` as base58.
    pub authority: String,
    /// The last round this authority had a certificate committed in.
    pub round: u32,
}

/// `ReputationScores` in `authorities_by_score_desc()` order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiReputationScores {
    /// Scores, highest first (ties broken by authority identifier, descending).
    pub scores: Vec<ApiAuthorityScore>,
    /// Whether these are the final scores of the current leader schedule.
    pub final_of_schedule: bool,
}

/// A `ConsensusHeader` (one committed sub-dag) without its embedded headers.
/// The list route (`/consensus/blocks`) and the detail route both carry it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConsensusHeader {
    /// Consensus block number (`ConsensusHeader::number`; the first stored
    /// output is 1, 0 is the pre-genesis anchor).
    pub number: u64,
    /// `ConsensusHeader::digest()` as 0x hex. Every execution block this output
    /// produced carries it as `parent_beacon_block_root`.
    pub digest: String,
    /// `digest` as full base58 (first 16 chars = node-log form).
    pub digest_bs58: String,
    /// `ConsensusHeader::parent_hash` as 0x hex.
    pub parent_digest: String,
    /// `parent_digest` as full base58.
    pub parent_digest_bs58: String,
    /// Epoch of the committing leader (`CommittedSubDag::leader_epoch`).
    pub epoch: u32,
    /// Round of the committing leader (`CommittedSubDag::leader_round`).
    pub round: u32,
    /// The leader's `AuthorityIdentifier` as base58.
    pub leader: String,
    /// The leader `Header`'s digest as 0x hex.
    pub leader_header_digest: String,
    /// `CommittedSubDag::commit_timestamp()` (unix seconds) — equals the
    /// `timestamp` of every execution block this output produced.
    pub committed_at: u64,
    /// Number of primary headers in the sub-dag including the leader
    /// (`CommittedSubDag::len`).
    pub sub_dag_header_count: usize,
    /// Number of batches referenced by the sub-dag's headers
    /// (`CommittedSubDag::num_primary_batches`).
    pub batch_count: usize,
    /// `CommittedSubDag::randomness()` — the committee-shuffle seed chain value
    /// as of this commit (0x hex).
    pub randomness: String,
    /// `ConsensusHeader::extra` (0x hex; currently unused by the protocol).
    pub extra: String,
    /// Execution blocks this output produced, from the indexer's
    /// `consensus_blocks` table. `null` = not indexed yet OR an empty
    /// non-closing output (0 blocks) — disambiguate via `/health.last_indexed`.
    pub exec_blocks: Option<ApiExecRange>,
}

/// One primary `Header` inside a committed sub-dag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiSubDagHeader {
    /// `Header::digest()` as 0x hex.
    pub digest: String,
    /// `Header::author()` (`AuthorityIdentifier`) as base58.
    pub author: String,
    /// The author's BLS public key (base58) when it resolves against the
    /// epoch committee; `null` on routes that do not load the committee.
    pub author_bls: Option<String>,
    /// `Header::round()`.
    pub round: u32,
    /// `Header::epoch()`.
    pub epoch: u32,
    /// `Header::created_at()` (unix seconds).
    pub created_at: u64,
    /// `Header::parents()` — parent header digests (0x hex) in `BTreeSet`
    /// order.
    pub parents: Vec<String>,
    /// `Header::payload()` entries in payload (insertion) order.
    pub payload: Vec<ApiPayloadEntry>,
    /// `Header::latest_execution_block()` — the author's execution tip when it
    /// built the header.
    pub latest_execution_block: ApiNumHash,
    /// Whether this header is the sub-dag's committing leader.
    pub is_leader: bool,
}

/// One batch of a `ConsensusOutput`, in `flatten_batches()` order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConsensusBatch {
    /// Position in `ConsensusOutput::flatten_batches()` order — which is also
    /// the batch's position among the execution blocks this output produced.
    pub index: u64,
    /// The batch digest (`BlockHash`) as 0x hex, taken from the output's
    /// `batch_digests()` deque at this position — the pairing the engine
    /// executes with, so it always equals the execution block's
    /// `ommers_hash`. It equals `Batch::digest()` of this row's batch EXCEPT
    /// on adiri outputs in epochs <= `ADIRI_DUP_BATCH_EPOCH` with duplicate
    /// payload keys, where the engine deliberately mispairs the same way.
    pub digest: String,
    /// `Batch::worker_id`.
    pub worker_id: u16,
    /// `Batch::beneficiary` — the PRODUCING worker's configured execution
    /// address (from that node's node-info), 0x hex. Not necessarily the
    /// execution block's coinbase, which is `authority_address`.
    pub beneficiary: String,
    /// `CertifiedBatch::address` — the authority whose header certified the
    /// batch (0x hex; may repeat within an output). This is the execution
    /// block's coinbase, `/blocks/{n}.miner`.
    pub authority_address: String,
    /// `Batch::base_fee_per_gas`.
    pub base_fee_per_gas: u64,
    /// `Batch::transactions.len()`.
    pub tx_count: usize,
    /// `Batch::size()` — struct size plus raw transaction bytes.
    pub size_bytes: usize,
    /// `exec_blocks.first + index` when the output's execution range is
    /// known AND already covers this batch (`<= exec_blocks.last`); `null`
    /// otherwise. The indexer commits one block per SQLite transaction, so
    /// mid-output the range can be shorter than the batch list and the
    /// trailing batches are `null` like `exec_blocks` itself.
    pub exec_block_number: Option<u64>,
    /// `keccak256` of each raw EIP-2718 transaction (== its tx hash).
    /// Present on `/consensus/blocks/{n}/batches` only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tx_hashes: Option<Vec<String>>,
}

/// `GET /consensus/blocks/{number}`: a full `ConsensusOutput`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConsensusBlock {
    /// The consensus header.
    pub header: ApiConsensusHeader,
    /// The leader's BLS public key (base58) when it resolves against the epoch
    /// committee (`get_committee_keys(epoch)`); `null` otherwise.
    pub leader_bls: Option<String>,
    /// Every primary header in the sub-dag (`CommittedSubDag::headers()`,
    /// leader last).
    pub sub_dag: Vec<ApiSubDagHeader>,
    /// `CommittedSubDag::reputation_scores()`.
    pub reputation_scores: ApiReputationScores,
    /// Batch summaries in `flatten_batches()` order (no tx hashes — see
    /// `/consensus/blocks/{n}/batches`).
    pub batches: Vec<ApiConsensusBatch>,
    /// `record(epoch).final_consensus.number == number`; `null` when the
    /// epoch record does not exist yet. (`ConsensusOutput::close_epoch` is NOT
    /// serialized — a deserialized value is never trusted.)
    pub closes_epoch: Option<bool>,
}

/// `GET /consensus/blocks/{number}/batches`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConsensusBatches {
    /// Consensus block number.
    pub number: u64,
    /// `ConsensusHeader::digest()` as 0x hex.
    pub digest: String,
    /// Batches in `flatten_batches()` order, each with `tx_hashes`.
    pub batches: Vec<ApiConsensusBatch>,
}

/// `GET /consensus/latest`: the newest consensus header plus the execution tip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConsensusLatest {
    /// `ConsensusChain::latest_consensus_number()` — the header is read by
    /// that number (not the pack's tail, which can be one output ahead while
    /// a write is in flight), so it agrees with `/consensus/blocks.total` and
    /// the bounds of `/consensus/blocks/{n}`.
    pub number: u64,
    /// Leader epoch of the latest header.
    pub epoch: u32,
    /// Leader round of the latest header.
    pub round: u32,
    /// Latest header digest as 0x hex.
    pub digest: String,
    /// Latest header digest as full base58.
    pub digest_bs58: String,
    /// Leader `AuthorityIdentifier` as base58.
    pub leader: String,
    /// `CommittedSubDag::commit_timestamp()`.
    pub committed_at: u64,
    /// Canonical execution tip block number.
    pub exec_tip: u64,
    /// Consensus fields decoded from the execution tip header (`null` when
    /// the tip is genesis).
    pub exec_tip_consensus: Option<ApiBlockConsensus>,
}

/// A stored `EpochRecord` (`EpochRecordDb::record_by_epoch`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiEpochRecord {
    /// `EpochRecord::epoch`.
    pub epoch: u32,
    /// `EpochRecord::digest()` as 0x hex.
    pub digest: String,
    /// `digest` as full base58 (first 16 chars = node-log form).
    pub digest_bs58: String,
    /// `EpochRecord::parent_hash` (previous record's digest) as 0x hex.
    pub parent_hash: String,
    /// `parent_hash` as full base58.
    pub parent_hash_bs58: String,
    /// `EpochRecord::committee` — this epoch's BLS public keys (base58), in
    /// record order (the certificate bitmap indexes this list).
    pub committee: Vec<String>,
    /// `EpochRecord::next_committee` — the following epoch's BLS keys.
    pub next_committee: Vec<String>,
    /// `EpochRecord::final_state` — the epoch's last execution block.
    pub final_state: ApiNumHash,
    /// `EpochRecord::final_consensus` — the epoch's last consensus header.
    pub final_consensus: ApiConsensusNumHash,
    /// `EpochRecord::super_quorum()` — `2/3 * committee.len() + 1`.
    pub super_quorum: usize,
}

/// A stored `EpochCertificate` over an [`ApiEpochRecord`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiEpochCertificate {
    /// `EpochCertificate::epoch_hash` (the record digest signed) as 0x hex.
    pub epoch_hash: String,
    /// `epoch_hash` as full base58.
    pub epoch_hash_bs58: String,
    /// `EpochCertificate::signature` — aggregate BLS signature (base58).
    pub signature: String,
    /// `EpochCertificate::signed_authorities` bitmap expanded to ascending
    /// indexes into `record.committee`.
    pub signer_indices: Vec<u32>,
    /// The BLS public keys (base58) at `signer_indices` that resolve against
    /// `record.committee`, in index order.
    pub signers: Vec<String>,
    /// Number of `signer_indices` with no `record.committee` entry (always 0
    /// for a well-formed certificate).
    pub unresolved_signers: u32,
    /// `signed_authorities.len()` — total set bits.
    pub signer_count: u64,
    /// `EpochRecord::super_quorum()` — signers needed for validity.
    pub super_quorum: usize,
    /// `EpochRecord::verify_with_cert` (BLS pairing check). Populated on
    /// detail routes only; `null` on lists.
    pub verified: Option<bool>,
}

/// `GET /consensus/epochs` row / `GET /consensus/epochs/{n}` detail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConsensusEpoch {
    /// Epoch number.
    pub epoch: u64,
    /// Whether this is the in-progress epoch. Its record, if the node has
    /// already written it, is hidden until the epoch stops being current.
    pub is_current: bool,
    /// The epoch record; `null` for the current epoch.
    pub record: Option<ApiEpochRecord>,
    /// The certificate over `record`; `null` while uncertified.
    pub certificate: Option<ApiEpochCertificate>,
    /// Consensus block numbers: `{prev.final_consensus.number + 1 (1 for epoch
    /// 0), record.final_consensus.number}`; `last` null while current,
    /// `first` null only if the predecessor record is unreadable.
    pub consensus_range: ApiRange,
    /// Execution block numbers: `{prev.final_state.number + 1 (0 for epoch
    /// 0), record.final_state.number}`; `last` null while current, `first`
    /// null only if the predecessor record is unreadable.
    pub exec_range: ApiRange,
    /// Timestamp of the epoch's last execution block; `null` while current.
    pub end_time: Option<u64>,
    /// Committee BLS public keys (base58).
    pub committee_bls: Vec<String>,
    /// Detail only: committee execution addresses from the registry pin;
    /// `null` on lists or when the pin cannot be resolved.
    pub committee_addresses: Option<Vec<String>>,
    /// Detail only: `ConsensusChain::is_epoch_complete(record)` — whether the
    /// epoch's final output is readable from its local pack. `false` when the
    /// pack is absent, truncated or corrupt as well as when it is genuinely
    /// incomplete (TN collapses those); `null` only for the current epoch
    /// (no record yet) and on lists.
    pub pack_complete: Option<bool>,
    /// Detail only: `ConsensusChain::read_last_committed(epoch)`, sorted;
    /// `null` on lists or when the pack is not local.
    pub last_committed_rounds: Option<Vec<ApiAuthorityRound>>,
    /// Detail only: the epoch's final `ReputationScores`
    /// (`read_latest_commit_with_final_reputation_scores`); `null` on lists
    /// or when unavailable.
    pub final_reputation_scores: Option<ApiReputationScores>,
}

/// Consensus provenance of one execution block, decoded purely from its header
/// (TN `evm/config.rs`, `payload.rs`): no consensus DB read.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiBlockConsensus {
    /// `parent_beacon_block_root` = the producing `ConsensusHeader`'s digest
    /// (0x hex).
    pub digest: String,
    /// `digest` as full base58 (first 16 chars = node-log form).
    pub digest_bs58: String,
    /// High 32 bits of `nonce` (`deconstruct_nonce`).
    pub epoch: u32,
    /// Low 32 bits of `nonce` (`deconstruct_nonce`).
    pub round: u32,
    /// `difficulty >> 16` — the batch's `flatten_batches()` index; the single
    /// empty block of an empty epoch-closing output carries the placeholder 0.
    pub batch_index: u64,
    /// `difficulty & 0xffff` — the producing worker; placeholder 0 on that
    /// empty epoch-closing block.
    pub worker_id: u16,
    /// `ommers_hash` — the digest the engine paired with this block from the
    /// output's `batch_digests()` deque (see `ApiConsensusBatch.digest`), 0x
    /// hex; ZERO for the empty block an epoch-closing output with no batches
    /// produces.
    pub batch_digest: String,
    /// `mix_hash` = prev_randao (0x hex), when set.
    pub prev_randao: Option<String>,
    /// Whether this block closed its epoch (non-empty `extra_data` = the
    /// committee shuffle seed).
    pub closes_epoch: bool,
    /// The consensus block number that produced this block
    /// (`consensus_header_by_digest(epoch, digest).number`). Resolved on
    /// `/blocks/{number}` only; `null` on lists.
    pub consensus_number: Option<u64>,
}

/// Which input representation a transaction response carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    /// List endpoints: `"0x"` or the 4-byte selector hex.
    List,
    /// `/txs/{hash}`: the full calldata hex (deployments run to ~24KB).
    Detail,
}

// ---------------------------------------------------------------------------
// Encoding helpers
// ---------------------------------------------------------------------------

/// Lowercase `0x` hex for a 32-byte hash.
pub fn hex_b256(value: &B256) -> String {
    format!("{value:#x}")
}

/// Lowercase `0x` hex for an address.
pub fn hex_address(value: &Address) -> String {
    format!("{value:#x}")
}

/// Lowercase `0x` hex for arbitrary bytes.
pub fn hex_bytes(value: &[u8]) -> String {
    format!("0x{}", hex::encode(value))
}

/// Lowercase `0x` hex for any TN digest (`ConsensusHeaderDigest`,
/// `EpochDigest`, `HeaderDigest`, `Digest<32>`, `B256`, ...): all of them are
/// `AsRef<[u8]>`.
pub fn hex_digest<D: AsRef<[u8]>>(d: &D) -> String {
    hex_bytes(d.as_ref())
}

/// FULL base58 of a 32-byte digest via the inner `Digest`'s `Display`.
///
/// The digest newtypes (`ConsensusHeaderDigest`, `EpochDigest`,
/// `HeaderDigest`) truncate their own `Display` to 16 characters — the
/// node-log form — so callers convert to the inner digest first
/// (`Digest::from(newtype)`; every newtype implements `From<Self> for
/// Digest<32>`). The 16-char form is a prefix of this string.
pub fn bs58_digest(d: &Digest32) -> String {
    d.to_string()
}

/// [`bs58_digest`] for any digest newtype convertible into the inner digest.
fn bs58_of<D: Into<Digest32>>(d: D) -> String {
    bs58_digest(&d.into())
}

/// [`bs58_digest`] for a `B256` that carries a consensus digest
/// (`parent_beacon_block_root`).
fn bs58_b256(b: &B256) -> String {
    bs58_digest(&Digest::new(b.0))
}

/// The 4-byte selector of calldata as `0x`-prefixed hex, or `None` when the
/// input is shorter than 4 bytes.
pub fn selector_hex(input: &[u8]) -> Option<String> {
    (input.len() >= 4).then(|| hex_bytes(&input[..4]))
}

/// Clamp a requested page size into `1..=MAX_PER_PAGE` (default
/// [`DEFAULT_PER_PAGE`]).
pub fn clamp_per_page(requested: Option<u64>) -> u64 {
    requested.unwrap_or(DEFAULT_PER_PAGE).clamp(1, MAX_PER_PAGE)
}

/// Lossy TEL display value for a wei amount (`wei / 1e18`, the explorer's own
/// formula).
pub fn wei_to_tel(wei: &U256) -> f64 {
    wei.to_string().parse::<f64>().unwrap_or(0.0) / 1e18
}

/// Human token amount from the exact decimal string and the token's decimals.
pub fn token_amount(value_decimal: &str, decimals: u8) -> f64 {
    value_decimal.parse::<f64>().unwrap_or(0.0) / 10f64.powi(i32::from(decimals))
}

// ---------------------------------------------------------------------------
// Builders: execution side
// ---------------------------------------------------------------------------

/// Map hydrated transaction data to the wire shape (shared by `/txs`,
/// `/txs/{hash}`, and `/address/{addr}/txs`). `token_transfers` /
/// `token_transfer_count` start `None`; the detail handler fills them.
pub fn build_api_transaction(data: &TxData, mode: InputMode) -> ApiTransaction {
    let input_bytes: &[u8] = data.tx.input().as_ref();
    let input = match mode {
        InputMode::Detail => hex_bytes(input_bytes),
        InputMode::List => selector_hex(input_bytes).unwrap_or_else(|| "0x".to_string()),
    };
    let value = u128::try_from(data.tx.value()).unwrap_or_else(|_| {
        // mainnet TEL supply fits u128 comfortably; clamp and flag if it ever happens
        warn!(target: "indexer::api", hash = %data.tx.hash(), "tx value exceeds u128; clamping");
        u128::MAX
    });
    let tx_type = tn_types::Typed2718::ty(&data.tx);
    ApiTransaction {
        hash: hex_b256(data.tx.hash()),
        from: hex_address(&data.sender),
        to: data.tx.to().map(|to| hex_address(&to)),
        value,
        value_tel: value as f64 / 1e18,
        gas: data.tx.gas_limit(),
        gas_price: u64::try_from(data.tx.effective_gas_price(data.base_fee)).unwrap_or(u64::MAX),
        gas_used: data.gas_used,
        status: Some(data.success),
        input,
        decoded_input: None,
        block_number: Some(data.block_number),
        transaction_index: Some(data.tx_index),
        nonce: data.tx.nonce(),
        tx_type,
        tx_type_name: tx_type_name(tx_type).to_string(),
        token_transfers: None,
        token_transfer_count: None,
    }
}

/// Map a sealed block to the wire shape (shared by `/blocks` and
/// `/blocks/{number}`). `consensus` is decoded purely from the header with
/// `consensus_number: None`; the detail handler resolves the number.
pub fn build_api_block(block: &SealedBlock) -> ApiBlock {
    let header = block.header();
    let transactions: Vec<String> = block
        .body()
        .transactions()
        .map(|tx| hex_b256(tx.hash()))
        .collect();
    ApiBlock {
        number: header.number(),
        hash: hex_b256(&block.hash()),
        parent_hash: hex_b256(&header.parent_hash()),
        timestamp: header.timestamp(),
        transaction_count: transactions.len(),
        transactions,
        gas_used: header.gas_used(),
        gas_limit: header.gas_limit(),
        miner: hex_address(&header.beneficiary()),
        validator: hex_address(&header.beneficiary()),
        extra_data: hex_bytes(header.extra_data()),
        base_fee: header.base_fee_per_gas(),
        size: block.rlp_length() as u64,
        consensus: decode_consensus_fields(header).map(|f| build_block_consensus(&f, None)),
    }
}

/// Map decoded execution-header consensus fields to the wire shape.
pub fn build_block_consensus(
    f: &HeaderConsensusFields,
    consensus_number: Option<u64>,
) -> ApiBlockConsensus {
    ApiBlockConsensus {
        digest: hex_b256(&f.digest),
        digest_bs58: bs58_b256(&f.digest),
        epoch: f.epoch,
        round: f.round,
        batch_index: f.batch_index,
        worker_id: f.worker_id,
        batch_digest: hex_b256(&f.batch_digest),
        prev_randao: f.prev_randao.as_ref().map(hex_b256),
        closes_epoch: f.closes_epoch,
        consensus_number,
    }
}

/// Map a stored transfer row + its hydrated transaction + the cached token
/// metadata to the wire shape (shared by all transfer feeds).
pub fn build_token_transfer(
    row: &crate::storage::StoredTransfer,
    hydrated: &TxData,
    token: Option<&crate::storage::StoredToken>,
) -> ApiTokenTransfer {
    // saturating u128 JSON number; `value_exact` carries the lossless string
    let value = row.value.parse::<u128>().unwrap_or(u128::MAX);
    let decimals = token.and_then(|t| t.decimals).unwrap_or(0); // 0-decimals fallback
    ApiTokenTransfer {
        tx_hash: hex_b256(hydrated.tx.hash()),
        from: row.from.clone(),
        to: row.to.clone(),
        value,
        value_exact: row.value.clone(),
        amount: token_amount(&row.value, decimals),
        block_number: row.block_number,
        timestamp: hydrated.timestamp,
        token_address: row.token.clone(),
        token_symbol: token.and_then(|t| t.symbol.clone()),
        log_index: row.log_index,
    }
}

// ---------------------------------------------------------------------------
// Builders: consensus side (pure — no I/O; the api layer supplies every read)
// ---------------------------------------------------------------------------

/// Map a `ConsensusHeader` to the wire shape. `exec_blocks` comes from the
/// indexer's `consensus_blocks` table (`None` when unknown).
pub fn build_consensus_header(
    h: &ConsensusHeader,
    exec_blocks: Option<ApiExecRange>,
) -> ApiConsensusHeader {
    let digest = h.digest();
    let leader = h.sub_dag.leader();
    ApiConsensusHeader {
        number: h.number,
        digest: hex_digest(&digest),
        digest_bs58: bs58_of(digest),
        parent_digest: hex_digest(&h.parent_hash),
        parent_digest_bs58: bs58_of(h.parent_hash),
        epoch: h.sub_dag.leader_epoch(),
        round: h.sub_dag.leader_round(),
        leader: leader.author().to_string(),
        leader_header_digest: hex_digest(&leader.digest()),
        committed_at: h.sub_dag.commit_timestamp(),
        sub_dag_header_count: h.sub_dag.len(),
        batch_count: h.sub_dag.num_primary_batches(),
        randomness: hex_b256(&h.sub_dag.randomness()),
        extra: hex_b256(&h.extra),
        exec_blocks,
    }
}

/// Map one primary `Header` of a sub-dag to the wire shape. `bls` is the
/// author's committee key when the caller resolved it.
pub fn build_sub_dag_header(
    h: &Header,
    is_leader: bool,
    bls: Option<&BlsPublicKey>,
) -> ApiSubDagHeader {
    let exec = h.latest_execution_block();
    ApiSubDagHeader {
        digest: hex_digest(&h.digest()),
        author: h.author().to_string(),
        author_bls: bls.map(ToString::to_string),
        round: h.round(),
        epoch: h.epoch(),
        created_at: *h.created_at(),
        parents: h.parents().iter().map(hex_digest).collect(),
        payload: h
            .payload()
            .iter()
            .map(|(digest, worker_id)| ApiPayloadEntry {
                batch_digest: hex_b256(digest),
                worker_id: *worker_id,
            })
            .collect(),
        latest_execution_block: ApiNumHash {
            number: exec.number,
            hash: hex_b256(&exec.hash),
        },
        is_leader,
    }
}

/// Map `ReputationScores` to the wire shape in `authorities_by_score_desc()`
/// order.
pub fn build_reputation_scores(scores: &ReputationScores) -> ApiReputationScores {
    ApiReputationScores {
        scores: scores
            .authorities_by_score_desc()
            .into_iter()
            .map(|(authority, score)| ApiAuthorityScore {
                authority: authority.to_string(),
                score,
            })
            .collect(),
        final_of_schedule: scores.final_of_schedule,
    }
}

/// `keccak256` of every raw EIP-2718 transaction in a batch — each equals
/// the transaction's hash.
pub fn batch_tx_hashes(batch: &Batch) -> Vec<String> {
    batch
        .transactions
        .iter()
        .map(|raw| hex_b256(&keccak256(raw)))
        .collect()
}

/// Map a `ConsensusOutput`'s batches to the wire shape in
/// `flatten_batches()` order (`(cert_idx, batch_idx)` →
/// `out.batches()[cert_idx].batches[batch_idx]`). `exec_blocks` numbers the
/// execution block each batch became; `with_tx_hashes` adds per-tx hashes.
pub fn build_consensus_batches(
    out: &ConsensusOutput,
    exec_blocks: Option<&ApiExecRange>,
    with_tx_hashes: bool,
) -> Vec<ApiConsensusBatch> {
    let certified = out.batches();
    out.flatten_batches()
        .into_iter()
        .enumerate()
        .map(|(index, (cert_idx, batch_idx))| {
            let cert = &certified[cert_idx];
            let batch = &cert.batches[batch_idx];
            // the output's digest deque, paired by position exactly as the
            // engine does (`get_batch_digest(batch_index)` in TN's payload
            // builder) — on adiri outputs at or below `ADIRI_DUP_BATCH_EPOCH`
            // with duplicate payload keys this differs from `batch.digest()`,
            // and the engine's pairing is what the exec block records. The
            // deque is never shorter than the flatten list; the recompute is
            // defensive only.
            let digest = out
                .get_batch_digest(index)
                .unwrap_or_else(|| batch.digest());
            let index = index as u64;
            ApiConsensusBatch {
                index,
                digest: hex_b256(&digest),
                worker_id: batch.worker_id,
                beneficiary: hex_address(&batch.beneficiary),
                authority_address: hex_address(&cert.address),
                base_fee_per_gas: batch.base_fee_per_gas,
                tx_count: batch.transactions.len(),
                size_bytes: batch.size(),
                // only blocks the indexer has committed: the range grows one
                // block at a time, so mid-output it can be shorter than the
                // batch list
                exec_block_number: exec_blocks.and_then(|r| {
                    let n = r.first + index;
                    (n <= r.last).then_some(n)
                }),
                tx_hashes: with_tx_hashes.then(|| batch_tx_hashes(batch)),
            }
        })
        .collect()
}

/// Assemble the `/consensus/blocks/{number}` body. `committee` (the epoch's
/// `get_committee_keys`) resolves `leader_bls` / `author_bls` by matching
/// `AuthorityIdentifier::from(key)`; `None` leaves them `null`.
pub fn build_consensus_block(
    header: ApiConsensusHeader,
    out: &ConsensusOutput,
    committee: Option<&BTreeSet<BlsPublicKey>>,
    batches: Vec<ApiConsensusBatch>,
    closes_epoch: Option<bool>,
) -> ApiConsensusBlock {
    let by_identifier: BTreeMap<AuthorityIdentifier, &BlsPublicKey> = committee
        .map(|keys| {
            keys.iter()
                .map(|key| (AuthorityIdentifier::from(*key), key))
                .collect()
        })
        .unwrap_or_default();
    let sub_dag = out.sub_dag();
    let headers = sub_dag.headers();
    let leader = sub_dag.leader();
    ApiConsensusBlock {
        header,
        leader_bls: by_identifier
            .get(leader.author())
            .map(|key| key.to_string()),
        sub_dag: headers
            .iter()
            .enumerate()
            .map(|(i, h)| {
                // the leader is always the LAST header of a committed sub-dag
                let is_leader = i + 1 == headers.len();
                build_sub_dag_header(h, is_leader, by_identifier.get(h.author()).copied())
            })
            .collect(),
        reputation_scores: build_reputation_scores(sub_dag.reputation_scores()),
        batches,
        closes_epoch,
    }
}

/// Map an `EpochRecord` to the wire shape.
pub fn build_epoch_record(r: &EpochRecord) -> ApiEpochRecord {
    let digest = r.digest();
    ApiEpochRecord {
        epoch: r.epoch,
        digest: hex_digest(&digest),
        digest_bs58: bs58_of(digest),
        parent_hash: hex_digest(&r.parent_hash),
        parent_hash_bs58: bs58_of(r.parent_hash),
        committee: r.committee.iter().map(ToString::to_string).collect(),
        next_committee: r.next_committee.iter().map(ToString::to_string).collect(),
        final_state: ApiNumHash {
            number: r.final_state.number,
            hash: hex_b256(&r.final_state.hash),
        },
        final_consensus: ApiConsensusNumHash {
            number: r.final_consensus.number,
            hash: hex_digest(&r.final_consensus.hash),
            hash_bs58: bs58_of(r.final_consensus.hash),
        },
        super_quorum: r.super_quorum(),
    }
}

/// Map an `EpochCertificate` to the wire shape, expanding the signer bitmap
/// against `r.committee` (mirrors `EpochRecord::verify_with_cert`). `verified`
/// is the caller's BLS pairing result (`None` on list routes).
pub fn build_epoch_certificate(
    r: &EpochRecord,
    c: &EpochCertificate,
    verified: Option<bool>,
) -> ApiEpochCertificate {
    let signer_indices: Vec<u32> = c.signed_authorities.iter().collect();
    let mut signers = Vec::with_capacity(signer_indices.len());
    let mut unresolved_signers = 0u32;
    for &index in &signer_indices {
        match r.committee.get(index as usize) {
            Some(key) => signers.push(key.to_string()),
            None => unresolved_signers += 1,
        }
    }
    ApiEpochCertificate {
        epoch_hash: hex_digest(&c.epoch_hash),
        epoch_hash_bs58: bs58_of(c.epoch_hash),
        signature: c.signature.to_string(),
        signer_indices,
        signers,
        unresolved_signers,
        signer_count: c.signed_authorities.len(),
        super_quorum: r.super_quorum(),
        verified,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{StoredToken, StoredTransfer};
    use std::collections::VecDeque;
    use tn_types::{
        BlockNumHash, BlsKeypair, BlsSignature, Bytes, CertifiedBatch, CommittedSubDag,
        ConsensusHeaderDigest, ConsensusNumHash, EpochDigest, EthSignature, ExecHeader,
        HeaderBuilder, HeaderDigest, TransactionSigned, TxEip1559, TxKind, U256,
    };

    /// VERBATIM copies of the explorer's structs — the wire-compatibility
    /// gate: every field name and type byte-matches.
    ///
    /// Source: `tn-block-explorer/src/services/rpc.rs` at commit `45379ba`
    /// (v0.2.4): `Block` lines 37-52, `DecodedInput` 53-58, `Transaction`
    /// 59-75, `ApiTokenTransfer` 172-184, `ApiEpoch` 236-247. The explorer
    /// does not use `deny_unknown_fields`, so additive fields are ignored.
    #[allow(dead_code)] // Deserialize-only copies: fields are parsed, never read
    mod explorer {
        use serde::{Deserialize, Serialize};

        #[rustfmt::skip]
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct Block {
            pub number:            u64,
            pub hash:              String,
            pub parent_hash:       String,
            pub timestamp:         u64,
            pub transactions:      Vec<String>,
            pub transaction_count: usize,
            pub gas_used:          u64,
            pub gas_limit:         u64,
            pub miner:             String,
            pub validator:         String,
            pub extra_data:        String,
            pub base_fee:          Option<u64>,
            pub size:              u64,
        }
        #[rustfmt::skip]
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct DecodedInput {
            pub method:    String,
            pub signature: String,
            pub params:    Vec<(String, String)>,
        }
        #[rustfmt::skip]
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct Transaction {
            pub hash:              String,
            pub from:              String,
            pub to:                Option<String>,
            pub value:             u128,
            pub value_tel:         f64,
            pub gas:               u64,
            pub gas_price:         u64,
            pub gas_used:          u64,
            pub status:            Option<bool>,
            pub input:             String,
            pub decoded_input:     Option<DecodedInput>,
            pub block_number:      Option<u64>,
            pub transaction_index: Option<u64>,
            pub nonce:             u64,
        }
        #[rustfmt::skip]
        #[derive(Debug, Clone, Deserialize)]
        pub struct ApiTokenTransfer {
            pub tx_hash:       String,
            pub from:          String,
            pub to:            String,
            pub value:         u128,
            pub value_exact:   String,
            pub amount:        f64,
            pub block_number:  u64,
            pub timestamp:     u64,
            pub token_address: String,
            pub token_symbol:  Option<String>,
        }
        #[rustfmt::skip]
        #[derive(Debug, Clone, Deserialize)]
        pub struct ApiEpoch {
            pub epoch:               u64,
            pub start_block:         u64,
            pub end_block:           Option<u64>,
            pub end_time:            Option<u64>,
            pub committee_size:      usize,
            pub committee_bls:       Vec<String>,
            pub certified:           bool,
            pub is_current:          bool,
            pub committee_addresses: Option<Vec<String>>,
        }
    }

    fn addr(byte: u8) -> Address {
        Address::repeat_byte(byte)
    }

    fn make_signed_tx(input: Vec<u8>, value: U256) -> TransactionSigned {
        let tx = TxEip1559 {
            chain_id: 0x1e7,
            nonce: 3,
            gas_limit: 100_000,
            max_fee_per_gas: 2_000_000_000,
            max_priority_fee_per_gas: 0,
            to: TxKind::Call(addr(0x22)),
            value,
            access_list: Default::default(),
            input: Bytes::from(input),
        };
        TransactionSigned::new_unhashed(
            tn_types::Transaction::Eip1559(tx),
            EthSignature::new(U256::from(1u64), U256::from(1u64), false),
        )
    }

    fn make_tx_data(input: Vec<u8>, value: U256) -> TxData {
        TxData {
            sender: addr(0x11),
            tx: make_signed_tx(input, value),
            success: true,
            gas_used: 21_000,
            block_number: 9,
            tx_index: 0,
            base_fee: Some(1_000_000_000),
            timestamp: 1_700_000_000,
        }
    }

    fn make_transfer_row() -> StoredTransfer {
        StoredTransfer {
            block_number: 9,
            tx_index: 0,
            log_index: 4,
            token: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
            from: "0x1111111111111111111111111111111111111111".into(),
            to: "0x2222222222222222222222222222222222222222".into(),
            value: big_value().to_string(),
        }
    }

    fn make_token() -> StoredToken {
        StoredToken {
            name: Some("Token".into()),
            symbol: Some("TOK".into()),
            decimals: Some(18),
            status: 0,
        }
    }

    /// Deterministic BLS keys (no `rand` dependency): the 32-byte scalars
    /// `[1;32]`, `[2;32]`, ... are all below the BLS12-381 group order.
    fn bls_keys(n: u8) -> Vec<BlsPublicKey> {
        (1..=n)
            .map(|i| *BlsKeypair::from_bytes(&[i; 32]).expect("valid sk").public())
            .collect()
    }

    fn make_batch(txs: Vec<Vec<u8>>, beneficiary: u8, worker_id: u16) -> Batch {
        Batch {
            transactions: txs,
            epoch: 3,
            beneficiary: addr(beneficiary),
            base_fee_per_gas: 7,
            worker_id,
            ..Default::default()
        }
    }

    fn make_header(
        author: AuthorityIdentifier,
        round: u32,
        created_at: u64,
        batches: &[&Batch],
    ) -> Header {
        let mut builder = HeaderBuilder::default()
            .author(author)
            .round(round)
            .epoch(3)
            .created_at(created_at)
            .latest_execution_block(BlockNumHash::new(77, B256::repeat_byte(0x77)));
        for batch in batches {
            builder = builder.with_payload_batch(batch, batch.worker_id);
        }
        builder.build()
    }

    /// A 2-header sub-dag (follower + leader) with three batches across two
    /// certified authorities. Returns `(output, raw txs, leader author)`.
    fn make_output(
        leader_author: AuthorityIdentifier,
    ) -> (ConsensusOutput, Vec<Vec<u8>>, Vec<Batch>) {
        let raw_a = tn_types::Encodable2718::encoded_2718(&make_signed_tx(
            vec![1, 2, 3, 4],
            U256::from(5u64),
        ));
        let raw_b =
            tn_types::Encodable2718::encoded_2718(&make_signed_tx(vec![], U256::from(6u64)));
        let batch_a = make_batch(vec![raw_a.clone(), raw_b.clone()], 0x31, 1);
        let batch_b = make_batch(vec![], 0x32, 0);
        let batch_c = make_batch(vec![raw_b.clone()], 0x33, 2);
        let certified = vec![
            CertifiedBatch {
                address: addr(0x41),
                batches: vec![batch_a.clone(), batch_b.clone()],
            },
            CertifiedBatch {
                address: addr(0x42),
                batches: vec![batch_c.clone()],
            },
        ];
        let digests: VecDeque<B256> = certified
            .iter()
            .flat_map(|c| c.batches.iter().map(Batch::digest))
            .collect();
        let follower = make_header(
            AuthorityIdentifier::dummy_for_test(0x0f),
            3,
            1_700_000_100,
            &[&batch_c],
        );
        let leader = make_header(leader_author, 4, 1_700_000_123, &[&batch_a, &batch_b]);
        let sub_dag = CommittedSubDag::new_with_headers_for_test(vec![follower, leader]);
        let out = ConsensusOutput::new(
            sub_dag,
            ConsensusHeaderDigest::from(B256::repeat_byte(0xcc)),
            12,
            false,
            digests,
            certified,
        );
        (out, vec![raw_a, raw_b], vec![batch_a, batch_b, batch_c])
    }

    /// A wei value strictly above u64::MAX (the corruption threshold).
    fn big_value() -> u128 {
        u128::from(u64::MAX) + 12_345
    }

    // ------------------------------------------------------------------
    // Encoding helpers
    // ------------------------------------------------------------------

    /// The `consensus_blocks` key (`storage::digest_hex`) and the wire digest
    /// helpers are one encoding, so a range looked up by a header's `B256`
    /// digest is the row the indexer wrote from `parent_beacon_block_root`.
    #[test]
    fn storage_digest_key_matches_wire_digest_encoding() {
        let b = B256::from(core::array::from_fn::<u8, 32, _>(|i| {
            (i as u8).wrapping_mul(0x1f).wrapping_add(0xa5)
        }));
        assert_eq!(crate::storage::digest_hex(&b), hex_digest(&b));
        assert_eq!(hex_digest(&b), hex_b256(&b));
        assert_eq!(hex_b256(&b).len(), 66);
    }

    #[test]
    fn bs58_digest_is_full_and_newtype_display_is_its_prefix() {
        let bytes = [7u8; 32];
        let full = bs58_digest(&Digest::new(bytes));
        assert!(
            full.len() > 16,
            "full bs58 of 32 bytes is 43-44 chars: {full}"
        );

        let consensus = ConsensusHeaderDigest::from(bytes);
        let epoch = EpochDigest::from(bytes);
        let header = HeaderDigest::new(bytes);
        for short in [consensus.to_string(), epoch.to_string(), header.to_string()] {
            assert_eq!(short.len(), 16, "newtype Display truncates to 16 chars");
            assert!(full.starts_with(&short));
        }
        // newtype -> inner digest via `From<Newtype> for Digest<32>`
        assert_eq!(bs58_of(consensus), full);
        assert_eq!(bs58_of(epoch), full);
        assert_eq!(bs58_of(header), full);
        assert_eq!(bs58_b256(&B256::from(bytes)), full);

        // hex side: every digest form renders the same 0x hex as the B256
        let hex = hex_b256(&B256::from(bytes));
        assert_eq!(hex_digest(&consensus), hex);
        assert_eq!(hex_digest(&epoch), hex);
        assert_eq!(hex_digest(&header), hex);
        assert_eq!(hex_digest(&Digest::new(bytes)), hex);
        assert!(hex.starts_with("0x07070707"));
    }

    // ------------------------------------------------------------------
    // Wire compatibility with the explorer (additive fields are safe)
    // ------------------------------------------------------------------

    #[test]
    fn u128_value_round_trips_direct_but_not_via_value() {
        let api_tx = build_api_transaction(
            &make_tx_data(vec![], U256::from(big_value())),
            InputMode::List,
        );
        assert_eq!(api_tx.value, big_value());
        let json = serde_json::to_string(&api_tx).expect("serialize");

        // (a) direct text parse into the VERBATIM explorer struct: preserved
        let direct: explorer::Transaction = serde_json::from_str(&json).expect("direct parse");
        assert_eq!(direct.value, big_value());
        assert_eq!(direct.to.as_deref(), api_tx.to.as_deref());

        // (b) the legacy rpc_call path: text -> serde_json::Value -> struct.
        // Without `arbitrary_precision`, Value stores integers above u64::MAX
        // as f64 — the exact number is destroyed in the intermediate...
        let value: serde_json::Value = serde_json::from_str(&json).expect("value parse");
        assert!(
            value["value"].as_u64().is_none(),
            "Value must NOT hold the number as an integer anymore"
        );
        // ...and re-deserializing does NOT reproduce the original: it either
        // errors (f64 into u128) or yields a corrupted value. Assert the real
        // behavior without forcing which failure mode serde_json picks.
        match serde_json::from_value::<explorer::Transaction>(value) {
            Err(_) => {} // rejected outright — the documented breakage
            Ok(routed) => assert_ne!(
                routed.value,
                big_value(),
                "Value round-trip unexpectedly preserved a >u64::MAX integer"
            ),
        }
    }

    #[test]
    fn enriched_transaction_round_trips_into_explorer_transaction() {
        let mut api_tx =
            build_api_transaction(&make_tx_data(vec![], U256::ZERO), InputMode::Detail);
        // tx type read via Typed2718 from the signed tx (EIP-1559 fixture)
        assert_eq!(api_tx.tx_type, 2);
        assert_eq!(api_tx.tx_type_name, "eip1559");
        assert!(api_tx.token_transfers.is_none());
        assert!(api_tx.token_transfer_count.is_none());

        // list shape: the detail-only keys are absent, not null
        let list = serde_json::to_value(&api_tx).expect("to_value");
        assert_eq!(list["tx_type"], 2);
        assert_eq!(list["tx_type_name"], "eip1559");
        assert!(list.get("token_transfers").is_none());
        assert!(list.get("token_transfer_count").is_none());
        let direct: explorer::Transaction =
            serde_json::from_str(&serde_json::to_string(&api_tx).expect("serialize"))
                .expect("list parse");
        assert_eq!(direct.nonce, 3);

        // detail shape: embedded transfers + count present
        // a >u64::MAX transfer value cannot pass through `serde_json::to_value` (the
        // documented hazard), so the embedded fixture uses a small value
        let mut row = make_transfer_row();
        row.value = "1500".into();
        let transfer =
            build_token_transfer(&row, &make_tx_data(vec![], U256::ZERO), Some(&make_token()));
        api_tx.token_transfers = Some(vec![transfer]);
        api_tx.token_transfer_count = Some(1);
        let json = serde_json::to_string(&api_tx).expect("serialize");
        let detail = serde_json::to_value(&api_tx).expect("to_value");
        assert_eq!(detail["token_transfer_count"], 1);
        assert_eq!(detail["token_transfers"][0]["log_index"], 4);
        let direct: explorer::Transaction = serde_json::from_str(&json).expect("detail parse");
        assert_eq!(direct.hash, api_tx.hash);
        assert_eq!(direct.status, Some(true));
    }

    #[test]
    fn token_transfer_round_trips_direct_with_saturating_value() {
        let row = make_transfer_row();
        let token = make_token();
        let hydrated = make_tx_data(vec![], U256::ZERO);
        let transfer = build_token_transfer(&row, &hydrated, Some(&token));
        assert_eq!(transfer.value, big_value());
        assert_eq!(transfer.value_exact, big_value().to_string());
        assert_eq!(transfer.log_index, 4);

        let json = serde_json::to_string(&transfer).expect("serialize");
        // direct parse into the VERBATIM explorer struct: value preserved,
        // the extra `log_index` field ignored by serde
        let direct: explorer::ApiTokenTransfer = serde_json::from_str(&json).expect("direct parse");
        assert_eq!(direct.value, big_value());
        assert_eq!(direct.value_exact, big_value().to_string());
        assert_eq!(direct.token_symbol.as_deref(), Some("TOK"));
        assert_eq!(direct.timestamp, 1_700_000_000);

        // a value beyond u128::MAX saturates on the number field while
        // value_exact stays lossless
        let mut huge = row.clone();
        huge.value = U256::MAX.to_string();
        let transfer = build_token_transfer(&huge, &hydrated, Some(&token));
        assert_eq!(transfer.value, u128::MAX);
        assert_eq!(transfer.value_exact, U256::MAX.to_string());

        // the legacy Value route breaks the number here too
        let json = serde_json::to_string(&transfer).expect("serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("value parse");
        match serde_json::from_value::<explorer::ApiTokenTransfer>(value) {
            Err(_) => {}
            Ok(routed) => assert_ne!(routed.value, u128::MAX),
        }
    }

    fn sample_block_consensus() -> ApiBlockConsensus {
        build_block_consensus(
            &HeaderConsensusFields {
                digest: B256::repeat_byte(0xaa),
                epoch: 7,
                round: 9,
                batch_index: 3,
                worker_id: 1,
                batch_digest: B256::repeat_byte(0xbb),
                prev_randao: Some(B256::repeat_byte(0xcc)),
                closes_epoch: true,
            },
            Some(4242),
        )
    }

    #[test]
    fn enriched_block_round_trips_into_explorer_block() {
        // hand-build the wire struct; `build_api_block` is covered below
        let api_block = ApiBlock {
            number: 42,
            hash: "0xabc".into(),
            parent_hash: "0xdef".into(),
            timestamp: 1_700_000_000,
            transactions: vec!["0x01".into()],
            transaction_count: 1,
            gas_used: 21_000,
            gas_limit: 30_000_000,
            miner: "0x11".into(),
            validator: "0x11".into(),
            extra_data: "0x".into(),
            base_fee: Some(7),
            size: 512,
            consensus: Some(sample_block_consensus()),
        };
        let json = serde_json::to_string(&api_block).expect("serialize");
        let direct: explorer::Block = serde_json::from_str(&json).expect("direct parse");
        assert_eq!(direct.number, 42);
        assert_eq!(direct.base_fee, Some(7));
        assert_eq!(direct.validator, "0x11");

        let value = serde_json::to_value(&api_block).expect("to_value");
        assert_eq!(value["consensus"]["epoch"], 7);
        assert_eq!(value["consensus"]["consensus_number"], 4242);
        // genesis: `consensus` is an explicit null, still parseable
        let mut genesis = api_block.clone();
        genesis.consensus = None;
        let value = serde_json::to_value(&genesis).expect("to_value");
        assert!(value["consensus"].is_null());
        let _: explorer::Block = serde_json::from_value(value).expect("genesis parse");
    }

    #[test]
    fn enriched_epoch_round_trips_into_explorer_epoch() {
        let keys = bls_keys(3);
        let record = EpochRecord {
            epoch: 4,
            committee: keys.clone(),
            next_committee: keys[..2].to_vec(),
            parent_hash: EpochDigest::from([0xab; 32]),
            final_state: BlockNumHash::new(100, B256::repeat_byte(1)),
            final_consensus: ConsensusNumHash::new(
                50,
                ConsensusHeaderDigest::from(B256::repeat_byte(2)),
            ),
        };
        let mut cert = EpochCertificate {
            epoch_hash: record.digest(),
            signature: BlsSignature::default(),
            signed_authorities: Default::default(),
        };
        cert.signed_authorities.insert(0);
        cert.signed_authorities.insert(2);

        let api_epoch = ApiEpoch {
            epoch: 4,
            start_block: 51,
            end_block: Some(100),
            end_time: Some(1_700_000_000),
            committee_size: 3,
            committee_bls: keys.iter().map(ToString::to_string).collect(),
            certified: true,
            is_current: false,
            committee_addresses: None,
            record: Some(build_epoch_record(&record)),
            certificate: Some(build_epoch_certificate(&record, &cert, Some(false))),
        };
        let json = serde_json::to_string(&api_epoch).expect("serialize");
        let direct: explorer::ApiEpoch = serde_json::from_str(&json).expect("direct parse");
        assert_eq!(direct.epoch, 4);
        assert!(direct.certified);
        assert_eq!(direct.committee_bls.len(), 3);

        let value = serde_json::to_value(&api_epoch).expect("to_value");
        assert_eq!(value["record"]["super_quorum"], 3);
        assert_eq!(value["record"]["final_consensus"]["number"], 50);
        assert_eq!(
            value["certificate"]["signer_indices"],
            serde_json::json!([0, 2])
        );
        assert_eq!(value["certificate"]["verified"], false);

        // current epoch: both null, still parseable
        let mut current = api_epoch.clone();
        current.record = None;
        current.certificate = None;
        let value = serde_json::to_value(&current).expect("to_value");
        assert!(value["record"].is_null() && value["certificate"].is_null());
        let _: explorer::ApiEpoch = serde_json::from_value(value).expect("current parse");
    }

    // ------------------------------------------------------------------
    // Consensus builders
    // ------------------------------------------------------------------

    #[test]
    fn build_block_consensus_maps_every_field() {
        let c = sample_block_consensus();
        assert_eq!(c.digest, hex_b256(&B256::repeat_byte(0xaa)));
        assert_eq!(c.digest_bs58, bs58_digest(&Digest::new([0xaa; 32])));
        assert_eq!(c.epoch, 7);
        assert_eq!(c.round, 9);
        assert_eq!(c.batch_index, 3);
        assert_eq!(c.worker_id, 1);
        assert_eq!(c.batch_digest, hex_b256(&B256::repeat_byte(0xbb)));
        assert_eq!(
            c.prev_randao.as_deref(),
            Some(hex_b256(&B256::repeat_byte(0xcc)).as_str())
        );
        assert!(c.closes_epoch);
        assert_eq!(c.consensus_number, Some(4242));

        let value = serde_json::to_value(&c).expect("to_value");
        for key in [
            "digest",
            "digest_bs58",
            "epoch",
            "round",
            "batch_index",
            "worker_id",
            "batch_digest",
            "prev_randao",
            "closes_epoch",
            "consensus_number",
        ] {
            assert!(value.get(key).is_some(), "missing JSON key {key}");
        }
    }

    #[test]
    fn build_api_block_decodes_consensus_from_header() {
        let mut header = ExecHeader {
            number: 5,
            parent_beacon_block_root: Some(B256::repeat_byte(0xaa)),
            ommers_hash: B256::repeat_byte(0xbb),
            mix_hash: B256::repeat_byte(0xcc),
            difficulty: U256::from((3u64 << 16) | 1),
            ..Default::default()
        };
        header.nonce = ((7u64 << 32) | 9).to_be_bytes().into();
        let body = tn_types::BlockBody {
            transactions: vec![],
            ommers: vec![],
            withdrawals: None,
        };
        let block = SealedBlock::seal_slow(tn_types::Block { header, body });
        let api = build_api_block(&block);
        let c = api
            .consensus
            .expect("non-genesis block carries consensus fields");
        assert_eq!(c.digest, hex_b256(&B256::repeat_byte(0xaa)));
        assert_eq!((c.epoch, c.round), (7, 9));
        assert_eq!((c.batch_index, c.worker_id), (3, 1));
        assert_eq!(c.batch_digest, hex_b256(&B256::repeat_byte(0xbb)));
        assert_eq!(
            c.consensus_number, None,
            "list builder never resolves the number"
        );

        // genesis: no parent_beacon_block_root => no consensus provenance
        let genesis = SealedBlock::seal_slow(tn_types::Block {
            header: ExecHeader::default(),
            body: tn_types::BlockBody {
                transactions: vec![],
                ommers: vec![],
                withdrawals: None,
            },
        });
        assert!(build_api_block(&genesis).consensus.is_none());
    }

    #[test]
    fn build_consensus_header_maps_sub_dag_summary() {
        let leader_author = AuthorityIdentifier::dummy_for_test(0x09);
        let (out, _, _) = make_output(leader_author.clone());
        let h = ConsensusHeader {
            parent_hash: out.parent_hash(),
            sub_dag: out.sub_dag().clone(),
            number: 12,
            extra: B256::repeat_byte(0xee),
        };
        let range = ApiExecRange {
            first: 100,
            last: 102,
        };
        let api = build_consensus_header(&h, Some(range));

        assert_eq!(api.number, 12);
        assert_eq!(api.digest, hex_digest(&h.digest()));
        assert!(api.digest_bs58.len() > 16);
        assert!(api.digest_bs58.starts_with(&h.digest().to_string()));
        assert_eq!(api.parent_digest, hex_b256(&B256::repeat_byte(0xcc)));
        assert!(api
            .parent_digest_bs58
            .starts_with(&out.parent_hash().to_string()));
        assert_eq!((api.epoch, api.round), (3, 4));
        assert_eq!(api.leader, leader_author.to_string());
        assert_eq!(api.leader_header_digest, hex_digest(&out.leader().digest()));
        // test sub-dags carry commit_timestamp 0 => falls back to leader created_at
        assert_eq!(api.committed_at, 1_700_000_123);
        assert_eq!(api.sub_dag_header_count, 2);
        assert_eq!(api.batch_count, 3, "follower 1 + leader 2 payload entries");
        assert_eq!(api.randomness, hex_b256(&out.sub_dag().randomness()));
        assert_eq!(api.extra, hex_b256(&B256::repeat_byte(0xee)));
        let exec = api.exec_blocks.as_ref().expect("range passed through");
        assert_eq!((exec.first, exec.last), (100, 102));

        let value = serde_json::to_value(&api).expect("to_value");
        assert_eq!(value["exec_blocks"]["first"], 100);
        assert_eq!(value["sub_dag_header_count"], 2);
        assert!(value["leader_header_digest"]
            .as_str()
            .unwrap()
            .starts_with("0x"));
    }

    #[test]
    fn build_consensus_batches_follows_flatten_order_and_hashes_txs() {
        let (out, raws, batches) = make_output(AuthorityIdentifier::dummy_for_test(0x09));
        let range = ApiExecRange {
            first: 100,
            last: 102,
        };

        let summaries = build_consensus_batches(&out, Some(&range), false);
        assert_eq!(summaries.len(), 3);
        for (i, (api, batch)) in summaries.iter().zip(&batches).enumerate() {
            assert_eq!(api.index, i as u64);
            assert_eq!(api.digest, hex_b256(&batch.digest()));
            assert_eq!(api.worker_id, batch.worker_id);
            assert_eq!(api.beneficiary, hex_address(&batch.beneficiary));
            assert_eq!(api.base_fee_per_gas, 7);
            assert_eq!(api.tx_count, batch.transactions.len());
            assert_eq!(api.size_bytes, batch.size());
            assert_eq!(api.exec_block_number, Some(100 + i as u64));
            assert!(api.tx_hashes.is_none());
        }
        // certified-batch grouping: batches 0,1 from authority 0x41; batch 2 from 0x42
        assert_eq!(summaries[0].authority_address, hex_address(&addr(0x41)));
        assert_eq!(summaries[1].authority_address, hex_address(&addr(0x41)));
        assert_eq!(summaries[2].authority_address, hex_address(&addr(0x42)));
        // summaries omit the key entirely (not null)
        let value = serde_json::to_value(&summaries[0]).expect("to_value");
        assert!(value.get("tx_hashes").is_none());
        assert_eq!(value["exec_block_number"], 100);

        // a range the indexer has only partly committed (one block per SQLite
        // transaction): batches past `last` render null, like `exec_blocks`
        let partial = ApiExecRange {
            first: 100,
            last: 101,
        };
        let numbers: Vec<Option<u64>> = build_consensus_batches(&out, Some(&partial), false)
            .iter()
            .map(|b| b.exec_block_number)
            .collect();
        assert_eq!(numbers, vec![Some(100), Some(101), None]);

        // unknown range => null exec numbers; with hashes => keccak256(raw) == tx.hash()
        let detailed = build_consensus_batches(&out, None, true);
        assert!(detailed.iter().all(|b| b.exec_block_number.is_none()));
        let expected: Vec<String> = raws
            .iter()
            .map(|raw| {
                let tx: TransactionSigned =
                    tn_types::Decodable2718::decode_2718(&mut raw.as_slice()).expect("decode");
                hex_b256(tx.hash())
            })
            .collect();
        assert_eq!(detailed[0].tx_hashes.as_deref(), Some(expected.as_slice()));
        assert_eq!(detailed[1].tx_hashes.as_deref(), Some(&[][..]));
        assert_eq!(detailed[2].tx_hashes.as_deref(), Some(&expected[1..]));
        assert_eq!(batch_tx_hashes(&batches[0]), expected);
        let value = serde_json::to_value(&detailed[0]).expect("to_value");
        assert_eq!(value["tx_hashes"].as_array().map(Vec::len), Some(2));
    }

    #[test]
    fn build_consensus_block_resolves_bls_keys_and_leader() {
        let keys = bls_keys(3);
        let leader_author = AuthorityIdentifier::from(keys[1]);
        let (out, _, _) = make_output(leader_author.clone());
        let header = build_consensus_header(&out.consensus_header(), None);
        let batches = build_consensus_batches(&out, None, false);
        let committee: BTreeSet<BlsPublicKey> = keys.iter().copied().collect();

        let block = build_consensus_block(
            header.clone(),
            &out,
            Some(&committee),
            batches.clone(),
            Some(true),
        );
        assert_eq!(
            block.leader_bls.as_deref(),
            Some(keys[1].to_string().as_str())
        );
        assert_eq!(block.sub_dag.len(), 2);
        let (follower, leader) = (&block.sub_dag[0], &block.sub_dag[1]);
        assert!(!follower.is_leader && leader.is_leader);
        assert_eq!(leader.author, leader_author.to_string());
        assert_eq!(
            leader.author_bls.as_deref(),
            Some(keys[1].to_string().as_str())
        );
        assert!(
            follower.author_bls.is_none(),
            "dummy author is not in the committee"
        );
        assert_eq!(
            (leader.round, leader.epoch, leader.created_at),
            (4, 3, 1_700_000_123)
        );
        assert_eq!(leader.payload.len(), 2);
        assert_eq!(leader.payload[0].worker_id, 1);
        assert_eq!(leader.latest_execution_block.number, 77);
        assert_eq!(
            leader.latest_execution_block.hash,
            hex_b256(&B256::repeat_byte(0x77))
        );
        assert!(leader.parents.is_empty());
        assert_eq!(block.batches.len(), 3);
        assert_eq!(block.closes_epoch, Some(true));
        assert!(!block.reputation_scores.final_of_schedule);
        assert!(block.reputation_scores.scores.is_empty());

        // no committee => no BLS resolution, everything else identical
        let bare = build_consensus_block(header, &out, None, batches, None);
        assert!(bare.leader_bls.is_none());
        assert!(bare.sub_dag.iter().all(|h| h.author_bls.is_none()));
        assert_eq!(bare.closes_epoch, None);

        let value = serde_json::to_value(&block).expect("to_value");
        assert_eq!(value["sub_dag"][1]["is_leader"], true);
        assert_eq!(value["reputation_scores"]["final_of_schedule"], false);
        assert_eq!(value["header"]["number"], 12);
    }

    #[test]
    fn build_reputation_scores_orders_descending() {
        let mut scores = ReputationScores::default();
        let (a, b, c) = (
            AuthorityIdentifier::dummy_for_test(1),
            AuthorityIdentifier::dummy_for_test(2),
            AuthorityIdentifier::dummy_for_test(3),
        );
        scores.add_score(&a, 5);
        scores.add_score(&b, 9);
        scores.add_score(&c, 5);
        scores.final_of_schedule = true;
        let api = build_reputation_scores(&scores);
        assert!(api.final_of_schedule);
        let got: Vec<(String, u64)> = api
            .scores
            .iter()
            .map(|s| (s.authority.clone(), s.score))
            .collect();
        let expected: Vec<(String, u64)> = scores
            .authorities_by_score_desc()
            .into_iter()
            .map(|(id, s)| (id.to_string(), s))
            .collect();
        assert_eq!(got, expected);
        assert_eq!(got[0], (b.to_string(), 9));
    }

    #[test]
    fn build_epoch_record_maps_every_field() {
        let keys = bls_keys(3);
        let record = EpochRecord {
            epoch: 4,
            committee: keys.clone(),
            next_committee: keys[1..].to_vec(),
            parent_hash: EpochDigest::from([0xab; 32]),
            final_state: BlockNumHash::new(100, B256::repeat_byte(1)),
            final_consensus: ConsensusNumHash::new(
                50,
                ConsensusHeaderDigest::from(B256::repeat_byte(2)),
            ),
        };
        let api = build_epoch_record(&record);
        assert_eq!(api.epoch, 4);
        assert_eq!(api.digest, hex_digest(&record.digest()));
        assert!(
            api.digest_bs58.starts_with(&record.digest().to_string()) && api.digest_bs58.len() > 16
        );
        assert_eq!(api.parent_hash, hex_b256(&B256::repeat_byte(0xab)));
        assert!(api
            .parent_hash_bs58
            .starts_with(&record.parent_hash.to_string()));
        let bs58_keys: Vec<String> = keys.iter().map(ToString::to_string).collect();
        assert_eq!(api.committee, bs58_keys);
        assert_eq!(api.next_committee, bs58_keys[1..]);
        assert_eq!(api.final_state.number, 100);
        assert_eq!(api.final_state.hash, hex_b256(&B256::repeat_byte(1)));
        assert_eq!(api.final_consensus.number, 50);
        assert_eq!(api.final_consensus.hash, hex_b256(&B256::repeat_byte(2)));
        assert!(api
            .final_consensus
            .hash_bs58
            .starts_with(&record.final_consensus.hash.to_string()));
        assert_eq!(api.super_quorum, 3);
    }

    #[test]
    fn build_epoch_certificate_expands_bitmap_against_committee() {
        let keys = bls_keys(3);
        let record = EpochRecord {
            epoch: 4,
            committee: keys.clone(),
            ..Default::default()
        };
        let mut cert = EpochCertificate {
            epoch_hash: record.digest(),
            signature: BlsSignature::default(),
            signed_authorities: Default::default(),
        };
        cert.signed_authorities.insert(0);
        cert.signed_authorities.insert(2);

        let api = build_epoch_certificate(&record, &cert, None);
        assert_eq!(api.signer_indices, vec![0, 2]);
        assert_eq!(api.signers, vec![keys[0].to_string(), keys[2].to_string()]);
        assert_eq!(api.unresolved_signers, 0);
        assert_eq!(api.signer_count, 2);
        assert_eq!(api.super_quorum, 3);
        assert_eq!(api.verified, None);
        assert_eq!(api.epoch_hash, hex_digest(&record.digest()));
        assert!(
            api.epoch_hash_bs58
                .starts_with(&record.digest().to_string())
                && api.epoch_hash_bs58.len() > 16
        );
        assert_eq!(api.signature, BlsSignature::default().to_string());

        // an index past the committee is counted, never invented
        cert.signed_authorities.insert(7);
        let api = build_epoch_certificate(&record, &cert, Some(true));
        assert_eq!(api.signer_indices, vec![0, 2, 7]);
        assert_eq!(api.signers, vec![keys[0].to_string(), keys[2].to_string()]);
        assert_eq!(api.unresolved_signers, 1);
        assert_eq!(api.signer_count, 3);
        assert_eq!(api.verified, Some(true));

        let value = serde_json::to_value(&api).expect("to_value");
        assert_eq!(value["unresolved_signers"], 1);
        assert_eq!(value["signer_count"], 3);
        assert_eq!(value["epoch_hash_bs58"], api.epoch_hash_bs58);
    }

    // ------------------------------------------------------------------
    // Existing helpers
    // ------------------------------------------------------------------

    #[test]
    fn input_policy_list_vs_detail() {
        // 3-byte input: selector is None => list serves "0x", detail serves full hex
        let short = make_tx_data(vec![0xaa, 0xbb, 0xcc], U256::ZERO);
        assert_eq!(selector_hex(&[0xaa, 0xbb, 0xcc]), None);
        assert_eq!(build_api_transaction(&short, InputMode::List).input, "0x");
        assert_eq!(
            build_api_transaction(&short, InputMode::Detail).input,
            "0xaabbcc"
        );

        // >= 4 bytes: list serves the selector only, detail serves everything
        let long = make_tx_data(vec![0xa9, 0x05, 0x9c, 0xbb, 0x01, 0x02], U256::ZERO);
        let list = build_api_transaction(&long, InputMode::List);
        let detail = build_api_transaction(&long, InputMode::Detail);
        assert_eq!(list.input, "0xa9059cbb");
        assert_eq!(detail.input, "0xa9059cbb0102");
        assert_ne!(list.input, detail.input);

        // empty input: both serve "0x"
        let empty = make_tx_data(vec![], U256::ZERO);
        assert_eq!(build_api_transaction(&empty, InputMode::List).input, "0x");
        assert_eq!(build_api_transaction(&empty, InputMode::Detail).input, "0x");
    }

    #[test]
    fn per_page_clamps() {
        assert_eq!(clamp_per_page(None), DEFAULT_PER_PAGE);
        assert_eq!(clamp_per_page(Some(1000)), MAX_PER_PAGE);
        assert_eq!(clamp_per_page(Some(0)), 1);
        assert_eq!(clamp_per_page(Some(40)), 40);
    }

    #[test]
    fn amount_and_tel_conversions() {
        assert_eq!(token_amount("1500000000000000000", 18), 1.5);
        assert_eq!(token_amount("1500", 0), 1500.0); // 0-decimals fallback shape
        assert_eq!(token_amount("garbage", 18), 0.0);
        assert_eq!(wei_to_tel(&U256::from(5_000_000_000_000_000_000u128)), 5.0);
    }
}
