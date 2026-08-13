//! Wire types and pure builders for the HTTP API.
//!
//! [`ApiTransaction`] and [`ApiBlock`] are field-for-field deserializable into
//! the explorer's existing `Transaction` and `Block` structs
//! (`telcoin-block-explorer-xyz/src/services/rpc.rs`) — asserted against
//! verbatim copies in this module's tests.
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

use crate::node_reads::TxData;
use serde::{Deserialize, Serialize};
use tn_types::{hex, Address, BlockHeader as _, SealedBlock, TransactionTrait as _, B256, U256};
use tracing::warn;

/// Default page size for list endpoints.
pub const DEFAULT_PER_PAGE: u64 = 25;
/// Maximum page size for list endpoints (requests above this clamp down).
pub const MAX_PER_PAGE: u64 = 100;

/// The list envelope every paged endpoint returns; `page` is 0-based and items
/// are newest-first.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope<T> {
    /// The page of items, newest first.
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
}

/// One ERC-20 transfer row — explorer `TokenTransfer`-compatible, plus
/// `value_exact` (ignored by older clients; the lossless decimal string).
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

/// Which input representation a transaction response carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    /// List endpoints: `"0x"` or the 4-byte selector hex.
    List,
    /// `/txs/{hash}`: the full calldata hex (deployments run to ~24KB).
    Detail,
}

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

/// Map hydrated transaction data to the wire shape (shared by `/txs`,
/// `/txs/{hash}`, and `/address/{addr}/txs`).
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
    }
}

/// Map a sealed block to the wire shape (shared by `/blocks` and
/// `/blocks/{number}`).
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
    }
}

/// Map a stored transfer row + its hydrated transaction + the cached token
/// metadata to the wire shape (shared by both transfer feeds).
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{StoredToken, StoredTransfer};
    use tn_types::{Bytes, EthSignature, TransactionSigned, TxEip1559, TxKind, U256};

    // ------------------------------------------------------------------
    // VERBATIM copies of the explorer's structs
    // (telcoin-block-explorer-xyz/src/services/rpc.rs v0.1.17). These are the
    // wire-compatibility gate: every field name and type byte-matches.
    // ------------------------------------------------------------------
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Block {
        pub number: u64,
        pub hash: String,
        pub parent_hash: String,
        pub timestamp: u64,
        pub transactions: Vec<String>,
        pub transaction_count: usize,
        pub gas_used: u64,
        pub gas_limit: u64,
        pub miner: String,
        pub validator: String,
        pub extra_data: String,
        pub base_fee: Option<u64>,
        pub size: u64,
    }
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DecodedInput {
        pub method: String,
        pub signature: String,
        pub params: Vec<(String, String)>,
    }
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Transaction {
        pub hash: String,
        pub from: String,
        pub to: Option<String>,
        pub value: u128,
        pub value_tel: f64,
        pub gas: u64,
        pub gas_price: u64,
        pub gas_used: u64,
        pub status: Option<bool>,
        pub input: String,
        pub decoded_input: Option<DecodedInput>,
        pub block_number: Option<u64>,
        pub transaction_index: Option<u64>,
        pub nonce: u64,
    }
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TokenTransfer {
        pub tx_hash: String,
        pub from: String,
        pub to: String,
        pub value: u128,
        pub amount: f64,
        pub block_number: u64,
        pub timestamp: u64,
        pub token_address: String,
        pub token_symbol: String,
    }

    fn addr(byte: u8) -> Address {
        Address::repeat_byte(byte)
    }

    fn make_tx_data(input: Vec<u8>, value: U256) -> TxData {
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
        TxData {
            sender: addr(0x11),
            tx: TransactionSigned::new_unhashed(
                tn_types::Transaction::Eip1559(tx),
                EthSignature::new(U256::from(1u64), U256::from(1u64), false),
            ),
            success: true,
            gas_used: 21_000,
            block_number: 9,
            tx_index: 0,
            base_fee: Some(1_000_000_000),
            timestamp: 1_700_000_000,
        }
    }

    /// A wei value strictly above u64::MAX (the corruption threshold).
    fn big_value() -> u128 {
        u128::from(u64::MAX) + 12_345
    }

    #[test]
    fn u128_value_round_trips_direct_but_not_via_value() {
        let api_tx = build_api_transaction(
            &make_tx_data(vec![], U256::from(big_value())),
            InputMode::List,
        );
        assert_eq!(api_tx.value, big_value());
        let json = serde_json::to_string(&api_tx).expect("serialize");

        // (a) direct text parse into the VERBATIM explorer struct: preserved
        let direct: Transaction = serde_json::from_str(&json).expect("direct parse");
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
        match serde_json::from_value::<Transaction>(value) {
            Err(_) => {} // rejected outright — the documented breakage
            Ok(routed) => assert_ne!(
                routed.value,
                big_value(),
                "Value round-trip unexpectedly preserved a >u64::MAX integer"
            ),
        }
    }

    #[test]
    fn token_transfer_round_trips_direct_with_saturating_value() {
        let row = StoredTransfer {
            block_number: 9,
            tx_index: 0,
            token: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
            from: "0x1111111111111111111111111111111111111111".into(),
            to: "0x2222222222222222222222222222222222222222".into(),
            value: big_value().to_string(),
        };
        let token = StoredToken {
            name: Some("Token".into()),
            symbol: Some("TOK".into()),
            decimals: Some(18),
            status: 0,
        };
        let hydrated = make_tx_data(vec![], U256::ZERO);
        let transfer = build_token_transfer(&row, &hydrated, Some(&token));
        assert_eq!(transfer.value, big_value());
        assert_eq!(transfer.value_exact, big_value().to_string());

        let json = serde_json::to_string(&transfer).expect("serialize");
        // direct parse into the VERBATIM explorer struct: value preserved,
        // the extra `value_exact` field ignored by serde
        let direct: TokenTransfer = serde_json::from_str(&json).expect("direct parse");
        assert_eq!(direct.value, big_value());
        assert_eq!(direct.token_symbol, "TOK");
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
        match serde_json::from_value::<TokenTransfer>(value) {
            Err(_) => {}
            Ok(routed) => assert_ne!(routed.value, u128::MAX),
        }
    }

    #[test]
    fn api_block_round_trips_into_explorer_block() {
        // hand-build the wire struct (SealedBlock construction is exercised in
        // integration; this asserts the serde shape byte-compatibility)
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
        };
        let json = serde_json::to_string(&api_block).expect("serialize");
        let direct: Block = serde_json::from_str(&json).expect("direct parse");
        assert_eq!(direct.number, 42);
        assert_eq!(direct.base_fee, Some(7));
        assert_eq!(direct.validator, "0x11");
    }

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
