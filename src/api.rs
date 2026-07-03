//! The explorer HTTP API: 17 routes served from direct node reads
//! (`node_reads`) and the derived SQLite indexes (`storage`), spawned by the
//! ExEx BEFORE catch-up so `/health` is observable during a long first replay.
//!
//! Error contract: missing resources are `404 {"error":"not found"}`; malformed
//! parameters are 400; internal failures are 500 and logged at `error!`. A
//! hydration failure for an INDEXED pointer is the R11 dangling-pointer
//! invariant violated — a bug by construction (ChainExecuted fires post-commit,
//! so every indexed pointer is hydratable), not load-shedding; it surfaces as
//! 500 + `error!` and is an alert condition.

use crate::{
    node_reads::{self, CallOutcome, NodeReader},
    status::IndexerStatus,
    storage::{self, ReadPool, StoredToken, StoredTransfer, TxPointer},
    types::{
        build_api_block, build_api_transaction, build_token_transfer, clamp_per_page, hex_address,
        hex_bytes, ApiAddress, ApiBlock, ApiEpoch, ApiEpochData, ApiStats, ApiTokenInfo,
        ApiTokenTransfer, ApiTransaction, ApiValidator, CallRequest, CallResponse, Envelope,
        HealthResponse, InputMode, LeaderEntry, LeadersResponse,
    },
};
use axum::{
    extract::{Path, Query, State},
    http::{header::CONTENT_TYPE, HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use eyre::eyre;
use serde::Deserialize;
use serde_json::json;
use std::{
    collections::BTreeMap,
    net::SocketAddr,
    str::FromStr,
    sync::Arc,
    time::{Duration, Instant},
};
use tn_reth::system_calls::ConsensusRegistry;
use tn_storage::epoch_records::EpochRecordDb;
use tn_types::{hex, Address, TxHash};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tracing::{error, info};

/// In-process TTL for the `/stats` cache — `/stats` is every client's
/// every-page poll (12-30s), so one node read serves them all.
const STATS_TTL: Duration = Duration::from_secs(5);

/// Maximum decoded calldata accepted by `POST /call`.
const MAX_CALLDATA_BYTES: usize = 128 * 1024;

/// Default and maximum `?window=` for `/validators/leaders`.
const DEFAULT_LEADER_WINDOW: u64 = 200;
const MAX_LEADER_WINDOW: u64 = 1000;

/// Shared state behind every handler.
#[derive(Debug)]
pub struct ApiState {
    /// Bounded direct reads over the node's databases.
    reader: NodeReader,
    /// Read-only SQLite pool for the derived indexes.
    pool: ReadPool,
    /// Indexing-loop health for `/health`.
    status: Arc<IndexerStatus>,
    /// `(refreshed_at, stats)` — see [`STATS_TTL`].
    stats_cache: RwLock<Option<(Instant, ApiStats)>>,
}

impl ApiState {
    /// Assemble the API state.
    pub fn new(reader: NodeReader, pool: ReadPool, status: Arc<IndexerStatus>) -> Self {
        Self {
            reader,
            pool,
            status,
            stats_cache: RwLock::new(None),
        }
    }
}

/// Handler error → HTTP response mapping (see the module docs for the
/// contract).
#[derive(Debug)]
enum ApiError {
    /// 404 `{"error":"not found"}`.
    NotFound,
    /// 400 with a specific message.
    BadRequest(String),
    /// 500; the cause is logged at `error!`, never leaked.
    Internal(eyre::Report),
}

impl From<eyre::Report> for ApiError {
    fn from(err: eyre::Report) -> Self {
        Self::Internal(err)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            Self::NotFound => {
                (StatusCode::NOT_FOUND, Json(json!({"error": "not found"}))).into_response()
            }
            Self::BadRequest(message) => {
                (StatusCode::BAD_REQUEST, Json(json!({"error": message}))).into_response()
            }
            Self::Internal(err) => {
                // dangling index pointers land here: a bug by construction
                // (R11), logged loudly — never load-shedding
                error!(target: "indexer::api", ?err, "internal error serving request");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": "internal error"})),
                )
                    .into_response()
            }
        }
    }
}

type ApiResult<T> = Result<Json<T>, ApiError>;

/// `?page=&per_page=` for the paged endpoints.
#[derive(Debug, Deserialize)]
struct PageQuery {
    page: Option<u64>,
    per_page: Option<u64>,
}

/// `(page, per_page, offset)` with the page size clamped to `1..=100`.
fn page_params(query: &PageQuery) -> (u64, u64, u64) {
    let page = query.page.unwrap_or(0);
    let per_page = clamp_per_page(query.per_page);
    (page, per_page, page.saturating_mul(per_page))
}

fn parse_address(raw: &str) -> Result<Address, ApiError> {
    Address::from_str(raw).map_err(|_| ApiError::BadRequest(format!("invalid address: {raw}")))
}

fn parse_hash(raw: &str) -> Result<TxHash, ApiError> {
    TxHash::from_str(raw)
        .map_err(|_| ApiError::BadRequest(format!("invalid transaction hash: {raw}")))
}

/// Epoch numbers travel as u64 on the wire but are u32 in consensus; inputs
/// are validated against the current epoch before conversion.
fn epoch_u32(epoch: u64) -> u32 {
    u32::try_from(epoch).unwrap_or(u32::MAX)
}

/// Build the router over shared state (CORS applied in [`serve`]).
pub fn router(state: Arc<ApiState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/stats", get(stats))
        .route("/txs", get(txs_list))
        .route("/txs/{hash}", get(tx_by_hash))
        .route("/blocks", get(blocks_list))
        .route("/blocks/{number}", get(block_by_number))
        .route("/address/{addr}", get(address_summary))
        .route("/address/{addr}/txs", get(address_txs))
        .route("/address/{addr}/transfers", get(address_transfers))
        .route("/tokens/{addr}", get(token_info))
        .route("/tokens/{addr}/transfers", get(token_transfers))
        .route("/epochs", get(epochs_list))
        .route("/epochs/current", get(epoch_current))
        .route("/epochs/{n}", get(epoch_by_number))
        .route("/validators", get(validators))
        .route("/validators/leaders", get(leaders))
        .route("/call", post(call))
        .fallback(|| async { ApiError::NotFound })
        .with_state(state)
}

/// Bind and serve until the cancellation token fires (node shutdown).
pub async fn serve(
    state: Arc<ApiState>,
    addr: SocketAddr,
    cors_origins: Vec<String>,
    shutdown: CancellationToken,
) -> eyre::Result<()> {
    let app = router(state).layer(build_cors(&cors_origins)?);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!(target: "indexer::api", %addr, "explorer indexer API listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(async move { shutdown.cancelled().await })
        .await?;
    Ok(())
}

/// CORS: `[GET, POST]` + `content-type` (the JSON `POST /call` triggers a
/// preflight, so GET-only CORS would break it). An empty origin list means
/// `Any` — the data is public and read-only.
fn build_cors(origins: &[String]) -> eyre::Result<CorsLayer> {
    let cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([CONTENT_TYPE]);
    if origins.is_empty() {
        return Ok(cors.allow_origin(Any));
    }
    let list = origins
        .iter()
        .map(|origin| {
            HeaderValue::from_str(origin).map_err(|_| eyre!("invalid CORS origin {origin:?}"))
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    Ok(cors.allow_origin(AllowOrigin::list(list)))
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `GET /health` — always 200; degradation is expressed in the body so the
/// explorer renders a banner instead of erroring.
async fn health(State(state): State<Arc<ApiState>>) -> Json<HealthResponse> {
    let live = state.status.live();
    Json(HealthResponse {
        status: if live { "ok" } else { "degraded" }.to_string(),
        indexing_live: live,
        last_indexed: state.status.last_indexed().unwrap_or(0),
        node_tip: state.status.node_tip(),
        lag: state.status.lag(),
    })
}

/// `GET /stats` — direct reads with a [`STATS_TTL`] in-process cache.
async fn stats(State(state): State<Arc<ApiState>>) -> ApiResult<ApiStats> {
    if let Some((refreshed_at, cached)) = state.stats_cache.read().await.as_ref() {
        if refreshed_at.elapsed() < STATS_TTL {
            return Ok(Json(cached.clone()));
        }
    }
    let snapshot = state.reader.stats_snapshot().await?;
    let epoch = state.reader.consensus().latest_consensus_epoch();
    // committee size from the epoch records (no EVM); registry fallback when
    // no record exists yet (genesis)
    let validator_count = match state
        .reader
        .consensus()
        .epochs()
        .get_committee_keys(epoch)
        .await
    {
        Some(keys) => keys.len(),
        None => state
            .reader
            .current_epoch_with_tip()
            .await?
            .0
            .validators
            .len(),
    };
    let stats = ApiStats {
        latest_block: snapshot.latest_block,
        gas_price_gwei: snapshot.gas_price_wei as f64 / 1e9,
        chain_id: snapshot.chain_id,
        epoch_number: Some(u64::from(epoch)),
        validator_count,
        total_txs: snapshot.total_txs,
    };
    *state.stats_cache.write().await = Some((Instant::now(), stats.clone()));
    Ok(Json(stats))
}

/// `GET /txs` — the chain-wide feed via arithmetic `TxNumber` pagination.
async fn txs_list(
    State(state): State<Arc<ApiState>>,
    Query(query): Query<PageQuery>,
) -> ApiResult<Envelope<ApiTransaction>> {
    let (page, per_page, _) = page_params(&query);
    let (rows, total) = state.reader.latest_txs_page(page, per_page).await?;
    let items = rows
        .iter()
        .map(|data| build_api_transaction(data, InputMode::List))
        .collect();
    Ok(Json(Envelope {
        items,
        total,
        page,
        per_page,
    }))
}

/// `GET /txs/{hash}` — mined transactions only (full input hex).
async fn tx_by_hash(
    State(state): State<Arc<ApiState>>,
    Path(raw): Path<String>,
) -> ApiResult<ApiTransaction> {
    let hash = parse_hash(&raw)?;
    match state.reader.tx_by_hash(hash).await? {
        Some(data) => Ok(Json(build_api_transaction(&data, InputMode::Detail))),
        None => Err(ApiError::NotFound),
    }
}

/// `GET /blocks` — dense-height arithmetic pagination (`total = tip + 1`).
async fn blocks_list(
    State(state): State<Arc<ApiState>>,
    Query(query): Query<PageQuery>,
) -> ApiResult<Envelope<ApiBlock>> {
    let (page, per_page, _) = page_params(&query);
    let (blocks, total) = state.reader.blocks_page(page, per_page).await?;
    let items = blocks.iter().map(build_api_block).collect();
    Ok(Json(Envelope {
        items,
        total,
        page,
        per_page,
    }))
}

/// `GET /blocks/{number}`.
async fn block_by_number(
    State(state): State<Arc<ApiState>>,
    Path(raw): Path<String>,
) -> ApiResult<ApiBlock> {
    let number: u64 = raw
        .parse()
        .map_err(|_| ApiError::BadRequest(format!("invalid block number: {raw}")))?;
    match state.reader.block_by_number(number).await? {
        Some(block) => Ok(Json(build_api_block(&block))),
        None => Err(ApiError::NotFound),
    }
}

/// `GET /address/{addr}` — balance/nonce/code direct from the node, pointer
/// count from SQLite.
async fn address_summary(
    State(state): State<Arc<ApiState>>,
    Path(raw): Path<String>,
) -> ApiResult<ApiAddress> {
    let address = parse_address(&raw)?;
    let (account, code) = state.reader.account_summary(address).await?;
    let key = storage::addr_hex(&address);
    let count_key = key.clone();
    let indexed_tx_count = state
        .pool
        .with_conn(move |conn| Ok(storage::count_address_txs(conn, &count_key)?))
        .await?;
    let balance = account
        .as_ref()
        .map(|acct| acct.balance)
        .unwrap_or_default();
    let code_hex = code
        .as_ref()
        .filter(|code| !code.is_empty())
        .map(|code| hex_bytes(code));
    Ok(Json(ApiAddress {
        address: key,
        balance_wei: balance.to_string(),
        balance_tel: crate::types::wei_to_tel(&balance),
        nonce: account.as_ref().map(|acct| acct.nonce).unwrap_or_default(),
        is_contract: code_hex.is_some(),
        code: code_hex,
        indexed_tx_count,
    }))
}

/// `GET /address/{addr}/txs` — SQLite reverse-PK scan, hydrated from the
/// node's DB (one replay per distinct block).
async fn address_txs(
    State(state): State<Arc<ApiState>>,
    Path(raw): Path<String>,
    Query(query): Query<PageQuery>,
) -> ApiResult<Envelope<ApiTransaction>> {
    let address = parse_address(&raw)?;
    let key = storage::addr_hex(&address);
    let (page, per_page, offset) = page_params(&query);
    let (pointers, total) = state
        .pool
        .with_conn(move |conn| {
            Ok((
                storage::address_txs_page(conn, &key, per_page, offset)?,
                storage::count_address_txs(conn, &key)?,
            ))
        })
        .await?;
    let hydrated = state.reader.hydrate_pointers(pointers).await?;
    let items = hydrated
        .iter()
        .map(|data| build_api_transaction(data, InputMode::List))
        .collect();
    Ok(Json(Envelope {
        items,
        total,
        page,
        per_page,
    }))
}

/// One page of stored transfers + the cached metadata of every token in the
/// page, in a single pooled read.
type TransferPage = (Vec<StoredTransfer>, u64, BTreeMap<String, StoredToken>);

/// Zip stored transfer rows with their hydrated transactions into wire rows.
async fn hydrate_transfers(
    state: &ApiState,
    rows: Vec<StoredTransfer>,
    tokens: BTreeMap<String, StoredToken>,
) -> Result<Vec<ApiTokenTransfer>, ApiError> {
    let pointers: Vec<TxPointer> = rows
        .iter()
        .map(|row| TxPointer {
            block_number: row.block_number,
            tx_index: row.tx_index,
        })
        .collect();
    let hydrated = state.reader.hydrate_pointers(pointers).await?;
    if hydrated.len() != rows.len() {
        return Err(ApiError::Internal(eyre!(
            "hydration returned {} rows for {} pointers",
            hydrated.len(),
            rows.len()
        )));
    }
    Ok(rows
        .iter()
        .zip(hydrated.iter())
        .map(|(row, data)| build_token_transfer(row, data, tokens.get(&row.token)))
        .collect())
}

fn page_tokens(
    conn: &rusqlite::Connection,
    rows: &[StoredTransfer],
) -> eyre::Result<BTreeMap<String, StoredToken>> {
    let mut tokens = BTreeMap::new();
    for row in rows {
        if !tokens.contains_key(&row.token) {
            if let Some(token) = storage::token_row(conn, &row.token)? {
                tokens.insert(row.token.clone(), token);
            }
        }
    }
    Ok(tokens)
}

/// `GET /address/{addr}/transfers` — transfers where the address is sender or
/// recipient (post-union newest-first), hydrated with tx hash + timestamp.
async fn address_transfers(
    State(state): State<Arc<ApiState>>,
    Path(raw): Path<String>,
    Query(query): Query<PageQuery>,
) -> ApiResult<Envelope<ApiTokenTransfer>> {
    let address = parse_address(&raw)?;
    let key = storage::addr_hex(&address);
    let (page, per_page, offset) = page_params(&query);
    let (rows, total, tokens): TransferPage = state
        .pool
        .with_conn(move |conn| {
            let rows = storage::address_transfers_page(conn, &key, per_page, offset)?;
            let total = storage::count_address_transfers(conn, &key)?;
            let tokens = page_tokens(conn, &rows)?;
            Ok((rows, total, tokens))
        })
        .await?;
    let items = hydrate_transfers(&state, rows, tokens).await?;
    Ok(Json(Envelope {
        items,
        total,
        page,
        per_page,
    }))
}

/// `GET /tokens/{addr}` — cache hit serves cached metadata + live supply; a
/// miss live-fetches WITHOUT write-back (the API never writes).
async fn token_info(
    State(state): State<Arc<ApiState>>,
    Path(raw): Path<String>,
) -> ApiResult<ApiTokenInfo> {
    let address = parse_address(&raw)?;
    let key = storage::addr_hex(&address);
    let cache_key = key.clone();
    let cached = state
        .pool
        .with_conn(move |conn| Ok(storage::token_row(conn, &cache_key)?))
        .await?;
    let info = match cached {
        Some(row) => {
            let total_supply = state.reader.token_supply(address).await?;
            ApiTokenInfo {
                address: key,
                name: row.name,
                symbol: row.symbol,
                decimals: row.decimals,
                total_supply,
                metadata_status: row.status,
            }
        }
        None => {
            let (metadata, total_supply) = state.reader.token_metadata_and_supply(address).await?;
            let metadata_status = u8::from(!metadata.any_decoded());
            ApiTokenInfo {
                address: key,
                name: metadata.name,
                symbol: metadata.symbol,
                decimals: metadata.decimals,
                total_supply,
                metadata_status,
            }
        }
    };
    Ok(Json(info))
}

/// `GET /tokens/{addr}/transfers` — the token's transfer feed (`idx_tt_token`).
async fn token_transfers(
    State(state): State<Arc<ApiState>>,
    Path(raw): Path<String>,
    Query(query): Query<PageQuery>,
) -> ApiResult<Envelope<ApiTokenTransfer>> {
    let address = parse_address(&raw)?;
    let key = storage::addr_hex(&address);
    let (page, per_page, offset) = page_params(&query);
    let (rows, total, tokens): TransferPage = state
        .pool
        .with_conn(move |conn| {
            let rows = storage::token_transfers_page(conn, &key, per_page, offset)?;
            let total = storage::count_token_transfers(conn, &key)?;
            let tokens = page_tokens(conn, &rows)?;
            Ok((rows, total, tokens))
        })
        .await?;
    let items = hydrate_transfers(&state, rows, tokens).await?;
    Ok(Json(Envelope {
        items,
        total,
        page,
        per_page,
    }))
}

/// First execution block of an epoch: `record(N-1).final_state.number + 1`
/// (epoch 0 starts at block 0). A missing predecessor record degrades to 0
/// rather than failing the page.
async fn epoch_start_block(db: &EpochRecordDb, epoch: u64) -> u64 {
    if epoch == 0 {
        return 0;
    }
    match db.record_by_epoch(epoch_u32(epoch - 1)).await {
        Some(previous) => previous.final_state.number.saturating_add(1),
        None => 0,
    }
}

/// Build one epoch row (without `end_time`, which needs a header read the
/// caller batches).
async fn epoch_row(db: &EpochRecordDb, epoch: u64, current: u64) -> ApiEpoch {
    let is_current = epoch == current;
    let (record, certificate) = if is_current {
        // the in-progress epoch has no record yet: synthesized
        (None, None)
    } else {
        match db.get_epoch_by_number(epoch_u32(epoch)).await {
            Some((record, certificate)) => (Some(record), certificate),
            None => (None, None),
        }
    };
    let start_block = epoch_start_block(db, epoch).await;
    let committee_bls: Vec<String> = match &record {
        Some(record) => record.committee.iter().map(|key| key.to_string()).collect(),
        // current epoch: previous record's next_committee via get_committee_keys
        None => db
            .get_committee_keys(epoch_u32(epoch))
            .await
            .map(|keys| keys.iter().map(|key| key.to_string()).collect())
            .unwrap_or_default(),
    };
    ApiEpoch {
        epoch,
        start_block,
        end_block: record.as_ref().map(|record| record.final_state.number),
        end_time: None,
        committee_size: committee_bls.len(),
        committee_bls,
        certified: certificate.is_some(),
        is_current,
        committee_addresses: None,
    }
}

/// `GET /epochs` — newest-first epoch history from the consensus DB's epoch
/// records; the current epoch is synthesized (no record exists yet).
async fn epochs_list(
    State(state): State<Arc<ApiState>>,
    Query(query): Query<PageQuery>,
) -> ApiResult<Envelope<ApiEpoch>> {
    let (page, per_page, _) = page_params(&query);
    let current = u64::from(state.reader.consensus().latest_consensus_epoch());
    let total = current.saturating_add(1);
    let numbers = node_reads::desc_page_items(total, page, per_page);
    let db = state.reader.consensus().epochs();

    let mut items = Vec::with_capacity(numbers.len());
    for number in numbers {
        items.push(epoch_row(db, number, current).await);
    }

    // boundary timestamps for the whole page in ONE blocking read
    let end_blocks: Vec<u64> = items.iter().filter_map(|item| item.end_block).collect();
    if !end_blocks.is_empty() {
        let times = state.reader.header_timestamps(end_blocks).await?;
        for item in &mut items {
            item.end_time = item.end_block.and_then(|end| times.get(&end).copied());
        }
    }
    Ok(Json(Envelope {
        items,
        total,
        page,
        per_page,
    }))
}

/// `GET /epochs/current` — one pinned registry read at the canonical tip.
async fn epoch_current(State(state): State<Arc<ApiState>>) -> ApiResult<ApiEpochData> {
    let (epoch_state, latest_block) = state.reader.current_epoch_with_tip().await?;
    Ok(Json(ApiEpochData {
        epoch: u64::from(epoch_state.epoch),
        epoch_duration: u64::from(epoch_state.epoch_info.epochDuration),
        validator_count: epoch_state.validators.len(),
        validators: epoch_state
            .validators
            .iter()
            .map(|info| hex_address(&info.validatorAddress))
            .collect(),
        start_block: epoch_state.epoch_info.blockHeight,
        latest_block,
    }))
}

/// `GET /epochs/{n}` — an epoch record + certificate presence, with
/// best-effort committee ADDRESSES while `n` is inside the registry's ring
/// buffer (BLS keys are always available).
async fn epoch_by_number(
    State(state): State<Arc<ApiState>>,
    Path(raw): Path<String>,
) -> ApiResult<ApiEpoch> {
    let number: u64 = raw
        .parse()
        .map_err(|_| ApiError::BadRequest(format!("invalid epoch number: {raw}")))?;
    let current = u64::from(state.reader.consensus().latest_consensus_epoch());
    if number > current {
        return Err(ApiError::NotFound);
    }
    let mut item = epoch_row(state.reader.consensus().epochs(), number, current).await;
    if let Some(end_block) = item.end_block {
        let times = state.reader.header_timestamps(vec![end_block]).await?;
        item.end_time = times.get(&end_block).copied();
    }
    item.committee_addresses = state
        .reader
        .committee_addresses(epoch_u32(number))
        .await
        .map(|addresses| addresses.iter().map(hex_address).collect());
    Ok(Json(item))
}

/// `GET /validators` — the current committee's registry `ValidatorInfo`s.
async fn validators(State(state): State<Arc<ApiState>>) -> ApiResult<Vec<ApiValidator>> {
    let epoch = state.reader.consensus().latest_consensus_epoch();
    let infos = state.reader.validators_for_epoch(epoch).await?;
    Ok(Json(infos.iter().map(api_validator).collect()))
}

fn api_validator(info: &ConsensusRegistry::ValidatorInfo) -> ApiValidator {
    ApiValidator {
        address: hex_address(&info.validatorAddress),
        activation_epoch: info.activationEpoch,
        exit_epoch: info.exitEpoch,
        status: info.currentStatus as u8,
        is_retired: info.isRetired,
        stake_version: info.stakeVersion,
        region: info.region,
    }
}

/// `?window=` for `/validators/leaders`.
#[derive(Debug, Deserialize)]
struct LeadersQuery {
    window: Option<u64>,
}

/// `GET /validators/leaders?window=200` — beneficiary counts over the trailing
/// window (clamped to [`MAX_LEADER_WINDOW`]).
async fn leaders(
    State(state): State<Arc<ApiState>>,
    Query(query): Query<LeadersQuery>,
) -> ApiResult<LeadersResponse> {
    let window = query
        .window
        .unwrap_or(DEFAULT_LEADER_WINDOW)
        .clamp(1, MAX_LEADER_WINDOW);
    let (from_block, to_block, counts) = state.reader.leader_counts(window).await?;
    Ok(Json(LeadersResponse {
        window,
        from_block,
        to_block,
        leaders: counts
            .into_iter()
            .map(|(address, blocks)| LeaderEntry {
                address: hex_address(&address),
                blocks,
            })
            .collect(),
    }))
}

/// `POST /call` — generic read-only contract call (`{to, data}`); read-only
/// and idempotent despite the verb (calldata routinely exceeds safe URL
/// lengths, and a JSON body is the eth_call convention). Cost is bounded by
/// tn-reth's inherent 30M gas cap.
async fn call(
    State(state): State<Arc<ApiState>>,
    Json(request): Json<CallRequest>,
) -> ApiResult<CallResponse> {
    let to = parse_address(&request.to)?;
    let raw = request.data.strip_prefix("0x").unwrap_or(&request.data);
    let data =
        hex::decode(raw).map_err(|_| ApiError::BadRequest("malformed hex in data".to_string()))?;
    if data.len() > MAX_CALLDATA_BYTES {
        return Err(ApiError::BadRequest(format!(
            "calldata exceeds {MAX_CALLDATA_BYTES} bytes"
        )));
    }
    let response = match state.reader.call(to, data.into()).await? {
        CallOutcome::Success(result) => CallResponse {
            ok: true,
            result: Some(hex_bytes(&result)),
            error: None,
            revert_data: None,
        },
        CallOutcome::Revert { reason, output } => CallResponse {
            ok: false,
            result: None,
            error: Some(reason.unwrap_or_else(|| "execution reverted".to_string())),
            revert_data: Some(hex_bytes(&output)),
        },
    };
    Ok(Json(response))
}
