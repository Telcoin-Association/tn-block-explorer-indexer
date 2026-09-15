//! The explorer HTTP API: 24 routes served from direct node reads
//! (`node_reads`) and the derived SQLite indexes (`storage`), spawned by the
//! ExEx BEFORE catch-up so `/health` is observable during a long first replay.
//!
//! Error contract: missing resources are `404 {"error":"not found"}`; malformed
//! parameters are 400; internal failures are 500 and logged at `error!`. A
//! hydration failure for an INDEXED pointer is the R11 dangling-pointer
//! invariant violated — a bug by construction (ChainExecuted fires post-commit,
//! so every indexed pointer is hydratable), not load-shedding; it surfaces as
//! 500 + `error!` and is an alert condition.
//!
//! The `/consensus/...` routes (and the consensus enrichment of `/blocks/{n}`
//! and `/epochs`) read the node's consensus DB and epoch records through
//! `NodeReader`'s actor round-trips: message-passing to the DB-owner thread,
//! awaited directly with no blocking permit. Number and epoch bounds are
//! checked against the in-memory tip BEFORE any pack is touched, and an epoch
//! pack this observer does not hold is a normal state, not a fault: the
//! resource is 404 and per-epoch fields are `null` (with a `warn!` from
//! `node_reads`), never 500. Only a pack that exists and cannot be read is an
//! internal error. Execution routes never depend on the consensus DB:
//! `/blocks/{n}` logs and drops `consensus_number` if that lookup fails.

use crate::{
    extract::{decode_consensus_fields, parse_tx_type},
    node_reads::{self, CallOutcome, NodeReader, TxData},
    status::IndexerStatus,
    storage::{self, ConsensusRange, ReadPool, StoredToken, StoredTransfer, TxPointer},
    types::{
        build_api_block, build_api_transaction, build_block_consensus, build_consensus_batches,
        build_consensus_block, build_consensus_header, build_epoch_certificate, build_epoch_record,
        build_reputation_scores, build_token_transfer, clamp_per_page, hex_address, hex_bytes,
        hex_digest, ApiAddress, ApiAuthorityRound, ApiBlock, ApiConsensusBatches,
        ApiConsensusBlock, ApiConsensusEpoch, ApiConsensusHeader, ApiConsensusLatest, ApiEpoch,
        ApiEpochCertificate, ApiEpochData, ApiExecRange, ApiRange, ApiStats, ApiTokenInfo,
        ApiTokenTransfer, ApiTransaction, ApiValidator, CallRequest, CallResponse, Envelope,
        HealthResponse, InputMode, LeaderEntry, LeadersResponse, MAX_PER_PAGE,
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
use tn_types::{hex, Address, EpochCertificate, EpochRecord, TxHash};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tracing::{error, info, warn};

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

/// `?page=&per_page=&type=` for `/txs` and `/address/{addr}/txs`: the paged
/// query plus an optional EIP-2718 type filter.
#[derive(Debug, Deserialize)]
struct TxsQuery {
    page: Option<u64>,
    per_page: Option<u64>,
    /// A `TX_TYPE_NAMES` entry (`legacy`, `eip2930`, `eip1559`, `eip4844`,
    /// `eip7702`, case-insensitive) or its type byte `0..=4`.
    #[serde(rename = "type")]
    tx_type: Option<String>,
}

impl TxsQuery {
    /// The paging part, for [`page_params`].
    fn paging(&self) -> PageQuery {
        PageQuery {
            page: self.page,
            per_page: self.per_page,
        }
    }

    /// The parsed `?type=` filter: `None` when absent (unfiltered), 400 when
    /// present but unknown — a filter never silently falls back to another
    /// type or to the unfiltered feed.
    fn tx_type(&self) -> Result<Option<u8>, ApiError> {
        self.tx_type
            .as_deref()
            .map(|raw| {
                parse_tx_type(raw).ok_or_else(|| {
                    ApiError::BadRequest(format!(
                        "unknown tx type: {raw}; expected \
                         legacy|eip2930|eip1559|eip4844|eip7702 or 0-4"
                    ))
                })
            })
            .transpose()
    }
}

/// Parse a decimal path segment; `what` names it in the 400 message.
fn parse_number(raw: &str, what: &str) -> Result<u64, ApiError> {
    raw.parse()
        .map_err(|_| ApiError::BadRequest(format!("invalid {what}: {raw}")))
}

/// A stored consensus-output block range in wire form.
fn exec_range(range: ConsensusRange) -> ApiExecRange {
    ApiExecRange {
        first: range.first_block,
        last: range.last_block,
    }
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
        .route("/txs/{hash}/transfers", get(tx_transfers))
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
        .route("/consensus/latest", get(consensus_latest))
        .route("/consensus/blocks", get(consensus_blocks_list))
        .route("/consensus/blocks/{number}", get(consensus_block))
        .route(
            "/consensus/blocks/{number}/batches",
            get(consensus_block_batches),
        )
        .route("/consensus/epochs", get(consensus_epochs_list))
        .route("/consensus/epochs/{n}", get(consensus_epoch))
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
/// With `?type=`, one SQLite read of the `tx_types` index (page + O(1) total
/// from `tx_type_counts`) hydrated exactly like an address page.
async fn txs_list(
    State(state): State<Arc<ApiState>>,
    Query(query): Query<TxsQuery>,
) -> ApiResult<Envelope<ApiTransaction>> {
    let (page, per_page, offset) = page_params(&query.paging());
    let Some(tx_type) = query.tx_type()? else {
        let (rows, total) = state.reader.latest_txs_page(page, per_page).await?;
        let items = rows
            .iter()
            .map(|data| build_api_transaction(data, InputMode::List))
            .collect();
        return Ok(Json(Envelope {
            items,
            total,
            page,
            per_page,
        }));
    };
    let (pointers, total) = state
        .pool
        .with_conn(move |conn| {
            Ok((
                storage::tx_type_page(conn, tx_type, per_page, offset)?,
                storage::count_tx_type(conn, tx_type)?,
            ))
        })
        .await?;
    let items = hydrate_tx_page(&state, pointers).await?;
    Ok(Json(Envelope {
        items,
        total,
        page,
        per_page,
    }))
}

/// `GET /txs/{hash}` — mined transactions only (full input hex), carrying
/// the first [`MAX_PER_PAGE`] ERC-20 transfers the transaction emitted and
/// their total from one SQLite read (the paginated feed is
/// `/txs/{hash}/transfers`).
async fn tx_by_hash(
    State(state): State<Arc<ApiState>>,
    Path(raw): Path<String>,
) -> ApiResult<ApiTransaction> {
    let hash = parse_hash(&raw)?;
    let Some(data) = state.reader.tx_by_hash(hash).await? else {
        return Err(ApiError::NotFound);
    };
    let (transfers, count) = tx_transfer_page(&state, &data, MAX_PER_PAGE, 0).await?;
    let mut item = build_api_transaction(&data, InputMode::Detail);
    item.token_transfers = Some(transfers);
    item.token_transfer_count = Some(count);
    Ok(Json(item))
}

/// `GET /txs/{hash}/transfers` — the ERC-20 transfers one transaction
/// emitted in emission order (`log_index` ascending): one node read for the
/// transaction, one SQLite read for the page.
async fn tx_transfers(
    State(state): State<Arc<ApiState>>,
    Path(raw): Path<String>,
    Query(query): Query<PageQuery>,
) -> ApiResult<Envelope<ApiTokenTransfer>> {
    let hash = parse_hash(&raw)?;
    let (page, per_page, offset) = page_params(&query);
    let Some(data) = state.reader.tx_by_hash(hash).await? else {
        return Err(ApiError::NotFound);
    };
    let (items, total) = tx_transfer_page(&state, &data, per_page, offset).await?;
    Ok(Json(Envelope {
        items,
        total,
        page,
        per_page,
    }))
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

/// `GET /blocks/{number}` — one node read, plus one consensus actor read
/// resolving `consensus.consensus_number` from the header's digest. That
/// lookup is best-effort: a consensus DB failure is logged at `warn!` and
/// leaves the field `null` rather than failing an execution route.
async fn block_by_number(
    State(state): State<Arc<ApiState>>,
    Path(raw): Path<String>,
) -> ApiResult<ApiBlock> {
    let number = parse_number(&raw, "block number")?;
    let Some(block) = state.reader.block_by_number(number).await? else {
        return Err(ApiError::NotFound);
    };
    let mut item = build_api_block(&block);
    if let (Some(consensus), Some(fields)) = (
        item.consensus.as_mut(),
        decode_consensus_fields(block.header()),
    ) {
        match state
            .reader
            .consensus_number_by_digest(fields.epoch, fields.digest)
            .await
        {
            Ok(resolved) => consensus.consensus_number = resolved,
            Err(err) => warn!(
                target: "indexer::api",
                number,
                ?err,
                "consensus number unresolved; serving block without it"
            ),
        }
    }
    Ok(Json(item))
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

/// `GET /address/{addr}/txs` — SQLite reverse-PK scan (`?type=` adds
/// `AND tx_type = ?` to both the page and the count), hydrated from the
/// node's DB (one replay per distinct block).
async fn address_txs(
    State(state): State<Arc<ApiState>>,
    Path(raw): Path<String>,
    Query(query): Query<TxsQuery>,
) -> ApiResult<Envelope<ApiTransaction>> {
    let address = parse_address(&raw)?;
    let key = storage::addr_hex(&address);
    let (page, per_page, offset) = page_params(&query.paging());
    let tx_type = query.tx_type()?;
    let (pointers, total) = state
        .pool
        .with_conn(move |conn| {
            Ok(match tx_type {
                None => (
                    storage::address_txs_page(conn, &key, per_page, offset)?,
                    storage::count_address_txs(conn, &key)?,
                ),
                Some(tx_type) => (
                    storage::address_txs_page_by_type(conn, &key, tx_type, per_page, offset)?,
                    storage::count_address_txs_by_type(conn, &key, tx_type)?,
                ),
            })
        })
        .await?;
    let items = hydrate_tx_page(&state, pointers).await?;
    Ok(Json(Envelope {
        items,
        total,
        page,
        per_page,
    }))
}

/// Hydrate one page of index pointers from the node's DB (one replay per
/// distinct block) into list rows.
async fn hydrate_tx_page(
    state: &ApiState,
    pointers: Vec<TxPointer>,
) -> Result<Vec<ApiTransaction>, ApiError> {
    let hydrated = state.reader.hydrate_pointers(pointers).await?;
    Ok(hydrated
        .iter()
        .map(|data| build_api_transaction(data, InputMode::List))
        .collect())
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

/// One page of a transaction's ERC-20 transfers (emission order) and their
/// total, in a single pooled read: the rows, the count and the cached
/// metadata of every token in the page. The caller already holds the
/// hydrated `TxData`, so no node read is needed to render the rows.
async fn tx_transfer_page(
    state: &ApiState,
    data: &TxData,
    limit: u64,
    offset: u64,
) -> Result<(Vec<ApiTokenTransfer>, u64), ApiError> {
    let (block_number, tx_index) = (data.block_number, data.tx_index);
    let (rows, total, tokens): TransferPage = state
        .pool
        .with_conn(move |conn| {
            let rows = storage::transfers_for_tx_page(conn, block_number, tx_index, limit, offset)?;
            let total = storage::count_transfers_for_tx(conn, block_number, tx_index)?;
            let tokens = page_tokens(conn, &rows)?;
            Ok((rows, total, tokens))
        })
        .await?;
    let items = rows
        .iter()
        .map(|row| build_token_transfer(row, data, tokens.get(&row.token)))
        .collect();
    Ok((items, total))
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

/// Committee BLS keys (base58) for an epoch: the record's own list, else
/// `committee_keys(epoch)` — the in-progress epoch, seated by the previous
/// record's `next_committee` — else empty.
async fn committee_bls(
    reader: &NodeReader,
    epoch: u32,
    record: Option<&EpochRecord>,
) -> Vec<String> {
    match record {
        Some(record) => record.committee.iter().map(ToString::to_string).collect(),
        None => reader
            .committee_keys(epoch)
            .await
            .map(|keys| keys.iter().map(ToString::to_string).collect())
            .unwrap_or_default(),
    }
}

/// The certificate over `record` in wire form. `verify` runs the BLS pairing
/// check (detail routes only); `verified` stays `null` on lists.
async fn api_certificate(
    reader: &NodeReader,
    record: &EpochRecord,
    certificate: Option<EpochCertificate>,
    verify: bool,
) -> Result<Option<ApiEpochCertificate>, ApiError> {
    let Some(certificate) = certificate else {
        return Ok(None);
    };
    let verified = if verify {
        Some(
            reader
                .verify_epoch_certificate(record.clone(), certificate.clone())
                .await?,
        )
    } else {
        None
    };
    Ok(Some(build_epoch_certificate(
        record,
        &certificate,
        verified,
    )))
}

/// Build one `/epochs` row (without `end_time`, which needs a header read the
/// caller batches) from the epoch's record pair. `verify` BLS-checks the
/// certificate (`/epochs/{n}` only).
///
/// The in-progress epoch is synthesized: `latest_consensus_epoch()` is the
/// epoch of the last PROCESSED output, so between an epoch's closing output
/// and the first output of the next its record can already exist while it is
/// still "current". That record is deliberately not shown until the epoch
/// stops being current, so `is_current` and `end_block` never disagree.
async fn epoch_row(
    reader: &NodeReader,
    epoch: u64,
    current: u64,
    verify: bool,
) -> Result<ApiEpoch, ApiError> {
    let is_current = epoch == current;
    let (this, previous) = reader.epoch_record_pair(epoch_u32(epoch)).await;
    let (record, certificate) = match this.filter(|_| !is_current) {
        Some((record, certificate)) => (Some(record), certificate),
        None => (None, None),
    };
    // epoch 0 starts at block 0; a missing predecessor record degrades to 0
    // rather than failing the page
    let start_block = previous.map_or(0, |previous| previous.final_state.number.saturating_add(1));
    let committee_bls = committee_bls(reader, epoch_u32(epoch), record.as_ref()).await;
    let certificate = match &record {
        Some(record) => api_certificate(reader, record, certificate, verify).await?,
        None => None,
    };
    Ok(ApiEpoch {
        epoch,
        start_block,
        end_block: record.as_ref().map(|record| record.final_state.number),
        end_time: None,
        committee_size: committee_bls.len(),
        committee_bls,
        certified: certificate.is_some(),
        is_current,
        committee_addresses: None,
        record: record.as_ref().map(build_epoch_record),
        certificate,
    })
}

/// `GET /epochs` — newest-first epoch history from the consensus DB's epoch
/// records; the current epoch is synthesized (no record exists yet).
async fn epochs_list(
    State(state): State<Arc<ApiState>>,
    Query(query): Query<PageQuery>,
) -> ApiResult<Envelope<ApiEpoch>> {
    let (page, per_page, _) = page_params(&query);
    let current = u64::from(state.reader.latest_consensus_epoch());
    let total = current.saturating_add(1);
    let numbers = node_reads::desc_page_items(total, page, per_page);

    let mut items = Vec::with_capacity(numbers.len());
    for number in numbers {
        items.push(epoch_row(&state.reader, number, current, false).await?);
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

/// `GET /epochs/{n}` — an epoch record + its certificate (BLS-verified
/// here), with committee ADDRESSES read from the registry pinned to the block
/// that seated `n`'s committee. `null` only when the predecessor epoch record
/// or the pin block itself is missing locally (BLS keys are always available).
async fn epoch_by_number(
    State(state): State<Arc<ApiState>>,
    Path(raw): Path<String>,
) -> ApiResult<ApiEpoch> {
    let number = parse_number(&raw, "epoch number")?;
    let current = u64::from(state.reader.latest_consensus_epoch());
    if number > current {
        return Err(ApiError::NotFound);
    }
    let mut item = epoch_row(&state.reader, number, current, true).await?;
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
    let infos = state.reader.current_committee_validators().await?;
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

// ---------------------------------------------------------------------------
// Consensus handlers (`/consensus/...`): actor round-trips, no permit
// ---------------------------------------------------------------------------

/// The execution blocks one consensus output produced, from the indexer's
/// `consensus_blocks` table keyed by the output's wire digest (one pooled
/// read). `None` = not indexed yet, or an empty non-closing output.
async fn consensus_exec_range(
    state: &ApiState,
    digest: String,
) -> Result<Option<ApiExecRange>, ApiError> {
    Ok(state
        .pool
        .with_conn(move |conn| {
            Ok(storage::consensus_range_by_digest(conn, &digest)?.map(exec_range))
        })
        .await?)
}

/// `GET /consensus/latest` — the newest consensus header (one actor read)
/// next to the canonical execution tip (one blocking read), so a client can
/// see how far execution trails consensus. 404 before the first output.
async fn consensus_latest(State(state): State<Arc<ApiState>>) -> ApiResult<ApiConsensusLatest> {
    let Some(latest) = state.reader.consensus_latest_header().await? else {
        return Err(ApiError::NotFound);
    };
    let tip = state.reader.tip_header().await?;
    // the same accessors and encodings as every other header on the wire
    let header = build_consensus_header(&latest, None);
    Ok(Json(ApiConsensusLatest {
        number: header.number,
        epoch: header.epoch,
        round: header.round,
        digest: header.digest,
        digest_bs58: header.digest_bs58,
        leader: header.leader,
        committed_at: header.committed_at,
        exec_tip: tip.header().number,
        exec_tip_consensus: decode_consensus_fields(tip.header())
            .map(|fields| build_block_consensus(&fields, None)),
    }))
}

/// `GET /consensus/blocks` — newest-first consensus headers (`total =
/// latest_consensus_number()`): `per_page` sequential actor header reads,
/// then ONE SQLite read for every header's execution-block range. Headers
/// this observer lacks are skipped and the page shrinks (see
/// `NodeReader::consensus_headers_page`).
async fn consensus_blocks_list(
    State(state): State<Arc<ApiState>>,
    Query(query): Query<PageQuery>,
) -> ApiResult<Envelope<ApiConsensusHeader>> {
    let (page, per_page, _) = page_params(&query);
    let (headers, total) = state.reader.consensus_headers_page(page, per_page).await?;
    let mut items: Vec<ApiConsensusHeader> = headers
        .iter()
        .map(|header| build_consensus_header(header, None))
        .collect();
    // execution ranges for the whole page in ONE pooled read, keyed by the
    // wire digest (the `consensus_blocks` key encoding)
    if !items.is_empty() {
        let digests: Vec<String> = items.iter().map(|item| item.digest.clone()).collect();
        let ranges = state
            .pool
            .with_conn(move |conn| Ok(storage::consensus_ranges_by_digests(conn, &digests)?))
            .await?;
        for item in &mut items {
            item.exec_blocks = ranges.get(&item.digest).copied().map(exec_range);
        }
    }
    Ok(Json(Envelope {
        items,
        total,
        page,
        per_page,
    }))
}

/// `GET /consensus/blocks/{number}` — one FULL output read (header, sub-dag
/// and every batch's raw transactions: one actor round-trip), one SQLite
/// read for the execution range, one `record_by_epoch` for `closes_epoch`
/// and one `committee_keys` to resolve leader/author BLS keys. Batch
/// summaries only; per-transaction hashes live on `/batches`.
async fn consensus_block(
    State(state): State<Arc<ApiState>>,
    Path(raw): Path<String>,
) -> ApiResult<ApiConsensusBlock> {
    let number = parse_number(&raw, "consensus block number")?;
    let Some(output) = state.reader.consensus_output(number).await? else {
        return Err(ApiError::NotFound);
    };
    // the output carries every header field (`consensus_header()` rebuilds it
    // over the Arc-backed sub-dag), so no second actor read is needed
    let mut header = build_consensus_header(&output.consensus_header(), None);
    let exec_blocks = consensus_exec_range(&state, header.digest.clone()).await?;
    let batches = build_consensus_batches(&output, exec_blocks.as_ref(), false);
    header.exec_blocks = exec_blocks;
    let epoch = header.epoch;
    // `ConsensusOutput::close_epoch` is not serialized: the epoch record is
    // the only truth, and `null` until it exists
    let closes_epoch = state
        .reader
        .consensus()
        .epochs()
        .record_by_epoch(epoch)
        .await
        .map(|record| record.final_consensus.number == number);
    let committee = state.reader.committee_keys(epoch).await;
    Ok(Json(build_consensus_block(
        header,
        &output,
        committee.as_ref(),
        batches,
        closes_epoch,
    )))
}

/// `GET /consensus/blocks/{number}/batches` — the same full output read as
/// the detail route, rendered as batches WITH per-transaction hashes
/// (`keccak256` of each raw transaction), plus one SQLite read to number the
/// execution block each batch became.
async fn consensus_block_batches(
    State(state): State<Arc<ApiState>>,
    Path(raw): Path<String>,
) -> ApiResult<ApiConsensusBatches> {
    let number = parse_number(&raw, "consensus block number")?;
    let Some(output) = state.reader.consensus_output(number).await? else {
        return Err(ApiError::NotFound);
    };
    let digest = hex_digest(&output.consensus_header_hash());
    let exec_blocks = consensus_exec_range(&state, digest.clone()).await?;
    Ok(Json(ApiConsensusBatches {
        number: output.number(),
        digest,
        batches: build_consensus_batches(&output, exec_blocks.as_ref(), true),
    }))
}

/// Build one `/consensus/epochs` row from the epoch's record pair (two to
/// three actor reads). `detail` BLS-verifies the certificate and adds the
/// fields that cost more reads — `committee_addresses` (a pinned registry
/// read), `pack_complete`, `last_committed_rounds` and
/// `final_reputation_scores` (one pack read each, `null` when the pack is
/// not local); lists leave them `null`. `end_time` is left for the caller to
/// batch.
async fn consensus_epoch_row(
    reader: &NodeReader,
    epoch: u32,
    detail: bool,
) -> Result<ApiConsensusEpoch, ApiError> {
    let (this, previous) = reader.epoch_record_pair(epoch).await;
    let is_current = epoch == reader.latest_consensus_epoch();
    let (record, certificate) = match this {
        Some((record, certificate)) => (Some(record), certificate),
        None => (None, None),
    };
    let certificate = match &record {
        Some(record) => api_certificate(reader, record, certificate, detail).await?,
        None => None,
    };
    let committee_bls = committee_bls(reader, epoch, record.as_ref()).await;
    let mut item = ApiConsensusEpoch {
        epoch: u64::from(epoch),
        is_current,
        record: record.as_ref().map(build_epoch_record),
        certificate,
        // the first stored output is 1; epoch 0 starts at block 0
        consensus_range: ApiRange {
            first: previous.as_ref().map_or(1, |previous| {
                previous.final_consensus.number.saturating_add(1)
            }),
            last: record.as_ref().map(|record| record.final_consensus.number),
        },
        exec_range: ApiRange {
            first: previous
                .as_ref()
                .map_or(0, |previous| previous.final_state.number.saturating_add(1)),
            last: record.as_ref().map(|record| record.final_state.number),
        },
        end_time: None,
        committee_bls,
        committee_addresses: None,
        pack_complete: None,
        last_committed_rounds: None,
        final_reputation_scores: None,
    };
    if detail {
        item.committee_addresses = reader
            .committee_addresses(epoch)
            .await
            .map(|addresses| addresses.iter().map(hex_address).collect());
        item.pack_complete = match &record {
            Some(record) => Some(reader.epoch_pack_complete(record).await),
            None => None,
        };
        item.last_committed_rounds = reader.epoch_last_committed(epoch).await.map(|rounds| {
            rounds
                .into_iter()
                .map(|(authority, round)| ApiAuthorityRound {
                    authority: authority.to_string(),
                    round,
                })
                .collect()
        });
        item.final_reputation_scores = reader
            .epoch_final_reputation(epoch)
            .await
            .map(|scores| build_reputation_scores(&scores));
    }
    Ok(item)
}

/// `GET /consensus/epochs` — newest-first over `0..=latest_consensus_epoch()`
/// (`total = latest + 1`): two to three actor reads per row, then ONE
/// blocking read for the page's end-block timestamps. Detail-only fields
/// are `null`.
async fn consensus_epochs_list(
    State(state): State<Arc<ApiState>>,
    Query(query): Query<PageQuery>,
) -> ApiResult<Envelope<ApiConsensusEpoch>> {
    let (page, per_page, _) = page_params(&query);
    let total = u64::from(state.reader.latest_consensus_epoch()).saturating_add(1);
    let numbers = node_reads::desc_page_items(total, page, per_page);

    let mut items = Vec::with_capacity(numbers.len());
    for number in numbers {
        items.push(consensus_epoch_row(&state.reader, epoch_u32(number), false).await?);
    }

    // boundary timestamps for the whole page in ONE blocking read
    let end_blocks: Vec<u64> = items
        .iter()
        .filter_map(|item| item.exec_range.last)
        .collect();
    if !end_blocks.is_empty() {
        let times = state.reader.header_timestamps(end_blocks).await?;
        for item in &mut items {
            item.end_time = item
                .exec_range
                .last
                .and_then(|end| times.get(&end).copied());
        }
    }
    Ok(Json(Envelope {
        items,
        total,
        page,
        per_page,
    }))
}

/// `GET /consensus/epochs/{n}` — the fully populated epoch row (~8 reads:
/// the record pair, BLS certificate verification, the pinned committee
/// registry read, three pack reads and the end-block timestamp). Epochs past
/// the current one are 404 before any read.
async fn consensus_epoch(
    State(state): State<Arc<ApiState>>,
    Path(raw): Path<String>,
) -> ApiResult<ApiConsensusEpoch> {
    let number = parse_number(&raw, "epoch number")?;
    if number > u64::from(state.reader.latest_consensus_epoch()) {
        return Err(ApiError::NotFound);
    }
    let mut item = consensus_epoch_row(&state.reader, epoch_u32(number), true).await?;
    if let Some(end_block) = item.exec_range.last {
        let times = state.reader.header_timestamps(vec![end_block]).await?;
        item.end_time = times.get(&end_block).copied();
    }
    Ok(Json(item))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn txs_query(tx_type: Option<&str>) -> TxsQuery {
        TxsQuery {
            page: None,
            per_page: None,
            tx_type: tx_type.map(str::to_string),
        }
    }

    /// `?type=` is optional, accepts names and digits, and rejects anything
    /// else with a 400 naming the value — never a silent fallback.
    #[test]
    fn txs_query_type_filter() {
        assert!(matches!(txs_query(None).tx_type(), Ok(None)));
        assert!(matches!(txs_query(Some("EIP1559")).tx_type(), Ok(Some(2))));
        assert!(matches!(txs_query(Some("0")).tx_type(), Ok(Some(0))));
        match txs_query(Some("blob")).tx_type() {
            Err(ApiError::BadRequest(message)) => {
                assert!(message.starts_with("unknown tx type: blob;"), "{message}");
            }
            other => panic!("expected 400, got {other:?}"),
        }
    }

    /// Path numbers keep the pre-existing 400 wording (`invalid <what>: <raw>`).
    #[test]
    fn parse_number_messages() {
        assert!(matches!(parse_number("42", "block number"), Ok(42)));
        match parse_number("-1", "epoch number") {
            Err(ApiError::BadRequest(message)) => {
                assert_eq!(message, "invalid epoch number: -1");
            }
            other => panic!("expected 400, got {other:?}"),
        }
    }
}
