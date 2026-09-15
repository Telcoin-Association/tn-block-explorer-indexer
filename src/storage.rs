//! SQLite persistence: exactly the derived indexes reth maintains in no table,
//! plus a cursor — nothing else.
//!
//! Tables (schema v3): `meta` (cursor, schema version, chain id), `address_txs`
//! (address → typed transaction pointers), `tx_types` (transaction type →
//! pointers) with `tx_type_counts` (exact per-type totals), `consensus_blocks`
//! (consensus output digest → execution block range), `token_transfers`
//! (ERC-20 transfer history), `tokens` (ERC-20 metadata cache). Blocks,
//! transactions, and receipts deliberately have NO tables here — they are
//! served by direct reads on the node's own databases (`node_reads`).
//!
//! # Consistency & disposability
//!
//! One SQLite transaction per block: pointer rows, type rows and counters, the
//! consensus range upsert, transfer rows, token upserts, and the cursor advance
//! commit atomically, so a `kill -9` loses rows and cursor **together** (WAL
//! rollback) and restart replays from `cursor + 1` with no holes. `INSERT OR
//! IGNORE` makes re-indexing the same block idempotent, and TN has no reorgs so
//! keys are permanently stable. The whole file is derived state: a
//! `schema_version` mismatch (or leftover v1 tables) drops every table and
//! re-replays from 0 — delete-the-file is the supported migration AND
//! corruption story.
//!
//! # Handles
//!
//! [`Writer`] is ONE connection in an `Arc<Mutex<_>>` used through
//! `tokio::task::spawn_blocking`; only the ExEx loop writes (single-writer
//! discipline — API handlers never upsert). [`ReadPool`] hands out 4 read-only
//! connections to the API behind a tokio [`Semaphore`].
//!
//! Postgres later = swap this file only (`value TEXT` → `NUMERIC(78,0)`,
//! `INSERT OR IGNORE` → `ON CONFLICT DO NOTHING`, `WITHOUT ROWID` → plain PK
//! tables).

use crate::extract::ExtractedBlock;
use eyre::{bail, eyre, WrapErr};
use rusqlite::{Connection, OpenFlags, OptionalExtension, TransactionBehavior};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tn_types::{Address, B256};
use tokio::sync::Semaphore;

/// The schema version this build writes and requires.
///
/// v3 added `address_txs.tx_type`, `tx_types`, `tx_type_counts`, and
/// `consensus_blocks`. A v2 file is dropped and replayed from 0 (see [`migrate`]).
const SCHEMA_VERSION: &str = "3";

/// Number of read-only connections in the [`ReadPool`].
const READ_POOL_SIZE: usize = 4;

/// Schema v3 DDL. `WITHOUT ROWID` keeps the composite primary keys clustered;
/// all hex values are lowercase `0x`-prefixed (the explorer compares with
/// `.to_lowercase()`).
const DDL: &str = "
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
-- rows: ('schema_version','3'), ('last_indexed','<u64>'), ('chain_id','<u64>')
-- 'last_indexed' ABSENT means fresh DB (replay from 0).

-- (a) address -> native-tx participation index. Pointer rows ONLY - no tx content.
CREATE TABLE IF NOT EXISTS address_txs (
    address      TEXT    NOT NULL,
    block_number INTEGER NOT NULL,
    tx_index     INTEGER NOT NULL,
    tx_type      INTEGER NOT NULL,
    PRIMARY KEY (address, block_number, tx_index)
) WITHOUT ROWID;
-- Newest-first pagination is a reverse scan of the composite PK; no secondary
-- index needed. Self-send (from==to) collapses onto one PK via INSERT OR IGNORE.
-- tx_type (the EIP-2718 type byte) is a payload column: `?type=` on an address
-- page filters that address's PK range, so no second index is needed.

-- (b) transaction type -> pointer index (`GET /txs?type=`).
CREATE TABLE IF NOT EXISTS tx_types (
    tx_type      INTEGER NOT NULL,
    block_number INTEGER NOT NULL,
    tx_index     INTEGER NOT NULL,
    PRIMARY KEY (tx_type, block_number, tx_index)
) WITHOUT ROWID;
-- Exact per-type totals maintained inside the block transaction; COUNT(*) over
-- tx_types would scan a whole type (millions of rows) on every list request.
CREATE TABLE IF NOT EXISTS tx_type_counts (
    tx_type INTEGER PRIMARY KEY,
    count   INTEGER NOT NULL
) WITHOUT ROWID;

-- (c) consensus output digest -> execution block range, decoded from each exec
-- header's parent_beacon_block_root (zero ConsensusChain reads while indexing).
CREATE TABLE IF NOT EXISTS consensus_blocks (
    digest      TEXT    PRIMARY KEY,
    epoch       INTEGER NOT NULL,
    round       INTEGER NOT NULL,
    first_block INTEGER NOT NULL,
    last_block  INTEGER NOT NULL
) WITHOUT ROWID;
-- digest: lowercase 0x-prefixed hex of the 32-byte digest (66 chars, see
-- `digest_hex`). Genesis names no output and has no row.

-- (d) ERC-20 transfer index, extracted from receipt logs.
CREATE TABLE IF NOT EXISTS token_transfers (
    block_number INTEGER NOT NULL,
    tx_index     INTEGER NOT NULL,
    log_index    INTEGER NOT NULL,
    token        TEXT    NOT NULL,
    from_addr    TEXT    NOT NULL,
    to_addr      TEXT    NOT NULL,
    value        TEXT    NOT NULL,
    PRIMARY KEY (block_number, tx_index, log_index)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_tt_token ON token_transfers(token,     block_number, tx_index, log_index);
CREATE INDEX IF NOT EXISTS idx_tt_from  ON token_transfers(from_addr, block_number, tx_index, log_index);
CREATE INDEX IF NOT EXISTS idx_tt_to    ON token_transfers(to_addr,   block_number, tx_index, log_index);
-- Per-tx transfers (`/txs/{hash}/transfers`) are a PK-prefix scan on
-- (block_number, tx_index); no extra index.

-- (e) token metadata cache. Written ONLY by the indexing path.
CREATE TABLE IF NOT EXISTS tokens (
    address    TEXT PRIMARY KEY,
    name       TEXT,
    symbol     TEXT,
    decimals   INTEGER,
    status     INTEGER NOT NULL,
    fetched_at INTEGER NOT NULL
) WITHOUT ROWID;
-- status: 0=ok, 1=retry_pending (one failed attempt), 2=failed (terminal).
-- totalSupply is NEVER cached (mutable) - served live via read_contract.
";

/// Lowercase `0x`-prefixed hex for an address — the storage key convention.
pub fn addr_hex(address: &Address) -> String {
    format!("{address:#x}")
}

/// Lowercase `0x`-prefixed hex for a 32-byte digest (`0x` + 64 hex chars) —
/// the `consensus_blocks.digest` key. This is the same encoding as
/// `types::hex_b256`, so a header's `parent_beacon_block_root` formatted by
/// either helper is a valid lookup key.
pub fn digest_hex(digest: &B256) -> String {
    format!("{digest:#x}")
}

/// One token-metadata row handed to [`Writer::index_block`] by the ExEx loop
/// after a fetch attempt.
///
/// The loop only ever hands in `status` 0 (fetch decoded at least one field;
/// per-field `None`s allowed) or 1 (fetch failed). The upsert escalates a
/// failure on an already-failed row to the terminal status 2.
#[derive(Debug, Clone)]
pub struct TokenRow {
    /// The token contract.
    pub address: Address,
    /// Decoded `name()`, if any.
    pub name: Option<String>,
    /// Decoded `symbol()`, if any.
    pub symbol: Option<String>,
    /// Decoded `decimals()`, if any.
    pub decimals: Option<u8>,
    /// 0 = fetch ok, 1 = fetch failed (storage escalates 1-on-1 to 2).
    pub status: u8,
    /// Unix seconds of this fetch attempt.
    pub fetched_at: u64,
}

/// A `(block_number, tx_index)` pointer read back from `address_txs` or
/// `tx_types`, hydrated into full transaction data by `node_reads`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TxPointer {
    /// Block containing the transaction.
    pub block_number: u64,
    /// Zero-based index of the transaction within its block.
    pub tx_index: u64,
}

/// One `token_transfers` row as stored (hex strings + exact decimal value).
#[derive(Debug, Clone)]
pub struct StoredTransfer {
    /// Block containing the transaction.
    pub block_number: u64,
    /// Zero-based index of the transaction within its block.
    pub tx_index: u64,
    /// Position of the log within THIS transaction's receipt logs (not the RPC
    /// `logIndex`). Orders the per-tx transfer list and is exposed on the wire
    /// so the explorer can show transfers in emission order.
    pub log_index: u64,
    /// Emitting token contract (lowercase hex).
    pub token: String,
    /// Sender (lowercase hex).
    pub from: String,
    /// Recipient (lowercase hex).
    pub to: String,
    /// Exact U256 decimal string.
    pub value: String,
}

/// One `tokens` cache row as stored.
#[derive(Debug, Clone)]
pub struct StoredToken {
    /// Cached `name()`, if decoded.
    pub name: Option<String>,
    /// Cached `symbol()`, if decoded.
    pub symbol: Option<String>,
    /// Cached `decimals()`, if decoded.
    pub decimals: Option<u8>,
    /// 0 = ok, 1 = retry pending, 2 = failed (terminal).
    pub status: u8,
}

/// The inclusive range of execution blocks one consensus output produced, read
/// back from `consensus_blocks`.
///
/// One block per batch in the output, so `first_block == last_block` for a
/// single-batch output and for the empty epoch-closing block. The range only
/// widens as blocks are applied; while an output is still being indexed the
/// stored `last_block` lags the true one (see `/health.last_indexed`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConsensusRange {
    /// Lowest execution block number produced by the output.
    pub first_block: u64,
    /// Highest execution block number produced by the output (so far).
    pub last_block: u64,
}

/// The single writing handle: ONE connection, used only by the ExEx loop.
#[derive(Debug, Clone)]
pub struct Writer {
    conn: Arc<Mutex<Connection>>,
}

impl Writer {
    /// Open (creating parent directories), apply PRAGMAs, run migrations
    /// (drop-and-rebuild on a schema-version mismatch or leftover v1 tables),
    /// and verify the stored chain id against the node's chainspec.
    pub async fn open(path: PathBuf, chain_id: u64) -> eyre::Result<Self> {
        tokio::task::spawn_blocking(move || {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .wrap_err("failed to create explorer-indexer db directory")?;
            }
            let conn = Connection::open(&path)
                .wrap_err_with(|| format!("failed to open sqlite db at {}", path.display()))?;
            configure(&conn)?;
            migrate(&conn, chain_id)?;
            Ok(Self {
                conn: Arc::new(Mutex::new(conn)),
            })
        })
        .await?
    }

    /// Run `f` on the writer connection on the blocking pool.
    async fn on_writer<T, F>(&self, f: F) -> eyre::Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut Connection) -> eyre::Result<T> + Send + 'static,
    {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = conn.lock().map_err(|_| eyre!("writer mutex poisoned"))?;
            f(&mut guard)
        })
        .await?
    }

    /// The highest durably indexed block, if any (`None` = fresh DB, replay from 0).
    pub async fn last_indexed(&self) -> eyre::Result<Option<u64>> {
        self.on_writer(|conn| read_cursor(conn)).await
    }

    /// Which of `candidates` need a metadata (re)fetch: unseen tokens and
    /// `status = 1` (retry pending). Never `status` 0 (ok) or 2 (terminal).
    pub async fn token_fetch_states(&self, candidates: Vec<Address>) -> eyre::Result<Vec<Address>> {
        self.on_writer(move |conn| {
            let mut out = Vec::new();
            let mut stmt = conn.prepare_cached("SELECT status FROM tokens WHERE address = ?1")?;
            for candidate in candidates {
                let status: Option<u8> = stmt
                    .query_row([addr_hex(&candidate)], |row| row.get(0))
                    .optional()?;
                if matches!(status, None | Some(1)) {
                    out.push(candidate);
                }
            }
            Ok(out)
        })
        .await
    }

    /// Index one block in ONE SQLite transaction: in-transaction cursor guard,
    /// pointer + type + consensus + transfer rows, token upserts, cursor
    /// advance, commit.
    ///
    /// Returns `false` when the block is at or below the cursor (skipped) —
    /// the authoritative monotonic guard for the replay/live overlap.
    pub async fn index_block(
        &self,
        extracted: ExtractedBlock,
        token_rows: Vec<TokenRow>,
    ) -> eyre::Result<bool> {
        self.on_writer(move |conn| {
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let indexed = apply_block(&tx, &extracted, &token_rows)?;
            if indexed {
                tx.commit()?;
            } else {
                tx.rollback()?;
            }
            Ok(indexed)
        })
        .await
    }
}

/// Apply PRAGMAs: WAL journaling, NORMAL sync (a crash rolls back to the last
/// commit — never corrupts; cursor+rows commit atomically so rollback just
/// means re-replay), a 5s busy timeout, and foreign keys.
fn configure(conn: &Connection) -> eyre::Result<()> {
    let _mode: String = conn.query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.busy_timeout(std::time::Duration::from_millis(5000))?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    Ok(())
}

/// Create or rebuild the schema and enforce the chain-id guard.
///
/// The DB is disposable derived state: a `schema_version` ≠ 3 (v2 lacks
/// `address_txs.tx_type` and the type/consensus tables; v1 left `blocks`/
/// `transactions` tables) drops every table so the indexer re-replays from 0.
/// A stored chain id different from the node's chainspec is a hard error
/// (guards against pointing an old sqlite file at another network's datadir).
fn migrate(conn: &Connection, chain_id: u64) -> eyre::Result<()> {
    let meta_exists = table_exists(conn, "meta")?;
    let v1_tables = table_exists(conn, "blocks")? || table_exists(conn, "transactions")?;
    let version = if meta_exists {
        read_meta(conn, "schema_version")?
    } else {
        None
    };

    if (meta_exists && version.as_deref() != Some(SCHEMA_VERSION)) || v1_tables {
        conn.execute_batch(
            "DROP TABLE IF EXISTS address_txs;
             DROP TABLE IF EXISTS tx_types;
             DROP TABLE IF EXISTS tx_type_counts;
             DROP TABLE IF EXISTS consensus_blocks;
             DROP TABLE IF EXISTS token_transfers;
             DROP TABLE IF EXISTS tokens;
             DROP TABLE IF EXISTS meta;
             DROP TABLE IF EXISTS blocks;
             DROP TABLE IF EXISTS transactions;",
        )
        .wrap_err("failed to drop outdated explorer-indexer schema")?;
    }

    conn.execute_batch(DDL)
        .wrap_err("failed to create explorer-indexer schema")?;
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?1)",
        [SCHEMA_VERSION],
    )?;

    match read_meta(conn, "chain_id")? {
        None => {
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('chain_id', ?1)",
                [chain_id.to_string()],
            )?;
        }
        Some(stored) => {
            let stored_id: u64 = stored
                .parse()
                .map_err(|_| eyre!("stored chain_id {stored:?} is not a valid u64"))?;
            if stored_id != chain_id {
                bail!(
                    "explorer-indexer db belongs to chain {stored_id} but the node runs chain \
                     {chain_id}; delete the db file to re-index this network"
                );
            }
        }
    }
    Ok(())
}

fn table_exists(conn: &Connection, name: &str) -> eyre::Result<bool> {
    let found: Option<i64> = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1",
            [name],
            |row| row.get(0),
        )
        .optional()?;
    Ok(found.is_some())
}

fn read_meta(conn: &Connection, key: &str) -> eyre::Result<Option<String>> {
    Ok(conn
        .query_row("SELECT value FROM meta WHERE key = ?1", [key], |row| {
            row.get(0)
        })
        .optional()?)
}

/// Read the cursor (`meta.last_indexed`) — called INSIDE the block transaction
/// so the guard is authoritative, not advisory.
fn read_cursor(conn: &Connection) -> eyre::Result<Option<u64>> {
    match read_meta(conn, "last_indexed")? {
        None => Ok(None),
        Some(raw) => {
            Ok(Some(raw.parse().map_err(|_| {
                eyre!("corrupt last_indexed value {raw:?}")
            })?))
        }
    }
}

/// Run every statement of the per-block write EXCEPT the commit, against an
/// open transaction. Split out so tests can prove that an error before commit
/// moves neither rows nor cursor.
///
/// Returns `false` (skip) when `extracted.number` is at or below the cursor.
fn apply_block(
    tx: &rusqlite::Transaction<'_>,
    extracted: &ExtractedBlock,
    token_rows: &[TokenRow],
) -> eyre::Result<bool> {
    // the in-transaction cursor read IS the monotonic guard
    if read_cursor(tx)?.is_some_and(|cursor| extracted.number <= cursor) {
        return Ok(false);
    }

    {
        let mut stmt = tx.prepare_cached(
            "INSERT OR IGNORE INTO address_txs (address, block_number, tx_index, tx_type) \
             VALUES (?1, ?2, ?3, ?4)",
        )?;
        for row in &extracted.address_rows {
            stmt.execute((
                addr_hex(&row.address),
                row.block_number,
                row.tx_index,
                row.tx_type,
            ))?;
        }
    }

    {
        // Type index + exact per-type totals. The cursor guard above admits
        // each block exactly once and this whole function commits or rolls
        // back with the cursor, so adding the number of rows the INSERT
        // actually stored (`execute` reports 0 for an ignored duplicate) keeps
        // `tx_type_counts` equal to `COUNT(*)` per type without ever scanning
        // `tx_types`.
        let mut inserted: BTreeMap<u8, u64> = BTreeMap::new();
        let mut rows = tx.prepare_cached(
            "INSERT OR IGNORE INTO tx_types (tx_type, block_number, tx_index) \
             VALUES (?1, ?2, ?3)",
        )?;
        for row in &extracted.tx_types {
            let changed = rows.execute((row.tx_type, row.block_number, row.tx_index))?;
            *inserted.entry(row.tx_type).or_default() += changed as u64;
        }
        let mut totals = tx.prepare_cached(
            "INSERT INTO tx_type_counts (tx_type, count) VALUES (?1, ?2) \
             ON CONFLICT(tx_type) DO UPDATE SET count = count + excluded.count",
        )?;
        for (tx_type, count) in inserted {
            totals.execute((tx_type, count))?;
        }
    }

    if let Some(consensus) = &extracted.consensus {
        // One row per output digest whose exec range widens as each block the
        // output produced is applied. min/max rather than "first writer wins"
        // makes the result independent of arrival order, even though the
        // cursor guard already forces ascending blocks.
        tx.prepare_cached(
            "INSERT INTO consensus_blocks (digest, epoch, round, first_block, last_block) \
             VALUES (?1, ?2, ?3, ?4, ?4) \
             ON CONFLICT(digest) DO UPDATE SET \
                first_block = min(first_block, excluded.first_block), \
                last_block = max(last_block, excluded.last_block)",
        )?
        .execute((
            digest_hex(&consensus.digest),
            consensus.epoch,
            consensus.round,
            consensus.block_number,
        ))?;
    }

    {
        let mut stmt = tx.prepare_cached(
            "INSERT OR IGNORE INTO token_transfers \
             (block_number, tx_index, log_index, token, from_addr, to_addr, value) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )?;
        for row in &extracted.transfers {
            stmt.execute((
                row.block_number,
                row.tx_index,
                row.log_index,
                addr_hex(&row.token),
                addr_hex(&row.from),
                addr_hex(&row.to),
                row.value.to_string(), // exact decimal string
            ))?;
        }
    }

    {
        // Token status machine, enforced at the write: a failed fetch (status 1)
        // over an already-failed row escalates to 2 (terminal). Rows with status
        // 0 or 2 are never handed back in by `token_fetch_states`, so ok rows
        // are not refetched and terminal rows never change.
        let mut stmt = tx.prepare_cached(
            "INSERT INTO tokens (address, name, symbol, decimals, status, fetched_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
             ON CONFLICT(address) DO UPDATE SET \
                name = excluded.name, \
                symbol = excluded.symbol, \
                decimals = excluded.decimals, \
                status = CASE \
                    WHEN excluded.status != 0 AND tokens.status != 0 THEN 2 \
                    ELSE excluded.status END, \
                fetched_at = excluded.fetched_at",
        )?;
        for row in token_rows {
            stmt.execute((
                addr_hex(&row.address),
                &row.name,
                &row.symbol,
                row.decimals,
                row.status,
                row.fetched_at,
            ))?;
        }
    }

    // advance the cursor in the SAME transaction as the rows
    tx.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_indexed', ?1)",
        [extracted.number.to_string()],
    )?;
    Ok(true)
}

// ---------------------------------------------------------------------------
// Read side (API queries; sync helpers shared by the pool and tests)
// ---------------------------------------------------------------------------

fn pointer_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<TxPointer> {
    Ok(TxPointer {
        block_number: row.get(0)?,
        tx_index: row.get(1)?,
    })
}

/// Count of pointer rows for an address (the address page's `total`).
pub fn count_address_txs(conn: &Connection, address: &str) -> rusqlite::Result<u64> {
    conn.prepare_cached("SELECT COUNT(*) FROM address_txs WHERE address = ?1")?
        .query_row([address], |row| row.get(0))
}

/// One newest-first page of an address's transaction pointers — a reverse scan
/// of the composite primary key.
///
/// OFFSET pagination is fine at MVP volumes; the upgrade path is keyset
/// pagination (`WHERE (block_number, tx_index) < (?, ?) ORDER BY ... DESC`).
pub fn address_txs_page(
    conn: &Connection,
    address: &str,
    limit: u64,
    offset: u64,
) -> rusqlite::Result<Vec<TxPointer>> {
    let mut stmt = conn.prepare_cached(
        "SELECT block_number, tx_index FROM address_txs WHERE address = ?1 \
         ORDER BY block_number DESC, tx_index DESC LIMIT ?2 OFFSET ?3",
    )?;
    let rows = stmt.query_map((address, limit, offset), pointer_from_row)?;
    rows.collect()
}

/// Count of an address's pointer rows with one EIP-2718 type (the `total` of
/// `/address/{addr}/txs?type=`).
///
/// A filtered walk of that address's PK range rather than a second index:
/// per-address volumes are small, and `tx_type` lives in the row.
pub fn count_address_txs_by_type(
    conn: &Connection,
    address: &str,
    tx_type: u8,
) -> rusqlite::Result<u64> {
    conn.prepare_cached("SELECT COUNT(*) FROM address_txs WHERE address = ?1 AND tx_type = ?2")?
        .query_row((address, tx_type), |row| row.get(0))
}

/// One newest-first page of an address's pointers restricted to one EIP-2718
/// type — [`address_txs_page`] with `AND tx_type = ?`.
pub fn address_txs_page_by_type(
    conn: &Connection,
    address: &str,
    tx_type: u8,
    limit: u64,
    offset: u64,
) -> rusqlite::Result<Vec<TxPointer>> {
    let mut stmt = conn.prepare_cached(
        "SELECT block_number, tx_index FROM address_txs WHERE address = ?1 AND tx_type = ?2 \
         ORDER BY block_number DESC, tx_index DESC LIMIT ?3 OFFSET ?4",
    )?;
    let rows = stmt.query_map((address, tx_type, limit, offset), pointer_from_row)?;
    rows.collect()
}

/// Total transactions of one EIP-2718 type (the `total` of `/txs?type=`), from
/// the `tx_type_counts` counter — O(1), exact because the counter is updated
/// in the same transaction that inserts the rows (see [`apply_block`]). A type
/// never seen (e.g. EIP-4844, which TN's batch allowlist rejects) has no row
/// and counts as 0.
pub fn count_tx_type(conn: &Connection, tx_type: u8) -> rusqlite::Result<u64> {
    let count: Option<u64> = conn
        .prepare_cached("SELECT count FROM tx_type_counts WHERE tx_type = ?1")?
        .query_row([tx_type], |row| row.get(0))
        .optional()?;
    Ok(count.unwrap_or(0))
}

/// One newest-first page of all transactions of one EIP-2718 type — a reverse
/// scan of the `tx_types` primary key (`tx_type, block_number, tx_index`), so
/// no sort step regardless of table size.
pub fn tx_type_page(
    conn: &Connection,
    tx_type: u8,
    limit: u64,
    offset: u64,
) -> rusqlite::Result<Vec<TxPointer>> {
    let mut stmt = conn.prepare_cached(
        "SELECT block_number, tx_index FROM tx_types WHERE tx_type = ?1 \
         ORDER BY block_number DESC, tx_index DESC LIMIT ?2 OFFSET ?3",
    )?;
    let rows = stmt.query_map((tx_type, limit, offset), pointer_from_row)?;
    rows.collect()
}

/// The execution block range recorded for one consensus output digest
/// (`digest` as produced by [`digest_hex`]), or `None` when no block naming
/// that digest has been indexed yet.
pub fn consensus_range_by_digest(
    conn: &Connection,
    digest: &str,
) -> rusqlite::Result<Option<ConsensusRange>> {
    conn.prepare_cached("SELECT first_block, last_block FROM consensus_blocks WHERE digest = ?1")?
        .query_row([digest], |row| {
            Ok(ConsensusRange {
                first_block: row.get(0)?,
                last_block: row.get(1)?,
            })
        })
        .optional()
}

/// [`consensus_range_by_digest`] for a whole list page (≤ `MAX_PER_PAGE`
/// digests): one cached point lookup per digest, keyed back by the input
/// string. Digests without a row are simply absent from the map — the caller
/// renders them as `exec_blocks: null`.
pub fn consensus_ranges_by_digests(
    conn: &Connection,
    digests: &[String],
) -> rusqlite::Result<BTreeMap<String, ConsensusRange>> {
    let mut stmt = conn
        .prepare_cached("SELECT first_block, last_block FROM consensus_blocks WHERE digest = ?1")?;
    let mut out = BTreeMap::new();
    for digest in digests {
        let range = stmt
            .query_row([digest], |row| {
                Ok(ConsensusRange {
                    first_block: row.get(0)?,
                    last_block: row.get(1)?,
                })
            })
            .optional()?;
        if let Some(range) = range {
            out.insert(digest.clone(), range);
        }
    }
    Ok(out)
}

/// Column list shared by the transfer page queries, in [`transfer_from_row`]
/// order.
const TRANSFER_COLUMNS: &str =
    "block_number, tx_index, log_index, token, from_addr, to_addr, value";

fn transfer_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredTransfer> {
    Ok(StoredTransfer {
        block_number: row.get(0)?,
        tx_index: row.get(1)?,
        log_index: row.get(2)?,
        token: row.get(3)?,
        from: row.get(4)?,
        to: row.get(5)?,
        value: row.get(6)?,
    })
}

/// Count of transfers where the address is sender or recipient (self-transfers
/// counted once — the UNION collapses them).
pub fn count_address_transfers(conn: &Connection, address: &str) -> rusqlite::Result<u64> {
    conn.prepare_cached(
        "SELECT COUNT(*) FROM ( \
            SELECT block_number, tx_index, log_index FROM token_transfers WHERE from_addr = ?1 \
            UNION \
            SELECT block_number, tx_index, log_index FROM token_transfers WHERE to_addr = ?1)",
    )?
    .query_row([address], |row| row.get(0))
}

/// One newest-first page of transfers where the address is sender or recipient
/// (two indexed arms, ordered post-union).
pub fn address_transfers_page(
    conn: &Connection,
    address: &str,
    limit: u64,
    offset: u64,
) -> rusqlite::Result<Vec<StoredTransfer>> {
    let sql = format!(
        "SELECT {TRANSFER_COLUMNS} FROM ( \
            SELECT {TRANSFER_COLUMNS} FROM token_transfers WHERE from_addr = ?1 \
            UNION \
            SELECT {TRANSFER_COLUMNS} FROM token_transfers WHERE to_addr = ?1) \
         ORDER BY block_number DESC, tx_index DESC, log_index DESC LIMIT ?2 OFFSET ?3"
    );
    let mut stmt = conn.prepare_cached(&sql)?;
    let rows = stmt.query_map((address, limit, offset), transfer_from_row)?;
    rows.collect()
}

/// Count of transfers emitted by a token contract.
pub fn count_token_transfers(conn: &Connection, token: &str) -> rusqlite::Result<u64> {
    conn.prepare_cached("SELECT COUNT(*) FROM token_transfers WHERE token = ?1")?
        .query_row([token], |row| row.get(0))
}

/// One newest-first page of a token's transfers (`idx_tt_token`).
pub fn token_transfers_page(
    conn: &Connection,
    token: &str,
    limit: u64,
    offset: u64,
) -> rusqlite::Result<Vec<StoredTransfer>> {
    let sql = format!(
        "SELECT {TRANSFER_COLUMNS} FROM token_transfers WHERE token = ?1 \
         ORDER BY block_number DESC, tx_index DESC, log_index DESC LIMIT ?2 OFFSET ?3"
    );
    let mut stmt = conn.prepare_cached(&sql)?;
    let rows = stmt.query_map((token, limit, offset), transfer_from_row)?;
    rows.collect()
}

/// Count of ERC-20 transfers one transaction emitted (the `total` of
/// `/txs/{hash}/transfers`) — a prefix probe of the `token_transfers` PK.
pub fn count_transfers_for_tx(
    conn: &Connection,
    block_number: u64,
    tx_index: u64,
) -> rusqlite::Result<u64> {
    conn.prepare_cached(
        "SELECT COUNT(*) FROM token_transfers WHERE block_number = ?1 AND tx_index = ?2",
    )?
    .query_row((block_number, tx_index), |row| row.get(0))
}

/// One page of a transaction's ERC-20 transfers in emission order
/// (`log_index ASC`) — a forward scan of the `token_transfers` PK prefix
/// `(block_number, tx_index)`, so the order is free and rows from neighbouring
/// transactions never enter the scan. Oldest-first here (unlike the feeds)
/// because a transaction's internal transfers read top-to-bottom like a trace.
pub fn transfers_for_tx_page(
    conn: &Connection,
    block_number: u64,
    tx_index: u64,
    limit: u64,
    offset: u64,
) -> rusqlite::Result<Vec<StoredTransfer>> {
    let sql = format!(
        "SELECT {TRANSFER_COLUMNS} FROM token_transfers \
         WHERE block_number = ?1 AND tx_index = ?2 \
         ORDER BY log_index ASC LIMIT ?3 OFFSET ?4"
    );
    let mut stmt = conn.prepare_cached(&sql)?;
    let rows = stmt.query_map((block_number, tx_index, limit, offset), transfer_from_row)?;
    rows.collect()
}

/// The cached metadata row for a token, if any.
pub fn token_row(conn: &Connection, address: &str) -> rusqlite::Result<Option<StoredToken>> {
    conn.prepare_cached("SELECT name, symbol, decimals, status FROM tokens WHERE address = ?1")?
        .query_row([address], |row| {
            Ok(StoredToken {
                name: row.get(0)?,
                symbol: row.get(1)?,
                decimals: row.get(2)?,
                status: row.get(3)?,
            })
        })
        .optional()
}

// ---------------------------------------------------------------------------
// Read pool
// ---------------------------------------------------------------------------

/// Four read-only connections handed out via a [`Semaphore`], used by the API.
#[derive(Debug, Clone)]
pub struct ReadPool {
    conns: Arc<Mutex<Vec<Connection>>>,
    permits: Arc<Semaphore>,
}

impl ReadPool {
    /// Open [`READ_POOL_SIZE`] read-only connections to an existing db (the
    /// [`Writer`] must have been opened first — it creates the file and the
    /// WAL side files).
    pub async fn open(path: PathBuf) -> eyre::Result<Self> {
        tokio::task::spawn_blocking(move || {
            let mut conns = Vec::with_capacity(READ_POOL_SIZE);
            for _ in 0..READ_POOL_SIZE {
                let conn = Connection::open_with_flags(
                    &path,
                    OpenFlags::SQLITE_OPEN_READ_ONLY
                        | OpenFlags::SQLITE_OPEN_URI
                        | OpenFlags::SQLITE_OPEN_NO_MUTEX,
                )
                .wrap_err_with(|| {
                    format!("failed to open read-only sqlite conn at {}", path.display())
                })?;
                conn.busy_timeout(std::time::Duration::from_millis(5000))?;
                conns.push(conn);
            }
            Ok(Self {
                conns: Arc::new(Mutex::new(conns)),
                permits: Arc::new(Semaphore::new(READ_POOL_SIZE)),
            })
        })
        .await?
    }

    /// Run `f` with one pooled read-only connection on the blocking pool. The
    /// semaphore permit is held for the full blocking lifetime, so pool
    /// occupancy is bounded even if the caller disconnects.
    pub async fn with_conn<T, F>(&self, f: F) -> eyre::Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&Connection) -> eyre::Result<T> + Send + 'static,
    {
        let permit = self.permits.clone().acquire_owned().await?;
        let conns = self.conns.clone();
        tokio::task::spawn_blocking(move || {
            let _permit = permit; // held until the blocking work completes
            let conn = conns
                .lock()
                .map_err(|_| eyre!("read pool mutex poisoned"))?
                .pop()
                .ok_or_else(|| eyre!("read pool exhausted"))?;
            let result = f(&conn);
            conns
                .lock()
                .map_err(|_| eyre!("read pool mutex poisoned"))?
                .push(conn);
            result
        })
        .await?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::{AddressRow, ConsensusRow, TransferRow, TxTypeRow};
    use tn_types::U256;

    fn addr(byte: u8) -> Address {
        Address::repeat_byte(byte)
    }

    fn digest(byte: u8) -> B256 {
        B256::repeat_byte(byte)
    }

    fn ptr(block_number: u64, tx_index: u64) -> TxPointer {
        TxPointer {
            block_number,
            tx_index,
        }
    }

    fn temp_db() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("explorer.sqlite");
        (dir, path)
    }

    /// A block with no derived rows at all; the other builders fill it in.
    /// Every field is spelled out so adding one to `ExtractedBlock` fails here
    /// until storage decides what to persist.
    fn empty_block(number: u64) -> ExtractedBlock {
        ExtractedBlock {
            number,
            address_rows: vec![],
            transfers: vec![],
            token_candidates: vec![],
            tx_types: vec![],
            consensus: None,
        }
    }

    /// Legacy (type 0) pointer rows only.
    fn block_with_rows(number: u64, addresses: &[(Address, u64)]) -> ExtractedBlock {
        let mut block = empty_block(number);
        block.address_rows = addresses
            .iter()
            .map(|(address, tx_index)| AddressRow {
                address: *address,
                block_number: number,
                tx_index: *tx_index,
                tx_type: 0,
            })
            .collect();
        block
    }

    /// One transaction per `(tx_index, tx_type, participant)`: the `tx_types`
    /// row plus the participant's typed pointer row — what `extract_block`
    /// emits for a single-party transaction.
    fn block_with_typed_txs(number: u64, txs: &[(u64, u8, Address)]) -> ExtractedBlock {
        let mut block = empty_block(number);
        for (tx_index, tx_type, address) in txs {
            block.tx_types.push(TxTypeRow {
                tx_type: *tx_type,
                block_number: number,
                tx_index: *tx_index,
            });
            block.address_rows.push(AddressRow {
                address: *address,
                block_number: number,
                tx_index: *tx_index,
                tx_type: *tx_type,
            });
        }
        block
    }

    /// Attach the consensus output (`parent_beacon_block_root` decode) the
    /// block came from.
    fn with_consensus(
        mut block: ExtractedBlock,
        digest: B256,
        epoch: u32,
        round: u32,
    ) -> ExtractedBlock {
        block.consensus = Some(ConsensusRow {
            digest,
            epoch,
            round,
            block_number: block.number,
        });
        block
    }

    /// Transfers `(token, from, to, value)` all in tx 0, log-indexed in order.
    fn block_with_transfers(
        number: u64,
        transfers: &[(Address, Address, Address, u64)],
    ) -> ExtractedBlock {
        let mut block = empty_block(number);
        block.transfers = transfers
            .iter()
            .enumerate()
            .map(|(log_index, (token, from, to, value))| TransferRow {
                block_number: number,
                tx_index: 0,
                log_index: log_index as u64,
                token: *token,
                from: *from,
                to: *to,
                value: U256::from(*value),
            })
            .collect();
        block
    }

    /// Transfers at explicit `(tx_index, log_index, value)` positions (one
    /// token, one sender/recipient pair) for the per-tx prefix scans.
    fn block_with_positioned_transfers(
        number: u64,
        positions: &[(u64, u64, u64)],
    ) -> ExtractedBlock {
        let mut block = empty_block(number);
        block.transfers = positions
            .iter()
            .map(|(tx_index, log_index, value)| TransferRow {
                block_number: number,
                tx_index: *tx_index,
                log_index: *log_index,
                token: addr(0xa1),
                from: addr(0x0b),
                to: addr(0x0c),
                value: U256::from(*value),
            })
            .collect();
        block
    }

    fn token_row(address: Address, ok: bool) -> TokenRow {
        TokenRow {
            address,
            name: ok.then(|| "Token".to_string()),
            symbol: ok.then(|| "TOK".to_string()),
            decimals: ok.then_some(18),
            status: u8::from(!ok),
            fetched_at: 1,
        }
    }

    /// Row counts of every derived table plus the cursor — the "nothing moved"
    /// witness for the cursor-guard and rollback tests. Extend it whenever a
    /// table is added so those tests cover it for free.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Snapshot {
        address_rows: u64,
        transfers: u64,
        tx_types: u64,
        tx_type_count_sum: u64,
        consensus_blocks: u64,
        cursor: Option<u64>,
    }

    fn snapshot(writer: &Writer) -> Snapshot {
        let conn = writer.conn.lock().expect("lock");
        let count = |sql: &str| -> u64 { conn.query_row(sql, [], |r| r.get(0)).expect("count") };
        Snapshot {
            address_rows: count("SELECT COUNT(*) FROM address_txs"),
            transfers: count("SELECT COUNT(*) FROM token_transfers"),
            tx_types: count("SELECT COUNT(*) FROM tx_types"),
            tx_type_count_sum: count("SELECT COALESCE(SUM(count), 0) FROM tx_type_counts"),
            consensus_blocks: count("SELECT COUNT(*) FROM consensus_blocks"),
            cursor: read_cursor(&conn).expect("cursor"),
        }
    }

    #[tokio::test]
    async fn transfer_pages_union_and_order_newest_first() {
        let (_dir, path) = temp_db();
        let writer = Writer::open(path, 0x1e7).await.expect("open");
        let token = addr(0xa1);
        let (me, other) = (addr(0x0b), addr(0x0c));
        // block 1: me -> other, then a self-transfer (dedup: counted once)
        writer
            .index_block(
                block_with_transfers(1, &[(token, me, other, 10), (token, me, me, 5)]),
                vec![],
            )
            .await
            .expect("index 1");
        // block 2: other -> me (recipient arm of the union)
        writer
            .index_block(block_with_transfers(2, &[(token, other, me, 7)]), vec![])
            .await
            .expect("index 2");

        let conn = writer.conn.lock().expect("lock");
        let key = addr_hex(&me);
        assert_eq!(count_address_transfers(&conn, &key).expect("count"), 3);
        let page = address_transfers_page(&conn, &key, 25, 0).expect("page");
        // newest first across both union arms; log_index rides along
        assert_eq!(
            page.iter()
                .map(|t| (t.block_number, t.log_index, t.value.as_str()))
                .collect::<Vec<_>>(),
            vec![(2, 0, "7"), (1, 1, "5"), (1, 0, "10")]
        );
        // token feed sees every transfer
        let token_key = addr_hex(&token);
        assert_eq!(count_token_transfers(&conn, &token_key).expect("count"), 3);
        let token_page = token_transfers_page(&conn, &token_key, 2, 1).expect("page");
        assert_eq!(token_page.len(), 2); // offset pagination applies
    }

    #[tokio::test]
    async fn transfers_for_tx_page_orders_by_log_index_within_one_tx() {
        let (_dir, path) = temp_db();
        let writer = Writer::open(path, 0x1e7).await.expect("open");
        // block 5: tx 0 emits logs 3, 1, 7 (handed in out of order); tx 1 emits 0, 2
        writer
            .index_block(
                block_with_positioned_transfers(
                    5,
                    &[(0, 3, 30), (1, 0, 100), (0, 1, 10), (1, 2, 120), (0, 7, 70)],
                ),
                vec![],
            )
            .await
            .expect("index 5");
        // block 6: tx 0 emits log 5 — same tx_index as (5, 0), different block
        writer
            .index_block(block_with_positioned_transfers(6, &[(0, 5, 50)]), vec![])
            .await
            .expect("index 6");

        let conn = writer.conn.lock().expect("lock");
        let logs = |block_number, tx_index| -> Vec<(u64, String)> {
            transfers_for_tx_page(&conn, block_number, tx_index, 25, 0)
                .expect("page")
                .iter()
                .map(|t| {
                    assert_eq!((t.block_number, t.tx_index), (block_number, tx_index));
                    (t.log_index, t.value.clone())
                })
                .collect()
        };
        let expected = |rows: &[(u64, &str)]| -> Vec<(u64, String)> {
            rows.iter().map(|(i, v)| (*i, v.to_string())).collect()
        };

        // emission order, regardless of insert order
        assert_eq!(count_transfers_for_tx(&conn, 5, 0).expect("count"), 3);
        assert_eq!(logs(5, 0), expected(&[(1, "10"), (3, "30"), (7, "70")]));
        // the PK prefix isolates neighbouring transactions and blocks
        assert_eq!(count_transfers_for_tx(&conn, 5, 1).expect("count"), 2);
        assert_eq!(logs(5, 1), expected(&[(0, "100"), (2, "120")]));
        assert_eq!(count_transfers_for_tx(&conn, 6, 0).expect("count"), 1);
        assert_eq!(logs(6, 0), expected(&[(5, "50")]));
        assert_eq!(count_transfers_for_tx(&conn, 6, 1).expect("count"), 0);
        assert!(logs(6, 1).is_empty());
        // offset pagination
        let second = transfers_for_tx_page(&conn, 5, 0, 1, 1).expect("page");
        assert_eq!(
            second.iter().map(|t| t.log_index).collect::<Vec<_>>(),
            vec![3]
        );
    }

    #[tokio::test]
    async fn consensus_blocks_merge_exec_range_per_digest() {
        let (_dir, path) = temp_db();
        let writer = Writer::open(path, 0x1e7).await.expect("open");
        let (output_a, output_b) = (digest(0xd1), digest(0xd2));
        // one output -> three exec blocks (one per batch): the range widens
        for number in 10..=12 {
            writer
                .index_block(with_consensus(empty_block(number), output_a, 3, 41), vec![])
                .await
                .expect("index");
        }
        // the next output gets its own row
        writer
            .index_block(with_consensus(empty_block(13), output_b, 3, 42), vec![])
            .await
            .expect("index 13");
        // a block naming no output (genesis shape): no row anywhere
        writer
            .index_block(empty_block(14), vec![])
            .await
            .expect("index 14");
        assert_eq!(snapshot(&writer).consensus_blocks, 2);

        let conn = writer.conn.lock().expect("lock");
        let (key_a, key_b) = (digest_hex(&output_a), digest_hex(&output_b));
        // 0x + 64 lowercase hex chars
        assert_eq!(key_a, format!("0x{}", "d1".repeat(32)));
        assert_eq!(
            consensus_range_by_digest(&conn, &key_a).expect("read"),
            Some(ConsensusRange {
                first_block: 10,
                last_block: 12
            })
        );
        assert_eq!(
            consensus_range_by_digest(&conn, &key_b).expect("read"),
            Some(ConsensusRange {
                first_block: 13,
                last_block: 13
            })
        );
        assert_eq!(
            consensus_range_by_digest(&conn, &digest_hex(&digest(0xd3))).expect("read"),
            None
        );
        // epoch/round persist alongside the range
        let (epoch, round): (u32, u32) = conn
            .query_row(
                "SELECT epoch, round FROM consensus_blocks WHERE digest = ?1",
                [&key_a],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .expect("row");
        assert_eq!((epoch, round), (3, 41));

        // batch lookup keeps only the digests that have rows
        let ranges = consensus_ranges_by_digests(
            &conn,
            &[key_a.clone(), digest_hex(&digest(0xd3)), key_b.clone()],
        )
        .expect("batch");
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[&key_a].last_block, 12);
        assert_eq!(ranges[&key_b].first_block, 13);
    }

    #[tokio::test]
    async fn tx_type_pages_newest_first_and_counts_exact() {
        let (_dir, path) = temp_db();
        let writer = Writer::open(path, 0x1e7).await.expect("open");
        let me = addr(0x0a);
        // block 1: legacy, eip1559, legacy; block 2: eip1559, eip2930
        writer
            .index_block(
                block_with_typed_txs(1, &[(0, 0, me), (1, 2, me), (2, 0, me)]),
                vec![],
            )
            .await
            .expect("index 1");
        writer
            .index_block(block_with_typed_txs(2, &[(0, 2, me), (1, 1, me)]), vec![])
            .await
            .expect("index 2");

        let snap = snapshot(&writer);
        assert_eq!(snap.tx_types, 5);
        assert_eq!(
            snap.tx_type_count_sum, snap.tx_types,
            "counters must equal the rows they count"
        );

        let conn = writer.conn.lock().expect("lock");
        assert_eq!(count_tx_type(&conn, 0).expect("count"), 2);
        assert_eq!(count_tx_type(&conn, 1).expect("count"), 1);
        assert_eq!(count_tx_type(&conn, 2).expect("count"), 2);
        assert_eq!(
            count_tx_type(&conn, 3).expect("count"),
            0,
            "unseen type is 0"
        );
        assert_eq!(
            tx_type_page(&conn, 0, 25, 0).expect("page"),
            vec![ptr(1, 2), ptr(1, 0)]
        );
        assert_eq!(
            tx_type_page(&conn, 2, 25, 0).expect("page"),
            vec![ptr(2, 0), ptr(1, 1)]
        );
        assert_eq!(
            tx_type_page(&conn, 2, 1, 1).expect("page"),
            vec![ptr(1, 1)],
            "offset applies"
        );
        assert!(tx_type_page(&conn, 4, 25, 0).expect("page").is_empty());
    }

    #[tokio::test]
    async fn address_txs_page_by_type_filters_one_type() {
        let (_dir, path) = temp_db();
        let writer = Writer::open(path, 0x1e7).await.expect("open");
        let (me, other) = (addr(0x0a), addr(0x0b));
        writer
            .index_block(
                block_with_typed_txs(1, &[(0, 0, me), (1, 2, me), (2, 2, other)]),
                vec![],
            )
            .await
            .expect("index 1");
        writer
            .index_block(
                block_with_typed_txs(3, &[(0, 2, me), (1, 0, other)]),
                vec![],
            )
            .await
            .expect("index 3");

        let conn = writer.conn.lock().expect("lock");
        let key = addr_hex(&me);
        // the untyped page is unchanged by the filter column
        assert_eq!(count_address_txs(&conn, &key).expect("count"), 3);
        assert_eq!(
            address_txs_page(&conn, &key, 25, 0).expect("page"),
            vec![ptr(3, 0), ptr(1, 1), ptr(1, 0)]
        );
        // typed: newest first within one type, other addresses never leak in
        assert_eq!(count_address_txs_by_type(&conn, &key, 2).expect("count"), 2);
        assert_eq!(
            address_txs_page_by_type(&conn, &key, 2, 25, 0).expect("page"),
            vec![ptr(3, 0), ptr(1, 1)]
        );
        assert_eq!(count_address_txs_by_type(&conn, &key, 0).expect("count"), 1);
        assert_eq!(
            address_txs_page_by_type(&conn, &key, 0, 25, 0).expect("page"),
            vec![ptr(1, 0)]
        );
        assert_eq!(count_address_txs_by_type(&conn, &key, 4).expect("count"), 0);
        assert!(address_txs_page_by_type(&conn, &key, 4, 25, 0)
            .expect("page")
            .is_empty());
        assert_eq!(
            count_address_txs_by_type(&conn, &addr_hex(&other), 2).expect("count"),
            1
        );
    }

    #[tokio::test]
    async fn self_send_collapses_to_one_row_via_insert_or_ignore() {
        let (_dir, path) = temp_db();
        let writer = Writer::open(path, 0x1e7).await.expect("open");
        let me = addr(0x0a);
        // extraction emits BOTH rows for a self-send; the PK dedups at write time
        let block = block_with_rows(1, &[(me, 0), (me, 0)]);
        assert!(writer.index_block(block, vec![]).await.expect("index"));
        let snap = snapshot(&writer);
        assert_eq!(snap.address_rows, 1);
        assert_eq!(snap.cursor, Some(1));
    }

    #[tokio::test]
    async fn reverse_pk_scan_returns_merged_newest_first() {
        let (_dir, path) = temp_db();
        let writer = Writer::open(path.clone(), 0x1e7).await.expect("open");
        let me = addr(0x0a);
        let other = addr(0x0b);
        // interleaved from/to activity for one address across blocks: the
        // write-time merge (one table, one PK) replaces v1's two-arm UNION
        writer
            .index_block(block_with_rows(1, &[(me, 0), (other, 0), (me, 2)]), vec![])
            .await
            .expect("index 1");
        writer
            .index_block(block_with_rows(3, &[(other, 0), (me, 0)]), vec![])
            .await
            .expect("index 3");
        writer
            .index_block(block_with_rows(5, &[(me, 1)]), vec![])
            .await
            .expect("index 5");

        let conn = writer.conn.lock().expect("lock");
        let page = address_txs_page(&conn, &addr_hex(&me), 25, 0).expect("page");
        assert_eq!(page, vec![ptr(5, 1), ptr(3, 0), ptr(1, 2), ptr(1, 0)]);
        assert_eq!(count_address_txs(&conn, &addr_hex(&me)).expect("count"), 4);
    }

    #[tokio::test]
    async fn token_status_machine_transitions() {
        let (_dir, path) = temp_db();
        let writer = Writer::open(path, 0x1e7).await.expect("open");
        let ok_token = addr(0xa1);
        let bad_token = addr(0xa2);

        // unseen tokens are fetch candidates
        let need = writer
            .token_fetch_states(vec![ok_token, bad_token])
            .await
            .expect("states");
        assert_eq!(need, vec![ok_token, bad_token]);

        // unseen -> 0 on success, unseen -> 1 on failure
        writer
            .index_block(
                block_with_rows(1, &[]),
                vec![token_row(ok_token, true), token_row(bad_token, false)],
            )
            .await
            .expect("index");
        let conn = writer.conn.clone();
        let (ok_status, bad_status) = {
            let conn = conn.lock().expect("lock");
            (
                super::token_row(&conn, &addr_hex(&ok_token))
                    .expect("row")
                    .expect("some")
                    .status,
                super::token_row(&conn, &addr_hex(&bad_token))
                    .expect("row")
                    .expect("some")
                    .status,
            )
        };
        assert_eq!(ok_status, 0);
        assert_eq!(bad_status, 1);

        // ok (0) is never refetched; failed (1) retries on next sighting
        let need = writer
            .token_fetch_states(vec![ok_token, bad_token])
            .await
            .expect("states");
        assert_eq!(need, vec![bad_token]);

        // 1 -> 2 on second failure: terminal
        writer
            .index_block(block_with_rows(2, &[]), vec![token_row(bad_token, false)])
            .await
            .expect("index");
        let status = {
            let conn = conn.lock().expect("lock");
            super::token_row(&conn, &addr_hex(&bad_token))
                .expect("row")
                .expect("some")
                .status
        };
        assert_eq!(status, 2);

        // 2 is terminal: never handed back for refetch
        let need = writer
            .token_fetch_states(vec![ok_token, bad_token])
            .await
            .expect("states");
        assert!(need.is_empty());
    }

    #[tokio::test]
    async fn retry_pending_token_recovers_to_ok() {
        let (_dir, path) = temp_db();
        let writer = Writer::open(path, 0x1e7).await.expect("open");
        let token = addr(0xa3);
        writer
            .index_block(block_with_rows(1, &[]), vec![token_row(token, false)])
            .await
            .expect("index");
        // 1 -> 0 when the retry succeeds
        writer
            .index_block(block_with_rows(2, &[]), vec![token_row(token, true)])
            .await
            .expect("index");
        let conn = writer.conn.lock().expect("lock");
        let row = super::token_row(&conn, &addr_hex(&token))
            .expect("row")
            .expect("some");
        assert_eq!(row.status, 0);
        assert_eq!(row.symbol.as_deref(), Some("TOK"));
    }

    #[tokio::test]
    async fn partial_decode_is_ok_with_null_field() {
        let (_dir, path) = temp_db();
        let writer = Writer::open(path, 0x1e7).await.expect("open");
        let token = addr(0xa4);
        // name reverted, symbol + decimals decoded => status 0 with NULL name
        let row = TokenRow {
            address: token,
            name: None,
            symbol: Some("TOK".into()),
            decimals: Some(6),
            status: 0,
            fetched_at: 1,
        };
        writer
            .index_block(block_with_rows(1, &[]), vec![row])
            .await
            .expect("index");
        let conn = writer.conn.lock().expect("lock");
        let stored = super::token_row(&conn, &addr_hex(&token))
            .expect("row")
            .expect("some");
        assert_eq!(stored.status, 0);
        assert_eq!(stored.name, None);
        assert_eq!(stored.symbol.as_deref(), Some("TOK"));
        assert_eq!(stored.decimals, Some(6));
    }

    #[tokio::test]
    async fn cursor_guard_is_monotonic_and_idempotent() {
        let (_dir, path) = temp_db();
        let writer = Writer::open(path, 0x1e7).await.expect("open");
        let me = addr(0x0a);
        let output = digest(0xd1);

        assert!(writer
            .index_block(
                with_consensus(block_with_typed_txs(5, &[(0, 0, me)]), output, 1, 1),
                vec![]
            )
            .await
            .expect("index 5"));
        let before = snapshot(&writer);

        // same block twice: skipped, every table unchanged
        assert!(!writer
            .index_block(
                with_consensus(
                    block_with_typed_txs(5, &[(0, 0, me), (1, 2, addr(0x0b))]),
                    output,
                    1,
                    1
                ),
                vec![]
            )
            .await
            .expect("re-index 5"));
        assert_eq!(snapshot(&writer), before);

        // older block: skipped
        assert!(!writer
            .index_block(
                with_consensus(block_with_typed_txs(4, &[(0, 2, addr(0x0c))]), output, 1, 1),
                vec![]
            )
            .await
            .expect("index 4"));
        assert_eq!(snapshot(&writer), before);

        // newer block advances
        assert!(writer
            .index_block(block_with_rows(6, &[(me, 0)]), vec![])
            .await
            .expect("index 6"));
        assert_eq!(snapshot(&writer).cursor, Some(6));
    }

    #[tokio::test]
    async fn error_before_commit_moves_neither_rows_nor_cursor() {
        let (_dir, path) = temp_db();
        let writer = Writer::open(path, 0x1e7).await.expect("open");
        let me = addr(0x0a);
        let output = digest(0xd1);
        assert!(writer
            .index_block(
                with_consensus(block_with_typed_txs(1, &[(0, 0, me)]), output, 1, 1),
                vec![]
            )
            .await
            .expect("index 1"));
        let before = snapshot(&writer);
        assert_eq!((before.tx_types, before.consensus_blocks), (1, 1));

        // Run the full statement sequence for block 2 — typed pointers, type
        // rows + counters, the consensus range upsert, transfers, tokens, and
        // the cursor advance — then fail before COMMIT. The drop rolls back,
        // exactly what any mid-transaction error produces.
        {
            let mut conn = writer.conn.lock().expect("lock");
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .expect("begin");
            let mut block =
                with_consensus(block_with_typed_txs(2, &[(0, 2, addr(0x0b))]), output, 1, 1);
            block.transfers.push(TransferRow {
                block_number: 2,
                tx_index: 0,
                log_index: 0,
                token: addr(0xa1),
                from: me,
                to: addr(0x0b),
                value: U256::from(1u64),
            });
            assert!(
                apply_block(&tx, &block, &[token_row(addr(0xa1), true)]).expect("statements run")
            );
            // simulated failure: tx dropped without commit
        }

        assert_eq!(
            snapshot(&writer),
            before,
            "rollback must leave every table AND the cursor untouched"
        );
        let conn = writer.conn.lock().expect("lock");
        assert!(super::token_row(&conn, &addr_hex(&addr(0xa1)))
            .expect("row")
            .is_none());
        // in-place updates roll back too: the range did not widen and the new
        // type's counter was never created
        assert_eq!(
            consensus_range_by_digest(&conn, &digest_hex(&output)).expect("read"),
            Some(ConsensusRange {
                first_block: 1,
                last_block: 1
            })
        );
        assert_eq!(count_tx_type(&conn, 2).expect("count"), 0);
    }

    #[tokio::test]
    async fn schema_mismatch_drops_and_rebuilds() {
        let (_dir, path) = temp_db();
        {
            let writer = Writer::open(path.clone(), 0x1e7).await.expect("open");
            writer
                .index_block(block_with_rows(9, &[(addr(0x0a), 0)]), vec![])
                .await
                .expect("index");
            let conn = writer.conn.lock().expect("lock");
            conn.execute(
                "UPDATE meta SET value = '1' WHERE key = 'schema_version'",
                [],
            )
            .expect("downgrade version");
        }
        // reopen: version mismatch drops everything -> fresh cursor, empty tables
        let writer = Writer::open(path, 0x1e7).await.expect("reopen");
        assert_eq!(writer.last_indexed().await.expect("cursor"), None);
        assert_eq!(snapshot(&writer).address_rows, 0);
    }

    #[tokio::test]
    async fn schema_v2_file_is_rebuilt_as_v3() {
        let (_dir, path) = temp_db();
        {
            let writer = Writer::open(path.clone(), 0x1e7).await.expect("open");
            let conn = writer.conn.lock().expect("lock");
            // Turn the fresh file into a v2 one: the v2 `address_txs` shape (no
            // tx_type), none of the v3 tables, a cursor, and version '2'.
            conn.execute_batch(
                "DROP TABLE address_txs;
                 DROP TABLE tx_types;
                 DROP TABLE tx_type_counts;
                 DROP TABLE consensus_blocks;
                 CREATE TABLE address_txs (
                     address      TEXT    NOT NULL,
                     block_number INTEGER NOT NULL,
                     tx_index     INTEGER NOT NULL,
                     PRIMARY KEY (address, block_number, tx_index)
                 ) WITHOUT ROWID;
                 INSERT INTO address_txs VALUES ('0x0a', 9, 0);
                 UPDATE meta SET value = '2' WHERE key = 'schema_version';
                 INSERT OR REPLACE INTO meta (key, value) VALUES ('last_indexed', '9');",
            )
            .expect("downgrade to v2");
        }

        let writer = Writer::open(path, 0x1e7).await.expect("reopen");
        // cursor reset -> replay from 0; the v2 rows are gone
        assert_eq!(writer.last_indexed().await.expect("cursor"), None);
        assert_eq!(snapshot(&writer).address_rows, 0);
        {
            let conn = writer.conn.lock().expect("lock");
            assert_eq!(
                read_meta(&conn, "schema_version").expect("meta").as_deref(),
                Some("3")
            );
            // v3 shape: address_txs carries tx_type and the new tables exist
            let mut stmt = conn
                .prepare("PRAGMA table_info(address_txs)")
                .expect("pragma");
            let columns: Vec<String> = stmt
                .query_map([], |r| r.get::<_, String>(1))
                .expect("query")
                .collect::<Result<_, _>>()
                .expect("columns");
            assert!(
                columns.iter().any(|c| c == "tx_type"),
                "address_txs columns: {columns:?}"
            );
            for table in ["tx_types", "tx_type_counts", "consensus_blocks"] {
                assert!(
                    table_exists(&conn, table).expect("exists"),
                    "{table} must exist after the rebuild"
                );
            }
        }
        // and the rebuilt file indexes typed rows normally
        assert!(writer
            .index_block(
                with_consensus(
                    block_with_typed_txs(1, &[(0, 2, addr(0x0a))]),
                    digest(0xd1),
                    0,
                    1
                ),
                vec![]
            )
            .await
            .expect("index"));
        let snap = snapshot(&writer);
        assert_eq!(
            (snap.address_rows, snap.tx_types, snap.consensus_blocks),
            (1, 1, 1)
        );
    }

    #[tokio::test]
    async fn chain_id_mismatch_is_a_hard_error() {
        let (_dir, path) = temp_db();
        drop(Writer::open(path.clone(), 0x1e7).await.expect("open"));
        let err = Writer::open(path, 2017)
            .await
            .expect_err("chain mismatch must fail");
        assert!(err.to_string().contains("chain"), "unexpected error: {err}");
    }

    #[tokio::test]
    async fn read_pool_serves_queries() {
        let (_dir, path) = temp_db();
        let writer = Writer::open(path.clone(), 0x1e7).await.expect("open");
        writer
            .index_block(block_with_rows(1, &[(addr(0x0a), 0)]), vec![])
            .await
            .expect("index");
        let pool = ReadPool::open(path).await.expect("pool");
        let key = addr_hex(&addr(0x0a));
        let total = pool
            .with_conn(move |conn| Ok(count_address_txs(conn, &key)?))
            .await
            .expect("query");
        assert_eq!(total, 1);
    }
}
