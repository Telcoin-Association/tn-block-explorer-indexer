//! SQLite persistence: exactly the derived indexes reth maintains in no table,
//! plus a cursor — nothing else.
//!
//! Tables (schema v2): `meta` (cursor, schema version, chain id), `address_txs`
//! (address → transaction pointers), `token_transfers` (ERC-20 transfer
//! history), `tokens` (ERC-20 metadata cache). Blocks, transactions, and
//! receipts deliberately have NO tables here — they are served by direct reads
//! on the node's own databases (`node_reads`).
//!
//! # Consistency & disposability
//!
//! One SQLite transaction per block: pointer rows, transfer rows, token upserts,
//! and the cursor advance commit atomically, so a `kill -9` loses rows and
//! cursor **together** (WAL rollback) and restart replays from `cursor + 1` with
//! no holes. `INSERT OR IGNORE` makes re-indexing the same block idempotent, and
//! TN has no reorgs so keys are permanently stable. The whole file is derived
//! state: a `schema_version` mismatch (or leftover v1 tables) drops every table
//! and re-replays from 0 — delete-the-file is the supported migration AND
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
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tn_types::Address;
use tokio::sync::Semaphore;

/// The schema version this build writes and requires.
const SCHEMA_VERSION: &str = "2";

/// Number of read-only connections in the [`ReadPool`].
const READ_POOL_SIZE: usize = 4;

/// Schema v2 DDL. `WITHOUT ROWID` keeps the composite primary keys clustered;
/// all hex values are lowercase `0x`-prefixed (the explorer compares with
/// `.to_lowercase()`).
const DDL: &str = "
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
-- rows: ('schema_version','2'), ('last_indexed','<u64>'), ('chain_id','<u64>')
-- 'last_indexed' ABSENT means fresh DB (replay from 0).

-- (a) address -> native-tx participation index. Pointer rows ONLY - no tx content.
CREATE TABLE IF NOT EXISTS address_txs (
    address      TEXT    NOT NULL,
    block_number INTEGER NOT NULL,
    tx_index     INTEGER NOT NULL,
    PRIMARY KEY (address, block_number, tx_index)
) WITHOUT ROWID;
-- Newest-first pagination is a reverse scan of the composite PK; no secondary
-- index needed. Self-send (from==to) collapses onto one PK via INSERT OR IGNORE.

-- (b) ERC-20 transfer index, extracted from receipt logs.
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

-- (c) token metadata cache. Written ONLY by the indexing path.
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

/// A `(block_number, tx_index)` pointer read back from `address_txs`, hydrated
/// into full transaction data by `node_reads`.
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
    /// Index of the log within its transaction's receipt logs.
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
    /// pointer + transfer rows, token upserts, cursor advance, commit.
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
/// The DB is disposable derived state: a `schema_version` ≠ 2 (or leftover v1
/// `blocks`/`transactions` tables) drops every table so the indexer re-replays
/// from 0. A stored chain id different from the node's chainspec is a hard
/// error (guards against pointing an old sqlite file at another network's
/// datadir).
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
            "INSERT OR IGNORE INTO address_txs (address, block_number, tx_index) \
             VALUES (?1, ?2, ?3)",
        )?;
        for row in &extracted.address_rows {
            stmt.execute((addr_hex(&row.address), row.block_number, row.tx_index))?;
        }
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
    let rows = stmt.query_map((address, limit, offset), |row| {
        Ok(TxPointer {
            block_number: row.get(0)?,
            tx_index: row.get(1)?,
        })
    })?;
    rows.collect()
}

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
    use crate::extract::{AddressRow, TransferRow};
    use tn_types::U256;

    fn addr(byte: u8) -> Address {
        Address::repeat_byte(byte)
    }

    fn temp_db() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("explorer.sqlite");
        (dir, path)
    }

    fn block_with_rows(number: u64, addresses: &[(Address, u64)]) -> ExtractedBlock {
        ExtractedBlock {
            number,
            address_rows: addresses
                .iter()
                .map(|(address, tx_index)| AddressRow {
                    address: *address,
                    block_number: number,
                    tx_index: *tx_index,
                })
                .collect(),
            transfers: vec![],
            token_candidates: vec![],
        }
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

    fn counts(writer: &Writer) -> (u64, u64, Option<u64>) {
        let conn = writer.conn.lock().expect("lock");
        let address_rows: u64 = conn
            .query_row("SELECT COUNT(*) FROM address_txs", [], |r| r.get(0))
            .expect("count");
        let transfers: u64 = conn
            .query_row("SELECT COUNT(*) FROM token_transfers", [], |r| r.get(0))
            .expect("count");
        let cursor = read_cursor(&conn).expect("cursor");
        (address_rows, transfers, cursor)
    }

    #[tokio::test]
    async fn self_send_collapses_to_one_row_via_insert_or_ignore() {
        let (_dir, path) = temp_db();
        let writer = Writer::open(path, 0x1e7).await.expect("open");
        let me = addr(0x0a);
        // extraction emits BOTH rows for a self-send; the PK dedups at write time
        let block = block_with_rows(1, &[(me, 0), (me, 0)]);
        assert!(writer.index_block(block, vec![]).await.expect("index"));
        let (address_rows, _, cursor) = counts(&writer);
        assert_eq!(address_rows, 1);
        assert_eq!(cursor, Some(1));
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
        assert_eq!(
            page,
            vec![
                TxPointer {
                    block_number: 5,
                    tx_index: 1
                },
                TxPointer {
                    block_number: 3,
                    tx_index: 0
                },
                TxPointer {
                    block_number: 1,
                    tx_index: 2
                },
                TxPointer {
                    block_number: 1,
                    tx_index: 0
                },
            ]
        );
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

        assert!(writer
            .index_block(block_with_rows(5, &[(me, 0)]), vec![])
            .await
            .expect("index 5"));
        let before = counts(&writer);

        // same block twice: skipped, counts unchanged
        assert!(!writer
            .index_block(block_with_rows(5, &[(me, 0), (addr(0x0b), 1)]), vec![])
            .await
            .expect("re-index 5"));
        assert_eq!(counts(&writer), before);

        // older block: skipped
        assert!(!writer
            .index_block(block_with_rows(4, &[(addr(0x0c), 0)]), vec![])
            .await
            .expect("index 4"));
        assert_eq!(counts(&writer), before);

        // newer block advances
        assert!(writer
            .index_block(block_with_rows(6, &[(me, 0)]), vec![])
            .await
            .expect("index 6"));
        assert_eq!(counts(&writer).2, Some(6));
    }

    #[tokio::test]
    async fn error_before_commit_moves_neither_rows_nor_cursor() {
        let (_dir, path) = temp_db();
        let writer = Writer::open(path, 0x1e7).await.expect("open");
        let me = addr(0x0a);
        assert!(writer
            .index_block(block_with_rows(1, &[(me, 0)]), vec![])
            .await
            .expect("index 1"));
        let before = counts(&writer);

        // Run the full statement sequence for block 2 — rows, transfers, and the
        // cursor advance — then fail before COMMIT. The drop rolls back, exactly
        // what any mid-transaction error produces.
        {
            let mut conn = writer.conn.lock().expect("lock");
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .expect("begin");
            let mut block = block_with_rows(2, &[(addr(0x0b), 0)]);
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
            counts(&writer),
            before,
            "rollback must leave rows AND cursor untouched"
        );
        let conn = writer.conn.lock().expect("lock");
        assert!(super::token_row(&conn, &addr_hex(&addr(0xa1)))
            .expect("row")
            .is_none());
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
        assert_eq!(counts(&writer).0, 0);
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
