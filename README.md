# tn-block-explorer-indexer

A block-explorer indexer for Telcoin Network, built on tn-3's ExEx plugin system.
The binary is a full observer node that runs an indexing ExEx and serves a small HTTP API from the same process.
[telscan.xyz](https://telscan.xyz) reads everything it renders from this API; it needs zero node JSON-RPC.

The design rule: never re-index what the node's own databases already contain.
Transactions, receipts, blocks, balances, code, gas price, epochs, validators, and generic read-only contract calls are answered by direct function calls on `RethEnv` (reth MDBX execution DB) and `ConsensusChain` (consensus pack files).
SQLite holds only the derived indexes reth maintains in no table:

- `address_txs`: address to transaction pointers (reth's `AccountsHistory` is a state-change index, not a tx index)
- `token_transfers`: ERC-20 transfer history by participant and by token
- `tokens`: ERC-20 metadata cache with a retry/terminal status machine
- `meta`: indexing cursor, schema version, chain id

tn-3 is consumed as a pinned git dependency.
Nothing here forks the node; updating means bumping one git rev (see below).

## Quickstart

Build (a C toolchain is the only native requirement; the tn-contracts submodule of the git dependency is fetched by cargo automatically):

```sh
cargo build --release
```

Run against a tn-3 local testnet.
From a tn-3 checkout, provision and start the validators:

```sh
./etc/local-testnet.sh --dev-funds <ADDR> --start
```

Kill the stock observer it started, then run this binary as the observer against the same datadir.
Note there is no `--http`: the indexer API replaces node RPC for the explorer.

```sh
TN_BLS_PASSPHRASE=local target/release/tn-block-explorer-indexer node \
    --datadir ../tn-3/local-validators/observer \
    --observer --instance 5 \
    --indexer.api-addr 127.0.0.1:8560 \
    --indexer.cors http://localhost:8080
```

The indexer flags, flattened into the `node` subcommand alongside all stock node flags:

| Flag | Default | Meaning |
|---|---|---|
| `--indexer.db-path <PATH>` | `<datadir>/explorer-indexer/explorer.sqlite` | SQLite file for the derived indexes |
| `--indexer.api-addr <SOCKET>` | `127.0.0.1:8560` | HTTP API bind address |
| `--indexer.cors <ORIGINS>` | empty (= any origin) | comma-separated allowed CORS origins |

For a public deployment, bind loopback and put a TLS-terminating reverse proxy in front, and pass the real origins: `--indexer.cors https://telscan.xyz,https://telcoin-explorer.netlify.app`.
Testnet (adiri, chain id 2017) builds need the feature: `cargo build --release --features adiri`.

## Endpoints

All list endpoints return `{items, total, page, per_page}` with 0-based `page`, `per_page` default 25 (max 100), newest first.
Missing resources are `404 {"error":"not found"}`.

| Route | Returns |
|---|---|
| `GET /health` | `{status, indexing_live, last_indexed, node_tip, lag}`; always 200, degradation is in the body |
| `GET /stats` | latest block, gas price (gwei), chain id, epoch, committee size, total txs; 5s server-side cache |
| `GET /txs` | the chain-wide transaction feed |
| `GET /txs/{hash}` | one mined transaction with receipt fields (full input hex) |
| `GET /blocks`, `GET /blocks/{number}` | block pages / one block |
| `GET /address/{addr}` | balance, nonce, `is_contract`, full code hex, indexed tx count |
| `GET /address/{addr}/txs` | the address's transaction history |
| `GET /address/{addr}/transfers` | ERC-20 transfers where the address is sender or recipient |
| `GET /tokens/{addr}` | token name/symbol/decimals + live `totalSupply` |
| `GET /tokens/{addr}/transfers` | the token's transfer feed |
| `GET /epochs`, `GET /epochs/{n}` | epoch history from the consensus DB's epoch records |
| `GET /epochs/current` | the in-progress epoch from the on-chain registry |
| `GET /validators` | the current committee's registry `ValidatorInfo`s |
| `GET /validators/leaders?window=200` | blocks per leader over the trailing window (max 1000) |
| `POST /call` | read-only `eth_call`-style contract read: `{to, data}` in, `{ok, result}` or `{ok:false, error, revert_data}` out |

Two wire details clients must respect.
Transaction and token-transfer `value` fields are u128 JSON numbers; parse responses directly with `serde_json::from_str::<T>`, never through a `serde_json::Value` intermediate, which destroys integers above `u64::MAX`.
Token transfers also carry `value_exact`, the lossless decimal string.

## Schema and disposability

The SQLite file is pure derived state, rebuilt from the node's own databases.
On open, a `schema_version` other than 2 (or leftover v1 tables) drops every table and the indexer re-replays from block 0.
A stored chain id that differs from the node's chainspec is a hard error, so an old file can't silently index a different network.
Delete the file and restart to rebuild from scratch; that is both the migration and the corruption story.

Writes are one SQLite transaction per block (pointer rows, transfer rows, token upserts, cursor advance), so a crash can't leave a half-indexed block: restart replays from `cursor + 1` with no holes.

## Bumping the tn-3 dependency

All eight tn git dependencies pin the same rev and must be bumped together in one commit; mixed revs give cargo two checkouts of the same crates and duplicate-type errors everywhere.
On every bump, re-check the `adiri` feature crate set against tn-3 (`grep -rn '^adiri' --include=Cargo.toml`), confirm the clap/eyre/tokio/futures/tracing majors still match tn-3's workspace, then run `cargo update && cargo check --all-targets && cargo check --features adiri && cargo test`.
The full procedure, including the ops runbook this repo follows, lives in tn-3's `integrate-exex.md` (Part 6).
