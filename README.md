# tn-block-explorer-indexer

A block-explorer indexer for Telcoin Network, built on tn-3's ExEx plugin system.
The binary is a full observer node that runs an indexing ExEx and serves a small HTTP API from the same process.
[telscan.xyz](https://telscan.xyz) reads everything it renders from this API; it needs zero node JSON-RPC.

The design rule: never re-index what the node's own databases already contain.
Transactions, receipts, blocks, balances, code, gas price, consensus headers and outputs, epochs, validators, and generic read-only contract calls are answered by direct function calls on `RethEnv` (reth MDBX execution DB) and `ConsensusChain` (consensus pack files).
SQLite holds only the derived indexes reth maintains in no table:

- `address_txs`: address to transaction pointers, each tagged with its EIP-2718 type byte (reth's `AccountsHistory` is a state-change index, not a tx index)
- `tx_types`, `tx_type_counts`: transaction type to transaction pointers, plus exact per-type totals
- `consensus_blocks`: consensus output digest to the execution block range it produced, decoded from each block header
- `token_transfers`: ERC-20 transfer history by participant, by token, and by transaction
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

All list endpoints return `{items, total, page, per_page}` with 0-based `page`, `per_page` default 25 (max 100), newest first, with one exception: `/txs/{hash}/transfers` is in ascending `log_index` order.
Missing resources are `404 {"error":"not found"}`; malformed query parameters are 400.

| Route | Returns |
|---|---|
| `GET /health` | `{status, indexing_live, last_indexed, node_tip, lag}`; always 200, degradation is in the body |
| `GET /stats` | latest block, gas price (gwei), chain id, epoch, committee size, total txs; 5s server-side cache |
| `GET /txs` | the chain-wide transaction feed; `?type=` filters by transaction type (see below) |
| `GET /txs/{hash}` | one mined transaction with receipt fields (full input hex) and its first page of ERC-20 transfers |
| `GET /txs/{hash}/transfers` | the ERC-20 transfers one transaction emitted, paginated, in `log_index` order |
| `GET /blocks`, `GET /blocks/{number}` | block pages / one block |
| `GET /address/{addr}` | balance, nonce, `is_contract`, full code hex, indexed tx count |
| `GET /address/{addr}/txs` | the address's transaction history; accepts `?type=` |
| `GET /address/{addr}/transfers` | ERC-20 transfers where the address is sender or recipient |
| `GET /tokens/{addr}` | token name/symbol/decimals + live `totalSupply` |
| `GET /tokens/{addr}/transfers` | the token's transfer feed |
| `GET /epochs`, `GET /epochs/{n}` | epoch history from the consensus DB's epoch records; the detail route adds committee addresses from a registry read pinned to that epoch's seating block |
| `GET /epochs/current` | the in-progress epoch from the on-chain registry |
| `GET /validators` | the current committee's registry `ValidatorInfo`s |
| `GET /validators/leaders?window=200` | blocks per leader over the trailing window (max 1000) |
| `GET /consensus/latest` | the newest consensus header (number, epoch, round, digest, leader, `committed_at`) plus the execution tip and the consensus fields decoded from its header |
| `GET /consensus/blocks` | consensus header pages; `total` is the latest consensus number |
| `GET /consensus/blocks/{number}` | one full consensus output: header, every primary header in the sub-dag, reputation scores, batch summaries, `closes_epoch` |
| `GET /consensus/blocks/{number}/batches` | the output's batches with their transaction hashes |
| `GET /consensus/epochs`, `GET /consensus/epochs/{n}` | epoch records and certificates with consensus and execution block ranges; the detail route adds committee addresses, pack completeness, last committed rounds, final reputation scores, and certificate verification |
| `POST /call` | read-only `eth_call`-style contract read: `{to, data}` in, `{ok, result}` or `{ok:false, error, revert_data}` out |

Two wire details clients must respect.
Transaction and token-transfer `value` fields are u128 JSON numbers; parse responses directly with `serde_json::from_str::<T>`, never through a `serde_json::Value` intermediate, which destroys integers above `u64::MAX`.
Token transfers also carry `value_exact`, the lossless decimal string.

### Transaction type filter

`GET /txs` and `GET /address/{addr}/txs` accept `?type=`.
Values are `legacy`, `eip2930`, `eip1559`, `eip4844`, `eip7702` (case-insensitive) or the type byte `0` to `4`; anything else is 400.
Without `?type=` both routes behave exactly as before.
Telcoin Network's batch allowlist admits only legacy, EIP-2930 and EIP-1559 transactions today, so the `eip4844` and `eip7702` pages are empty.
The index is generic, so nothing here changes if the allowlist widens.

### Additive response fields

Every change to an existing response is a new field; existing clients are unaffected.

- `ApiBlock.consensus` on `/blocks` and `/blocks/{number}`: decoded from the execution header; `/blocks/{n}` adds one consensus-pack read for `consensus_number`, null if that epoch's pack is not held. Carries `digest` (with `digest_bs58`), `epoch`, `round`, `batch_index`, `worker_id`, `batch_digest`, `prev_randao`, `closes_epoch`. `consensus_number` is `null` on lists. The whole object is `null` for genesis. The single empty block of an empty epoch-closing output carries placeholder values: `batch_index` 0, `worker_id` 0 and an all-zero `batch_digest`.
- `ApiEpoch.record` and `ApiEpoch.certificate` on `/epochs` and `/epochs/{n}`: the stored epoch record and its certificate. `record` is `null` for the current epoch, `certificate` is `null` while uncertified, and `certificate.verified` (a BLS pairing check) is populated on `/epochs/{n}` only.
- `ApiTransaction.tx_type` (the EIP-2718 type byte) and `tx_type_name` on every transaction row.
- `/txs/{hash}` only: `token_transfers`, the first page (up to 100) of the transaction's ERC-20 transfers in `log_index` order, and `token_transfer_count`, the full count. Both are absent from list rows; `/txs/{hash}/transfers` pages through the rest. Both are also absent for a transaction the index has not reached yet (`/health.lag > 0`), so "not indexed yet" and "no transfers" stay distinguishable; `/txs/{hash}/transfers` returns an empty page in that state.
- `ApiTokenTransfer.log_index`: the position of the `Transfer` log within its transaction's receipt.

### Internal transactions

Per-transaction token transfers are the ERC-20 `Transfer` events in the receipt logs: a log with exactly three topics (`Transfer(address,address,uint256)`, `from`, `to`) and 32 bytes of data.
ERC-721's four-topic `Transfer` is excluded.
`token_address` is always the emitting contract; the indexer writes no synthetic rows for native TEL, so value moved by internal calls is not visible here.
There are no EVM call traces: that would require tn-reth to export its EVM config, and is out of scope.

### Consensus routes

Consensus block numbers start at 1; 0 is the pre-genesis anchor and returns 404, as does any number above the latest consensus number or any epoch above the current one.
`exec_blocks` on a consensus header is `null` when the indexer has not reached those blocks yet and also when the output produced no execution blocks (an empty output that did not close an epoch); compare against `last_indexed` from `/health` to tell the two apart.
A batch's `exec_block_number` follows the same rule and is `null` for batches past the end of the indexed range, which happens mid-output because the indexer commits one block per SQLite transaction.
`closes_epoch` on a consensus block is `null` until the epoch record exists.
On `/consensus/epochs` rows the current epoch has no `record`, `certificate`, `end_time` or range `last`, even if the node has already written its record; `consensus_range.first` and `exec_range.first` are `null` only when the previous epoch's record could not be read, which does not happen on a healthy node (the epoch record store is dense).
List routes leave `verified`, `committee_addresses`, `pack_complete`, `last_committed_rounds`, and `final_reputation_scores` as `null`; each costs extra reads or a BLS verification, so only the detail routes populate them.
An observer that does not hold an older epoch's consensus pack returns 404 for its blocks and `null` for `last_committed_rounds` and `final_reputation_scores`, never a 500; `pack_complete` is `false` when the pack is absent or incomplete, and `null` only for the current epoch and on list routes.
The by-number routes cannot tell an absent pack from one that cannot be read (truncated or corrupt): the node answers both the same way, so both are 404 / `null`.
When that looks wrong for an epoch the observer should hold, check the node log and `/blocks/{n}.consensus.consensus_number` for a block of that epoch; it resolves the pack by digest, which surfaces the real error as a `warn!` in the indexer log.

Encoding: all 32-byte digests are `0x` hex.
Consensus-header and epoch digests also carry a `*_bs58` companion, the full base58 form; its first 16 characters match what the node prints in its logs.
BLS public keys, BLS signatures, and authority identifiers are base58 strings, the same encoding as the existing `committee_bls`.

## Schema and disposability

The SQLite file is pure derived state, rebuilt from the node's own databases.
On open, a `schema_version` other than 3 (or leftover v1 tables) drops every table and the indexer re-replays from block 0.
A stored chain id that differs from the node's chainspec is a hard error, so an old file can't silently index a different network.
Delete the file and restart to rebuild from scratch; that is both the migration and the corruption story.

Writes are one SQLite transaction per block (pointer rows, type rows and counters, the consensus range upsert, transfer rows, token upserts, cursor advance), so a crash can't leave a half-indexed block: restart replays from `cursor + 1` with no holes.

### Upgrading to schema v3

Schema v3 adds `tx_types`, `tx_type_counts`, `consensus_blocks`, and the `address_txs.tx_type` column.
The first start on v3 drops the v2 tables and replays from block 0 (about 4.3M blocks on testnet).
The API keeps serving during the replay.
Node-backed routes are complete immediately; everything SQLite-backed is partial until `/health` reports `lag == 0`: `?type=`, `/txs/{hash}/transfers` and `token_transfers`, `exec_blocks` (the only SQLite-backed consensus field; `consensus_number` is resolved through the consensus pack by digest, never SQLite), and the existing address and transfer feeds.
Deploy in a low-traffic window.

## Bumping the tn-3 dependency

All eight tn git dependencies pin the same rev and must be bumped together in one commit; mixed revs give cargo two checkouts of the same crates and duplicate-type errors everywhere.
On every bump, re-check the `adiri` feature crate set against tn-3 (`grep -rn '^adiri' --include=Cargo.toml`), confirm the clap/eyre/tokio/futures/tracing majors still match tn-3's workspace, then run `cargo update && cargo check --all-targets && cargo check --features adiri && cargo test`.
The full procedure, including the ops runbook this repo follows, lives in tn-3's `integrate-exex.md` (Part 6).
