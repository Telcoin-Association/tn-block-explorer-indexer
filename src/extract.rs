//! Pure per-block extraction: turns an executed block + its receipts into the
//! derived index rows SQLite persists (address participation pointers, EIP-2718
//! tx-type pointers, ERC-20 transfer rows, token-metadata fetch candidates, and
//! the consensus-output ↔ execution-block mapping decoded from the header).
//!
//! Everything in this module is pure — no IO, no clocks, no node access — which
//! makes it the primary unit-test target for the indexer's decode rules.

use tn_types::{
    deconstruct_nonce, hex_literal, Address, Block, BlockHeader as _, ExecHeader, Log, Receipt,
    RecoveredBlock, TransactionTrait as _, Typed2718 as _, B256, U256,
};
use tracing::debug;

/// keccak256("Transfer(address,address,uint256)") — unit-tested against
/// [`tn_types::keccak256`].
pub const TRANSFER_TOPIC: B256 = B256::new(hex_literal::hex!(
    "ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
));

/// ERC-20 `name()` selector — unit-tested equal to `keccak256("name()")[..4]`.
pub const SEL_NAME: [u8; 4] = hex_literal::hex!("06fdde03");
/// ERC-20 `symbol()` selector — unit-tested equal to `keccak256("symbol()")[..4]`.
pub const SEL_SYMBOL: [u8; 4] = hex_literal::hex!("95d89b41");
/// ERC-20 `decimals()` selector — unit-tested equal to `keccak256("decimals()")[..4]`.
pub const SEL_DECIMALS: [u8; 4] = hex_literal::hex!("313ce567");
/// ERC-20 `totalSupply()` selector — unit-tested equal to `keccak256("totalSupply()")[..4]`.
pub const SEL_TOTAL_SUPPLY: [u8; 4] = hex_literal::hex!("18160ddd");

/// Names of the EIP-2718 transaction types the index knows, positioned by type
/// byte (`TX_TYPE_NAMES[ty as usize]`). These are the `?type=` values the API
/// accepts and the `tx_type_name` it emits.
///
/// TN's batch allowlist (`batch_allowlisted_tx_type`, `tn-types/src/lib.rs`)
/// admits only legacy / EIP-2930 / EIP-1559 today, so the 4844 and 7702 pages
/// are empty; the index stays generic so widening the allowlist needs no
/// schema change.
pub const TX_TYPE_NAMES: [&str; 5] = ["legacy", "eip2930", "eip1559", "eip4844", "eip7702"];

/// Name for an EIP-2718 type byte; `"unknown"` above EIP-7702 so an unexpected
/// future type can never panic a response builder.
pub fn tx_type_name(tx_type: u8) -> &'static str {
    TX_TYPE_NAMES
        .get(usize::from(tx_type))
        .copied()
        .unwrap_or("unknown")
}

/// Parse a `?type=` query value: a case-insensitive [`TX_TYPE_NAMES`] entry or
/// exactly one decimal digit `0..=4`, surrounding whitespace ignored.
///
/// Anything else (`"blob"`, `"5"`, `"02"`, `"+2"`, `""`) is `None`, which the
/// API maps to 400 — a filter must never silently fall back to another type.
pub fn parse_tx_type(s: &str) -> Option<u8> {
    let s = s.trim();
    if let Some(idx) = TX_TYPE_NAMES
        .iter()
        .position(|name| name.eq_ignore_ascii_case(s))
    {
        return Some(idx as u8);
    }
    match s.as_bytes() {
        // exactly one ASCII digit below the table length: rejects "+2", "02", "5"
        [d @ b'0'..=b'9'] if usize::from(d - b'0') < TX_TYPE_NAMES.len() => Some(d - b'0'),
        _ => None,
    }
}

/// One `address_txs` pointer row: an address participated in the transaction at
/// `(block_number, tx_index)` as sender, recipient, or created contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddressRow {
    /// Participating address.
    pub address: Address,
    /// Block containing the transaction.
    pub block_number: u64,
    /// Zero-based index of the transaction within its block.
    pub tx_index: u64,
    /// EIP-2718 type byte of the transaction (`Typed2718::ty`), denormalised
    /// onto the pointer so `/address/{addr}/txs?type=` filters with a single
    /// `AND tx_type = ?` instead of a join against `tx_types`.
    pub tx_type: u8,
}

/// One `tx_types` pointer row: the transaction at `(block_number, tx_index)`
/// has EIP-2718 type `tx_type`. Keyed `(tx_type, block_number, tx_index)` so
/// a `/txs?type=` page is a reverse primary-key scan, newest first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TxTypeRow {
    /// EIP-2718 type byte (`Typed2718::ty`): 0 legacy … 4 EIP-7702.
    pub tx_type: u8,
    /// Block containing the transaction.
    pub block_number: u64,
    /// Zero-based index of the transaction within its block.
    pub tx_index: u64,
}

/// One decoded ERC-20 `Transfer` log, keyed by `(block_number, tx_index, log_index)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferRow {
    /// Block containing the transaction that emitted the log.
    pub block_number: u64,
    /// Zero-based index of the transaction within its block.
    pub tx_index: u64,
    /// Index of the log within THIS transaction's receipt logs (not the RPC
    /// `logIndex`; only uniqueness + ordering matter).
    pub log_index: u64,
    /// The emitting ERC-20 contract.
    pub token: Address,
    /// `topics[1]` truncated to its last 20 bytes.
    pub from: Address,
    /// `topics[2]` truncated to its last 20 bytes.
    pub to: Address,
    /// Transferred amount (persisted as its exact decimal string).
    pub value: U256,
}

/// One `consensus_blocks` row: this block was produced by the consensus output
/// whose `ConsensusHeader::digest()` is `digest`, decoded from the exec header
/// alone ([`decode_consensus_fields`]) so indexing never reads `ConsensusChain`.
/// Consecutive blocks from one output share a digest; storage merges them into
/// a `[first_block, last_block]` range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConsensusRow {
    /// `ConsensusHeader::digest()` (the header's `parent_beacon_block_root`).
    pub digest: B256,
    /// Consensus epoch of the output's leader certificate.
    pub epoch: u32,
    /// Consensus round of the output's leader certificate.
    pub round: u32,
    /// The execution block this row was decoded from.
    pub block_number: u64,
}

/// The consensus provenance TN packs into every executed block's header.
///
/// Written by the header assembly in `tn-reth/src/evm/block.rs` (~:1493-1525)
/// from `TNPayload::new` (`tn-reth/src/payload.rs`, ~:102-138) and
/// `context_for_next_block` (`tn-reth/src/evm/config.rs`, ~:231-235):
/// - `parent_beacon_block_root` = `ConsensusHeader::digest()` for every
///   post-genesis block (genesis carries `Some(B256::ZERO)`, a reth Cancun
///   genesis artefact, and is excluded by number) → [`digest`](Self::digest)
/// - `nonce` = sub-dag leader's `(epoch << 32) | round`
///   (`tn-types/src/primary/header.rs:213`) → [`epoch`](Self::epoch), [`round`](Self::round)
/// - `difficulty` = `(batch_index << 16) | worker_id` → [`batch_index`](Self::batch_index),
///   [`worker_id`](Self::worker_id)
/// - `ommers_hash` = `Batch::digest()`; `B256::ZERO` for the empty epoch-closing
///   block → [`batch_digest`](Self::batch_digest)
/// - `mix_hash` = EIP-4399 prev_randao → [`prev_randao`](Self::prev_randao)
/// - `extra_data` = committee-shuffle seed, non-empty ONLY on the epoch-closing
///   block → [`closes_epoch`](Self::closes_epoch)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeaderConsensusFields {
    /// `ConsensusHeader::digest()` of the output that produced this block.
    pub digest: B256,
    /// Consensus epoch of the leader certificate (upper 32 bits of `nonce`).
    pub epoch: u32,
    /// Consensus round of the leader certificate (lower 32 bits of `nonce`).
    pub round: u32,
    /// Position of this block's batch within its output (`difficulty >> 16`).
    pub batch_index: u64,
    /// Worker that built the batch (`difficulty & 0xffff`).
    pub worker_id: u16,
    /// `Batch::digest()` of the executed batch (`ommers_hash`).
    pub batch_digest: B256,
    /// EIP-4399 prev_randao (`mix_hash`). Always `Some` for an alloy header;
    /// the `Option` mirrors `BlockHeader::mix_hash()` so the wire type can
    /// carry `null` should a future header format drop the field.
    pub prev_randao: Option<B256>,
    /// `true` iff this is the last block of its epoch (`extra_data` non-empty).
    pub closes_epoch: bool,
}

/// Decode [`HeaderConsensusFields`] from an executed block header.
///
/// `None` for genesis and for a `difficulty` that does not fit `u64`. Genesis
/// is block 0: reth writes a Cancun-style genesis header whose
/// `parent_beacon_block_root` is `Some(B256::ZERO)`, not `None`, so the number
/// is the only reliable marker (a missing root is also treated as genesis for
/// pre-Cancun chain specs). TN always packs a `usize` into `difficulty`
/// (`tn-reth/src/evm/config.rs:235`), so overflow means this is not a TN block
/// and none of its fields should be trusted. The nonce bytes are big-endian,
/// matching alloy's `B64: From<u64>` used on the write side (`ctx.nonce.into()`),
/// and `deconstruct_nonce` (`tn-types/src/helpers.rs:284`) returns `(epoch, round)`.
pub fn decode_consensus_fields(header: &ExecHeader) -> Option<HeaderConsensusFields> {
    if header.number == 0 {
        return None;
    }
    let digest = header.parent_beacon_block_root?;
    let (epoch, round) = deconstruct_nonce(u64::from_be_bytes(header.nonce.0));
    let difficulty = u64::try_from(header.difficulty).ok()?;
    Some(HeaderConsensusFields {
        digest,
        epoch,
        round,
        batch_index: difficulty >> 16,
        worker_id: (difficulty & 0xffff) as u16,
        batch_digest: header.ommers_hash,
        prev_randao: Some(header.mix_hash),
        closes_epoch: !header.extra_data.is_empty(),
    })
}

/// Everything the indexer persists for one executed block. Pure; no IO.
#[derive(Debug, Clone, Default)]
pub struct ExtractedBlock {
    /// The block number these rows were derived from.
    pub number: u64,
    /// `(address, block_number, tx_index, tx_type)` pointer rows.
    pub address_rows: Vec<AddressRow>,
    /// `(tx_type, block_number, tx_index)` pointer rows, exactly one per transaction.
    pub tx_types: Vec<TxTypeRow>,
    /// Decoded ERC-20 `Transfer` logs.
    pub transfers: Vec<TransferRow>,
    /// Distinct contracts that emitted a conforming `Transfer` in this block,
    /// in first-seen order — the token-metadata fetch candidates.
    pub token_candidates: Vec<Address>,
    /// The consensus output that produced this block; `None` only for genesis.
    pub consensus: Option<ConsensusRow>,
}

/// Extract every derived index row from one executed block and its receipts.
///
/// Address rows per transaction: the sender; `to` when `Some`; and for
/// creations (`to == None`) the created contract address `sender.create(nonce)`
/// so a contract's page shows its own deployment transaction. Self-sends
/// produce two identical rows that collapse onto one primary key at write time
/// (`INSERT OR IGNORE`). Every address row carries the transaction's EIP-2718
/// type, and one [`TxTypeRow`] is emitted per transaction.
///
/// Transfer rows use the strict ERC-20 decode rule — see [`decode_transfer_log`].
/// The consensus row comes from the header alone — see [`decode_consensus_fields`].
pub fn extract_block(block: &RecoveredBlock<Block>, receipts: &[Receipt]) -> ExtractedBlock {
    let number = block.number();
    let mut address_rows = Vec::new();
    let mut tx_types = Vec::new();
    let mut transfers = Vec::new();
    let mut token_candidates: Vec<Address> = Vec::new();

    for (tx_index, (sender, tx)) in block.transactions_with_sender().enumerate() {
        let tx_index = tx_index as u64;
        let tx_type = tx.ty();
        tx_types.push(TxTypeRow {
            tx_type,
            block_number: number,
            tx_index,
        });
        address_rows.push(AddressRow {
            address: *sender,
            block_number: number,
            tx_index,
            tx_type,
        });
        match tx.to() {
            Some(to) => address_rows.push(AddressRow {
                address: to,
                block_number: number,
                tx_index,
                tx_type,
            }),
            // Contract creation: index the created address so the contract's
            // own page lists its deployment transaction.
            None => address_rows.push(AddressRow {
                address: sender.create(tx.nonce()),
                block_number: number,
                tx_index,
                tx_type,
            }),
        }

        if let Some(receipt) = receipts.get(tx_index as usize) {
            for (log_index, log) in receipt.logs.iter().enumerate() {
                if let Some(row) = decode_transfer_log(log, number, tx_index, log_index as u64) {
                    if !token_candidates.contains(&row.token) {
                        token_candidates.push(row.token);
                    }
                    transfers.push(row);
                }
            }
        }
    }

    let consensus = decode_consensus_fields(block.header()).map(|fields| ConsensusRow {
        digest: fields.digest,
        epoch: fields.epoch,
        round: fields.round,
        block_number: number,
    });

    ExtractedBlock {
        number,
        address_rows,
        tx_types,
        transfers,
        token_candidates,
        consensus,
    }
}

/// Decode one receipt log as an ERC-20 `Transfer`, strictly.
///
/// A log is an ERC-20 Transfer iff `topics.len() == 3 && topics[0] ==
/// TRANSFER_TOPIC && data.len() == 32`. ERC-721's `Transfer` carries 4 topics
/// (indexed tokenId) and is excluded by the 3-topic check; non-32-byte data is
/// skipped (counted at `debug!`).
pub fn decode_transfer_log(
    log: &Log,
    block_number: u64,
    tx_index: u64,
    log_index: u64,
) -> Option<TransferRow> {
    let topics = log.topics();
    if topics.len() != 3 || topics[0] != TRANSFER_TOPIC {
        return None;
    }
    let data: &[u8] = log.data.data.as_ref();
    if data.len() != 32 {
        debug!(
            target: "exex::explorer",
            token = ?log.address,
            block_number,
            tx_index,
            data_len = data.len(),
            "skipping Transfer-topic log with non-32-byte data"
        );
        return None;
    }
    Some(TransferRow {
        block_number,
        tx_index,
        log_index,
        token: log.address,
        from: Address::from_word(topics[1]),
        to: Address::from_word(topics[2]),
        value: U256::from_be_slice(data),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tn_types::{
        keccak256, Bytes, EthSignature, Transaction, TransactionSigned, TxEip1559, TxKind,
    };

    fn addr(byte: u8) -> Address {
        Address::repeat_byte(byte)
    }

    /// A word whose last 20 bytes are `address` but whose upper 12 bytes are dirty.
    fn dirty_topic(address: Address) -> B256 {
        let mut word = [0xffu8; 32];
        word[12..].copy_from_slice(address.as_slice());
        B256::new(word)
    }

    fn clean_topic(address: Address) -> B256 {
        let mut word = [0u8; 32];
        word[12..].copy_from_slice(address.as_slice());
        B256::new(word)
    }

    fn value_word(value: u64) -> Bytes {
        Bytes::from(U256::from(value).to_be_bytes::<32>().to_vec())
    }

    fn transfer_log(token: Address, from: Address, to: Address, value: u64) -> Log {
        Log::new_unchecked(
            token,
            vec![TRANSFER_TOPIC, clean_topic(from), clean_topic(to)],
            value_word(value),
        )
    }

    /// Wrap any typed body in a dummy signature; extraction never recovers it.
    fn make_signed(tx: Transaction) -> TransactionSigned {
        TransactionSigned::new_unhashed(
            tx,
            EthSignature::new(U256::from(1u64), U256::from(1u64), false),
        )
    }

    fn make_tx(nonce: u64, to: Option<Address>) -> TransactionSigned {
        make_signed(Transaction::Eip1559(TxEip1559 {
            chain_id: 0x1e7,
            nonce,
            gas_limit: 100_000,
            max_fee_per_gas: 1_000_000_000,
            max_priority_fee_per_gas: 0,
            to: to.map(TxKind::Call).unwrap_or(TxKind::Create),
            value: U256::from(1u64),
            access_list: Default::default(),
            input: Bytes::new(),
        }))
    }

    /// A header carrying TN's consensus encoding: digest 0x01…, epoch 7 /
    /// round 9 in the nonce, batch 3 / worker 1 in difficulty, batch digest
    /// 0x02…, prev_randao 0x03…, not epoch-closing.
    fn consensus_header(number: u64) -> ExecHeader {
        ExecHeader {
            number,
            parent_beacon_block_root: Some(B256::repeat_byte(1)),
            nonce: ((7u64 << 32) | 9).into(),
            difficulty: U256::from((3u64 << 16) | 1),
            ommers_hash: B256::repeat_byte(2),
            mix_hash: B256::repeat_byte(3),
            ..Default::default()
        }
    }

    fn make_block_with_header(
        header: ExecHeader,
        txs: Vec<TransactionSigned>,
        senders: Vec<Address>,
    ) -> RecoveredBlock<Block> {
        let body = tn_types::BlockBody {
            transactions: txs,
            ommers: vec![],
            withdrawals: None,
        };
        RecoveredBlock::new_unhashed(Block { header, body }, senders)
    }

    /// A block with a default (genesis-like, no consensus fields) header.
    fn make_block(
        number: u64,
        txs: Vec<TransactionSigned>,
        senders: Vec<Address>,
    ) -> RecoveredBlock<Block> {
        make_block_with_header(
            ExecHeader {
                number,
                ..Default::default()
            },
            txs,
            senders,
        )
    }

    fn receipt_with_logs(logs: Vec<Log>) -> Receipt {
        Receipt {
            success: true,
            cumulative_gas_used: 21_000,
            logs,
            ..Default::default()
        }
    }

    #[test]
    fn transfer_topic_matches_keccak_of_signature() {
        assert_eq!(
            TRANSFER_TOPIC,
            keccak256("Transfer(address,address,uint256)")
        );
    }

    #[test]
    fn selector_consts_match_keccak_derivation() {
        for (sel, sig) in [
            (SEL_NAME, "name()"),
            (SEL_SYMBOL, "symbol()"),
            (SEL_DECIMALS, "decimals()"),
            (SEL_TOTAL_SUPPLY, "totalSupply()"),
        ] {
            assert_eq!(sel, keccak256(sig)[..4], "selector mismatch for {sig}");
        }
    }

    #[test]
    fn tx_type_names_and_parse_round_trip() {
        for (ty, name) in TX_TYPE_NAMES.iter().enumerate() {
            let ty = ty as u8;
            assert_eq!(tx_type_name(ty), *name);
            assert_eq!(parse_tx_type(name), Some(ty), "name {name}");
            assert_eq!(
                parse_tx_type(&name.to_uppercase()),
                Some(ty),
                "upper-case {name}"
            );
            assert_eq!(parse_tx_type(&ty.to_string()), Some(ty), "digit {ty}");
        }
        assert_eq!(tx_type_name(5), "unknown");
        assert_eq!(tx_type_name(u8::MAX), "unknown");

        for (input, expected) in [
            ("EIP1559", Some(2)),
            ("Legacy", Some(0)),
            ("2", Some(2)),
            (" 1 ", Some(1)),
            ("\teip7702\n", Some(4)),
            ("blob", None),
            ("5", None),
            ("unknown", None),
            ("", None),
            ("02", None),
            ("+2", None),
            ("-1", None),
            ("eip 1559", None),
        ] {
            assert_eq!(parse_tx_type(input), expected, "input {input:?}");
        }
    }

    #[test]
    fn decode_consensus_fields_round_trips_tn_header_encoding() {
        let mut header = consensus_header(42);
        assert_eq!(
            decode_consensus_fields(&header),
            Some(HeaderConsensusFields {
                digest: B256::repeat_byte(1),
                epoch: 7,
                round: 9,
                batch_index: 3,
                worker_id: 1,
                batch_digest: B256::repeat_byte(2),
                prev_randao: Some(B256::repeat_byte(3)),
                closes_epoch: false,
            })
        );

        // the committee-shuffle seed in extra_data marks the epoch-closing block
        header.extra_data = Bytes::from(vec![0xaa; 32]);
        assert!(
            decode_consensus_fields(&header)
                .expect("closing header decodes")
                .closes_epoch
        );

        // genesis: no consensus output produced it
        header.parent_beacon_block_root = None;
        assert_eq!(decode_consensus_fields(&header), None);
    }

    #[test]
    fn decode_consensus_fields_splits_nonce_and_difficulty_at_bit_boundaries() {
        let header = ExecHeader {
            nonce: u64::MAX.into(),
            difficulty: U256::from(u64::MAX),
            ..consensus_header(1)
        };
        let fields = decode_consensus_fields(&header).expect("decodes");
        assert_eq!((fields.epoch, fields.round), (u32::MAX, u32::MAX));
        assert_eq!(
            (fields.batch_index, fields.worker_id),
            (u64::MAX >> 16, u16::MAX)
        );
    }

    #[test]
    fn decode_consensus_fields_rejects_difficulty_wider_than_u64() {
        let header = ExecHeader {
            difficulty: U256::from(1u128 << 64),
            ..consensus_header(1)
        };
        assert_eq!(decode_consensus_fields(&header), None);
    }

    #[test]
    fn conforming_transfer_log_decodes() {
        let (token, from, to) = (addr(0xaa), addr(0xbb), addr(0xcc));
        let row = decode_transfer_log(&transfer_log(token, from, to, 12345), 7, 1, 2)
            .expect("conforming log decodes");
        assert_eq!(row.token, token);
        assert_eq!(row.from, from);
        assert_eq!(row.to, to);
        assert_eq!(row.value, U256::from(12345u64));
        assert_eq!((row.block_number, row.tx_index, row.log_index), (7, 1, 2));
    }

    #[test]
    fn four_topic_erc721_transfer_is_excluded() {
        let log = Log::new_unchecked(
            addr(0xaa),
            vec![
                TRANSFER_TOPIC,
                clean_topic(addr(0xbb)),
                clean_topic(addr(0xcc)),
                B256::with_last_byte(1), // indexed tokenId
            ],
            Bytes::new(),
        );
        assert!(decode_transfer_log(&log, 0, 0, 0).is_none());
    }

    #[test]
    fn two_topic_log_is_excluded() {
        let log = Log::new_unchecked(
            addr(0xaa),
            vec![TRANSFER_TOPIC, clean_topic(addr(0xbb))],
            value_word(1),
        );
        assert!(decode_transfer_log(&log, 0, 0, 0).is_none());
    }

    #[test]
    fn wrong_data_length_is_excluded() {
        let mut data = U256::from(5u64).to_be_bytes::<32>().to_vec();
        data.push(0); // 33 bytes
        let log = Log::new_unchecked(
            addr(0xaa),
            vec![
                TRANSFER_TOPIC,
                clean_topic(addr(0xbb)),
                clean_topic(addr(0xcc)),
            ],
            Bytes::from(data),
        );
        assert!(decode_transfer_log(&log, 0, 0, 0).is_none());
    }

    #[test]
    fn wrong_event_topic_is_excluded() {
        let log = Log::new_unchecked(
            addr(0xaa),
            vec![
                keccak256("Approval(address,address,uint256)"),
                clean_topic(addr(0xbb)),
                clean_topic(addr(0xcc)),
            ],
            value_word(1),
        );
        assert!(decode_transfer_log(&log, 0, 0, 0).is_none());
    }

    #[test]
    fn topic_to_address_truncates_last_twenty_bytes() {
        let from = addr(0x11);
        let to = addr(0x22);
        let log = Log::new_unchecked(
            addr(0xaa),
            vec![TRANSFER_TOPIC, dirty_topic(from), dirty_topic(to)],
            value_word(9),
        );
        let row = decode_transfer_log(&log, 0, 0, 0).expect("decodes despite dirty upper bytes");
        assert_eq!(row.from, from);
        assert_eq!(row.to, to);
    }

    #[test]
    fn address_rows_cover_sender_recipient_and_creation() {
        let (sender_a, recipient, sender_b) = (addr(0x01), addr(0x02), addr(0x03));
        let call_tx = make_tx(0, Some(recipient));
        let create_tx = make_tx(7, None);
        let block = make_block(5, vec![call_tx, create_tx], vec![sender_a, sender_b]);
        let receipts = vec![receipt_with_logs(vec![]), receipt_with_logs(vec![])];

        let extracted = extract_block(&block, &receipts);

        let expected_created = sender_b.create(7);
        assert_eq!(
            extracted.address_rows,
            vec![
                AddressRow {
                    address: sender_a,
                    block_number: 5,
                    tx_index: 0,
                    tx_type: 2,
                },
                AddressRow {
                    address: recipient,
                    block_number: 5,
                    tx_index: 0,
                    tx_type: 2,
                },
                AddressRow {
                    address: sender_b,
                    block_number: 5,
                    tx_index: 1,
                    tx_type: 2,
                },
                AddressRow {
                    address: expected_created,
                    block_number: 5,
                    tx_index: 1,
                    tx_type: 2,
                },
            ]
        );
        // creation adds the created address, never a recipient row
        assert!(!extracted
            .address_rows
            .iter()
            .any(|r| r.tx_index == 1 && r.address != sender_b && r.address != expected_created));
        // one tx_types row per transaction, both EIP-1559
        assert_eq!(
            extracted.tx_types,
            vec![
                TxTypeRow {
                    tx_type: 2,
                    block_number: 5,
                    tx_index: 0,
                },
                TxTypeRow {
                    tx_type: 2,
                    block_number: 5,
                    tx_index: 1,
                },
            ]
        );
    }

    #[test]
    fn tx_types_carry_each_transactions_eip2718_type() {
        let sender = addr(0x0d);
        // Default legacy / EIP-2930 bodies are creations (TxKind::Create); only
        // their type byte matters here.
        let txs = vec![
            make_signed(Transaction::Legacy(Default::default())),
            make_tx(1, Some(addr(0x0e))),
            make_signed(Transaction::Eip2930(Default::default())),
        ];
        let block = make_block(11, txs, vec![sender; 3]);

        let extracted = extract_block(&block, &[]);

        assert_eq!(
            extracted.tx_types,
            vec![
                TxTypeRow {
                    tx_type: 0,
                    block_number: 11,
                    tx_index: 0,
                },
                TxTypeRow {
                    tx_type: 2,
                    block_number: 11,
                    tx_index: 1,
                },
                TxTypeRow {
                    tx_type: 1,
                    block_number: 11,
                    tx_index: 2,
                },
            ]
        );
        // every address row of a transaction carries that transaction's type
        assert_eq!(extracted.address_rows.len(), 6);
        for row in &extracted.address_rows {
            let expected = extracted.tx_types[row.tx_index as usize].tx_type;
            assert_eq!(row.tx_type, expected, "row {row:?}");
        }
    }

    #[test]
    fn consensus_row_decoded_from_header_and_absent_for_genesis() {
        let block = make_block_with_header(
            consensus_header(21),
            vec![make_tx(0, Some(addr(0x02)))],
            vec![addr(0x01)],
        );
        assert_eq!(
            extract_block(&block, &[]).consensus,
            Some(ConsensusRow {
                digest: B256::repeat_byte(1),
                epoch: 7,
                round: 9,
                block_number: 21,
            })
        );

        // genesis has no parent_beacon_block_root: no consensus output produced it
        let genesis = make_block(0, vec![], vec![]);
        assert_eq!(extract_block(&genesis, &[]).consensus, None);

        // the real reth genesis header carries Some(B256::ZERO) (Cancun genesis
        // artefact) with zeroed nonce/difficulty — still no consensus output
        let mut real_genesis = consensus_header(0);
        real_genesis.parent_beacon_block_root = Some(B256::ZERO);
        assert_eq!(decode_consensus_fields(&real_genesis), None);
        let block = make_block_with_header(real_genesis, vec![], vec![]);
        assert_eq!(extract_block(&block, &[]).consensus, None);
    }

    #[test]
    fn self_send_yields_two_identical_rows_pre_dedup() {
        // The PK collapse happens at write time (INSERT OR IGNORE); extraction
        // deliberately emits both rows. The SQLite-level dedup is asserted in
        // storage.rs tests.
        let me = addr(0x0a);
        let block = make_block(1, vec![make_tx(0, Some(me))], vec![me]);
        let extracted = extract_block(&block, &[receipt_with_logs(vec![])]);
        assert_eq!(extracted.address_rows.len(), 2);
        assert_eq!(extracted.address_rows[0], extracted.address_rows[1]);
    }

    #[test]
    fn transfers_and_candidates_extracted_from_receipt_logs() {
        let (token_a, token_b) = (addr(0xa1), addr(0xa2));
        let (from, to) = (addr(0xb1), addr(0xb2));
        let sender = addr(0xc1);
        let block = make_block(9, vec![make_tx(0, Some(token_a))], vec![sender]);
        let receipts = vec![receipt_with_logs(vec![
            transfer_log(token_a, from, to, 100),
            // non-conforming log interleaved: ignored
            Log::new_unchecked(token_a, vec![TRANSFER_TOPIC], Bytes::new()),
            transfer_log(token_b, to, from, 200),
            transfer_log(token_a, from, to, 300),
        ])];

        let extracted = extract_block(&block, &receipts);

        assert_eq!(extracted.transfers.len(), 3);
        // log_index is the position within the receipt's log array
        assert_eq!(
            extracted
                .transfers
                .iter()
                .map(|t| t.log_index)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
        // distinct emitting contracts, first-seen order
        assert_eq!(extracted.token_candidates, vec![token_a, token_b]);
    }
}
