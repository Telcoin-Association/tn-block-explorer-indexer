//! Pure per-block extraction: turns an executed block + its receipts into the
//! derived index rows SQLite persists (address participation pointers, ERC-20
//! transfer rows, and token-metadata fetch candidates).
//!
//! Everything in this module is pure — no IO, no clocks, no node access — which
//! makes it the primary unit-test target for the indexer's decode rules.

use tn_types::{
    hex_literal, Address, Block, BlockHeader as _, Log, Receipt, RecoveredBlock,
    TransactionTrait as _, B256, U256,
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

/// Everything the indexer persists for one executed block. Pure; no IO.
#[derive(Debug, Clone, Default)]
pub struct ExtractedBlock {
    /// The block number these rows were derived from.
    pub number: u64,
    /// `(address, block_number, tx_index)` pointer rows.
    pub address_rows: Vec<AddressRow>,
    /// Decoded ERC-20 `Transfer` logs.
    pub transfers: Vec<TransferRow>,
    /// Distinct contracts that emitted a conforming `Transfer` in this block,
    /// in first-seen order — the token-metadata fetch candidates.
    pub token_candidates: Vec<Address>,
}

/// Extract every derived index row from one executed block and its receipts.
///
/// Address rows per transaction: the sender; `to` when `Some`; and for
/// creations (`to == None`) the created contract address `sender.create(nonce)`
/// so a contract's page shows its own deployment transaction. Self-sends
/// produce two identical rows that collapse onto one primary key at write time
/// (`INSERT OR IGNORE`).
///
/// Transfer rows use the strict ERC-20 decode rule — see [`decode_transfer_log`].
pub fn extract_block(block: &RecoveredBlock<Block>, receipts: &[Receipt]) -> ExtractedBlock {
    let number = block.number();
    let mut address_rows = Vec::new();
    let mut transfers = Vec::new();
    let mut token_candidates: Vec<Address> = Vec::new();

    for (tx_index, (sender, tx)) in block.transactions_with_sender().enumerate() {
        let tx_index = tx_index as u64;
        address_rows.push(AddressRow {
            address: *sender,
            block_number: number,
            tx_index,
        });
        match tx.to() {
            Some(to) => address_rows.push(AddressRow {
                address: to,
                block_number: number,
                tx_index,
            }),
            // Contract creation: index the created address so the contract's
            // own page lists its deployment transaction.
            None => address_rows.push(AddressRow {
                address: sender.create(tx.nonce()),
                block_number: number,
                tx_index,
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

    ExtractedBlock {
        number,
        address_rows,
        transfers,
        token_candidates,
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

    fn make_tx(nonce: u64, to: Option<Address>) -> TransactionSigned {
        let tx = TxEip1559 {
            chain_id: 0x1e7,
            nonce,
            gas_limit: 100_000,
            max_fee_per_gas: 1_000_000_000,
            max_priority_fee_per_gas: 0,
            to: to.map(TxKind::Call).unwrap_or(TxKind::Create),
            value: U256::from(1u64),
            access_list: Default::default(),
            input: Bytes::new(),
        };
        TransactionSigned::new_unhashed(
            Transaction::Eip1559(tx),
            EthSignature::new(U256::from(1u64), U256::from(1u64), false),
        )
    }

    fn make_block(
        number: u64,
        txs: Vec<TransactionSigned>,
        senders: Vec<Address>,
    ) -> RecoveredBlock<Block> {
        let header = tn_types::ExecHeader {
            number,
            ..Default::default()
        };
        let body = tn_types::BlockBody {
            transactions: txs,
            ommers: vec![],
            withdrawals: None,
        };
        RecoveredBlock::new_unhashed(Block { header, body }, senders)
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
                    tx_index: 0
                },
                AddressRow {
                    address: recipient,
                    block_number: 5,
                    tx_index: 0
                },
                AddressRow {
                    address: sender_b,
                    block_number: 5,
                    tx_index: 1
                },
                AddressRow {
                    address: expected_created,
                    block_number: 5,
                    tx_index: 1
                },
            ]
        );
        // creation adds the created address, never a recipient row
        assert!(!extracted
            .address_rows
            .iter()
            .any(|r| r.tx_index == 1 && r.address != sender_b && r.address != expected_created));
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
