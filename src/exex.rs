//! The indexing loop: an ExEx that catches up via replay, follows live
//! `ChainExecuted` notifications, reconciles `Lagged` gaps by re-replaying,
//! and reports durable progress — the `examples/exex-indexer` model with a
//! real SQLite seam.
//!
//! Per executed block: `extract_block` → `token_fetch_states` → token-metadata
//! fetches through the SAME bounded blocking path as the API → `index_block`
//! in ONE SQLite transaction (rows + token upserts + cursor). A crash between
//! the metadata fetch and the commit just refetches on replay — idempotent.
//!
//! The axum API is spawned BEFORE catch-up so `/health` (and every direct-read
//! endpoint) is observable during a long first replay. The ExEx runs isolated
//! (`catch_unwind`, non-critical): if this loop errors or panics, the API task
//! stays up and degrades honestly — [`LiveGuard`] flips `indexing_live` to
//! false and `/health.lag` grows.

use crate::{
    api::{self, ApiState},
    extract::extract_block,
    node_reads::NodeReader,
    status::{IndexerStatus, LiveGuard},
    storage::{ReadPool, TokenRow, Writer},
    IndexerArgs,
};
use futures::StreamExt as _;
use std::{path::PathBuf, sync::Arc, time::SystemTime};
use tn_exex::{Chain, TnExExContext, TnExExNotification};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

/// The explorer-indexer ExEx entrypoint (installed by `main.rs`).
pub async fn indexer_exex(
    mut ctx: TnExExContext,
    args: IndexerArgs,
    db_path: PathBuf,
) -> eyre::Result<()> {
    info!(
        target: "exex::explorer",
        db = %db_path.display(),
        api = %args.api_addr,
        "explorer-indexer ExEx starting"
    );

    // 1. storage: migrations (drop-and-rebuild on version mismatch), chain-id
    //    guard, cursor load
    let chain_id = ctx.reth_env().chainspec().chain_id();
    let writer = Writer::open(db_path.clone(), chain_id).await?;
    let read_pool = ReadPool::open(db_path).await?;

    // 2. the bounded direct-read seam, shared by the API and this loop's
    //    token-metadata step
    let reader = NodeReader::new(ctx.reth_env().clone(), ctx.consensus_chain().clone());

    let status = Arc::new(IndexerStatus::default());
    if let Some(cursor) = writer.last_indexed().await? {
        status.set_last_indexed(cursor);
    }

    // 3. spawn the API BEFORE catch-up, with its own cancellation token, so
    //    /health is observable during a long first replay
    let shutdown = CancellationToken::new();
    let api_state = Arc::new(ApiState::new(reader.clone(), read_pool, status.clone()));
    let api_task = tokio::spawn({
        let shutdown = shutdown.clone();
        let addr = args.api_addr;
        let cors = args.cors_origins.clone();
        async move {
            if let Err(err) = api::serve(api_state, addr, cors, shutdown).await {
                tracing::error!(target: "indexer::api", %err, "API server exited with error");
            }
        }
    });

    // 4. liveness guard: ANY exit from this function — including an error
    //    return below — drops it and flips /health to degraded while the API
    //    task deliberately stays up
    let _live = LiveGuard::new(status.clone());

    // 5. catch up on history, then follow live notifications
    catch_up(&writer, &reader, &ctx, &status).await?;

    while let Some(notification) = ctx.next_notification().await {
        match notification {
            TnExExNotification::ChainExecuted { new } => {
                index_chain(&writer, &reader, &new, &status).await?;
                report_progress(&ctx, &status);
            }
            TnExExNotification::Lagged { missed } => {
                // `missed` is a best-effort magnitude — never used
                // arithmetically; the durable cursor drives the re-replay.
                warn!(target: "exex::explorer", missed, "lagged; reconciling via replay");
                catch_up(&writer, &reader, &ctx, &status).await?;
            }
            // Explicit match so a future variant is a compile error, not a
            // silent drop. Only ChainExecuted is ordered + replayable; this
            // indexer derives everything from it.
            TnExExNotification::CertificateAccepted { .. }
            | TnExExNotification::ConsensusOutput { .. } => {}
        }
    }

    // channel returned None => node is shutting down: take the API down too
    info!(
        target: "exex::explorer",
        last_indexed = ?status.last_indexed(),
        "node shutting down; stopping explorer-indexer"
    );
    shutdown.cancel();
    let _ = api_task.await;
    Ok(())
}

/// Replay from the durable cursor to the current chain tip.
///
/// Convergence on `Lagged`: each re-replay covers a strictly smaller window
/// (the cursor advanced), and live notifications buffered during replay whose
/// heights are at or below the cursor are skipped by the in-transaction guard.
async fn catch_up(
    writer: &Writer,
    reader: &NodeReader,
    ctx: &TnExExContext,
    status: &Arc<IndexerStatus>,
) -> eyre::Result<()> {
    let start = writer.last_indexed().await?.map_or(0, |cursor| cursor + 1);
    let tip = ctx.reth_env().last_block_number()?;
    status.set_node_tip(tip);
    info!(target: "exex::explorer", start, tip, "catching up via replay");

    let mut replay = ctx.replay_from(start)?; // inclusive [start, tip-at-call]
    while let Some(result) = replay.next().await {
        // surface replay errors instead of silently stalling
        if let TnExExNotification::ChainExecuted { new } = result? {
            index_chain(writer, reader, &new, status).await?;
        }
    }
    report_progress(ctx, status);
    Ok(())
}

/// Index every `(block, receipts)` pair in an executed chain segment
/// (replayed chains carry exactly one block; live chains may carry more).
async fn index_chain(
    writer: &Writer,
    reader: &NodeReader,
    chain: &Chain,
    status: &Arc<IndexerStatus>,
) -> eyre::Result<()> {
    for (block, receipts) in chain.blocks_and_receipts() {
        let extracted = extract_block(block, receipts);
        let number = extracted.number;

        // token-metadata step: only unseen or retry-pending (status 1)
        // candidates; fetches go through the SAME semaphore + spawn_blocking
        // path as the API — never EVM work inline on the async runtime
        let need_fetch = writer
            .token_fetch_states(extracted.token_candidates.clone())
            .await?;
        let mut token_rows = Vec::with_capacity(need_fetch.len());
        for token in need_fetch {
            let metadata = reader.token_metadata(token).await?;
            let ok = metadata.any_decoded();
            if !ok {
                debug!(
                    target: "exex::explorer",
                    %token,
                    block = number,
                    "token metadata fetch failed; marking for retry/terminal"
                );
            }
            token_rows.push(TokenRow {
                address: token,
                name: metadata.name,
                symbol: metadata.symbol,
                decimals: metadata.decimals,
                // 1 escalates to the terminal 2 in storage when already 1
                status: u8::from(!ok),
                fetched_at: unix_now(),
            });
        }

        // ONE SQLite transaction: cursor guard, rows, token upserts, cursor
        if writer.index_block(extracted, token_rows).await? {
            status.set_last_indexed(number);
        }
        status.set_node_tip(number);
    }
    Ok(())
}

/// Report durable progress to the node (non-blocking; latest-wins).
fn report_progress(ctx: &TnExExContext, status: &Arc<IndexerStatus>) {
    if let Some(height) = status.last_indexed() {
        ctx.report_finished_height(height);
    }
}

/// Unix seconds for `tokens.fetched_at`.
fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}
