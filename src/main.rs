//! Observer-node binary hosting the explorer-indexer ExEx.
//!
//! The canonical host-binary shape: read (and clear) the BLS passphrase env
//! var before any threads exist, parse the CLI with the indexer's extension
//! flags flattened into the `node` subcommand, resolve the passphrase policy,
//! install the ExEx on the builder, and launch the node. The indexer serves
//! its own HTTP API — the observer runs WITHOUT `--http`; nothing here depends
//! on node JSON-RPC.

use clap::Parser;
use std::{net::SocketAddr, path::PathBuf};
use telcoin_network_cli::{cli::Cli, passphrase::get_bls_passphrase_from_env};
use tn_node::launch_node;

mod api;
mod exex;
mod extract;
mod node_reads;
mod status;
mod storage;
mod types;

/// Explorer-indexer CLI extension (flattened into the `node` subcommand).
#[derive(Debug, Clone, clap::Args)]
pub struct IndexerArgs {
    /// SQLite file. Defaults to `<datadir>/explorer-indexer/explorer.sqlite`.
    #[arg(long = "indexer.db-path", value_name = "PATH")]
    pub db_path: Option<PathBuf>,
    /// HTTP API bind address.
    #[arg(
        long = "indexer.api-addr",
        value_name = "SOCKET",
        default_value = "127.0.0.1:8560"
    )]
    pub api_addr: SocketAddr,
    /// Allowed CORS origins (comma-separated). Empty = any origin (read-only data).
    #[arg(long = "indexer.cors", value_name = "ORIGINS", value_delimiter = ',')]
    pub cors_origins: Vec<String>,
}

fn main() {
    // Must be the first statement of main: reads and clears TN_BLS_PASSPHRASE
    // before any threads exist (see `get_bls_passphrase_from_env`).
    let preloaded = get_bls_passphrase_from_env();
    let cli = Cli::<IndexerArgs>::parse();

    let passphrase = cli.resolve_bls_passphrase(preloaded).unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    });

    if let Err(err) = cli.run(
        passphrase,
        |mut builder, args, tn_datadir, key_config, version| {
            let db_path = args
                .db_path
                .clone()
                .unwrap_or_else(|| tn_datadir.join("explorer-indexer").join("explorer.sqlite"));
            builder.install_exex("explorer-indexer", move |ctx| {
                exex::indexer_exex(ctx, args, db_path)
            });
            launch_node(builder, tn_datadir, key_config, version)
        },
    ) {
        eprintln!("Error: {err:?}");
        std::process::exit(1);
    }
}
