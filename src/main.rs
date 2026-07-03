//! Observer-node binary hosting the explorer-indexer ExEx.
//!
//! Scaffold stage: the canonical three-step host-binary shape with a no-op ExEx,
//! proving the git-dep seam (submodule auto-fetch, vergen, resolution) end to end.
//! The real indexer modules replace the no-op in the next commits.

use clap::Parser;
use telcoin_network_cli::{cli::Cli, passphrase::get_bls_passphrase_from_env};
use tn_node::launch_node;

fn main() {
    // Must be the first statement of main: reads and clears TN_BLS_PASSPHRASE
    // before any threads exist (see `get_bls_passphrase_from_env`).
    let preloaded = get_bls_passphrase_from_env();
    let cli = Cli::<telcoin_network_cli::NoArgs>::parse();

    let passphrase = cli.resolve_bls_passphrase(preloaded).unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    });

    if let Err(err) = cli.run(passphrase, |mut builder, _, tn_datadir, key_config, version| {
        builder.install_exex("explorer-indexer", |_ctx| async move { Ok(()) });
        launch_node(builder, tn_datadir, key_config, version)
    }) {
        eprintln!("Error: {err:?}");
        std::process::exit(1);
    }
}
