#![cfg_attr(not(test), warn(unused_crate_dependencies))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

use std::{io::Write, time::Duration};

use alloy::{
    consensus::{Transaction, constants::GWEI_TO_WEI},
    network::TransactionBuilder,
    primitives::{Address, Bytes, U256},
    providers::{Provider, ProviderBuilder, WalletProvider},
    rpc::types::TransactionRequest,
};
use clap::Parser;
use flate2::{Compression, write::ZlibEncoder};
use futures::StreamExt;
use tokio::time::sleep;

mod blob;
use blob::create_blob_sidecar_from_data_async;

mod cli;
use cli::{Cli, Cmd, SendCmdOptions, SpamCmdOptions};

mod chainio;
use chainio::{
    DefaultWalletProvider,
    IForcedInclusionStore::{
        self, ForcedInclusionConsumed, ForcedInclusionStored, IForcedInclusionStoreErrors,
        IForcedInclusionStoreInstance,
    },
};

#[tokio::main]
async fn main() -> eyre::Result<()> {
    if let Ok(custom_env_file) = std::env::var("ENV_FILE") {
        // Try from custom env file, and abort if it fails
        dotenvy::from_filename(custom_env_file)?;
    } else {
        // Try from default .env file, and ignore if it fails. It might
        // be that the user isn't using it.
        dotenvy::dotenv().ok();
    }

    let cli = Cli::parse();

    let l1 = ProviderBuilder::new()
        .wallet(cli.l1_private_key)
        .connect_http(cli.l1_rpc_url);
    let l2 = ProviderBuilder::new()
        .wallet(cli.l2_private_key)
        .connect_http(cli.l2_rpc_url);

    let store = IForcedInclusionStore::new(cli.forced_inclusion_store_address, l1);

    match cli.command {
        Cmd::ReadQueue => read_queue(store).await,
        Cmd::MonitorQueue => monitor_queue(store).await,
        Cmd::Send(opts) => send_one(opts, &l2, &store).await,
        Cmd::Spam(opts) => spam(opts, l2, store).await,
    }
}

/// Send a forced inclusion transaction.
async fn send_one(
    opts: SendCmdOptions,
    l2: &DefaultWalletProvider,
    store: &IForcedInclusionStoreInstance<DefaultWalletProvider>,
) -> eyre::Result<()> {
    // Generate the L2 transaction to be force-included. Make it a simple transfer of 1 gwei.
    let mut l2_tx_req = TransactionRequest::default()
        .to(Address::ZERO)
        .value(U256::from(GWEI_TO_WEI));

    // If a nonce delta is provided, calculate the nonce manually instead of using the
    // default `CachedNonceManager` value.
    if opts.nonce_delta > 0 {
        let sender = l2.wallet().default_signer().address();
        let pending_nonce = l2.get_transaction_count(sender).pending().await?;
        l2_tx_req.set_nonce(pending_nonce + opts.nonce_delta);
    }

    let l2_tx = l2.fill(l2_tx_req).await?.try_into_envelope()?;
    println!(
        "🔍 L2 tx to be force-included: nonce={}, hash={}",
        l2_tx.nonce(),
        l2_tx.hash()
    );

    // Prepare the sidecar for the forced inclusion
    let compressed_batch = rlp_encode_and_compress(&vec![l2_tx])?;
    let byte_size = compressed_batch.len() as u32;
    let sidecar = create_blob_sidecar_from_data_async(compressed_batch).await?;

    // Get the required fee for the forced inclusion
    let fee_wei = U256::from(store.feeInGwei().call().await? * GWEI_TO_WEI);

    // Send the forced inclusion transaction on L1
    match store
        .storeForcedInclusion(0, 0, byte_size)
        .sidecar(sidecar)
        .value(fee_wei)
        .send()
        .await
    {
        Ok(tx) => {
            let receipt = tx.get_receipt().await?;
            if receipt.status() {
                println!(
                    "✅ Forced inclusion batch sent successfully! Hash: {}",
                    receipt.transaction_hash
                );
            } else {
                println!(
                    "❌ Forced inclusion batch failed! Status: {}",
                    receipt.transaction_hash
                );
            }
        }
        Err(e) => {
            let decoded_error = e
                .as_decoded_interface_error::<IForcedInclusionStoreErrors>()
                .ok_or(e)?;

            println!("❌ Forced inclusion batch failed! Error: {decoded_error:?}",);
        }
    }

    Ok(())
}

/// Read the forced inclusion queue from the contract.
async fn read_queue(
    store: IForcedInclusionStoreInstance<DefaultWalletProvider>,
) -> eyre::Result<()> {
    let tail = store.tail().call().await?;
    let head = store.head().call().await?;
    let size = tail.saturating_sub(head);

    if size == 0 {
        println!("Forced inclusion queue is empty");
        return Ok(());
    }

    for i in head..tail {
        match store.getForcedInclusion(U256::from(i)).call().await {
            Ok(fi) => println!("Forced inclusion {i}: {fi:?}\n"),
            Err(e) => {
                if let Some(dec) = e.as_decoded_interface_error::<IForcedInclusionStoreErrors>() {
                    println!("Error reading forced inclusion {i}: {dec:?}");
                } else {
                    println!("Error reading forced inclusion {i}: {e:?}");
                }
            }
        }
    }

    Ok(())
}

/// Monitor events in the forced inclusion queue
async fn monitor_queue(
    store: IForcedInclusionStoreInstance<DefaultWalletProvider>,
) -> eyre::Result<()> {
    let stored = store.ForcedInclusionStored_filter().filter;
    let consumed = store.ForcedInclusionConsumed_filter().filter;

    let mut stored_sub = store.provider().watch_logs(&stored).await?.into_stream();
    let mut consumed_sub = store.provider().watch_logs(&consumed).await?.into_stream();

    println!("Monitoring forced inclusion queue...");
    loop {
        tokio::select! {
            Some(events) = stored_sub.next() => {
                if let Some(event) = events.first() {
                    let decoded = event.log_decode::<ForcedInclusionStored>()?;
                    println!("New forced inclusion stored: {:?}", decoded.data().forcedInclusion);
                }
            }
            Some(consumed_event) = consumed_sub.next() => {
                if let Some(event) = consumed_event.first() {
                    let decoded = event.log_decode::<ForcedInclusionConsumed>()?;
                    println!("Forced inclusion consumed: {:?}", decoded.data().forcedInclusion);
                }
            }
        }
    }
}

/// Send forced inclusion transactions in a loop.
async fn spam(
    opts: SpamCmdOptions,
    l2: DefaultWalletProvider,
    store: IForcedInclusionStoreInstance<DefaultWalletProvider>,
) -> eyre::Result<()> {
    let send_opts = SendCmdOptions::default();

    loop {
        // NOTE: by using the default `CachedNonceManager`, the nonce will be incremented
        // automatically by the provider without making new RPC calls.
        if let Err(e) = send_one(send_opts, &l2, &store).await {
            eprintln!("Error sending forced-inclusion: {e:?}");
            return Err(e);
        }

        sleep(Duration::from_secs(opts.interval_secs)).await;
    }
}

/// RLP-encode and compress with zlib a given encodable object.
pub fn rlp_encode_and_compress<E: alloy_rlp::Encodable>(b: &E) -> std::io::Result<Bytes> {
    let rlp_encoded_tx_list = alloy_rlp::encode(b);
    zlib_compress(&rlp_encoded_tx_list)
}

/// Compress the input bytes using `zlib`.
pub fn zlib_compress(input: &[u8]) -> std::io::Result<Bytes> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(input)?;
    encoder.finish().map(Bytes::from)
}
