use std::{io::Write, time::Duration};

use alloy::{
    consensus::constants::GWEI_TO_WEI,
    network::TransactionBuilder,
    primitives::{Address, Bytes, U256},
    providers::{Provider, ProviderBuilder},
    rpc::types::TransactionRequest,
    signers::local::PrivateKeySigner,
    transports::http::reqwest::Url,
};
use clap::Parser;
use flate2::{Compression, write::ZlibEncoder};
use tokio::time::sleep;

use crate::chainio::IForcedInclusionStore::{self, IForcedInclusionStoreErrors};

mod blob;
mod chainio;

/// CLI for the forced inclusion tx sender tool.
#[derive(Debug, Parser)]
struct Cli {
    #[clap(subcommand)]
    command: Cmd,

    /// RPC URL of the L1 execution layer network.
    #[clap(long, env)]
    l1_rpc_url: Url,
    /// RPC URL of the L2 execution layer network.
    #[clap(long, env)]
    l2_rpc_url: Url,
    /// Private key of the forced inclusion tx signer. Needs to be funded with ETH on L1.
    #[clap(long, env)]
    l1_private_key: PrivateKeySigner,
    /// Private key of the forced inclusion tx signer. Needs to be funded with ETH on L2.
    #[clap(long, env)]
    l2_private_key: PrivateKeySigner,
    /// Address of the forced inclusion store contract on L1.
    #[clap(long, env)]
    forced_inclusion_store_address: Address,
}

/// Command to execute.
#[derive(Debug, Parser)]
enum Cmd {
    /// Read the forced inclusion queue from the contract.
    ReadQueue,
    /// Send a forced inclusion transaction.
    Send(SendCmdOptions),
    /// Send forced inclusion transactions in a loop.
    Spam(SpamCmdOptions),
}

/// Options for the send command.
#[derive(Debug, Clone, Copy, Default, Parser)]
struct SendCmdOptions {
    /// The nonce delta to use for the forced inclusion transactions.
    ///
    /// This is useful to send multiple forced batches with valid transactions
    /// from the same account.
    #[clap(long, default_value_t = 0)]
    nonce_delta: u64,
}

/// Options for the spam command.
#[derive(Debug, Clone, Copy, Default, Parser)]
struct SpamCmdOptions {
    /// The interval in seconds between forced inclusion transactions.
    #[clap(long, default_value_t = 24)]
    interval_secs: u64,
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    match cli.command {
        Cmd::ReadQueue => cli.read_queue().await,
        Cmd::Send(opts) => cli.send(opts).await,
        Cmd::Spam(opts) => cli.spam(opts).await,
    }
}

impl Cli {
    /// Send a forced inclusion transaction.
    async fn send(&self, opts: SendCmdOptions) -> eyre::Result<()> {
        let l1_provider = ProviderBuilder::new()
            .wallet(self.l1_private_key.clone())
            .connect_http(self.l1_rpc_url.clone());
        let l2_provider = ProviderBuilder::new()
            .wallet(self.l2_private_key.clone())
            .connect_http(self.l2_rpc_url.clone());

        let store = IForcedInclusionStore::new(self.forced_inclusion_store_address, l1_provider);

        let sender = self.l2_private_key.address();
        let current_nonce = l2_provider.get_transaction_count(sender).pending().await?;
        let starting_nonce = current_nonce + opts.nonce_delta;

        // Generate the L2 transaction to be force-included. Make it a simple transfer of 1 gwei.
        let l2_tx_req = TransactionRequest::default()
            .to(Address::ZERO)
            .with_nonce(starting_nonce)
            .value(U256::from(GWEI_TO_WEI));

        let l2_tx = l2_provider.fill(l2_tx_req).await?.try_into_envelope()?;
        println!("L2 transasction to be force-included: {:?}", l2_tx.hash());

        // Prepare the sidecar for the forced inclusion
        let compressed_batch = rlp_encode_and_compress(&vec![l2_tx])?;
        let byte_size = compressed_batch.len() as u32;
        let sidecar = blob::create_blob_sidecar_from_data_async(compressed_batch).await?;

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
                println!(
                    "❌ Forced inclusion batch failed! Error: {:?}",
                    decoded_error
                );
            }
        }

        Ok(())
    }

    /// Read the forced inclusion queue from the contract.
    async fn read_queue(self) -> eyre::Result<()> {
        let l1_provider = ProviderBuilder::new()
            .wallet(self.l1_private_key)
            .connect_http(self.l1_rpc_url);
        let store = IForcedInclusionStore::new(self.forced_inclusion_store_address, l1_provider);

        let tail = store.tail().call().await?;
        let head = store.head().call().await?;
        let size = tail.saturating_sub(head);

        if size == 0 {
            println!("Forced inclusion queue is empty");
            return Ok(());
        }

        for i in head..tail {
            match store.getForcedInclusion(U256::from(i)).call().await {
                Ok(fi) => println!("Forced inclusion {}: {:?}\n", i, fi),
                Err(e) => {
                    if let Some(dec) = e.as_decoded_interface_error::<IForcedInclusionStoreErrors>()
                    {
                        println!("Error reading forced inclusion {}: {:?}", i, dec);
                        continue;
                    } else {
                        println!("Error reading forced inclusion {}: {:?}", i, e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Send forced inclusion transactions in a loop.
    async fn spam(self, opts: SpamCmdOptions) -> eyre::Result<()> {
        let mut send_opts = SendCmdOptions::default();

        loop {
            println!("Sending forced-inclusion (nonce={})", send_opts.nonce_delta);
            if let Err(e) = self.send(send_opts).await {
                eprintln!("Error sednding forced-inclusion transaction: {:?}", e);
                return Err(e);
            }

            send_opts.nonce_delta += 1;
            sleep(Duration::from_secs(opts.interval_secs)).await;
        }
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
