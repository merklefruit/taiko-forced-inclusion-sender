use std::io::Write;

use alloy::{
    consensus::constants::GWEI_TO_WEI,
    primitives::{Address, Bytes, U256},
    providers::ProviderBuilder,
    rpc::types::TransactionRequest,
    signers::local::PrivateKeySigner,
    transports::http::reqwest::Url,
};
use clap::Parser;
use flate2::{Compression, write::ZlibEncoder};

use crate::chainio::IForcedInclusionStore::{self, IForcedInclusionStoreErrors};

mod blob;
mod chainio;

/// CLI for the forced inclusion tx sender tool.
#[derive(Debug, Parser)]
struct Cli {
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

#[tokio::main]
async fn main() -> eyre::Result<()> {
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    let l1_provider = ProviderBuilder::new()
        .wallet(cli.l1_private_key)
        .connect_http(cli.l1_rpc_url);
    let l2_provider = ProviderBuilder::new()
        .wallet(cli.l2_private_key)
        .connect_http(cli.l2_rpc_url);

    let store = IForcedInclusionStore::new(cli.forced_inclusion_store_address, l1_provider);

    // Generate the L2 transaction to be force-included. Make it a simple transfer of 1 gwei.
    let l2_tx_req = TransactionRequest::default()
        .to(Address::ZERO)
        .value(U256::from(1_000_000_000));
    let l2_tx = l2_provider.fill(l2_tx_req).await?.try_into_envelope()?;

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
