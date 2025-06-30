use alloy::{
    primitives::Address, signers::local::PrivateKeySigner, transports::http::reqwest::Url,
};
use clap::Parser;

/// CLI for the forced inclusion toolbox.
#[derive(Debug, Parser)]
pub struct Cli {
    /// The command to execute.
    #[clap(subcommand)]
    pub command: Cmd,

    /// RPC URL of the L1 execution layer network.
    #[clap(long, env)]
    pub l1_rpc_url: Url,
    /// RPC URL of the L2 execution layer network.
    #[clap(long, env)]
    pub l2_rpc_url: Url,
    /// Private key of the forced inclusion tx signer. Needs to be funded with ETH on L1.
    #[clap(long, env)]
    pub l1_private_key: PrivateKeySigner,
    /// Private key of the forced inclusion tx signer. Needs to be funded with ETH on L2.
    #[clap(long, env)]
    pub l2_private_key: PrivateKeySigner,
    /// Address of the forced inclusion store contract on L1.
    #[clap(long, env)]
    pub forced_inclusion_store_address: Address,
}

/// Command to execute.
#[derive(Debug, Parser)]
pub enum Cmd {
    /// Read the forced inclusion queue from the contract.
    ReadQueue,
    /// Monitor the forced inclusion queue, printing new additions/removals.
    MonitorQueue,
    /// Send a forced inclusion transaction.
    Send(SendCmdOptions),
    /// Send forced inclusion transactions in a loop.
    Spam(SpamCmdOptions),
}

/// Options for the send command.
#[derive(Debug, Clone, Copy, Default, Parser)]
pub struct SendCmdOptions {
    /// The nonce delta to use for the forced inclusion transactions.
    ///
    /// This is useful to send multiple forced batches with valid transactions
    /// from the same account.
    #[clap(long, default_value_t = 0)]
    pub nonce_delta: u64,
}

/// Options for the spam command.
#[derive(Debug, Clone, Copy, Default, Parser)]
pub struct SpamCmdOptions {
    /// The interval in seconds between forced inclusion transactions.
    #[clap(long, default_value_t = 24)]
    pub interval_secs: u64,
}
