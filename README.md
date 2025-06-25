# taiko forced-inclusion toolbox 🧰

Simple CLI tool to interact with the Taiko `ForcedInclusionStore` contract.

### Installation

Requires Rust and Cargo to be installed. You can install them from [rustup.rs](https://rustup.rs/).

```shell
# clone the repo locally
git clone git@github.com:merklefruit/taiko-forced-inclusion-sender.git && cd taiko-forced-inclusion-sender

# fill out the .env file with the required variables
cp .env.example .env
vim .env
```

### Usage

```shell
# to send a forced included transaction:
cargo run send

# to read the current queue from the contract:
cargo run read-queue

# to monitor the queue for new events as they are emitted live:
cargo run monitor-queue

# to periodically send a forced inclusion transaction in a loop:
cargo run spam
```
