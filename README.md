# taiko forced-inclusion transaction sender

Usage:

```shell
# clone the repo locally
git clone git@github.com:merklefruit/taiko-forced-inclusion-sender.git && cd taiko-forced-inclusion-sender

# fill out the .env file with the required variables
cp .env.example .env
vim .env

# to send a forced included transaction:
cargo run send

# to read the current queue from the contract:
cargo run read-queue
```
