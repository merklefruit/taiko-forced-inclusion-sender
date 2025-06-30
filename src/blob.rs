// Imported from: mk1 src/primitives/blob.rs

use std::sync::LazyLock;

use alloy::primitives::Bytes;
use alloy::{
    consensus::{Blob, BlobTransactionSidecar},
    eips::eip4844::BYTES_PER_BLOB,
};
use tokio::runtime::{Builder, Runtime};

// Constants
const ENCODING_VERSION: u8 = 0;
const ROUNDS: usize = 1024;

/// The maximum size of a blob's data, in bytes. Corresponds to:
/// - 4 bytes per field element * 31 field elements per row
/// - 3 additional bytes for correct field element alignment
/// - multiplied by 1024 rows
/// - minus 1 byte for version, and 3 bytes for length prefix
/// - This gives us 130044 bytes of usable space per blob
pub const MAX_BLOB_DATA_SIZE: usize = (4 * 31 + 3) * 1024 - 4; // (127 * 1024) - 4 = 130044

/// Define a custom thread pool with larger stack size for blob encoding.
///
/// The reason for this is that if we tried to `tokio::spawn()` a task and call
/// `BlobTransactionSidecar::try_from_blobs_bytes` inside it, the binary panics with a
/// "tokio runtime: stack overflow" error. The default stack size is 2MB which is not enough here.
static BLOB_THREAD_POOL: LazyLock<Runtime> = LazyLock::new(|| {
    Builder::new_multi_thread()
        .thread_name("blob-worker")
        .worker_threads(2)
        .thread_stack_size(8 * 1024 * 1024) // 8MB stack size
        .build()
        .expect("Failed to create blob worker thread pool")
});

/// An error type for blob encoding errors.
#[derive(Debug, thiserror::Error)]
#[allow(missing_docs)]
pub enum BlobError {
    #[error("too much data to encode in one blob: len={0}")]
    InputTooLarge(usize),
    #[error("data did not fit in blob: read_offset={read_offset}, data_len={data_len}")]
    DataDidNotFit { read_offset: usize, data_len: usize },
    #[error("KZG error: {0}")]
    KZGError(Box<dyn std::error::Error + Sync + Send>),
    #[error("thread panicked: {0}")]
    ThreadPanicked(#[from] tokio::task::JoinError),
}

/// Encodes the provided input data into a list of blobs, and returns a sidecar.
///
/// This operation blocks the current thread until the encoding is complete.
pub fn create_blob_sidecar_from_data_blocking(
    data: &[u8],
) -> Result<BlobTransactionSidecar, BlobError> {
    // Split the input data into chunks of `MAX_BLOB_DATA_SIZE` and encode each chunk into a blob
    let blobs = data
        .chunks(MAX_BLOB_DATA_SIZE)
        .map(create_blob_from_data)
        .collect::<Result<Vec<_>, _>>()?;

    // Create a sidecar from the blob bytes (blocking)
    BlobTransactionSidecar::try_from_blobs_bytes(blobs)
        .map_err(|e| BlobError::KZGError(Box::new(e)))
}

/// Encodes the provided input data into a list of blobs, and returns a sidecar.
///
/// This function is async and uses a blocking thread pool with custom stack size for blob encoding.
pub async fn create_blob_sidecar_from_data_async(
    data: Bytes,
) -> Result<BlobTransactionSidecar, BlobError> {
    BLOB_THREAD_POOL
        .spawn_blocking(move || create_blob_sidecar_from_data_blocking(&data))
        .await
        .map_err(BlobError::ThreadPanicked)?
}

/// Encodes the provided input data into the blob.
///
/// The encoding scheme works in rounds. In each round we process 4 field elements (each 32 bytes).
/// Each field element is written in two parts: a 6‑bit value (at the beginning) and a 31‑byte
/// chunk. In round 0 the first field element reserves bytes [1..5] to encode the version and data
/// length.
///
/// Data bounds: 0 <= `data.len()` <= [`MAX_BLOB_DATA_SIZE`]
///
/// Ported from: <https://github.com/ethereum-optimism/optimism/blob/0e4b867e08ed4dfcb5f1a76693f17392b189a7f6/op-service/eth/blob.go#L90>
pub fn create_blob_from_data(data: &[u8]) -> Result<Bytes, BlobError> {
    let mut out = Blob::default();

    if data.len() > MAX_BLOB_DATA_SIZE {
        return Err(BlobError::InputTooLarge(data.len()));
    }

    let mut read_offset: usize = 0;
    let mut write_offset: usize = 0;

    // Process the input data in rounds.
    for round in 0..ROUNDS {
        if read_offset >= data.len() {
            break;
        }

        let x = if round == 0 {
            // For round 0, reserve the first 4 bytes of the first field element.
            let mut buf = [0u8; 31];
            buf[0] = ENCODING_VERSION;
            let ilen = data.len() as u32;
            buf[1] = ((ilen >> 16) & 0xFF) as u8;
            buf[2] = ((ilen >> 8) & 0xFF) as u8;
            buf[3] = (ilen & 0xFF) as u8;

            // Copy as many bytes as possible into buf starting at index 4.
            let available = 31 - 4;
            let n = std::cmp::min(available, data.len() - read_offset);
            buf[4..4 + n].copy_from_slice(&data[read_offset..read_offset + n]);
            read_offset += n;

            // First field element: encode one 6‑bit value from input.
            let x = read_one_byte(data, &mut read_offset);
            let six_bits_of_x = x & 0b0011_1111;
            write_one_byte(&mut out.0, &mut write_offset, six_bits_of_x);
            write_31_bytes(&mut out.0, &mut write_offset, &buf);

            x
        } else {
            // For subsequent rounds, fill buf from data.
            let buf = read_31_bytes(data, &mut read_offset);
            let x = read_one_byte(data, &mut read_offset);
            let six_bits_of_x = x & 0b0011_1111;
            write_one_byte(&mut out.0, &mut write_offset, six_bits_of_x);
            write_31_bytes(&mut out.0, &mut write_offset, &buf);

            x
        };

        // Second field element: combine bits from x and a new byte.
        let buf = read_31_bytes(data, &mut read_offset);
        let y = read_one_byte(data, &mut read_offset);
        let b = (y & 0b0000_1111) | ((x & 0b1100_0000) >> 2);
        write_one_byte(&mut out.0, &mut write_offset, b);
        write_31_bytes(&mut out.0, &mut write_offset, &buf);

        // Third field element: encode another 6‑bit value.
        let buf = read_31_bytes(data, &mut read_offset);
        let z = read_one_byte(data, &mut read_offset);
        let six_bits_of_z = z & 0b0011_1111;
        write_one_byte(&mut out.0, &mut write_offset, six_bits_of_z);
        write_31_bytes(&mut out.0, &mut write_offset, &buf);

        // Fourth field element: combine bits from y and z.
        let buf = read_31_bytes(data, &mut read_offset);
        let d = ((z & 0b1100_0000) >> 2) | ((y & 0b1111_0000) >> 4);
        write_one_byte(&mut out.0, &mut write_offset, d);
        write_31_bytes(&mut out.0, &mut write_offset, &buf);
    }

    if read_offset < data.len() {
        return Err(BlobError::DataDidNotFit {
            read_offset,
            data_len: data.len(),
        });
    }

    Ok(Bytes::from(out.0))
}

/// Helper functions for reading from a single byte from the input data,
/// while advancing the read offset.
fn read_one_byte(data: &[u8], read_offset: &mut usize) -> u8 {
    if *read_offset < data.len() {
        let b = data[*read_offset];
        *read_offset += 1;
        b
    } else {
        0
    }
}

/// Helper functions for reading 31 bytes from the input data,
/// while advancing the read offset.
fn read_31_bytes(data: &[u8], read_offset: &mut usize) -> [u8; 31] {
    let mut buf = [0u8; 31];
    if *read_offset < data.len() {
        let n = std::cmp::min(31, data.len() - *read_offset);
        buf[..n].copy_from_slice(&data[*read_offset..*read_offset + n]);
        *read_offset += n;
    }
    buf
}

/// Helper functions for writing one byte to the output blob, while
/// advancing the write offset.
fn write_one_byte(out: &mut [u8; BYTES_PER_BLOB], write_offset: &mut usize, v: u8) {
    assert!(
        (*write_offset % 32 == 0),
        "blob encoding: invalid byte write offset: {}",
        *write_offset
    );

    assert!(
        (v & 0b1100_0000 == 0),
        "blob encoding: invalid 6-bit value: {v:08b}"
    );

    out[*write_offset] = v;
    *write_offset += 1;
}

/// Helper function for writing a 31-byte chunk to the output blob, while
/// advancing the write offset.
fn write_31_bytes(out: &mut [u8; BYTES_PER_BLOB], write_offset: &mut usize, buf: &[u8; 31]) {
    assert!(
        (*write_offset % 32 == 1),
        "blob encoding: invalid bytes31 write offset: {}",
        *write_offset
    );

    out[*write_offset..*write_offset + 31].copy_from_slice(buf);
    *write_offset += 31;
}
