// Custom reader for MPEG-in-WAV files (BWF files with MPEG audio)
use eyre::Result;
use std::io::{Read, Seek, SeekFrom};

#[derive(Debug)]
struct RiffChunk {
    size: u32,
    data_offset: u64,
}

fn read_fourcc<R: Read>(reader: &mut R) -> Result<[u8; 4]> {
    let mut fourcc = [0u8; 4];
    reader.read_exact(&mut fourcc)?;
    Ok(fourcc)
}

fn read_u32_le<R: Read>(reader: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u16_le<R: Read>(reader: &mut R) -> Result<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

/// Checks if a WAV file contains MPEG audio and extracts the MPEG data
pub fn extract_mpeg_from_wav<R: Read + Seek>(reader: &mut R) -> Result<Option<Vec<u8>>> {
    // Read RIFF header
    let riff = read_fourcc(reader)?;
    if &riff != b"RIFF" {
        return Ok(None); // Not a RIFF file
    }

    let _file_size = read_u32_le(reader)?;
    let wave = read_fourcc(reader)?;
    if &wave != b"WAVE" {
        return Ok(None); // Not a WAVE file
    }

    let mut is_mpeg = false;
    let mut data_chunk: Option<RiffChunk> = None;

    // Parse chunks
    loop {
        let chunk_id = match read_fourcc(reader) {
            Ok(id) => id,
            Err(_) => break, // End of file
        };

        let chunk_size = match read_u32_le(reader) {
            Ok(size) => size,
            Err(_) => break,
        };

        let data_offset = reader.stream_position()?;

        match &chunk_id {
            b"fmt " => {
                // Read format tag
                let format_tag = read_u16_le(reader)?;

                // 0x0050 = MPEG, 0x0055 = MP3
                if format_tag == 0x0050 || format_tag == 0x0055 {
                    is_mpeg = true;
                }

                // Skip rest of format chunk
                reader.seek(SeekFrom::Start(data_offset + chunk_size as u64))?;
            }
            b"data" => {
                data_chunk = Some(RiffChunk {
                    size: chunk_size,
                    data_offset,
                });

                // Skip data chunk for now
                reader.seek(SeekFrom::Start(data_offset + chunk_size as u64))?;
            }
            _ => {
                // Skip unknown chunk
                reader.seek(SeekFrom::Start(data_offset + chunk_size as u64))?;
            }
        }

        // Ensure we're aligned to even byte boundary (RIFF requirement)
        if chunk_size % 2 != 0 {
            reader.seek(SeekFrom::Current(1))?;
        }
    }

    // If this is an MPEG WAV file, extract the MPEG data
    if is_mpeg {
        if let Some(data) = data_chunk {
            reader.seek(SeekFrom::Start(data.data_offset))?;
            let mut mpeg_data = vec![0u8; data.size as usize];
            reader.read_exact(&mut mpeg_data)?;
            return Ok(Some(mpeg_data));
        }
    }

    Ok(None)
}
