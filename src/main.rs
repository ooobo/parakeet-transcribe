use clap::Parser;
use eyre::{Context, Result};
use hf_hub::api::sync::Api;
use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};
use rubato::{FftFixedIn, Resampler};
use serde::Serialize;
use std::fs::{self, File};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

mod mpeg_wav_reader;

const SAMPLE_RATE: u32 = 16_000;
const OVERLAP_DURATION: f32 = 15.0;

#[derive(Parser, Debug)]
#[command(name = "parakeet-transcribe")]
#[command(about = "Transcribe audio using Parakeet TDT")]
struct Args {
    /// Path to audio file
    audio_file: String,

    /// Model name (v2 or v3)
    #[arg(long, default_value = "nemo-parakeet-tdt-0.6b-v2")]
    model: String,

    /// Chunk duration in seconds for long files
    #[arg(long, default_value = "120.0")]
    chunk_duration: f32,

    /// Model quantization (int8 or none)
    #[arg(long, default_value = "int8")]
    quantization: String,

    /// Output JSON lines instead of plain text
    #[arg(long)]
    json: bool,

    /// Include timestamps in text output (ignored with --json)
    #[arg(long)]
    timestamps: bool,

    /// File to create when transcription is complete
    #[arg(long)]
    completion_marker: Option<String>,
}

#[derive(Serialize, Debug)]
struct Segment {
    text: String,
    start: f32,
    end: f32,
}

// ------------------------------------------------------------
// Model helpers
// ------------------------------------------------------------

fn get_repo_id(model: &str) -> Result<&'static str> {
    match model {
        "nemo-parakeet-tdt-0.6b-v2" => Ok("istupakov/parakeet-tdt-0.6b-v2-onnx"),
        "nemo-parakeet-tdt-0.6b-v3" => Ok("istupakov/parakeet-tdt-0.6b-v3-onnx"),
        _ => Err(eyre::eyre!(
            "Unknown model: {}. Supported: nemo-parakeet-tdt-0.6b-v2, nemo-parakeet-tdt-0.6b-v3",
            model
        )),
    }
}

fn get_model_dir(model: &str, quantization: &str) -> Result<PathBuf> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| eyre::eyre!("Could not find cache directory"))?
        .join("parakeet-tdt")
        .join(format!("{model}-{quantization}"));
    Ok(cache_dir)
}

fn ensure_model_files(model: &str, quantization: &str) -> Result<PathBuf> {
    let repo_id = get_repo_id(model)?;
    let model_dir = get_model_dir(model, quantization)?;
    fs::create_dir_all(&model_dir)?;

    let use_int8 = quantization.to_lowercase() == "int8";

    let encoder_path = model_dir.join("encoder-model.onnx");
    let decoder_path = model_dir.join("decoder_joint-model.onnx");
    let vocab_path = model_dir.join("vocab.txt");

    if encoder_path.exists() && decoder_path.exists() && vocab_path.exists() {
        eprintln!("Using cached model files from {model_dir:?}");
        return Ok(model_dir);
    }

    eprintln!("Downloading model files from {repo_id}...");
    let api = Api::new().wrap_err("Failed to create HuggingFace API client")?;
    let repo = api.model(repo_id.to_string());

    eprintln!("  Downloading vocab.txt...");
    let vocab_src = repo
        .get("vocab.txt")
        .wrap_err("Failed to download vocab.txt")?;
    fs::copy(&vocab_src, &vocab_path).wrap_err("Failed to copy vocab.txt")?;

    if use_int8 {
        eprintln!("  Downloading encoder-model.int8.onnx...");
        let encoder_src = repo
            .get("encoder-model.int8.onnx")
            .wrap_err("Failed to download encoder-model.int8.onnx")?;
        fs::copy(&encoder_src, &encoder_path).wrap_err("Failed to copy encoder model")?;

        eprintln!("  Downloading decoder_joint-model.int8.onnx...");
        let decoder_src = repo
            .get("decoder_joint-model.int8.onnx")
            .wrap_err("Failed to download decoder_joint-model.int8.onnx")?;
        fs::copy(&decoder_src, &decoder_path).wrap_err("Failed to copy decoder model")?;
    } else {
        eprintln!("  Downloading encoder-model.onnx...");
        let encoder_src = repo
            .get("encoder-model.onnx")
            .wrap_err("Failed to download encoder-model.onnx")?;
        fs::copy(&encoder_src, &encoder_path).wrap_err("Failed to copy encoder model")?;

        eprintln!("  Downloading encoder-model.onnx.data...");
        let encoder_data_src = repo
            .get("encoder-model.onnx.data")
            .wrap_err("Failed to download encoder-model.onnx.data")?;
        let encoder_data_path = model_dir.join("encoder-model.onnx.data");
        fs::copy(&encoder_data_src, &encoder_data_path)
            .wrap_err("Failed to copy encoder model data")?;

        eprintln!("  Downloading decoder_joint-model.onnx...");
        let decoder_src = repo
            .get("decoder_joint-model.onnx")
            .wrap_err("Failed to download decoder_joint-model.onnx")?;
        fs::copy(&decoder_src, &decoder_path).wrap_err("Failed to copy decoder model")?;
    }

    eprintln!("Model files downloaded to {model_dir:?}");
    Ok(model_dir)
}

// ------------------------------------------------------------
// Audio loading / resampling
// ------------------------------------------------------------

fn load_audio_native(path: &Path) -> Result<Vec<f32>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("unknown");

    // Check for MPEG-in-WAV (common in BWF files)
    if ext.eq_ignore_ascii_case("wav") || ext.eq_ignore_ascii_case("bwf") {
        let mut file = File::open(path).wrap_err("Failed to open audio file")?;
        if let Some(mpeg_data) = mpeg_wav_reader::extract_mpeg_from_wav(&mut file)? {
            eprintln!("Detected MPEG audio in WAV container, extracting...");
            let cursor = Cursor::new(mpeg_data);
            let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

            let mut hint = Hint::new();
            hint.with_extension("mp3");

            return load_audio_from_stream(mss, hint, ext);
        }
    }

    let file = File::open(path).wrap_err("Failed to open audio file")?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    let hint_ext = if ext.eq_ignore_ascii_case("bwf") {
        "wav"
    } else {
        ext
    };
    hint.with_extension(hint_ext);

    load_audio_from_stream(mss, hint, ext)
}

fn load_audio_from_stream(
    mss: MediaSourceStream,
    hint: Hint,
    original_ext: &str,
) -> Result<Vec<f32>> {
    let unsupported_msg = format!(
        "Cannot decode .{} files. Supported: WAV, BWF, MP3, FLAC, AAC/M4A, OGG. \
        In REAPER, select the item and use 'Glue items' (Cmd+Shift+G) to convert it first.",
        original_ext
    );
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|_| eyre::eyre!("{unsupported_msg}"))?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| eyre::eyre!("No audio track found"))?;

    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let source_sample_rate = codec_params
        .sample_rate
        .ok_or_else(|| eyre::eyre!("Unknown sample rate"))?;
    let channels = codec_params.channels.map_or(1, |c| c.count());

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|_| eyre::eyre!("{unsupported_msg}"))?;

    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(e).wrap_err("Failed to read packet"),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(e).wrap_err("Failed to decode packet"),
        };

        let spec = *decoded.spec();
        let num_frames = decoded.frames();

        if num_frames == 0 {
            continue;
        }

        let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);

        all_samples.extend(sample_buf.samples());
    }

    if all_samples.is_empty() {
        return Err(eyre::eyre!("No audio samples decoded"));
    }

    // Convert to mono if multi-channel
    let mono_samples = if channels > 1 {
        all_samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        all_samples
    };

    // Resample to 16kHz if needed
    if source_sample_rate != SAMPLE_RATE {
        resample_audio(&mono_samples, source_sample_rate, SAMPLE_RATE)
    } else {
        Ok(mono_samples)
    }
}

fn resample_audio(samples: &[f32], source_rate: u32, target_rate: u32) -> Result<Vec<f32>> {
    if source_rate == target_rate {
        return Ok(samples.to_vec());
    }

    let chunk_size = 4096;
    let mut resampler = FftFixedIn::<f32>::new(
        source_rate as usize,
        target_rate as usize,
        chunk_size,
        2,
        1,
    )
    .wrap_err("Failed to create resampler")?;

    let mut output = Vec::new();
    let mut pos = 0;

    while pos < samples.len() {
        let end = (pos + chunk_size).min(samples.len());
        let mut chunk = samples[pos..end].to_vec();

        if chunk.len() < chunk_size {
            chunk.resize(chunk_size, 0.0);
        }

        let resampled = resampler
            .process(&[chunk], None)
            .wrap_err("Resampling failed")?;

        if !resampled.is_empty() {
            output.extend(&resampled[0]);
        }

        pos += chunk_size;
    }

    // Trim to expected length (avoid padding artifacts at the end)
    let expected_len = (samples.len() as f64 * target_rate as f64 / source_rate as f64) as usize;
    output.truncate(expected_len);

    Ok(output)
}

// ------------------------------------------------------------
// Transcription
// ------------------------------------------------------------

fn transcribe_with_chunking(
    parakeet: &mut ParakeetTDT,
    audio_samples: Vec<f32>,
    chunk_duration: f32,
) -> Result<Vec<Segment>> {
    let duration = audio_samples.len() as f32 / SAMPLE_RATE as f32;

    if duration <= chunk_duration {
        let result = parakeet.transcribe_samples(
            audio_samples,
            SAMPLE_RATE,
            1,
            Some(TimestampMode::Sentences),
        )?;

        return Ok(result
            .tokens
            .into_iter()
            .map(|t| Segment {
                text: t.text,
                start: t.start,
                end: t.end,
            })
            .filter(|s| !s.text.trim().is_empty())
            .collect());
    }

    // Long file - process in chunks
    let chunk_samples = (chunk_duration * SAMPLE_RATE as f32) as usize;
    let overlap_samples = (OVERLAP_DURATION * SAMPLE_RATE as f32) as usize;
    let stride = chunk_samples - overlap_samples;

    let mut all_segments: Vec<Segment> = Vec::new();
    let total_samples = audio_samples.len();

    let mut start = 0;
    let mut chunk_idx = 0;

    while start < total_samples {
        let end = (start + chunk_samples).min(total_samples);
        let chunk: Vec<f32> = audio_samples[start..end].to_vec();
        let chunk_start_time = start as f32 / SAMPLE_RATE as f32;

        eprintln!(
            "Processing chunk {} ({:.1}s - {:.1}s)...",
            chunk_idx + 1,
            chunk_start_time,
            end as f32 / SAMPLE_RATE as f32
        );

        let result =
            parakeet.transcribe_samples(chunk, SAMPLE_RATE, 1, Some(TimestampMode::Sentences))?;

        for token in result.tokens {
            let adjusted_start = token.start + chunk_start_time;
            let adjusted_end = token.end + chunk_start_time;

            // Skip segments in overlap region that were already captured
            if chunk_idx > 0 {
                let overlap_end = chunk_start_time + OVERLAP_DURATION;
                if adjusted_start < overlap_end {
                    if let Some(last) = all_segments.last() {
                        if adjusted_start < last.end {
                            continue;
                        }
                    }
                }
            }

            let text = token.text;
            if !text.trim().is_empty() {
                all_segments.push(Segment {
                    text,
                    start: adjusted_start,
                    end: adjusted_end,
                });
            }
        }

        chunk_idx += 1;
        start += stride;

        if end >= total_samples {
            break;
        }
    }

    Ok(all_segments)
}

// ------------------------------------------------------------
// Main logic
// ------------------------------------------------------------

fn run(args: &Args) -> Result<()> {
    let start_time = Instant::now();

    let audio_path = Path::new(&args.audio_file);
    if !audio_path.exists() {
        return Err(eyre::eyre!("Audio file not found: {}", args.audio_file));
    }

    // Validate quantization
    let quantization = args.quantization.to_lowercase();
    if quantization != "int8" && quantization != "none" {
        return Err(eyre::eyre!(
            "Invalid quantization '{}'. Use 'int8' or 'none'",
            args.quantization
        ));
    }

    eprintln!(
        "Using model: {} with quantization: {}",
        args.model, quantization
    );
    let model_dir = ensure_model_files(&args.model, &quantization)
        .wrap_err("Failed to download model files")?;

    eprintln!("Loading audio...");
    let audio_samples = load_audio_native(audio_path)?;
    let duration = audio_samples.len() as f32 / SAMPLE_RATE as f32;
    eprintln!(
        "Loaded {:.1}s of audio ({} samples)",
        duration,
        audio_samples.len()
    );

    eprintln!("Loading model...");
    let mut parakeet = ParakeetTDT::from_pretrained(&model_dir, None)
        .wrap_err("Failed to load Parakeet TDT model")?;

    eprintln!("Transcribing...");
    let segments = transcribe_with_chunking(&mut parakeet, audio_samples, args.chunk_duration)?;

    if args.json {
        // JSON lines output for scripting / reaspeech integration
        for segment in &segments {
            println!("{}", serde_json::to_string(segment)?);
        }
        std::io::stdout().flush()?;
    } else {
        // Human-readable text output for drag-and-drop use
        let transcript = if args.timestamps {
            segments
                .iter()
                .map(|s| {
                    let minutes = (s.start / 60.0) as u32;
                    let seconds = (s.start % 60.0) as u32;
                    format!("[{:02}:{:02}] {}", minutes, seconds, s.text)
                })
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            segments
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        };

        println!();
        println!("====================");
        println!();
        println!("{}", transcript);
        println!();
        println!("====================");
        println!();

        match arboard::Clipboard::new() {
            Ok(mut clipboard) => {
                if let Err(e) = clipboard.set_text(&transcript) {
                    eprintln!("Warning: could not copy to clipboard: {}", e);
                }
            }
            Err(e) => {
                eprintln!("Warning: could not access clipboard: {}", e);
            }
        }

        eprintln!("Transcript copied to clipboard.");
    }

    eprintln!(
        "\nDone in {:.2}s",
        start_time.elapsed().as_secs_f32()
    );

    Ok(())
}

fn wait_for_keypress() {
    eprintln!("\nPress any key to close...");
    let _ = std::io::stdin().read(&mut [0u8]);
}

fn main() {
    let args = Args::parse();
    let marker_path = args.completion_marker.clone();
    let is_json = args.json;

    let result = run(&args);

    // Always write completion marker if specified (even on error)
    if let Some(ref path) = marker_path {
        let _ = fs::write(path, "done\n");
    }

    if let Err(e) = result {
        eprintln!("ERROR: {:#}", e);
        if !is_json {
            wait_for_keypress();
        }
        std::process::exit(1);
    }

    if !is_json {
        wait_for_keypress();
    }
}
