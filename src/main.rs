use clap::Parser;
use eyre::{Context, Result};
use hf_hub::api::sync::Api;
use parakeet_rs::sortformer::{DiarizationConfig, Sortformer, SpeakerSegment};
use parakeet_rs::{ExecutionConfig, ParakeetTDT, TimestampMode, Transcriber};
use rubato::{FftFixedIn, Resampler};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, Cursor, Read, Write};
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

macro_rules! status {
    ($debug:expr, $($arg:tt)*) => {
        if $debug { eprintln!($($arg)*); }
    };
}

#[derive(Parser, Debug)]
#[command(name = "parakeet-transcribe")]
#[command(about = "Transcribe audio using Parakeet TDT")]
struct Args {
    /// Path to audio file (omit to configure default settings)
    audio_file: Option<String>,

    /// Model name (v2 or v3)
    #[arg(long)]
    model: Option<String>,

    /// Chunk duration in seconds for long files
    #[arg(long)]
    chunk_duration: Option<f32>,

    /// Model quantization (int8 or none)
    #[arg(long)]
    quantization: Option<String>,

    /// Output JSON lines instead of plain text
    #[arg(long)]
    json: bool,

    /// Include timestamps in text output (ignored with --json)
    #[arg(long)]
    timestamps: bool,

    /// Enable speaker diarization (text output only, not supported with --json)
    #[arg(long)]
    diarize: bool,

    /// Show verbose status messages on stderr
    #[arg(long)]
    debug: bool,

    /// File to create when transcription is complete
    #[arg(long)]
    completion_marker: Option<String>,
}

// ------------------------------------------------------------
// Saved defaults
// ------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone)]
struct SavedConfig {
    #[serde(default = "default_model")]
    model: String,
    #[serde(default = "default_chunk_duration")]
    chunk_duration: f32,
    #[serde(default = "default_quantization")]
    quantization: String,
    #[serde(default)]
    timestamps: bool,
    #[serde(default)]
    diarize: bool,
    #[serde(default)]
    debug: bool,
}

fn default_model() -> String {
    "nemo-parakeet-tdt-0.6b-v2".to_string()
}
fn default_chunk_duration() -> f32 {
    120.0
}
fn default_quantization() -> String {
    "int8".to_string()
}

impl Default for SavedConfig {
    fn default() -> Self {
        Self {
            model: default_model(),
            chunk_duration: default_chunk_duration(),
            quantization: default_quantization(),
            timestamps: false,
            diarize: false,
            debug: false,
        }
    }
}

fn config_path() -> Result<PathBuf> {
    let dir = dirs::config_dir()
        .ok_or_else(|| eyre::eyre!("Could not find config directory"))?
        .join("parakeet-transcribe");
    Ok(dir.join("config.json"))
}

fn load_config() -> SavedConfig {
    config_path()
        .ok()
        .and_then(|p| fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_config(config: &SavedConfig) -> Result<()> {
    let path = config_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(config)?;
    fs::write(&path, json)?;
    Ok(())
}

/// Resolve effective settings: CLI args override saved config defaults.
struct ResolvedArgs {
    audio_file: String,
    model: String,
    chunk_duration: f32,
    quantization: String,
    json: bool,
    timestamps: bool,
    diarize: bool,
    debug: bool,
    completion_marker: Option<String>,
}

impl ResolvedArgs {
    fn from_args_and_config(args: &Args, config: &SavedConfig) -> Self {
        Self {
            audio_file: args.audio_file.clone().unwrap_or_default(),
            model: args.model.clone().unwrap_or_else(|| config.model.clone()),
            chunk_duration: args.chunk_duration.unwrap_or(config.chunk_duration),
            quantization: args
                .quantization
                .clone()
                .unwrap_or_else(|| config.quantization.clone()),
            json: args.json,
            timestamps: args.timestamps || config.timestamps,
            diarize: args.diarize || config.diarize,
            debug: args.debug || config.debug,
            completion_marker: args.completion_marker.clone(),
        }
    }
}

fn prompt_line(prompt: &str) -> String {
    eprint!("{}", prompt);
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    let _ = std::io::stdin().lock().read_line(&mut line);
    line.trim().to_string()
}

fn run_settings_wizard() -> Result<()> {
    let current = load_config();

    eprintln!("=== Parakeet Transcribe - Default Settings ===");
    eprintln!();

    // Model
    eprintln!("Model options:");
    eprintln!("  [1] nemo-parakeet-tdt-0.6b-v2 (English only)");
    eprintln!("  [2] nemo-parakeet-tdt-0.6b-v3 (less accurate for English, supports 10 European languages)");
    let current_model_num = if current.model == "nemo-parakeet-tdt-0.6b-v3" {
        "2"
    } else {
        "1"
    };
    let model_choice = prompt_line(&format!("Choose model [{}]: ", current_model_num));
    let model = match model_choice.as_str() {
        "1" => "nemo-parakeet-tdt-0.6b-v2".to_string(),
        "2" => "nemo-parakeet-tdt-0.6b-v3".to_string(),
        "" => current.model.clone(),
        _ => {
            eprintln!("Invalid choice, keeping current: {}", current.model);
            current.model.clone()
        }
    };

    // Quantization
    eprintln!();
    eprintln!("Quantization options:");
    eprintln!("  [1] int8 (smaller, faster)");
    eprintln!("  [2] none (more accurate, much slower)");
    let current_quant_num = if current.quantization == "none" {
        "2"
    } else {
        "1"
    };
    let quant_choice = prompt_line(&format!("Choose quantization [{}]: ", current_quant_num));
    let quantization = match quant_choice.as_str() {
        "1" => "int8".to_string(),
        "2" => "none".to_string(),
        "" => current.quantization.clone(),
        _ => {
            eprintln!(
                "Invalid choice, keeping current: {}",
                current.quantization
            );
            current.quantization.clone()
        }
    };

    // Output mode
    eprintln!();
    eprintln!("Default output mode:");
    eprintln!("  [1] Plain text");
    eprintln!("  [2] With timestamps");
    eprintln!("  [3] With speakers (diarization)");
    eprintln!("  [4] With speakers + timestamps");
    let current_mode = match (current.diarize, current.timestamps) {
        (false, false) => "1",
        (false, true) => "2",
        (true, false) => "3",
        (true, true) => "4",
    };
    let mode_choice = prompt_line(&format!("Choose output mode [{}]: ", current_mode));
    let (diarize, timestamps) = match mode_choice.as_str() {
        "1" => (false, false),
        "2" => (false, true),
        "3" => (true, false),
        "4" => (true, true),
        "" => (current.diarize, current.timestamps),
        _ => {
            eprintln!("Invalid choice, keeping current mode.");
            (current.diarize, current.timestamps)
        }
    };

    // Chunk duration
    eprintln!();
    let chunk_input = prompt_line(&format!(
        "Chunk duration in seconds for long files [{}]: ",
        current.chunk_duration
    ));
    let chunk_duration = if chunk_input.is_empty() {
        current.chunk_duration
    } else {
        chunk_input
            .parse::<f32>()
            .unwrap_or_else(|_| {
                eprintln!(
                    "Invalid number, keeping current: {}",
                    current.chunk_duration
                );
                current.chunk_duration
            })
    };

    // Debug mode
    eprintln!();
    eprintln!("Enable debug mode?");
    eprintln!("  [1] No");
    eprintln!("  [2] Yes (show verbose status messages)");
    let current_debug_num = if current.debug { "2" } else { "1" };
    let debug_choice = prompt_line(&format!("Choose [{}]: ", current_debug_num));
    let debug = match debug_choice.as_str() {
        "1" => false,
        "2" => true,
        "" => current.debug,
        _ => {
            eprintln!("Invalid choice, keeping current.");
            current.debug
        }
    };

    let config = SavedConfig {
        model,
        chunk_duration,
        quantization,
        timestamps,
        diarize,
        debug,
    };

    save_config(&config)?;

    eprintln!();
    eprintln!("Settings saved! These will be used as defaults for future transcriptions.");
    eprintln!("You can override any setting with command-line flags (e.g. --timestamps).");
    let path = config_path()?;
    eprintln!("Config file: {}", path.display());

    Ok(())
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

fn ensure_model_files(model: &str, quantization: &str, debug: bool) -> Result<PathBuf> {
    let repo_id = get_repo_id(model)?;
    let model_dir = get_model_dir(model, quantization)?;
    fs::create_dir_all(&model_dir)?;

    let use_int8 = quantization.to_lowercase() == "int8";

    let encoder_path = model_dir.join("encoder-model.onnx");
    let decoder_path = model_dir.join("decoder_joint-model.onnx");
    let vocab_path = model_dir.join("vocab.txt");

    if encoder_path.exists() && decoder_path.exists() && vocab_path.exists() {
        status!(debug, "Using cached model files from {model_dir:?}");
        return Ok(model_dir);
    }

    status!(debug, "Downloading model files from {repo_id}...");
    let api = Api::new().wrap_err("Failed to create HuggingFace API client")?;
    let repo = api.model(repo_id.to_string());

    status!(debug, "  Downloading vocab.txt...");
    let vocab_src = repo
        .get("vocab.txt")
        .wrap_err("Failed to download vocab.txt")?;
    fs::copy(&vocab_src, &vocab_path).wrap_err("Failed to copy vocab.txt")?;

    if use_int8 {
        status!(debug, "  Downloading encoder-model.int8.onnx...");
        let encoder_src = repo
            .get("encoder-model.int8.onnx")
            .wrap_err("Failed to download encoder-model.int8.onnx")?;
        fs::copy(&encoder_src, &encoder_path).wrap_err("Failed to copy encoder model")?;

        status!(debug, "  Downloading decoder_joint-model.int8.onnx...");
        let decoder_src = repo
            .get("decoder_joint-model.int8.onnx")
            .wrap_err("Failed to download decoder_joint-model.int8.onnx")?;
        fs::copy(&decoder_src, &decoder_path).wrap_err("Failed to copy decoder model")?;
    } else {
        status!(debug, "  Downloading encoder-model.onnx...");
        let encoder_src = repo
            .get("encoder-model.onnx")
            .wrap_err("Failed to download encoder-model.onnx")?;
        fs::copy(&encoder_src, &encoder_path).wrap_err("Failed to copy encoder model")?;

        status!(debug, "  Downloading encoder-model.onnx.data...");
        let encoder_data_src = repo
            .get("encoder-model.onnx.data")
            .wrap_err("Failed to download encoder-model.onnx.data")?;
        let encoder_data_path = model_dir.join("encoder-model.onnx.data");
        fs::copy(&encoder_data_src, &encoder_data_path)
            .wrap_err("Failed to copy encoder model data")?;

        status!(debug, "  Downloading decoder_joint-model.onnx...");
        let decoder_src = repo
            .get("decoder_joint-model.onnx")
            .wrap_err("Failed to download decoder_joint-model.onnx")?;
        fs::copy(&decoder_src, &decoder_path).wrap_err("Failed to copy decoder model")?;
    }

    status!(debug, "Model files downloaded to {model_dir:?}");
    Ok(model_dir)
}

fn ensure_sortformer_model(debug: bool) -> Result<PathBuf> {
    let model_dir = dirs::cache_dir()
        .ok_or_else(|| eyre::eyre!("Could not find cache directory"))?
        .join("parakeet-tdt")
        .join("sortformer-v2.1");
    fs::create_dir_all(&model_dir)?;

    let model_path = model_dir.join("diar_streaming_sortformer_4spk-v2.1.onnx");

    if model_path.exists() {
        status!(debug, "Using cached sortformer model from {model_dir:?}");
        return Ok(model_path);
    }

    status!(debug, "Downloading sortformer v2.1 model...");
    let api = Api::new().wrap_err("Failed to create HuggingFace API client")?;
    let repo = api.model("altunenes/parakeet-rs".to_string());

    let src = repo
        .get("diar_streaming_sortformer_4spk-v2.1.onnx")
        .wrap_err("Failed to download sortformer model")?;
    fs::copy(&src, &model_path).wrap_err("Failed to copy sortformer model")?;

    status!(debug, "Sortformer model downloaded to {model_dir:?}");
    Ok(model_path)
}

// ------------------------------------------------------------
// Audio loading / resampling
// ------------------------------------------------------------

fn load_audio_native(path: &Path, debug: bool) -> Result<Vec<f32>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("unknown");

    // Check for MPEG-in-WAV (common in BWF files)
    if ext.eq_ignore_ascii_case("wav") || ext.eq_ignore_ascii_case("bwf") {
        let mut file = File::open(path).wrap_err("Failed to open audio file")?;
        if let Some(mpeg_data) = mpeg_wav_reader::extract_mpeg_from_wav(&mut file)? {
            status!(debug, "Detected MPEG audio in WAV container, extracting...");
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
// Text cleanup
// ------------------------------------------------------------

/// Fix punctuation spacing artifacts from the model's tokenization.
/// The model sometimes separates periods and commas from adjacent text,
/// producing "D .C ." instead of "D.C." and "$30 ,000" instead of "$30,000".
fn clean_segment_text(text: &str) -> String {
    let mut chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i + 2 < chars.len() {
        if chars[i] == ' ' {
            // Space before period followed by a letter: " .C" → ".C"
            if chars[i + 1] == '.' && chars.get(i + 2).map_or(false, |c| c.is_alphabetic()) {
                chars.remove(i);
                continue;
            }
            // Space before comma followed by a digit: " ,000" → ",000"
            if chars[i + 1] == ',' && chars.get(i + 2).map_or(false, |c| c.is_ascii_digit()) {
                chars.remove(i);
                continue;
            }
        }
        i += 1;
    }
    chars.into_iter().collect()
}

/// Merge orphaned punctuation-only segments into the previous segment's text.
fn merge_orphaned_punctuation(segments: &mut Vec<Segment>) {
    let mut i = 1;
    while i < segments.len() {
        let trimmed = segments[i].text.trim().to_string();
        if trimmed.len() <= 2 && trimmed.chars().all(|c| c.is_ascii_punctuation()) {
            // Append to previous segment
            segments[i - 1].text.push_str(&trimmed);
            segments[i - 1].end = segments[i].end;
            segments.remove(i);
        } else {
            i += 1;
        }
    }
}

// ------------------------------------------------------------
// Transcription
// ------------------------------------------------------------

/// Finalize a batch of segments: merge orphaned punctuation and clean text.
fn finalize_segments(segments: &mut Vec<Segment>) {
    merge_orphaned_punctuation(segments);
    for seg in segments {
        seg.text = clean_segment_text(&seg.text);
    }
}

fn transcribe_with_chunking(
    parakeet: &mut ParakeetTDT,
    audio_samples: Vec<f32>,
    chunk_duration: f32,
    mut on_segments: Option<&mut dyn FnMut(&[Segment])>,
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

    // Buffer one chunk's segments to handle cross-chunk orphaned punctuation
    let mut buffered_segments: Vec<Segment> = Vec::new();

    let mut start = 0;
    let mut chunk_idx = 0;

    while start < total_samples {
        let end = (start + chunk_samples).min(total_samples);
        let chunk: Vec<f32> = audio_samples[start..end].to_vec();
        let chunk_start_time = start as f32 / SAMPLE_RATE as f32;

        let result =
            parakeet.transcribe_samples(chunk, SAMPLE_RATE, 1, Some(TimestampMode::Sentences))?;

        let mut chunk_segments: Vec<Segment> = Vec::new();

        for token in result.tokens {
            let adjusted_start = token.start + chunk_start_time;
            let adjusted_end = token.end + chunk_start_time;

            // Skip segments in overlap region that were already captured
            if chunk_idx > 0 {
                let overlap_end = chunk_start_time + OVERLAP_DURATION;
                if adjusted_start < overlap_end {
                    if let Some(last) = buffered_segments.last().or(all_segments.last()) {
                        if adjusted_start < last.end {
                            continue;
                        }
                    }
                }
            }

            let text = token.text;
            if !text.trim().is_empty() {
                chunk_segments.push(Segment {
                    text,
                    start: adjusted_start,
                    end: adjusted_end,
                });
            }
        }

        if !buffered_segments.is_empty() {
            // Merge orphaned punctuation at the boundary: check if the first
            // new segment is just punctuation that belongs on the last buffered one
            let mut combined = buffered_segments;
            combined.append(&mut chunk_segments);
            finalize_segments(&mut combined);

            // Split back: the original buffered segments are now finalized
            // Find the split point — segments whose start time is >= chunk_start_time
            // belong to the new chunk (accounting for overlap dedup)
            let split_idx = combined
                .iter()
                .position(|s| s.start >= chunk_start_time - OVERLAP_DURATION)
                .unwrap_or(combined.len());

            chunk_segments = combined.split_off(split_idx);
            let finalized = combined;

            if let Some(ref mut cb) = on_segments {
                cb(&finalized);
            }
            all_segments.extend(finalized);
        }

        buffered_segments = chunk_segments;

        chunk_idx += 1;
        start += stride;

        if end >= total_samples {
            break;
        }
    }

    // Flush remaining buffered segments
    if !buffered_segments.is_empty() {
        finalize_segments(&mut buffered_segments);
        if let Some(cb) = on_segments {
            cb(&buffered_segments);
        }
        all_segments.extend(buffered_segments);
    }

    Ok(all_segments)
}

// ------------------------------------------------------------
// Diarization helpers
// ------------------------------------------------------------

/// For each transcript segment, find the speaker with the most overlap.
fn assign_speakers(segments: &[Segment], speaker_segments: &[SpeakerSegment]) -> Vec<Option<usize>> {
    segments
        .iter()
        .map(|seg| {
            speaker_segments
                .iter()
                .filter_map(|s| {
                    let s_start = s.start as f32 / 16_000.0;
                    let s_end = s.end as f32 / 16_000.0;
                    let overlap_start = seg.start.max(s_start);
                    let overlap_end = seg.end.min(s_end);
                    let overlap = (overlap_end - overlap_start).max(0.0);
                    if overlap > 0.0 {
                        Some((s.speaker_id, overlap))
                    } else {
                        None
                    }
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(id, _)| id)
        })
        .collect()
}

/// A speaker turn: one or more consecutive segments from the same speaker.
struct SpeakerTurn {
    text: String,
    start: f32,
}

/// Group consecutive segments by speaker into turns.
fn group_by_speaker(segments: &[Segment], speakers: &[Option<usize>]) -> Vec<SpeakerTurn> {
    let mut turns: Vec<SpeakerTurn> = Vec::new();
    let mut last_speaker: Option<usize> = None;

    for (seg, speaker) in segments.iter().zip(speakers.iter()) {
        if *speaker == last_speaker && speaker.is_some() {
            if let Some(last_turn) = turns.last_mut() {
                last_turn.text.push(' ');
                last_turn.text.push_str(seg.text.trim());
                continue;
            }
        }

        last_speaker = *speaker;
        turns.push(SpeakerTurn {
            text: seg.text.trim().to_string(),
            start: seg.start,
        });
    }

    turns
}

// ------------------------------------------------------------
// Main logic
// ------------------------------------------------------------

fn run(args: &ResolvedArgs) -> Result<()> {
    let start_time = Instant::now();

    let audio_path = Path::new(&args.audio_file);
    if !audio_path.exists() {
        return Err(eyre::eyre!("Audio file not found: {}", args.audio_file));
    }

    if args.diarize && args.json {
        return Err(eyre::eyre!(
            "--diarize is not supported with --json yet"
        ));
    }

    // Validate quantization
    let quantization = args.quantization.to_lowercase();
    if quantization != "int8" && quantization != "none" {
        return Err(eyre::eyre!(
            "Invalid quantization '{}'. Use 'int8' or 'none'",
            args.quantization
        ));
    }

    let debug = args.debug;

    let filename = Path::new(&args.audio_file)
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| args.audio_file.clone());

    status!(
        debug,
        "Using model: {} with quantization: {}",
        args.model, quantization
    );
    let model_dir = ensure_model_files(&args.model, &quantization, debug)
        .wrap_err("Failed to download model files")?;

    status!(debug, "Loading audio...");
    let audio_samples = load_audio_native(audio_path, debug)?;
    let duration = audio_samples.len() as f32 / SAMPLE_RATE as f32;
    status!(
        debug,
        "Loaded {:.1}s of audio ({} samples)",
        duration,
        audio_samples.len()
    );

    eprintln!("Transcribing {} of {:.0}s...", filename, duration);

    let is_json = args.json;
    let use_timestamps = args.timestamps;
    let chunk_duration = args.chunk_duration;

    let total_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    // When diarizing, run diarization and transcription in parallel to cut wall time.
    // When not diarizing, run transcription with streaming output.
    let (speaker_segments, mut segments) = if args.diarize {
        let audio_for_diarize = audio_samples.clone();
        let half_threads = (total_threads / 2).max(1);

        status!(debug, "Running diarization and transcription in parallel...");

        let result: Result<(Vec<SpeakerSegment>, Vec<Segment>)> =
            std::thread::scope(|s| {
                let diarize_handle = s.spawn(|| -> Result<Vec<SpeakerSegment>> {
                    status!(debug, "Loading sortformer model...");
                    let sortformer_path = ensure_sortformer_model(debug)
                        .wrap_err("Failed to download sortformer model")?;

                    let exec_config = ExecutionConfig::new()
                        .with_intra_threads(half_threads);
                    let mut sortformer = Sortformer::with_config(
                        sortformer_path.to_str().unwrap(),
                        Some(exec_config),
                        DiarizationConfig::callhome(),
                    )
                    .wrap_err("Failed to load sortformer model")?;

                    // Use smaller chunks for faster inference — halves the input
                    // size per ONNX call (attention is quadratic in chunk size)
                    sortformer.chunk_len = 62;
                    sortformer.fifo_len = 62;
                    sortformer.spkcache_len = 94;

                    status!(debug, "Running speaker diarization...");
                    let segs = sortformer
                        .diarize(audio_for_diarize, SAMPLE_RATE, 1)
                        .wrap_err("Diarization failed")?;
                    status!(debug, "Found {} speaker segments", segs.len());
                    Ok(segs)
                });

                let transcribe_handle = s.spawn(|| -> Result<Vec<Segment>> {
                    status!(debug, "Loading model...");
                    let exec_config = ExecutionConfig::new()
                        .with_intra_threads(half_threads);
                    let mut parakeet = ParakeetTDT::from_pretrained(&model_dir, Some(exec_config))
                        .wrap_err("Failed to load Parakeet TDT model")?;

                    status!(debug, "Transcribing...");
                    transcribe_with_chunking(
                        &mut parakeet,
                        audio_samples,
                        chunk_duration,
                        None,
                    )
                });

                let speaker_segs = diarize_handle
                    .join()
                    .map_err(|_| eyre::eyre!("Diarization thread panicked"))??;
                let transcript_segs = transcribe_handle
                    .join()
                    .map_err(|_| eyre::eyre!("Transcription thread panicked"))??;

                Ok((speaker_segs, transcript_segs))
            });

        let (speaker_segs, transcript_segs) = result?;
        (Some(speaker_segs), transcript_segs)
    } else {
        status!(debug, "Loading model...");
        let exec_config = ExecutionConfig::new().with_intra_threads(total_threads);
        let mut parakeet = ParakeetTDT::from_pretrained(&model_dir, Some(exec_config))
            .wrap_err("Failed to load Parakeet TDT model")?;

        status!(debug, "Transcribing...");

        let mut stream_cb = |segments: &[Segment]| {
            for seg in segments {
                if is_json {
                    if let Ok(json) = serde_json::to_string(seg) {
                        println!("{}", json);
                    }
                } else if use_timestamps {
                    let minutes = (seg.start / 60.0) as u32;
                    let seconds = (seg.start % 60.0) as u32;
                    println!("[{:02}:{:02}] {}", minutes, seconds, seg.text);
                } else {
                    println!("{}", seg.text);
                }
            }
            let _ = std::io::stdout().flush();
        };

        let segs = transcribe_with_chunking(
            &mut parakeet,
            audio_samples,
            chunk_duration,
            Some(&mut stream_cb),
        )?;

        (None, segs)
    };

    // For short files the callback isn't invoked, so finalize and output here
    let was_chunked = duration > args.chunk_duration;

    if !was_chunked || args.diarize {
        if !was_chunked {
            finalize_segments(&mut segments);
        }

        if args.diarize {
            // Match speakers and group into turns
            let speaker_segs = speaker_segments.as_ref().unwrap();
            let speakers = assign_speakers(&segments, speaker_segs);
            let turns = group_by_speaker(&segments, &speakers);

            for turn in &turns {
                if use_timestamps {
                    let minutes = (turn.start / 60.0) as u32;
                    let seconds = (turn.start % 60.0) as u32;
                    println!("[{}:{:02}] - {}", minutes, seconds, turn.text);
                } else {
                    println!("- {}", turn.text);
                }
            }

            // Clipboard
            let transcript = turns
                .iter()
                .map(|t| {
                    if use_timestamps {
                        let minutes = (t.start / 60.0) as u32;
                        let seconds = (t.start % 60.0) as u32;
                        format!("[{}:{:02}] - {}", minutes, seconds, t.text)
                    } else {
                        format!("- {}", t.text)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");

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
            eprintln!(
                "Transcribed in {:.2}s, transcript copied to clipboard",
                start_time.elapsed().as_secs_f32()
            );
        } else if args.json {
            for segment in &segments {
                println!("{}", serde_json::to_string(segment)?);
            }
            std::io::stdout().flush()?;
        } else if args.timestamps {
            for seg in &segments {
                let minutes = (seg.start / 60.0) as u32;
                let seconds = (seg.start % 60.0) as u32;
                println!("[{:02}:{:02}] {}", minutes, seconds, seg.text);
            }
        } else {
            let transcript = segments
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            println!("{}", transcript);
        }
    }

    if !args.json && !args.diarize {
        // Build full transcript for clipboard
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

        eprintln!(
            "Transcribed in {:.2}s, transcript copied to clipboard",
            start_time.elapsed().as_secs_f32()
        );
    }

    Ok(())
}

fn wait_for_keypress() {
    eprintln!("\nPress any key to close...");
    let _ = std::io::stdin().read(&mut [0u8]);
}

fn main() {
    let args = Args::parse();

    // No audio file → enter settings wizard
    if args.audio_file.is_none() {
        if let Err(e) = run_settings_wizard() {
            eprintln!("ERROR: {:#}", e);
        }
        wait_for_keypress();
        return;
    }

    let config = load_config();
    let resolved = ResolvedArgs::from_args_and_config(&args, &config);
    let marker_path = resolved.completion_marker.clone();
    let is_json = resolved.json;

    let result = run(&resolved);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_initials() {
        // The trailing "." comes as a separate segment (handled by merge_orphaned_punctuation)
        assert_eq!(clean_segment_text("Washington, D .C"), "Washington, D.C");
        assert_eq!(clean_segment_text("the U .S .A"), "the U.S.A");
    }

    #[test]
    fn test_clean_digit_separator() {
        assert_eq!(clean_segment_text("$30 ,000 difference"), "$30,000 difference");
        assert_eq!(clean_segment_text("$1 ,000 ,000"), "$1,000,000");
    }

    #[test]
    fn test_clean_no_false_positives() {
        // Space before period at end of sentence should stay
        assert_eq!(clean_segment_text("end of sentence ."), "end of sentence .");
        // Space before comma followed by a word should stay
        assert_eq!(clean_segment_text("hello , world"), "hello , world");
        // Normal text unchanged
        assert_eq!(clean_segment_text("hello world"), "hello world");
    }

    #[test]
    fn test_merge_orphaned_punctuation() {
        let mut segments = vec![
            Segment { text: "Washington, D.C".to_string(), start: 0.0, end: 1.0 },
            Segment { text: ".".to_string(), start: 1.0, end: 1.1 },
            Segment { text: "Next sentence.".to_string(), start: 1.1, end: 2.0 },
        ];
        merge_orphaned_punctuation(&mut segments);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].text, "Washington, D.C.");
        assert_eq!(segments[0].end, 1.1);
        assert_eq!(segments[1].text, "Next sentence.");
    }
}
