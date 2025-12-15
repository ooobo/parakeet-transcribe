use clap::Parser;
use eyre::Result;
use hf_hub::api::sync::Api;
use parakeet_rs::{ParakeetTDT, TimestampMode};
use rubato::{FftFixedIn, Resampler};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use arboard::Clipboard;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

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
}

struct Segment {
    text: String,
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

    let use_int8 = quantization == "int8";

    let encoder_path = model_dir.join("encoder-model.onnx");
    let decoder_path = model_dir.join("decoder_joint-model.onnx");
    let vocab_path = model_dir.join("vocab.txt");

    if encoder_path.exists() && decoder_path.exists() && vocab_path.exists() {
        return Ok(model_dir);
    }

    eprintln!("Downloading model files...");
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());

    fs::copy(repo.get("vocab.txt")?, &vocab_path)?;

    if use_int8 {
        fs::copy(repo.get("encoder-model.int8.onnx")?, &encoder_path)?;
        fs::copy(repo.get("decoder_joint-model.int8.onnx")?, &decoder_path)?;
    } else {
        fs::copy(repo.get("encoder-model.onnx")?, &encoder_path)?;
        fs::copy(
            repo.get("encoder-model.onnx.data")?,
            model_dir.join("encoder-model.onnx.data"),
        )?;
        fs::copy(repo.get("decoder_joint-model.onnx")?, &decoder_path)?;
    }

    Ok(model_dir)
}

// ------------------------------------------------------------
// Audio loading / resampling
// ------------------------------------------------------------

fn load_audio_native(path: &Path) -> Result<Vec<f32>> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| eyre::eyre!("No audio track found"))?;
    let track_id = track.id;

    let codec_params = &track.codec_params;
    let source_rate = codec_params
        .sample_rate
        .ok_or_else(|| eyre::eyre!("Unknown sample rate"))?;
    let channels = codec_params.channels.map_or(1, |c| c.count());

    let mut decoder =
        symphonia::default::get_codecs().make(codec_params, &DecoderOptions::default())?;

    let mut samples = Vec::new();

    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }

        if let Ok(decoded) = decoder.decode(&packet) {
            let mut buf = SampleBuffer::<f32>::new(decoded.frames() as u64, *decoded.spec());
            buf.copy_interleaved_ref(decoded);
            samples.extend(buf.samples());
        }
    }

    let mono: Vec<f32> = if channels > 1 {
        samples
            .chunks(channels)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    if source_rate != SAMPLE_RATE {
        resample_audio(&mono, source_rate, SAMPLE_RATE)
    } else {
        Ok(mono)
    }
}

fn resample_audio(samples: &[f32], src: u32, dst: u32) -> Result<Vec<f32>> {
    let mut resampler = FftFixedIn::<f32>::new(src as usize, dst as usize, 4096, 2, 1)?;
    let mut out = Vec::new();

    for chunk in samples.chunks(4096) {
        let mut c = chunk.to_vec();
        if c.len() < 4096 {
            c.resize(4096, 0.0);
        }
        let r = resampler.process(&[c], None)?;
        out.extend(&r[0]);
    }

    Ok(out)
}

// ------------------------------------------------------------
// Text cleanup
// ------------------------------------------------------------

fn clean_text(text: &str) -> String {
    text.replace(" .", ".")
        .replace(" ,", ",")
        .trim()
        .to_string()
}

fn is_valid_segment(text: &str) -> bool {
    text.chars().any(|c| c.is_alphanumeric())
}

// ------------------------------------------------------------
// Transcription
// ------------------------------------------------------------

fn transcribe(
    parakeet: &mut ParakeetTDT,
    audio: Vec<f32>,
    chunk_duration: f32,
) -> Result<Vec<Segment>> {
    let duration = audio.len() as f32 / SAMPLE_RATE as f32;

    if duration <= chunk_duration {
        let r =
            parakeet.transcribe_samples(audio, SAMPLE_RATE, 1, Some(TimestampMode::Sentences))?;

        return Ok(r
            .tokens
            .into_iter()
            .map(|t| Segment {
                text: clean_text(&t.text),
            })
            .filter(|s| is_valid_segment(&s.text))
            .collect());
    }

    let chunk_samples = (chunk_duration * SAMPLE_RATE as f32) as usize;
    let overlap = (OVERLAP_DURATION * SAMPLE_RATE as f32) as usize;
    let stride = chunk_samples - overlap;

    let mut all = Vec::new();
    let mut pos = 0;

    while pos < audio.len() {
        let end = (pos + chunk_samples).min(audio.len());
        let chunk = audio[pos..end].to_vec();

        let r =
            parakeet.transcribe_samples(chunk, SAMPLE_RATE, 1, Some(TimestampMode::Sentences))?;

        for t in r.tokens {
            let txt = clean_text(&t.text);
            if is_valid_segment(&txt) {
                all.push(Segment { text: txt });
            }
        }

        pos += stride;
        if end == audio.len() {
            break;
        }
    }

    Ok(all)
}

// ------------------------------------------------------------
// Main logic
// ------------------------------------------------------------

fn run(args: &Args) -> Result<()> {
    let start = Instant::now();

    let audio_path = Path::new(&args.audio_file);
    if !audio_path.exists() {
        return Err(eyre::eyre!("Audio file not found"));
    }

    let model_dir = ensure_model_files(&args.model, &args.quantization)?;

    eprintln!("Loading audio...");
    let audio = load_audio_native(audio_path)?;

    eprintln!("Loading model...");
    let mut model = ParakeetTDT::from_pretrained(&model_dir, None)?;

    eprintln!("Transcribing...");
    let segments = transcribe(&mut model, audio, args.chunk_duration)?;

    let transcript = segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    println!("{}", transcript);

    let mut clipboard = Clipboard::new()?;
    clipboard.set_text(transcript)?;

    eprintln!(
        "\nDone in {:.2}s — transcript copied to clipboard.",
        start.elapsed().as_secs_f32()
    );

    Ok(())
}

fn wait_for_keypress() {
    eprintln!("\nPress any key to close...");
    let _ = std::io::stdin().read(&mut [0u8]);
}

fn main() {
    let args = Args::parse();

    if let Err(e) = run(&args) {
        eprintln!("ERROR: {:#}", e);
        wait_for_keypress();
        std::process::exit(1);
    }

    wait_for_keypress();
}
