#!/usr/bin/env python3
"""
Minimal inference script for HH-Codec model.

This script loads a trained model checkpoint and demonstrates:
1. Loading an MP3 audio file
2. Encoding audio to tokens
3. Decoding tokens back to audio
4. Saving the reconstructed audio

Usage:
    python inference.py --checkpoint HH_codec/8k_ration_20_loss/epoch=4-step=65.ckpt --input audio.mp3 --output reconstructed.wav
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

from hhcodec.model import VQModel


def load_model_from_checkpoint(checkpoint_path: Path, device: str = "cuda") -> VQModel:
    """
    Load the VQModel from a checkpoint file.

    Args:
        checkpoint_path: Path to the .ckpt file
        device: Device to load the model on

    Returns:
        Loaded VQModel ready for inference
    """
    print(f"Loading model from {checkpoint_path}")

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    # Create model with the same configuration as training
    # Based on the config file, we can reconstruct the model parameters
    model_config = {
        "ddconfig": {"causal": True, "dimension": 512, "ratios": [8, 8, 4, 4]},
        "lossconfig": {
            "target": "hhcodec.losses.stft_simvq_mel.VQSTFTWithDiscriminator",
            "params": {
                "disc_conditional": False,
                "disc_in_channels": 1,
                "disc_start": 0,
                "codebook_enlarge_ratio": 0,
                "codebook_enlarge_steps": 2000,
                "sample_rate": 24000,
                "commit_weight": 1000.0,
                "gen_loss_weight": 1.0,
                "mel_loss_coeff": 45.0,
                "mrd_loss_coeff": 1.0,
            },
        },
        "quantconfig": {},
        "target_bandwidths": [1.5, 3.0, 6.0, 12.0],
        "segment": 65536,
        "sample_rate": 24000,
        "audio_normalize": False,
        "learning_rate": 1e-4,
        "scheduler_type": "None",
        "use_ema": True,
    }

    # Create the model
    model = VQModel(**model_config)

    # Load the state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model


def load_audio(audio_path: Path, target_sample_rate: int = 24000) -> torch.Tensor:
    """
    Load audio file and preprocess it for the model.

    Args:
        audio_path: Path to audio file (supports MP3, WAV, etc.)
        target_sample_rate: Target sample rate for the model

    Returns:
        Preprocessed audio tensor of shape [1, channels, samples]
    """
    print(f"Loading audio from {audio_path}")

    # Load audio file
    audio, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
        print("Converted stereo to mono")

    # Resample if necessary
    if sample_rate != target_sample_rate:
        print(f"Resampling from {sample_rate}Hz to {target_sample_rate}Hz")
        resampler = T.Resample(sample_rate, target_sample_rate)
        audio = resampler(audio)

    # Add batch dimension: [channels, samples] -> [1, channels, samples]
    audio = audio.unsqueeze(0)

    print(
        f"Audio shape: {audio.shape}, duration: {audio.shape[-1] / target_sample_rate:.2f}s"
    )
    return audio


def encode_audio(
    model: VQModel, audio: torch.Tensor, device: str
) -> Tuple[torch.Tensor, dict]:
    """
    Encode audio to tokens using the model.

    Args:
        model: Loaded VQModel
        audio: Audio tensor of shape [1, 1, samples]
        device: Device to run inference on

    Returns:
        Tuple of (quantized_features, info_dict)
    """
    audio = audio.to(device)

    with torch.no_grad():
        if hasattr(model, "ema_scope") and model.use_ema:
            with model.ema_scope():
                (
                    quant,
                    diff,
                    indices,
                    loss_breakdown,
                    first_quant,
                    second_quant,
                    first_index,
                ) = model.encode(audio)
        else:
            (
                quant,
                diff,
                indices,
                loss_breakdown,
                first_quant,
                second_quant,
                first_index,
            ) = model.encode(audio)

    info = {
        "indices": indices,
        "first_index": first_index,
        "num_tokens": indices.numel() if indices is not None else 0,
        "quantization_loss": diff.item() if hasattr(diff, "item") else diff,
    }

    print(f"Encoded to {info['num_tokens']} tokens")
    return first_quant, info


def decode_tokens(model: VQModel, tokens: torch.Tensor, device: str) -> torch.Tensor:
    """
    Decode tokens back to audio using the model.

    Args:
        model: Loaded VQModel
        tokens: Quantized features from encoding
        device: Device to run inference on

    Returns:
        Reconstructed audio tensor
    """
    tokens = tokens.to(device)

    with torch.no_grad():
        if hasattr(model, "ema_scope") and model.use_ema:
            with model.ema_scope():
                mel, reconstructed_audio = model.decode(tokens)
        else:
            mel, reconstructed_audio = model.decode(tokens)

    print("Decoded tokens to audio")
    return reconstructed_audio


def save_audio(audio: torch.Tensor, output_path: Path, sample_rate: int = 24000):
    """
    Save audio tensor to file.

    Args:
        audio: Audio tensor of shape [1, 1, samples]
        output_path: Output file path
        sample_rate: Sample rate for saving
    """
    # Remove batch dimension and clip to prevent clipping artifacts
    audio = audio.squeeze(0).cpu()
    audio = torch.clamp(audio, -0.99, 0.99)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as 16-bit PCM WAV
    torchaudio.save(
        output_path,
        audio,
        sample_rate=sample_rate,
        encoding="PCM_S",
        bits_per_sample=16,
    )

    print(f"Saved reconstructed audio to {output_path}")


def main():
    # parser = argparse.ArgumentParser(description="HH-Codec inference script")
    # parser.add_argument(
    #     "--checkpoint",
    #     type=Path,
    #     required=True,
    #     help="Path to model checkpoint (.ckpt file)"
    # )
    # parser.add_argument(
    #     "--input",
    #     type=Path,
    #     required=True,
    #     help="Path to input audio file (MP3, WAV, etc.)"
    # )
    # parser.add_argument(
    #     "--output",
    #     type=Path,
    #     required=True,
    #     help="Path to output reconstructed audio file (.wav)"
    # )
    # parser.add_argument(
    #     "--device",
    #     type=str,
    #     default="cuda" if torch.cuda.is_available() else "cpu",
    #     help="Device to run inference on (cuda/cpu)"
    # )
    # args = parser.parse_args()

    checkpoint = Path(
        "/mnt/c/Github/HH-Coded-fork/HH_codec/8k_ration_20_loss/epoch=1-step=11000.ckpt"
    )
    file_input = Path("/mnt/c/Github/LHF_voznet/examples/rickandmorty_0.wav")
    file_output = Path(
        "/mnt/c/Github/LHF_voznet/examples/rickandmorty_0_reconstructed_hhcodec.wav"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Validate inputs
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not file_input.exists():
        raise FileNotFoundError(f"Input audio not found: {file_input}")

    print("=" * 60)
    print("HH-Codec Audio Reconstruction Demo")
    print("=" * 60)

    # Load model
    model = load_model_from_checkpoint(checkpoint, device)

    # Load and preprocess audio
    audio = load_audio(file_input)

    # Encode audio to tokens
    print("\nEncoding audio to tokens...")
    tokens, encode_info = encode_audio(model, audio, device)
    print(f"Encoding info: {encode_info}")

    # Decode tokens back to audio
    print("\nDecoding tokens to audio...")
    reconstructed_audio = decode_tokens(model, tokens, device)

    # Save reconstructed audio
    print("\nSaving reconstructed audio...")
    save_audio(reconstructed_audio, file_output)

    print("\n" + "=" * 60)
    print("Reconstruction complete!")
    print(f"Original: {file_input}")
    print(f"Reconstructed: {file_output}")
    print(f"Tokens used: {encode_info['num_tokens']}")
    print(f"File used  : {file_input}")
    print(f"File output: {file_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
