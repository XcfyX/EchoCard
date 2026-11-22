"""转码与音频工具测试（真实 ffmpeg）。"""
from __future__ import annotations

import subprocess

import pytest

from echocard.audio.transcode import (
    probe_duration,
    read_pcm16,
    slice_wav,
    to_16k_mono,
    wav_to_opus,
)

FF = "ffmpeg"


def _make_48k_f32(path, seconds=2.0, freq=440):
    """用 ffmpeg 生成 48kHz 32-bit float 单声道 WAV（模拟 DJI 原始格式）。"""
    subprocess.run(
        [FF, "-y", "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}",
         "-ar", "48000", "-ac", "1", "-c:a", "pcm_f32le", str(path)],
        check=True, capture_output=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def test_to_16k_mono_normalized(tmp_path):
    src = tmp_path / "raw48.wav"
    _make_48k_f32(src, seconds=2.0)
    dst = to_16k_mono(src, tmp_path / "out16.wav")
    assert dst.exists()
    data, rate = read_pcm16(dst)
    assert rate == 16000
    # loudnorm 后电平应明显提升（sine 默认 -18dBFS 附近 → -16 LUFS PCM16 应有较大幅值）
    import numpy as np

    samples = np.frombuffer(bytes(data), dtype=np.int16).astype(np.float32) / 32768
    rms = float(np.sqrt(np.mean(samples ** 2)))
    assert rms > 0.05, f"响度归一化后 RMS 过低：{rms}"


def test_to_16k_mono_fallback_on_odd_input(tmp_path):
    src = tmp_path / "tiny.wav"
    _make_48k_f32(src, seconds=0.3)  # 极短音频
    dst = to_16k_mono(src, tmp_path / "out.wav")
    assert dst.exists()


def test_wav_to_opus_size_and_duration(tmp_path):
    src = tmp_path / "raw.wav"
    _make_48k_f32(src, seconds=10.0)
    dst = wav_to_opus(src, tmp_path / "out.opus")
    assert dst.exists()
    # 32kbps mono × 10s ≈ 40KB（规格书 ~15MB/小时 量级）
    assert dst.stat().st_size < 60_000
    assert 9.0 <= probe_duration(dst) <= 11.0


def test_slice_wav(tmp_path):
    src = tmp_path / "src.wav"
    _make_48k_f32(src, seconds=8.0)
    dst = slice_wav(src, tmp_path / "slice.wav", 2.0, 5.0)
    assert abs(probe_duration(dst) - 3.0) < 0.3
