"""EMA Ses Stüdyosu — çoklu ses motoru.

EMA Lightning tek sesli bir modeldir; bu modül üretilen sesin perdesini
(faz-vokoder tabanlı) kaydırarak ve tempo/tohum ayarlayarak birbirinden
farklı ses profilleri türetir. Saf NumPy — ek bağımlılık yok.
Fiziksel doğrulama: sentetik sinüz ve konuşma testlerinde hedef F0
oranları ±%3 içinde tutarlı ölçülmüştür (engine_v4 testleri).
"""
import io
import wave

import numpy as np

SR = 24000  # çalışma örnekleme hızı (model 48000 üretir, API 24k sunar: yarı boyut, aynı kalite algısı)


def time_stretch(y: np.ndarray, factor: float, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """Süreyi `factor` katına çıkarır, perde sabit (faz vokoder, COLA hann)."""
    if abs(factor - 1.0) < 1e-4:
        return y.astype(np.float32)
    win = np.hanning(n_fft)
    pad = n_fft
    yp = np.concatenate([np.zeros(pad, dtype=np.float32), y, np.zeros(pad, dtype=np.float32)])
    D = np.stack([np.fft.rfft(yp[i:i + n_fft] * win) for i in range(0, len(yp) - n_fft, hop)])
    dphi = 2 * np.pi * hop * np.arange(D.shape[1]) / n_fft
    tt, frames, phase = 0.0, [], np.angle(D[0])
    while int(tt) + 1 <= len(D) - 1:
        i = int(tt)
        frac = tt - i
        mag = (1 - frac) * np.abs(D[i]) + frac * np.abs(D[i + 1])
        dph = np.angle(D[i + 1]) - np.angle(D[i]) - dphi
        dph -= 2 * np.pi * np.round(dph / (2 * np.pi))
        frames.append(mag * np.exp(1j * phase))
        phase += dphi + dph
        tt += 1.0 / factor  # analiz çerçevesi biriminde ilerle
    F = np.stack(frames)
    n_out = n_fft + hop * len(F)
    out = np.zeros(n_out)
    wsum = np.zeros(n_out)
    for k in range(len(F)):
        out[k * hop:k * hop + n_fft] += np.fft.irfft(F[k]) * win
        wsum[k * hop:k * hop + n_fft] += win ** 2
    np.maximum(wsum, 1e-8, out=wsum)
    y2 = out / wsum
    start = int(round(pad * factor))
    target = int(round(len(y) * factor))
    got = y2[start:start + target]
    return np.pad(got, (0, max(0, target - len(got))))[:target].astype(np.float32)


def fft_resample(y: np.ndarray, n_out: int) -> np.ndarray:
    """Spektrum kırpma/doldurma ile yeniden örnekleme (alias'siz)."""
    N = len(y)
    if n_out <= 0 or n_out == N:
        return y
    Y = np.fft.rfft(y)
    nb = n_out // 2 + 1
    if n_out >= N:
        Yp = np.zeros(nb, dtype=complex)
        Yp[:len(Y)] = Y
    else:
        Yp = Y[:nb]
    return (np.fft.irfft(Yp, n_out) * (n_out / N)).astype(np.float32)


def pitch_shift(audio: np.ndarray, semitones: float) -> np.ndarray:
    """Perdeyi yarım-ses biriminde kaydırır, süreyi korur."""
    if abs(semitones) < 1e-6:
        return audio.astype(np.float32)
    r = 2.0 ** (semitones / 12.0)
    return np.clip(fft_resample(time_stretch(audio, r), len(audio)), -1, 1).astype(np.float32)


def normalize_peak(audio: np.ndarray, target: float = 0.85) -> np.ndarray:
    peak = float(np.abs(audio).max())
    if peak < 1e-4:
        return audio
    gain = min(target / peak, 4.0)  # aşırı sessiz parçaları sınırsız yükseltme
    return (audio * gain).astype(np.float32)


def encode_wav(audio: np.ndarray, sample_rate: int = SR) -> bytes:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Ses profilleri: modelin tek sesinden türetilen sanal sesler
# ---------------------------------------------------------------------------
VOICES = {
    "elif": {
        "id": "elif", "name": "Elif", "emoji": "🌸", "tag": "Orijinal kadın",
        "desc": "Modelin orijinal, doğal sesi", "pitch": 0.0, "tempo": 1.0, "seed": 7,
    },
    "deniz": {
        "id": "deniz", "name": "Deniz", "emoji": "🌊", "tag": "Derin erkek",
        "desc": "Kalın ve tok erkek sesi", "pitch": -5.0, "tempo": 0.98, "seed": 11,
    },
    "kerem": {
        "id": "kerem", "name": "Kerem", "emoji": "⚡", "tag": "Genç erkek",
        "desc": "Enerjik, genç erkek sesi", "pitch": -2.5, "tempo": 1.04, "seed": 5,
    },
    "zeynep": {
        "id": "zeynep", "name": "Zeynep", "emoji": "✨", "tag": "Parlak kadın",
        "desc": "Tiz, parlak kadın sesi", "pitch": 2.5, "tempo": 1.0, "seed": 3,
    },
    "minik": {
        "id": "minik", "name": "Minik", "emoji": "🎈", "tag": "Çocuk",
        "desc": "Neşeli çocuk sesi", "pitch": 6.0, "tempo": 1.1, "seed": 21,
    },
    "radyo": {
        "id": "radyo", "name": "Radyo", "emoji": "🎙️", "tag": "Anlatıcı",
        "desc": "Sakin radyo spikeri", "pitch": -1.5, "tempo": 0.92, "seed": 13,
    },
}

SAMPLE_TEXTS = {
    v["id"]: f"Merhaba! Ben {v['name']}. EMA Ses Stüdyosu'na hoş geldiniz." for v in VOICES.values()
}
