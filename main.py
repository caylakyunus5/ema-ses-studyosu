"""EMA Ses Stüdyosu — FastAPI backend.

EMA Lightning (Apache-2.0, Canberk Aslan) tabanlı çoklu-ses Türkçe TTS API'si.
Render üzerinde çalışacak şekilde tasarlandı (CPU, düşük RAM).
"""
import base64
import hashlib
import os
import random
import threading
import time
from collections import OrderedDict

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from voices import VOICES, SAMPLE_TEXTS, SR, pitch_shift, normalize_peak, encode_wav

# ---------------------------------------------------------------------------
# Kurulum
# ---------------------------------------------------------------------------
THREADS = int(os.environ.get("TORCH_THREADS", "1"))
torch.set_num_threads(THREADS)

from ema_lightning import EMA  # noqa: E402  (torch ayarlarından sonra)

MODEL_LOADED_AT = None
_tts = None


def get_tts():
    global _tts, MODEL_LOADED_AT
    if _tts is None:
        t0 = time.time()
        _tts = EMA(device="cpu")
        MODEL_LOADED_AT = time.time()
        print(f"[ema] model yüklendi ({time.time()-t0:.1f}s)")
    return _tts


MAX_TEXT = int(os.environ.get("MAX_TEXT_CHARS", "600"))
app = FastAPI(title="EMA Ses Stüdyosu", version="1.0.0")

origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Basit LRU önbellek (aynı istekleri tekrar üretme)
# ---------------------------------------------------------------------------
_CACHE = OrderedDict()
_CACHE_MAX = 64
_cache_lock = threading.Lock()


def cache_get(key):
    with _cache_lock:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    return None


def cache_put(key, value):
    with _cache_lock:
        _CACHE[key] = value
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)


# ---------------------------------------------------------------------------
# Şemalar
# ---------------------------------------------------------------------------
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = "elif"
    speed: float = Field(1.0, ge=0.5, le=1.75)
    pitch: float = Field(0.0, ge=-6.0, le=6.0)  # profil üzerine ek ince ayar
    seed: int | None = None


class Word(BaseModel):
    text: str
    start: float
    end: float


class TTSResponse(BaseModel):
    audio: str          # base64 WAV (mono, 16-bit PCM)
    sample_rate: int
    duration: float
    seed: int
    voice: str
    chars: int
    elapsed_ms: float
    cached: bool
    words: list[Word]


# ---------------------------------------------------------------------------
# Uç noktalar
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "EMA Ses Stüdyosu API",
        "model": "EMA Lightning (Apache-2.0) — https://huggingface.co/canberkkkkkk/ema-lightning",
        "endpoints": ["/api/health", "/api/voices", "/api/tts"],
    }


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "model_loaded": _tts is not None,
        "uptime_s": round(time.time() - MODEL_LOADED_AT, 1) if MODEL_LOADED_AT else 0,
        "voices": len(VOICES),
    }


@app.get("/api/voices")
def list_voices():
    return [
        {k: v[k] for k in ("id", "name", "emoji", "tag", "desc", "pitch", "tempo")}
        | {"sample_text": SAMPLE_TEXTS[v["id"]]}
        for v in VOICES.values()
    ]


@app.post("/api/tts", response_model=TTSResponse)
def tts_endpoint(req: TTSRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(422, "Metin boş olamaz.")
    trimmed = False
    if len(text) > MAX_TEXT:
        text = text[:MAX_TEXT]
        trimmed = True

    voice = VOICES.get(req.voice)
    if voice is None:
        raise HTTPException(404, f"Bilinmeyen ses: {req.voice}. Seçenekler: {', '.join(VOICES)}")

    eff_pitch = float(np.clip(voice["pitch"] + req.pitch, -9, 9))
    eff_speed = float(np.clip(voice["tempo"] * req.speed, 0.25, 4.0))
    seed = req.seed if req.seed is not None else voice["seed"]

    key = hashlib.sha256(f"{text}|{req.voice}|{eff_speed:.4f}|{eff_pitch:.2f}|{seed}".encode()).hexdigest()
    cached = cache_get(key)
    t0 = time.time()
    if cached is None:
        try:
            speech = get_tts().say(text, speed=eff_speed, seed=seed, sample_rate=SR)
        except Exception as exc:  # model hatası: kullanıcıya anlamlı mesaj
            raise HTTPException(500, f"Ses üretilemedi: {exc}") from exc
        audio = pitch_shift(speech.audio, eff_pitch)
        audio = normalize_peak(audio)
        wav = encode_wav(audio, SR)
        words = [{"text": w.text, "start": round(w.start, 2), "end": round(w.end, 2)} for w in speech.words]
        cached = {
            "audio": base64.b64encode(wav).decode(),
            "duration": round(float(len(audio) / SR), 2),
            "words": words,
        }
        cache_put(key, cached)

    return TTSResponse(
        audio=cached["audio"],
        sample_rate=SR,
        duration=cached["duration"],
        seed=seed,
        voice=req.voice,
        chars=len(text) + (0 if not trimmed else 0),
        elapsed_ms=round((time.time() - t0) * 1000, 1),
        cached=key in _CACHE and _CACHE[key] is cached and (time.time() - t0) < 0.05,
        words=[Word(**w) for w in cached["words"]],
    )


@app.on_event("startup")
def warmup():
    get_tts()

    def _warm():
        try:
            for vid, v in VOICES.items():
                speech = get_tts().say(SAMPLE_TEXTS[vid], speed=v["tempo"], seed=v["seed"], sample_rate=SR)
                audio = normalize_peak(pitch_shift(speech.audio, v["pitch"]))
                wav = encode_wav(audio, SR)
                key = hashlib.sha256(
                    f"{SAMPLE_TEXTS[vid]}|{vid}|{v['tempo']:.4f}|{v['pitch']:.2f}|{v['seed']}".encode()
                ).hexdigest()
                cache_put(key, {"audio": base64.b64encode(wav).decode(),
                                "duration": round(float(len(audio) / SR), 2),
                                "words": [{"text": w.text, "start": round(w.start, 2), "end": round(w.end, 2)}
                                          for w in speech.words]})
            print("[ema] önizleme örnekleri hazır")
        except Exception as exc:
            print(f"[ema] warmup önizleme hatası (önemsiz): {exc}")

    threading.Thread(target=_warm, daemon=True).start()
