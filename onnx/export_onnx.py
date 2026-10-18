"""EMA Lightning -> ONNX: text_stage / sound_stage / decoder, sayısal doğrulamalı.
JS portu için fixtures.npz de üretir."""
import json
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

from ema_lightning import EMA
from ema_lightning.chunker import chunk

OUT = "/home/user/ema-onnx"
dev = torch.device("cpu")

tts = EMA(device=dev)
eng = tts._engine
model, decoder = eng.model, eng.decoder

# ---------------------------------------------------------------- fixtures: gerçek ara tensörler
TEXT = "Merhaba! Bugün hava çok güzel, değil mi? Yarın görüşürüz."
norm = tts._frontend(TEXT)
print("normalize:", repr(norm))
pieces = chunk(norm, 1.0)
print("parçalar:", pieces)
ptext, ppause = pieces[0]
piece = eng.piece(ptext, ppause, 7)
eng.plan([piece], 1.0)          # piece.h, piece.dur, piece.fw, piece.fp dolar
noise = eng.noise(piece)        # [4,T,64]

L = piece.letters
T = piece.frames
ids1 = piece.ids[None].long()
mask1 = torch.ones(1, L, dtype=torch.bool)
cw1 = piece.cw[None].long()
wstart1 = piece.wstart[None].long()
fw1 = piece.fw[None].long()
fp1 = piece.fp[None].float()
fmask1 = torch.ones(1, T, dtype=torch.bool)
noise1 = noise[None]

with torch.no_grad():
    h_t, dur_t = model.text_stage(ids1, mask1)
    lat_t = model.sound_stage(piece.h[None], piece.dur[None], mask1, cw1, wstart1, fw1, fp1, fmask1, noise1)
    aud_t = decoder(lat_t.transpose(1, 2), None)   # [1, 64, T] -> [1, T*1920]
audio_np = aud_t[0].numpy()
print(f"L={L} T={T} frames, audio={len(audio_np)} samples ({len(audio_np)/48000:.2f}s)")

np.savez(f"{OUT}/fixtures.npz",
         ids=ids1.numpy(), mask=mask1.numpy(), cw=cw1.numpy(), wstart=wstart1.numpy(),
         fw=fw1.numpy(), fp=fp1.numpy(), fmask=fmask1.numpy(), noise=noise1.numpy(),
         h=h_t.numpy(), dur=dur_t.numpy(), latents=lat_t.numpy(), audio=audio_np,
         h_in=piece.h[None].numpy(), dur_in=piece.dur[None].numpy())
json.dump({"norm_text": ptext, "seed": 7, "L": int(L), "T": int(T)},
          open(f"{OUT}/fixtures.json", "w"))

# ---------------------------------------------------------------- export wrappers
class TextStage(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids, mask):
        return self.m.text_stage(ids, mask)


class SoundStage(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, h, dur, mask, cw, wstart, fw, fp, fmask, noise):
        return self.m.sound_stage(h, dur, mask, cw, wstart, fw, fp, fmask, noise)


class DecodeStage(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, z):                      # z [1,T,64]
        return self.d(z.transpose(1, 2), None)  # -> [1, T*1920]


def to_onnx(module, args, path, in_names, out_names, dyn):
    # aten::expm1 ONNX'te yok: exp(x)-1 olarak izle (matematiksel eşdeğer)
    _orig = torch.expm1
    torch.expm1 = lambda x: torch.exp(x) - 1.0
    try:
        torch.onnx.export(
            module, args, path,
            input_names=in_names, output_names=out_names,
            dynamic_axes=dyn, opset_version=18,
            dynamo=False,  # TorchScript tabanlı export: kontrol akışı yok, gayet stabil
        )
    finally:
        torch.expm1 = _orig
    print("exported:", path)


to_onnx(TextStage(model), (ids1, mask1), f"{OUT}/text_stage.onnx",
        ["ids", "mask"], ["h", "dur"],
        {"ids": {1: "L"}, "mask": {1: "L"}, "h": {1: "L"}, "dur": {1: "L"}})

to_onnx(SoundStage(model), (piece.h[None], piece.dur[None], mask1, cw1, wstart1, fw1, fp1, fmask1, noise1),
        f"{OUT}/sound_stage.onnx",
        ["h", "dur", "mask", "cw", "wstart", "fw", "fp", "fmask", "noise"], ["latents"],
        {"h": {1: "L"}, "dur": {1: "L"}, "mask": {1: "L"}, "cw": {1: "L"}, "wstart": {1: "L"},
         "fw": {1: "T"}, "fp": {1: "T"}, "fmask": {1: "T"}, "noise": {2: "T"}, "latents": {1: "T"}})

to_onnx(DecodeStage(decoder), (lat_t,), f"{OUT}/decoder.onnx",
        ["z"], ["audio"], {"z": {1: "T"}, "audio": {1: "S"}})

# ---------------------------------------------------------------- doğrulama
def run(sess_path, feed):
    sess = ort.InferenceSession(sess_path, providers=["CPUExecutionProvider"])
    return sess.run(None, feed)


def close(name, a, b, tol):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    diff = float(np.abs(a - b).max())
    print(f"{name}: max|delta| = {diff:.2e}  {'PASS' if diff < tol else 'FAIL'}")
    return diff < tol


h_o, dur_o = run(f"{OUT}/text_stage.onnx", {"ids": ids1.numpy(), "mask": mask1.numpy()})
ok1 = close("text h", h_o, h_t.numpy(), 2e-3) & close("text dur", dur_o, dur_t.numpy(), 2e-3)

lat_o = run(f"{OUT}/sound_stage.onnx",
            {"h": piece.h[None].numpy(), "dur": piece.dur[None].numpy(), "mask": mask1.numpy(),
             "cw": cw1.numpy(), "wstart": wstart1.numpy(), "fw": fw1.numpy(), "fp": fp1.numpy(),
             "fmask": fmask1.numpy(), "noise": noise1.numpy()})[0]
ok2 = close("latents", lat_o, lat_t.numpy(), 2e-3)

aud_o = run(f"{OUT}/decoder.onnx", {"z": lat_t.numpy().astype(np.float32)})[0]
ok3 = close("audio", aud_o, audio_np, 2e-3)

import os
for f in ("text_stage", "sound_stage", "decoder"):
    print(f, f"{os.path.getsize(f'{OUT}/{f}.onnx')/1e6:.1f} MB")
print("SONUÇ:", "HEPSİ GEÇTİ ✅" if (ok1 and ok2 and ok3) else "SORUN VAR ❌")
