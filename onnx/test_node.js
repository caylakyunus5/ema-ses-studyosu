/* engine.js uçtan uca doğrulama: Python (PyTorch) referanslarıyla karşılaştırma */
const ort = require("onnxruntime-web");
const E = require("./engine.js");
const fs = require("fs");

const DIR = "/home/user/ema-onnx";
const fx = JSON.parse(fs.readFileSync(`${DIR}/fixtures_js.json`));
const meta = JSON.parse(fs.readFileSync(`${DIR}/meta.json`));

const f32 = a => Float32Array.from(a);
const i64 = a => BigInt64Array.from(a.map(BigInt));
const flatten2 = a => a.flat(10);

function maxDiff(a, b) {
  let m = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m + Math.abs(a.length - b.length);
}
function pearson(a, b) {
  const n = Math.min(a.length, b.length);
  let sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { sa += a[i]; sb += b[i]; }
  const ma = sa / n, mb = sb / n;
  for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
  return sab / Math.sqrt(saa * sbb + 1e-12);
}
function f0Est(a, sr = 24000) {
  const f = [];
  for (let s = 0; s + 1024 < a.length; s += 480) {
    let e = 0; for (let i = 0; i < 1024; i++) e += a[s + i] ** 2;
    if (Math.sqrt(e / 1024) < 0.03) continue;
    const N = 2048, re = new Float32Array(N), im = new Float32Array(N);
    for (let i = 0; i < 1024; i++) re[i] = a[s + i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / 1023));
    E.fft(re, im, false); // otokorelasyon: |X|^2'nin ters FFT'si yerine FFT ve zaman-domain autocorr daha kolay:
    f.push(null);
    break;
  }
  return f; // kullanılmayacak, basit autocorr aşağıda
}
function f0(a, sr = 24000) {
  const out = [];
  for (let s = 0; s + 1024 < a.length; s += 480) {
    const fr = a.slice(s, s + 1024);
    let e = 0; for (const v of fr) e += v * v;
    if (Math.sqrt(e / 1024) < 0.03) continue;
    const c = new Float32Array(400);
    for (let lag = 0; lag < 400; lag++) { let s2 = 0; for (let i = 0; i + lag < 1024; i++) s2 += fr[i] * fr[i + lag]; c[lag] = s2; }
    let p = 60; for (let l = 61; l < 400; l++) if (c[l] > c[p]) p = l;
    if (c[p] > 0.3 * c[0]) out.push(sr / p);
  }
  out.sort((x, y) => x - y);
  return out.length ? out[out.length >> 1] : 0;
}

(async () => {
  const sess = {
    text: await ort.InferenceSession.create(`${DIR}/text_stage.onnx`),
    sound: await ort.InferenceSession.create(`${DIR}/sound_stage.onnx`),
    decoder: await ort.InferenceSession.create(`${DIR}/decoder.onnx`),
  };
  const engine = new E.EmaEngine(ort, sess, meta);
  console.log("— test 1: metin normalizasyonu —");
  const t = engine.normalize("Merhaba! Bugün hava çok güzel, değil mi? Yarın görüşürüz. Saat 14:30'da %15 indirim ve 1.250 TL ödeme.");
  console.log("normalize:", JSON.stringify(t));

  console.log("— test 2: text_stage (ids/cw/wstart → h,dur) —");
  const idsN = fx.ids[0].length;
  const ids = i64(fx.ids[0]);
  const hIn = f32(flatten2(fx.h_in));
  const durIn = f32(fx.dur_in[0]);
  // engine wordMap kontrolü: fixture cw/wstart ile aynı olmalı
  const { cw, wstart } = engine.cwForTest !== undefined ? {} : E.wordMap("merhaba! bugün hava çok güzel, değil mi? yarın görüşürüz.");
  const cwFix = i64(fx.cw[0]), wsFix = i64(fx.wstart[0]);
  console.log("cw eşit:", String(cw) === String(cwFix) && String(wstart) === String(wsFix));

  const L = fx.h_in[0].length;
  const textOut = await sess.text.run({
    ids: new ort.Tensor("int64", ids, [1, idsN]),
    mask: new ort.Tensor("bool", new Uint8Array(idsN).fill(1), [1, idsN]),
  });
  console.log("h farkı:", maxDiff(textOut.h.data, hIn).toExponential(2));
  console.log("dur farkı:", maxDiff(textOut.dur.data, durIn).toExponential(2));

  console.log("— test 3: planFrames ↔ python fw/fp —");
  const { fw, fp, wordStart, wordEnd } = E.planFrames(durIn, cwFix, 1.0);
  const fwFix = fx.fw[0], fpFix = fx.fp[0];
  const sameFw = String(Array.from(fw, Number)) === String(fwFix.map(Number));
  console.log("fw uzunluk:", fw.length, "vs", fwFix.length, "| içerik eşit:", sameFw);
  console.log("fp max fark:", maxDiff(fp, f32(fpFix)).toExponential(2));

  console.log("— test 4: sound_stage + decoder ↔ pytorch zinciri —");
  const T = fx.noise[0][0].length, CH = fx.noise[0][0][0].length;  // fx.noise: [1][4][T][64]
  const noise = f32(flatten2(fx.noise));
  const lat = (await sess.sound.run({
    h: new ort.Tensor("float32", hIn, [1, L, meta.cfg.d]),
    dur: new ort.Tensor("float32", durIn, [1, L]),
    mask: new ort.Tensor("bool", new Uint8Array(idsN).fill(1), [1, idsN]),
    cw: new ort.Tensor("int64", cwFix, [1, idsN]),
    wstart: new ort.Tensor("int64", wsFix, [1, idsN]),
    fw: new ort.Tensor("int64", i64(fx.fw[0]), [1, fx.fw[0].length]),
    fp: new ort.Tensor("float32", f32(fx.fp[0]), [1, fx.fp[0].length]),
    fmask: new ort.Tensor("bool", new Uint8Array(fx.fw[0].length).fill(1), [1, fx.fw[0].length]),
    noise: new ort.Tensor("float32", noise, [1, 4, T, CH]),
  })).latents;
  console.log("latents farkı:", maxDiff(lat.data, f32(flatten2(fx.latents))).toExponential(2));
  const aud = (await sess.decoder.run({ z: lat })).audio.data;
  const audioRef = f32(fx.audio);
  console.log("audio farkı:", maxDiff(aud, audioRef).toExponential(2), "| korelasyon:", pearson(aud, audioRef).toFixed(6));

  console.log("— test 5: 24k yeniden örnekleme + perde (-5) ↔ numpy —");
  const a24 = E.resampleSinc(aud, 0.5, 64);
  const a24Ref = f32(fx.a24_ref);
  console.log("a24 korelasyon:", pearson(a24, a24Ref).toFixed(4), "| uzunluk:", a24.length, "vs", a24Ref.length);
  const shifted = E.pitchShift(a24, -5.0);
  const shRef = f32(fx.shifted_m5_ref);
  console.log("shifted korelasyon:", pearson(shifted, shRef).toFixed(4), "| uzunluk:", shifted.length, "vs", shRef.length);
  console.log("f0: ref 24k", f0(a24).toFixed(1), "Hz → shifted", f0(shifted).toFixed(1), "Hz (beklenen oran 0.749)");
  console.log("f0 numpy ref:", f0(shRef).toFixed(1), "Hz");

  console.log("— test 6: tam boru hattı (synthesize) —");
  let doneMsg = null;
  for await (const ev of engine.synthesize("Merhaba! Bugün hava çok güzel, değil mi? Yarın görüşürüz.", { voice: { pitch: 0, tempo: 1, seed: 7 }, speed: 1 })) {
    if (ev.type === "progress") process.stdout.write(`\r  ilerleme %${(ev.done * 100).toFixed(0)}`);
    else doneMsg = ev;
  }
  console.log(`\nsüre: ${doneMsg.duration} sn | kelime: ${doneMsg.words.length} | wav: ${doneMsg.wav.length} bayt | ${doneMsg.elapsedMs} ms`);
  console.log("kelimeler:", doneMsg.words.slice(0, 4).map(w => `${w.text}@${w.start}`).join("  "));
  console.log("wav başlık RIFF:", String.fromCharCode(...doneMsg.wav.slice(0, 4)) === "RIFF");
})().catch(e => { console.error("HATA:", e); process.exit(1); });
