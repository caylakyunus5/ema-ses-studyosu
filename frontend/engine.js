/* EMA Ses Stüdyosu — tarayıcı içi TTS motoru (engine.js) v2
 * EMA Lightning ONNX portu + saf-JS normalizasyon + FORMANT destekli ses motoru + duygu katmanı.
 * Node (onnxruntime-web/node) ve tarayıcı (onnxruntime-web) ile çalışır.
 */
(function (root) {
  const FPS = 25, HOP = 1920, SR48 = 48000, SR24 = 24000;
  const MAX_WORD_FRAMES = 250, MAX_FRAMES = 3000;
  const LETTERS_PER_SECOND = 18.0, MAX_SECONDS = 10.0, MAX_LETTERS = 250;

  /* ================= Türkçe sayı->yazı ================= */
  const O = ["sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"];
  const ON = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"];
  function yuz(n) {
    let s = "";
    if (n >= 100) { if (n >= 200) s += O[Math.floor(n / 100)] + " "; s += "yüz "; n %= 100; }
    if (n >= 10) { s += ON[Math.floor(n / 10)] + " "; n %= 10; }
    if (n > 0) s += O[n] + " ";
    return s.trim();
  }
  function sayi(n) {
    if (n === 0) return "sıfır";
    const G = [[1e12, "trilyon"], [1e9, "milyar"], [1e6, "milyon"], [1e3, "bin"]];
    let out = [];
    for (const [v, ad] of G) {
      if (n >= v) {
        const k = Math.floor(n / v);
        if (!(v === 1000 && k === 1)) out.push(sayi(k));
        out.push(ad);
        n -= Math.floor(n / v) * v;
      }
    }
    if (n > 0) out.push(yuz(n));
    return out.join(" ");
  }

  /* ================= hafif Türkçe normalizasyon ================= */
  const KISALTMA = { "dr": "doktor", "prof": "profesör", "doç": "doçent", "sn": "sayın", "vb": "vesaire", "vs": "vesaire", "vd": "vesaire", "örn": "örneğin", "tl": "türk lirası", "km": "kilometre", "cm": "santimetre", "mm": "milimetre", "kg": "kilogram", "gr": "gram", "mg": "miligram", "lt": "litre", "ml": "mililitre", "cad": "cadde", "sok": "sokak", "apt": "apartman", "no": "numara", "tel": "telefon", "yak": "yaklaşık", "bkz": "bakınız", "av": "avukat", "müh": "mühendis" };
  const AY = { "ocak": 1, "şubat": 2, "mart": 3, "nisan": 4, "mayıs": 5, "haziran": 6, "temmuz": 7, "ağustos": 8, "eylül": 9, "ekim": 10, "kasım": 11, "aralık": 12 };
  const HARF = { a: "a", b: "be", c: "ce", "ç": "çe", d: "de", e: "e", f: "fe", g: "ge", "ğ": "yumuşak ge", h: "he", "ı": "ı", i: "i", j: "je", k: "ke", l: "le", m: "me", n: "ne", o: "o", "ö": "ö", p: "pe", r: "re", s: "se", "ş": "şe", t: "te", u: "u", "ü": "ü", v: "ve", y: "ye", z: "ze", w: "dublüve", q: "kü", x: "iks" };
  function hecele(kelime) { return [...kelime.toLowerCase()].map(c => HARF[c] || "").filter(Boolean).join(" "); }
  function paraKismi(str) {
    const [tam, kus] = str.split(",");
    let s = sayi(parseInt(tam.replace(/\./g, "") || "0", 10));
    if (kus) s += " virgül " + sayi(parseInt(kus, 10));
    return s;
  }
  function normalizeTR(text) {
    let t = " " + text.replace(/\s+/g, " ") + " ";
    t = t.replace(/\S+@\S+\.\S+/g, " e-posta adresi ");
    t = t.replace(/https?:\/\/\S+|www\.\S+/g, " internet adresi ");
    t = t.replace(/\b(\d{1,2})[:.](\d{2})(?!\d)/g, (m, a, b) => (+a < 24 && +b < 60) ? " " + sayi(+a) + (b === "00" ? "" : " " + sayi(+b)) + " " : m);
    t = t.replace(/\b(\d{1,2})[./](\d{1,2})[./](\d{2,4})\b/g, (m, g, a, y) => {
      const ayAd = Object.keys(AY).find(k => AY[k] === parseInt(a));
      return " " + sayi(+g) + " " + (ayAd || sayi(+a)) + " " + sayi(+(y.length === 2 ? "20" + y : y)) + " ";
    });
    t = t.replace(/\b(\d{1,2})\s+(ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)\s+(\d{4})/gi, (m, g, a, y) =>
      " " + sayi(+g) + " " + a.toLowerCase() + " " + sayi(+y) + " ");
    t = t.replace(/%\s*(\d+(?:[.,]\d+)?)/g, (m, n) => " yüzde " + paraKismi(n.replace(".", "")) + " ");
    t = t.replace(/(\d+(?:[.,]\d+)?)\s*%/g, (m, n) => " yüzde " + paraKismi(n.replace(".", "")) + " ");
    t = t.replace(/(\d+(?:\.\d{3})*(?:,\d+)?)\s*(tl|₺)/gi, (m, n) => " " + paraKismi(n) + " türk lirası ");
    t = t.replace(/[$£€]\s*(\d+(?:[.,]\d+)?)/g, (m, n) => {
      const birim = m[0] === "$" ? "dolar" : m[0] === "€" ? "avro" : "sterlin";
      return " " + paraKismi(n) + " " + birim + " ";
    });
    t = t.replace(/\b(\d+),(\d+)\b/g, (m, a, b) => " " + sayi(+a) + " virgül " + sayi(+b) + " ");
    t = t.replace(/\b(\d{1,3}(?:\.\d{3})+)\b/g, (m) => " " + sayi(parseInt(m.replace(/\./g, ""), 10)) + " ");
    t = t.replace(/\b(\d+)\b/g, (m) => " " + sayi(parseInt(m, 10)) + " ");
    t = t.replace(/\b([A-ZÇĞİÖŞÜ]{2,})\b/g, (m) => " " + hecele(m) + " ");
    t = t.replace(/\b(dr|prof|doç|sn|vb|vs|vd|örn|yak|bkz|av|müh|cad|sok|apt|tel|no)\.(?=\s|$)/gi, (m, k) => " " + (KISALTMA[k.toLowerCase()] || k.toLowerCase()) + " ");
    t = t.replace(/\b(kg|gr|mg|cm|mm|km|lt|ml)\b\.?/g, (m, k) => " " + KISALTMA[k] + " ");
    return t.replace(/\s+/g, " ").trim();
  }

  /* ================= NOKTALAMA OTOMATİK DÜZELTME ================= */
  function fixPunctuation(text) {
    // satır satır: her satırın sonuna cümle sonu imi ekle, boşluk/tekrar düzelt
    const lines = text.split(/\n+/).map(l => l.trim()).filter(Boolean);
    if (!lines.length) return text.trim();
    const fixed = lines.map(line => {
      let t = line
        .replace(/\s+/g, " ")
        .replace(/([.!?…])\1+/g, "$1")         // !!! -> !, ??? -> ?, …… -> …
        .replace(/…/g, ".")
        .replace(/\s+([,.!?;:%)»"'])/g, "$1")   // im öncesi boşluk yok
        .replace(/([(\[«《"'])\s+/g, "$1")       // açma imi sonrası boşluk yok
        .replace(/([,.!?;:])(?=\S)/g, "$1 ")    // im sonrası boşluk
        .replace(/,\s*\./g, ".")
        .trim();
      if (!/[.!?…:;]["'»)]*$/.test(t)) t += /[,]$/.test(t) ? "" : ".";
      t = t.replace(/,\s*$/, ".");
      return t;
    });
    return fixed.join(" ").replace(/\s+/g, " ").trim();
  }

  /* ================= alfabe ================= */
  const TIP = { "’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'", "“": '"', "”": '"', "„": '"', "«": '"', "»": '"', "–": "-", "—": "-", "−": "-", "…": "..." };
  const TUR = new Set([..."çğıöşüÇĞİÖŞÜ"]);
  function makeAlphabet(vocab) {
    const V = new Set(vocab);
    return function (text) {
      text = [...text].map(c => TIP[c] || c).join("").replaceAll("İ", "i").replaceAll("I", "ı").toLowerCase();
      let out = [];
      for (const ch of text) {
        let c = ch;
        if (!TUR.has(ch)) c = ch.normalize("NFKD").replace(/[̀-ͯ]/g, "");
        out.push(c && [...c].every(x => V.has(x)) ? c : " ");
      }
      return out.join("").replace(/\s+/g, " ").trim();
    };
  }

  /* ================= parçalama ================= */
  const CUTS = [
    { re: /[.!?]+["')]*(?= )/g, pause: 0.25 },
    { re: /[,;:](?= )/g, pause: 0.12 },
    { re: /\S(?= )/g, pause: 0.12 },
  ];
  function finish(piece) {
    const s = piece.replace(/[\"')]+\s*$/, "");
    if (/[.!?]$/.test(s)) return piece;
    return piece.replace(/[,;:\- ]+$/, "") + ".";
  }
  function chunkText(text, speed) {
    const limit = Math.trunc(Math.min(MAX_LETTERS, LETTERS_PER_SECOND * MAX_SECONDS * speed));
    const pieces = [];
    let rest = text.trim();
    while (rest.length) {
      let cut = rest.length, pause = 0;
      if (rest.length > limit) {
        cut = limit;
        for (const { re, pause: gap } of CUTS) {
          re.lastIndex = 0;
          const ends = [];
          let m;
          const region = rest.slice(0, limit + 1);
          while ((m = re.exec(region)) !== null) { ends.push(m.index + m[0].length); if (m[0] === "") re.lastIndex++; }
          if (ends.length) { cut = ends[ends.length - 1]; pause = gap; break; }
        }
      }
      const piece = rest.slice(0, cut).trim();
      rest = rest.slice(cut).trim();
      if (/[a-zçğıöşü]/.test(piece)) pieces.push([finish(piece), pause]);
    }
    if (pieces.length) pieces[pieces.length - 1][1] = 0;
    return pieces;
  }

  /* ================= tohumlu gürültü ================= */
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function randn(seed, n) {
    const rnd = mulberry32(seed >>> 0 || 1);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 2) {
      const u1 = Math.max(rnd(), 1e-12), u2 = rnd();
      const r = Math.sqrt(-2 * Math.log(u1)), th = 2 * Math.PI * u2;
      out[i] = r * Math.cos(th);
      if (i + 1 < n) out[i + 1] = r * Math.sin(th);
    }
    return out;
  }

  /* ================= FFT ================= */
  const _twid = {};
  function fftInit(n) {
    if (!_twid[n]) {
      const cos = new Float32Array(n / 2), sin = new Float32Array(n / 2), rev = new Uint32Array(n);
      for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos(-2 * Math.PI * i / n); sin[i] = Math.sin(-2 * Math.PI * i / n); }
      const bits = Math.log2(n);
      for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); rev[i] = r; }
      _twid[n] = { cos, sin, rev };
    }
    return _twid[n];
  }
  function fft(re, im, invert) {
    const n = re.length, { cos, sin, rev } = fftInit(n);
    for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1, step = n / len;
      for (let i = 0; i < n; i += len) {
        for (let j = 0; j < half; j++) {
          const wr = cos[j * step], wi = (invert ? -sin[j * step] : sin[j * step]);
          const a = i + j, b = a + half;
          const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
          re[b] = re[a] - tr; im[b] = im[a] - ti;
          re[a] += tr; im[a] += ti;
        }
      }
    }
    if (invert) { for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; } }
  }

  /* ================= FORMANT'LI faz vokoder =================
   * Genlik spektrumu = zarf (envelope) × ince yapı (harmonik).
   * Zarf, frekans ekseninde c_rel ile ölçeklenir; perde sonraki yeniden
   * örneklemede r ile çarpılacağından net formant = c_rel × r olur. */
  function hann(n) { const w = new Float32Array(n); for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)); return w; }
  const _hann = {}, _smk = {}, _fmap = {};
  function timeStretchFormant(y, factor, formantRel = 1, nFft = 1024, hop = 256, smoothBins = 11) {
    const win = _hann[nFft] || (_hann[nFft] = hann(nFft));
    const padded = new Float32Array(nFft + y.length + nFft);
    padded.set(y, nFft);
    const D = [];
    for (let i = 0; i + nFft <= padded.length; i += hop) {
      const re = new Float32Array(nFft), im = new Float32Array(nFft);
      for (let k = 0; k < nFft; k++) re[k] = padded[i + k] * win[k];
      fft(re, im, false);
      D.push({ re, im });
    }
    const nBins = nFft / 2 + 1;
    const dphi = new Float32Array(nBins);
    for (let i = 0; i < nBins; i++) dphi[i] = 2 * Math.PI * hop * i / nFft;
    // zarf yumuşatma çekirdeği
    const doFormant = Math.abs(formantRel - 1) > 1e-3;
    let smk = null, f_lo = null, f_hi = null, f_fr = null;
    if (doFormant) {
      if (!_smk[smoothBins]) {
        const kk = hann(smoothBins * 2 + 1);
        let s = 0; for (const v of kk) s += v;
        for (let i = 0; i < kk.length; i++) kk[i] /= s;
        _smk[smoothBins] = kk;
      }
      smk = _smk[smoothBins];
      const fkey = nBins + ":" + formantRel.toFixed(4);
      if (!_fmap[fkey]) {
        const lo = new Int32Array(nBins), hi = new Int32Array(nBins), fr = new Float32Array(nBins);
        for (let i = 0; i < nBins; i++) {
          const f = Math.min(Math.max(i / formantRel, 0), nBins - 1);
          lo[i] = Math.floor(f); hi[i] = Math.min(lo[i] + 1, nBins - 1); fr[i] = f - lo[i];
        }
        _fmap[fkey] = { lo, hi, fr };
      }
      ({ lo: f_lo, hi: f_hi, fr: f_fr } = _fmap[fkey]);
    }
    const mag = (f, i) => Math.hypot(f.re[i], f.im[i]);
    const ang = (f, i) => Math.atan2(f.im[i], f.re[i]);
    const phase = new Float32Array(nBins);
    for (let i = 0; i < nBins; i++) phase[i] = ang(D[0], i);
    const frames = [];
    const logm = new Float32Array(nBins), env = new Float32Array(nBins), fine = new Float32Array(nBins);
    for (let tt = 0; (tt | 0) + 1 <= D.length - 1; tt += 1 / factor) {
      const i = tt | 0, frac = tt - i;
      const ore = new Float32Array(nFft), oim = new Float32Array(nFft);
      let e0 = 0, e1 = 0;
      for (let b = 0; b < nBins; b++) {
        const m = (1 - frac) * mag(D[i], b) + frac * mag(D[i + 1], b);
        logm[b] = Math.log(m + 1e-9);
        e0 += m * m;
      }
      if (doFormant) {
        for (let b = 0; b < nBins; b++) { // "same" konvolüsyon
          let s = 0;
          for (let j = 0; j < smk.length; j++) { const x = b + j - smoothBins; s += smk[j] * (x >= 0 && x < nBins ? logm[x] : 0); }
          env[b] = s;
        }
        let emax = -1e9;
        for (let b = 0; b < nBins; b++) if (env[b] > emax) emax = env[b];
        const floor = emax - 9.2;
        for (let b = 0; b < nBins; b++) {
          const ev = Math.max(env[b], floor);
          fine[b] = logm[b] - ev;
        }
        for (let b = 0; b < nBins; b++) {
          const es = env[f_lo[b]] * (1 - f_fr[b]) + env[f_hi[b]] * f_fr[b];
          const nl = fine[b] + Math.max(es, floor);
          logm[b] = nl;
          e1 += Math.exp(2 * nl);
        }
        const scale = Math.min(2, Math.max(0.5, Math.sqrt(e0 / (e1 + 1e-12))));
        for (let b = 0; b < nBins; b++) logm[b] = Math.exp(logm[b]) * scale;
      } else {
        for (let b = 0; b < nBins; b++) logm[b] = Math.exp(logm[b]);
      }
      for (let b = 0; b < nBins; b++) {
        ore[b] = logm[b] * Math.cos(phase[b]); oim[b] = logm[b] * Math.sin(phase[b]);
        let d = ang(D[i + 1], b) - ang(D[i], b) - dphi[b];
        d -= 2 * Math.PI * Math.round(d / (2 * Math.PI));
        phase[b] += dphi[b] + d;
      }
      fft(ore, oim, true);
      frames.push(ore);
    }
    const nOut = nFft + hop * frames.length;
    const out = new Float32Array(nOut), ws = new Float32Array(nOut);
    frames.forEach((f, k) => { for (let i = 0; i < nFft; i++) { out[k * hop + i] += f[i] * win[i]; ws[k * hop + i] += win[i] * win[i]; } });
    const start = Math.round(nFft * factor), target = Math.round(y.length * factor);
    const res = new Float32Array(target);
    for (let i = 0; i < target; i++) { const v = out[start + i] / Math.max(ws[start + i] || 1, 1e-8); res[i] = isFinite(v) ? v : 0; }
    return res;
  }
  function resampleSinc(x, ratio, taps = 48) {
    if (ratio === 1) return x;
    const nOut = Math.max(1, Math.round(x.length * ratio));
    const fc = 0.45 * Math.min(1, ratio), half = taps >> 1;
    const out = new Float32Array(nOut);
    for (let m = 0; m < nOut; m++) {
      const center = m / ratio;
      const i0 = Math.max(0, Math.ceil(center - half)), i1 = Math.min(x.length - 1, Math.floor(center + half));
      let sum = 0, wsum = 0;
      for (let i = i0; i <= i1; i++) {
        const d = i - center;
        const w = 0.42 + 0.5 * Math.cos(Math.PI * d / half) + 0.08 * Math.cos(2 * Math.PI * d / half);
        const k = d === 0 ? 1 : Math.sin(2 * Math.PI * fc * d) / (2 * Math.PI * fc * d);
        const g = 2 * fc * k * w;
        sum += x[i] * g; wsum += g;
      }
      out[m] = Math.abs(wsum) > 1e-8 ? sum / wsum : (x[Math.round(center)] || 0);
    }
    return out;
  }
  function voiceShift(audio, pitchSt, formantSt = 0) {
    const r = Math.pow(2, pitchSt / 12), c = Math.pow(2, formantSt / 12);
    if (Math.abs(r - 1) < 1e-4 && Math.abs(c - 1) < 1e-4) return audio;
    const stretched = timeStretchFormant(audio, r, c / r);
    const back = resampleSinc(stretched, audio.length / stretched.length, 64);
    for (let i = 0; i < back.length; i++) back[i] = Math.max(-1, Math.min(1, back[i]));
    return back.subarray(0, audio.length);
  }

  /* ================= planlama ================= */
  function idsFor(text, stoi) {
    const ids = new BigInt64Array(text.length);
    for (let i = 0; i < text.length; i++) ids[i] = BigInt(stoi[text[i]] ?? 1);
    return ids;
  }
  function wordMap(text) {
    const starts = [];
    for (let i = 0; i < text.length; i++) if (text[i] !== " " && (i === 0 || text[i - 1] === " ")) starts.push(i);
    if (!starts.length) starts.push(0);
    const bounds = [0, ...starts.slice(1), text.length];
    const cw = new BigInt64Array(text.length), wstart = new BigInt64Array(text.length);
    bounds.slice(0, -1).forEach((a, w) => { const b = bounds[w + 1]; for (let i = a; i < b; i++) { cw[i] = BigInt(w); wstart[i] = BigInt(a); } });
    return { cw, wstart, nWords: bounds.length - 1 };
  }
  function planFrames(dur, cwBig, speed) {
    const L = dur.length, nWords = Number(cwBig[cwBig.length - 1]) + 1;
    const d = new Float32Array(L);
    for (let i = 0; i < L; i++) d[i] = dur[i] / speed;
    const counts = new Float32Array(nWords);
    for (let i = 0; i < L; i++) counts[Number(cwBig[i])] += d[i];
    const wordFrames = new Int32Array(nWords);
    let totalFrames = 0;
    for (let w = 0; w < nWords; w++) {
      wordFrames[w] = Math.max(1, Math.min(MAX_WORD_FRAMES, Math.round(counts[w])));
      totalFrames += wordFrames[w];
    }
    const T = Math.min(totalFrames, MAX_FRAMES);
    const fw = new BigInt64Array(T), fp = new Float32Array(T);
    let f = 0;
    for (let w = 0; w < nWords && f < T; w++)
      for (let k = 0; k < wordFrames[w] && f < T; k++, f++) { fw[f] = BigInt(w); fp[f] = k / wordFrames[w]; }
    const wordStart = new Float32Array(nWords), wordEnd = new Float32Array(nWords);
    let acc = 0;
    for (let w = 0; w < nWords; w++) { wordStart[w] = acc / FPS; acc += wordFrames[w]; wordEnd[w] = acc / FPS; }
    return { fw, fp, wordStart, wordEnd };
  }

  /* ================= WAV ================= */
  function wavBytes(audio, sr) {
    const n = audio.length, buf = new ArrayBuffer(44 + n * 2), dv = new DataView(buf);
    const ws = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
    ws(0, "RIFF"); dv.setUint32(4, 36 + n * 2, true); ws(8, "WAVE"); ws(12, "fmt ");
    dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
    dv.setUint32(24, sr, true); dv.setUint32(28, sr * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
    ws(36, "data"); dv.setUint32(40, n * 2, true);
    for (let i = 0; i < n; i++) dv.setInt16(44 + i * 2, Math.max(-1, Math.min(1, audio[i])) * 32767, true);
    return new Uint8Array(buf);
  }

  /* ================= DUYGU ÖNAYARLARI =================
   * Cümle başına hız/perde oynaması robotikliği kırar; duraklama ölçeği tonu verir. */
  const EMOTIONS = {
    notr:      { label: "Nötr",      emoji: "🙂", speed: 1.0,  pitchAdd: 0,    jitterS: 0,     jitterP: 0,    pauseMul: 1.0  },
    neseli:    { label: "Neşeli",    emoji: "😄", speed: 1.06, pitchAdd: 0.7,  jitterS: 0.045, jitterP: 0.5,  pauseMul: 0.85 },
    heyecanli: { label: "Heyecanlı", emoji: "🤩", speed: 1.12, pitchAdd: 1.0,  jitterS: 0.08,  jitterP: 0.9,  pauseMul: 0.6  },
    sakin:     { label: "Sakin",     emoji: "😌", speed: 0.94, pitchAdd: -0.3, jitterS: 0.02,  jitterP: 0.2,  pauseMul: 1.25 },
    uzgun:     { label: "Üzgün",     emoji: "😢", speed: 0.88, pitchAdd: -0.8, jitterS: 0.02,  jitterP: 0.25, pauseMul: 1.45 },
  };

  /* ================= SES PROFİLLERİ (formant destekli) ================= */
  const VOICES = {
    elif:   { id: "elif",   name: "Elif",   emoji: "🌸", tag: "Orijinal kadın", pitch: 0,    formant: 0,    tempo: 1.0,  seed: 7  },
    deniz:  { id: "deniz",  name: "Deniz",  emoji: "🌊", tag: "Derin erkek",    pitch: -4.5, formant: -2.0, tempo: 0.98, seed: 11 },
    kerem:  { id: "kerem",  name: "Kerem",  emoji: "⚡", tag: "Genç erkek",     pitch: -2.0, formant: -1.0, tempo: 1.04, seed: 5  },
    zeynep: { id: "zeynep", name: "Zeynep", emoji: "✨", tag: "Parlak kadın",   pitch: 2.0,  formant: 1.0,  tempo: 1.0,  seed: 3  },
    minik:  { id: "minik",  name: "Minik",  emoji: "🎈", tag: "Çocuk",          pitch: 5.0,  formant: 3.5,  tempo: 1.12, seed: 21 },
    radyo:  { id: "radyo",  name: "Radyo",  emoji: "🎙️", tag: "Anlatıcı",       pitch: -1.5, formant: -1.5, tempo: 0.9,  seed: 13 },
  };

  /* ================= ana motor ================= */
  class EmaEngine {
    constructor(ort, sessions, meta) {
      this.ort = ort; this.sessions = sessions; this.meta = meta;
      this.stoi = {}; meta.vocab.forEach((c, i) => this.stoi[c] = i);
      this.alphabet = makeAlphabet(meta.vocab);
    }
    normalize(text) { return this.alphabet(normalizeTR(text)); }

    async * synthesize(text, { voice = VOICES.elif, speed = 1, pitch = 0, emotion = EMOTIONS.notr, seed = null, onProgress } = {}) {
      const vseed = seed ?? voice.seed ?? 7;
      const effSpeedBase = Math.min(4, Math.max(0.25, speed * (voice.tempo || 1) * emotion.speed));
      const jit = mulberry32((vseed ^ 0x9e3779b9) >>> 0);
      const norm = this.normalize(text);
      if (!norm) throw new Error("Metin boş ya da okunabilir karakter içermiyor.");
      const pieces = chunkText(norm, effSpeedBase);
      const out24 = [], words = [];
      let offset = 0; const t0 = Date.now();
      for (let pi = 0; pi < pieces.length; pi++) {
        const [ptext, pause0] = pieces[pi];
        // duygu: parça başına hız + perde oynaması (tohumla tekrarlanabilir)
        const sj = (jit() * 2 - 1) * emotion.jitterS;
        const pj = (jit() * 2 - 1) * emotion.jitterP;
        const pieceSpeed = Math.min(4, Math.max(0.25, effSpeedBase * (1 + sj)));
        const piecePitch = (voice.pitch || 0) + pitch + emotion.pitchAdd + pj;
        const pieceSeed = (vseed * 1000003 + pi) % Number.MAX_SAFE_INTEGER;
        const ids = idsFor(ptext, this.stoi);
        const { cw, wstart } = wordMap(ptext);
        const L = ids.length;
        const feeds = {
          ids: new this.ort.Tensor("int64", ids, [1, L]),
          mask: new this.ort.Tensor("bool", new Uint8Array(L).fill(1), [1, L]),
        };
        const textOut = await this.sessions.text.run(feeds);
        const durArr = textOut.dur.data, hArr = textOut.h.data;
        const { fw, fp, wordStart, wordEnd } = planFrames(durArr, cw, pieceSpeed);
        const T = fw.length;
        const noise = randn(pieceSeed >>> 0, 4 * T * 64);
        const lat = (await this.sessions.sound.run({
          h: new this.ort.Tensor("float32", hArr, [1, L, this.meta.cfg.d]),
          dur: new this.ort.Tensor("float32", durArr, [1, L]),
          mask: feeds.mask,
          cw: new this.ort.Tensor("int64", cw, [1, L]),
          wstart: new this.ort.Tensor("int64", wstart, [1, L]),
          fw: new this.ort.Tensor("int64", fw, [1, T]),
          fp: new this.ort.Tensor("float32", fp, [1, T]),
          fmask: new this.ort.Tensor("bool", new Uint8Array(T).fill(1), [1, T]),
          noise: new this.ort.Tensor("float32", noise, [1, 4, T, 64]),
        })).latents;
        const audio48 = (await this.sessions.decoder.run({ z: lat })).audio.data.slice(0, T * HOP);
        // parça bazında 24k + formant/perde dönüşümü
        const a24 = resampleSinc(audio48, 0.5, 48);
        const shifted = voiceShift(a24, piecePitch, voice.formant || 0);
        out24.push(shifted);
        const pw = ptext.split(/ +/);
        for (let w = 0; w < pw.length && w < wordStart.length; w++)
          words.push({ text: pw[w], start: offset + wordStart[w], end: offset + wordEnd[w] });
        const pause = pause0 * emotion.pauseMul;
        const pauseSamples = Math.round(pause * SR24);
        if (pauseSamples) out24.push(new Float32Array(pauseSamples));
        offset += T / FPS + pause;
        if (onProgress) onProgress((pi + 1) / pieces.length);
        yield { type: "progress", done: (pi + 1) / pieces.length };
      }
      const total = out24.reduce((a, c) => a + c.length, 0);
      const full = new Float32Array(total);
      let o = 0; for (const c of out24) { full.set(c, o); o += c.length; }
      let peak = 0; for (const s of full) peak = Math.max(peak, Math.abs(s));
      const gain = peak > 1e-4 ? Math.min(0.85 / peak, 4) : 1;
      for (let i = 0; i < full.length; i++) full[i] *= gain;
      const finalWords = words.map(w => ({ text: w.text, start: Math.round(w.start * 100) / 100, end: Math.round(w.end * 100) / 100 }));
      yield {
        type: "done", wav: wavBytes(full, SR24), sampleRate: SR24,
        duration: Math.round(full.length / SR24 * 100) / 100,
        seed: vseed, words: finalWords, elapsedMs: Date.now() - t0, text: norm,
      };
    }
  }

  const api = { EmaEngine, VOICES, EMOTIONS, normalizeTR, fixPunctuation, chunkText, makeAlphabet, planFrames, wordMap, idsFor, randn, mulberry32, timeStretchFormant, resampleSinc, voiceShift, wavBytes, fft, SR48, SR24, FPS, HOP };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.EMA_ENGINE = api;
})(typeof self !== "undefined" ? self : globalThis);
