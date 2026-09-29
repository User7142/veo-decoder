# Veo Palm Camera PDB→JPEG Decoder — Complete Reverse-Engineering Documentation

**Status:** Done. Final decoder: `veo_pdb_decoder.py`

**Quality:** PSNR=24.46dB, correlation 0.984 vs. original JPEG

**Total duration:** September 2025 – March 2026 (~7 months of documented work, probably longer before that)

---

## Project overview

The Veo Photo Traveler for Palm Handhelds was a small digital camera from the early 2000s that plugs into the SD expansion slot of a Palm. It stored images in `.pdb` (Palm Database) files with proprietary Type 4 compression. After ~20 years, the original decoders and the Veo software are long gone.

**Goal:** Write a reproducible, working decoder that makes all old Veo images readable again.

**Challenge:** No documentation, no sources, only binary data and an x86 DLL as a starting point.

---

## Timeline and iterations

### Phase 1: Exploration and wrong hypotheses (Sept – Nov 2025)

#### Starting point
- 4 PDB files available (two of them, 3840520404.pdb and 3798526348.pdb, are included in this repository)
- 1 reference JPEG from the original Veo software (Thom_091225_001.JPG, 640×480; not included in this repository)
- No documentation
- Only intuition: "This looks like delta compression"

#### Hypothesis 1: Simple row-wise delta coding
**Attempt:** Decode linearly, treating each row as a sequence of delta values.

**Result:** Poor correlation (0.5-0.6), but interestingly better correlation for certain "row types" than for others.

**Findings:**
- The data does NOT simply follow linear order
- There is an internal structure that is being ignored
- The decoder evidently reads in a more complex way

**Code:** `veo_decoder_v1.py` (lost)

#### Hypothesis 2: Column interleaving (luma/chroma separation)
**Attempt:** "Maybe it is like JPEG — Y, Cb, Cr coded separately?"

Extracted the "luma" channel (assumed Y signal) and tried to decode it.

**Result:** Correlation 0.64 — better, but still far off.

**Findings:**
- A column pattern does exist
- But it is not YCbCr
- More likely Bayer-like interleaving

**Code:** `veo_decoder_v2.py` (lost)

#### Hypothesis 3: Concatenated bitstream across all records
**Attempt:** "Append all records to each other → one large bitstream → decode linearly"

**Result:** Correlation 0.6466 — even worse than before!

**Findings:**
- This was a classic mistake: records are **independent** bitstreams, not one continuous stream
- Each record has padding bits at the end
- When concatenating, "garbage" bits are read as data bits → misalignment

**Code:** `veo_decoder_v6.py` (preserved)

**Lesson:** Records are atomic. Each one needs its own BitReader.

---

### Phase 2: Structure hypotheses (Dec 2025 – Jan 2026)

#### DLL disassembly begins
In parallel with the trial-and-error coding: **extract and disassemble COMConduit.dll**.

```
VeoSetupEN.exe (Windows installer)
  → contains an LZSS-compressed DLL
  → COMConduit.dll, 147,456 bytes
  → x86 code, can be disassembled
```

Relevant functions identified:
- `fcn.100029c0`: 1076 bytes — **core decoder loop**
- `fcn.10002e00`: 71 bytes — row loop (H/2 iterations)
- `fcn.10002610`: 182 bytes — Type 4 handler

#### Hypothesis 4: 2-pass structure (even/odd columns)
**Attempt:** "Based on the DLL analysis: pass 1 = even columns, pass 2 = odd columns"

**Code:** `veo_decoder_v3.py`, `veo_decoder_v4.py`

**Result:** Correlation 0.8+ — **major progress!**

**Findings:**
- There really is a multi-pass structure
- But 2 passes are not enough
- The row order is subtle (0/1/2/3 in a complex order)

#### Hypothesis 5: 3-pass structure with zigzag
**Attempt:** "What if there are 3 passes? Pass 1 = odd columns of row 0, pass 2 = even columns of row 1, pass 3 = zigzag?"

**Breakthrough:** Instruction-by-instruction disassembly of `fcn.100029c0`

```asm
mov edx, [ebp + arg_width]  ; W in edx
lea eax, [edx - 1]         ; W-1
shr eax, 1                 ; (W-1)/2 = 319
; ... Pass 1 Schleife 319 Iterationen

; byte-align
; Pass 2 ...

; byte-align
; 2 Literal-Bytes
; Pass 3 Zigzag: Y0[even] ← pred(Y1[odd-1]), Y1[odd] ← pred(Y0[even])
```

(Comments in the listing: "Pass 1 Schleife 319 Iterationen" = pass 1 loop, 319 iterations; "2 Literal-Bytes" = 2 literal bytes.)

**Code:** `veo_decoder_v5.py`, `veo_decoder_v7.py`

**Result:** Correlation 0.87+ — even better, but...

**Findings:**
- The 3-pass structure is correct
- **BUT:** the 4 row types have very different mean values
  - Pass 1 (odd columns, row 0): mean=12
  - Pass 2 (even columns, row 1): mean=101
  - Pass 3 row 0 even: mean=64
  - Pass 3 row 1 odd: mean=64
- These are **not video luminance/chroma values**

---

### Phase 3: The critical breakthrough (Jan 2026)

#### Hypothesis 6: UYVY video format?
**Attempt:** "Maybe the different mean values are normal? Convert UYVY→RGB"

**Result:** PSNR 7.77 dB — **extremely poor**

The colors were completely wrong. This could not be right.

**Code:** `veo_decoder_v6.py`, `veo_decoder_v7.py`

**Findings:**
- It is definitely NOT UYVY
- But the different channel values are real
- They point to **separate color channels**

#### The insight: Bayer sensor data
**Hypothesis 7:** "What if the sensor has a **Bayer CFA** (color filter array)? GBRG pattern?"

In a Bayer sensor, each pixel has only one color:
```
Row 0: Gb  B  Gb  B  ...
Row 1: R   Gr R   Gr ...
```

**Per-pixel analysis:**
- Pass 1 (Row0 odd) = blue pixels → mean=12 (dark)
- Pass 2 (Row1 even) = red pixels → mean=101 (bright)
- Pass 3 Row0 even = Gb (green in the blue row) → mean=64
- Pass 3 Row1 odd = Gr (green in the red row) → mean=64

This matches the GBRG pattern **exactly**.

**Code:** `veo_decoder_v8.py`

**Result:**
- GBRG with auto white balance → correlation **0.9816** at half resolution (320×240)
- Colors suddenly look natural
- The iPhone image shows actual colors (wood, blue of the app icons, cable)

**Findings:**
- **This is raw sensor output, not processed video data**
- The 3-pass structure decodes the raw data pixel by pixel
- Auto white balance (gray world) works reliably
- Each sensor/lighting situation has different channel gains

---

### Phase 4: Full-resolution demosaicing (March 2026)

#### Dead end: slow Python loop decoder
**Attempt:** `veo_decoder_v9.py` with a for loop over every pixel

**Result:** Very slow (several minutes per image because of the nested loops)

**Code:** `demosaic_gbrg()` with y/x loops

**Fix:** Vectorized numpy version

**Code:** `demosaic_gbrg_fast()` — uses half-res channel extraction + padding + numpy slicing

**Result:** 0.4 seconds instead of minutes

#### The 320px mystery
**Anomaly:** The "320px" PDB files (metadata flag=0) produced artifacts

**Attempt 1:** "Maybe we need to decode with W=320?"

**Result:** Only 49% of the record data is consumed, heavily distorted

**Attempt 2:** "Maybe we decode with W=640, but the output image is 320?"

**Result:** 100% data usage, correct decoding

**Findings:**
- The metadata flag has nothing to do with the internal width
- **All PDB files use 640px internally**, regardless of the flag
- The flag probably means something else (output scaling? hardware variant?)
- All 4 test images decode correctly with W=640

---

## Dead ends and lessons

### Dead end 1: Concatenated bitstream
**What:** Treating all records in sequence as one bitstream

**Why wrong:** Each record has padding bits. Concatenating causes misalignment.

**Lesson:** **Data formats are usually modular. Records are atomic. The BitReader must be re-initialized for each record.**

### Dead end 2: UYVY video format
**What:** Assuming the data is compressed YCbCr video like DV tape or MJPEG

**Why wrong:** UYVY would have consistent channel means, not 12/101/64/64

**Lesson:** **If a hypothesis gives poor results (PSNR 7dB), the basic assumption is wrong. Go back to the raw data.**

### Dead end 3: 2 passes instead of 3
**What:** Assuming only even/odd columns are decoded

**Why wrong:** The disassembly showed 3 distinct byte-flush segments, each with different data patterns

**Lesson:** **Instruction-by-instruction disassembly is time-consuming, but indispensable when hypotheses do not converge.**

### Dead end 4: Loop-based demosaicing
**What:** for y in range(H): for x in range(W): ... for every pixel

**Why slow:** Python is slow. Two nested loops over 640×480 = 307k iterations

**Lesson:** **Numpy vectorization is > 100x faster. Half-res channel extraction is cleaner than per-pixel classification.**

### Dead end 5: W=320 decoding
**What:** Assuming "320px" means the records are only 320 pixels wide

**Why wrong:** 49% data usage strongly indicates that this does not fit. 100% data usage with W=640 is the sign of correct decoding.

**Lesson:** **Data usage efficiency is a gold standard for "correct" decoding. If only 50% of the data is consumed, the model is wrong.**

---

## Technical deep dives

### Understanding the delta decoding

Type 4 compression is a **prediction-based delta coding**:

```python
def decode_delta(reader, prev):
    flag = reader.read_bit()
    if flag == 1:
        return prev  # REPEAT: gleicher Wert wie vorher

    code = reader.read_bits(5)
    if code == 0:
        return reader.read_bits(8)  # LITERAL: nächste 8 Bits sind absoluter Wert

    sign = (code >> 4) & 1
    mag = code & 0x0F
    return (prev - mag) if sign else (prev + mag)  # DELTA: ±1 bis ±15 vom vorherigen
```

(Code comments: REPEAT = same value as before; LITERAL = the next 8 bits are the absolute value; DELTA = ±1 to ±15 from the previous value.)

**Why does this work?** Natural images have high spatial coherence. Neighboring pixels typically differ by only a few values. These 3 modes are highly efficient:
- REPEAT: very common in uniform areas (1 bit)
- DELTA: common in gradients (6 bits instead of 8)
- LITERAL: rare, only at abrupt transitions (13 bits)

### Understanding the 3-pass interleaving

The Veo sensor produces Bayer data, but the decoder stores it in **interleaved pass order** instead of "normal" row-by-row order:

```
Input buffer (memory):
Row 0: Gb B Gb B ...
Row 1: R  Gr R Gr ...
Row 2: Gb B Gb B ...
Row 3: R  Gr R Gr ...

Decoding order:
Pass 1: Row0[1,3,5,...], then byte-align
Pass 2: Row1[0,2,4,...], then byte-align
Pass 3: Row0[0,2,4,...] + Row1[1,3,5,...] zigzag, then done
       (= 2 calls, then Row2,Row3)
```

**Why this order?** Presumably for optimal availability of predictors. After passes 1 and 2 have run, the neighboring pixels are available for good cross-row prediction in pass 3.

### Bayer demosaicing without aliasing

Naive demosaicing would simply "take the nearest pixel of the same color". That produces **aliasing artifacts**.

Our approach: **bilinear interpolation**

```
Gb position:  R = (R_above + R_below)/2, B = (B_left + B_right)/2
B position:   R = 4 corner neighbors/4,  G = 4 corner neighbors/4
R position:   B = 4 corner neighbors/4,  G = 4 corner neighbors/4
Gr position:  R = (R_left + R_right)/2,  B = (B_above + B_below)/2
```

This is not "fancy", but it works: PSNR=24.46dB.

---

## Why 24.46dB and not higher?

JPEG is a **lossy** compression. When we compare against the JPEG decoding:

```
Original sensor → [our decode] → RGB → JPEG (quality 62)
                                         ↓
                                    JPEG decoder
                                         ↓
                                    RGB

vs.

Original sensor → [Veo software] → [processing] → JPEG (quality 62)
                                                   ↓
                                                JPEG decoder
                                                   ↓
                                                   RGB
```

The Veo software probably applied additional post-processing (gamma correction, sharpening, etc.). We do not. Therefore a deviation of ~1-2% is expected.

If we had compared against the **raw sensor data**, our decoder would probably reach >30dB.

---

## Why the discovery phase took so long

1. **No documentation:** Reverse-engineering everything is slow
2. **Subtle bugs:** e.g. record concatenation — looks "reasonable", but is wrong
3. **Several simultaneously plausible hypotheses:** UYVY vs. Bayer vs. raw vs. ???
4. **Translating the disassembly:** Translating x86 code into Python takes time
5. **Iterative verification:** Each hypothesis needs code + test + comparison with the reference

---

## Final findings

### What was critical

1. **DLL disassembly** — without it I would have been speculating forever
2. **Data usage analysis** — "Why is only 50% of the data read?" was the signal for W=640
3. **Bayer hypothesis** — the insight that different channel means indicate GBRG
4. **Per-channel white balance** — without it the colors would not have been right

### What was unnecessary

1. All UYVY theories — discarded
2. Column-interleaving hypotheses without the Bayer context — discarded
3. Loop-based demosaicing — works, but numpy is 100x faster

### What could have been faster

1. Disassembling the DLL earlier (time wasted on speculation)
2. Reading the 3-pass hypothesis directly from the DLL instead of guessing empirically
3. Testing the Bayer hypothesis earlier (the channel means were a very big hint)

---

## Summary for future projects

**When you need to crack an unknown binary codec:**

1. **Always do the disassembly** — it gives much more information than speculation
2. **Track data usage efficiency** — it is a strong indicator of correctness
3. **Test edge cases** (here: 320px metadata) — that is often where assumptions break
4. **Falsify hypotheses systematically** — UYVY with PSNR 7dB → completely wrong
5. **Look at the raw output** — the channel means (12, 101, 64, 64) were the key
6. **Compare against an existing reference** — without the reference JPEG we would never have known that we were right

---

## Files from the original working folder (March 2026)

### Production
- **`veo_pdb_decoder.py`** — final decoder, production-ready
- **`OVERVIEW.md`** — short summary
- **`final_3840520404.jpg`** — example output

### Development (chronological)
- `veo_decoder_v3.py` to `v9.py` — iterative improvements
- `veo_trace_record1.py` — byte-by-byte trace of a record
- `veo_analyze_passes.py` — detailed pass statistics
- `investigate_320.py` — analysis of the 320px mystery
- `check_pdb_meta.py` — metadata dumper
- `veo_decoder_v9_batch.py` — batch variant of the decoder

### Documentation
- `3_VEO_DECOMPRESSION_ALGORITHM.md` — early documentation of the decompression

---

## Listing of all outputs

```
März2026/
├── veo_pdb_decoder.py          ← FINAL DECODER
├── OVERVIEW.md                  ← short documentation
├── REVERSE_ENGINEERING.md       ← THIS FILE
├── final_3840520404.jpg         ← example 1
├── final_3840520404_thumb.jpg   ← thumbnail
├── 3798526348.jpg               ← example 2 (watering can)
│
├── [development files]
├── veo_decoder_v6.py to v9.py
├── investigate_320.py
├── veo_analyze_passes.py
├── veo_trace_record1.py
└── ...
```

In this repository, the decoder and development scripts are in `src/`, the documentation in `docs/`, and the sample PDB files in `data/`. The example JPEG outputs are not included.

---

## Epilogue

This decoder is an example of how persistence and systematic debugging work.

Do not give up, even if 10 hypotheses fail. The 11th one works.
