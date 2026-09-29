# Veo Decoder — Development History

All files in this directory document the complete reverse-engineering path.

## Final version
- **`veo_pdb_decoder.py`** — production-ready decoder, CLI tool

## Test/batch variants
- **`veo_decoder_v9_test.py`** — quick test on a single file
- **`veo_decoder_v9_batch.py`** — batch processing of multiple files

## Development iterations (v3–v9)

Each version shows one step in the reverse engineering:

### v3–v5: First hypotheses
- `veo_decoder_v3.py` — first multi-pass hypothesis (2-pass)
- `veo_decoder_v4.py` — various row-order tests
- `veo_decoder_v5.py` — first 3-pass structure implementation

### v6–v7: Wrong direction (concatenated bitstream & UYVY)
- `veo_decoder_v6.py` — concatenated record bitstream (gives poor results)
- `veo_decoder_v7.py` — UYVY format test (PSNR 7.77dB, completely wrong)

### v8–v9: Breakthrough (Bayer raw)
- `veo_decoder_v8.py` — first Bayer GBRG hypothesis (PSNR 23.88dB, 320×240)
- `veo_decoder_v9.py` — full-resolution demosaicing (PSNR 24.46dB, 640×480)

## Debug & analysis tools

- **`veo_trace_record1.py`** — byte-by-byte trace of a record
- **`veo_analyze_passes.py`** — detailed pass structure analysis
- **`veo_interleave_test.py`** — test of various interleaving strategies
- **`veo_interleave_v2.py`** — improved interleaving tests
- **`veo_pass_structure.py`** — pass structure experiments
- **`veo_byte_align_test.py`** — byte alignment edge cases
- **`veo_channel_test.py`** — channel analysis
- **`veo_channel_id.py`** — color channel identification
- **`veo_shift_test.py`** — experiments with pixel shifting
- **`veo_deep_analysis.py`** — in-depth statistical analysis
- **`veo_best_result.py`** — collects the best results so far

- **`investigate_320.py`** — mystery: why "320px" metadata?
- **`check_pdb_meta.py`** — metadata dumper for all PDB files

## How to follow the development

```bash
# Chronological walkthrough:
python3 veo_decoder_v3.py
python3 veo_decoder_v4.py
python3 veo_decoder_v5.py
python3 veo_decoder_v6.py  # ← poor results (concatenation bug)
python3 veo_decoder_v7.py  # ← poor results (UYVY wrong)
python3 veo_decoder_v8.py  # ← better, but only 320×240
python3 veo_decoder_v9.py  # ← final, 640×480, PSNR 24.46dB

# With analysis tools:
python3 veo_analyze_passes.py  # shows correlations per pass
python3 veo_trace_record1.py   # detailed trace
python3 investigate_320.py     # why is "320px" wrong?
```

## Key findings per version

| Version | Status | PSNR | Findings |
|---------|--------|------|----------|
| v3 | Good | ~0.7 | 3-pass structure exists |
| v4 | Better | 0.8+ | Row order is critical |
| v5 | OK | 0.85 | 3-pass interleaving works |
| v6 | Wrong | 0.65 | Records cannot be concatenated |
| v7 | Wrong | 7.77dB | It is NOT UYVY |
| v8 | Good | 23.88dB | **GBRG Bayer!** Auto WB works |
| v9 | Final | 24.46dB | **Full resolution + optimization** |

## Dead ends you can avoid

1. **Concatenated bitstream** — each record needs its own BitReader
2. **UYVY/YCbCr video format** — it is raw sensor Bayer data, not compressed video
3. **2 passes instead of 3** — the 3rd pass is essential for cross-row prediction
4. **Loop-based demosaicing** — numpy vectorization is 100x faster
5. **W=320 decoding** — all PDB files are 640px wide internally

## Learning outcome

Each version shows important debugging techniques:
- How to deal with poor PSNR values (falsify the hypothesis)
- How to reverse-engineer from disassembly code
- How to find structural bugs (record concatenation)
- How to find the right hypothesis through data usage analysis
- How to optimize performance (loop → vectorized)

See `docs/REVERSE_ENGINEERING.md` for the complete story.
