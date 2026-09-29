# Test data for the Veo decoder

Sample .pdb test files. The original analysis used four PDB files; two of them are included here.

## PDB test files

| File | Size | Subject | Width (meta) | Notes |
|------|------|---------|--------------|-------|
| `3840520404.pdb` | 221 KB | iPhone on a wooden table | 640px (meta=1) | **Main test file**, good quality |
| `3798526348.pdb` | 312 KB | Watering can | 640px (meta=0) | Largest file, outdoor |

**Important:** The metadata flag (0 or 1) does NOT affect the internal width. All files are 640px internally.

## Reference file

`Thom_091225_001.JPG` — original JPEG from the Veo software. It is not included in this repository.
- Size: 135 KB
- Dimensions: 640×480
- Quality: Veo default (libjpeg quality ~62)
- Use: verifying the decoding quality (PSNR, correlation)

## Test commands

```bash
# Decode a single file:
python3 ../src/veo_pdb_decoder.py 3840520404.pdb output.jpg

# All files at once:
python3 ../src/veo_pdb_decoder.py --batch . ./outputs/

# With thumbnail extraction:
python3 ../src/veo_pdb_decoder.py 3840520404.pdb --thumb
```

## Analysis scripts and older versions

The analysis scripts and the older decoder versions (`src/veo_decoder_v3.py` … `v9`) are
kept for reference; they document the way to the final decoder. They are run from the
repository root, read `data/3840520404.pdb` and write their results to `output/`:

```bash
mkdir -p output
python3 src/veo_trace_record1.py
python3 src/veo_decoder_v8.py      # v8: half resolution (320x240), PSNR 23.88 dB
python3 src/veo_decoder_v7.py      # v7: UYVY attempt (PSNR 7.77 dB, very poor)
python3 src/veo_decoder_v6.py      # v6: concatenated bitstream (correlation 0.6466)
```

Most of them compare their output against the reference JPEG `data/Thom_091225_001.JPG`
(see above), which is not included, so they stop at that point without it.

---

**Tip:** Start with `3840520404.pdb` — it is the best test file for beginners.
