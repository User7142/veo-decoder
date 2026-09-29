# Veo Palm Camera PDB→JPEG Decoder

## Result

The decoder `veo_pdb_decoder.py` converts .pdb image files from the Veo Photo Traveler
Palm OS camera into JPEG images, completely and reproducibly.

**Quality:** PSNR=24.46dB, correlation 0.984 against the original JPEG produced by the Veo software.

**Speed:** ~0.4 seconds per image (640x480, pure Python).

## Usage

```bash
# Single file
python3 veo_pdb_decoder.py image.pdb [output.jpg]

# Batch mode (whole folder)
python3 veo_pdb_decoder.py --batch path/to/pdbs/ path/to/output/

# With thumbnail extraction
python3 veo_pdb_decoder.py image.pdb --thumb

# Set JPEG quality (default: 85)
python3 veo_pdb_decoder.py image.pdb -q 95
```

Requirements: Python 3.8+, numpy, Pillow.

## Reverse-engineering results

### PDB file format

```
Offset  Size   Content
0       32     PDB name (null-terminated)
32      44     PDB header fields
76      2      Number of records (big-endian uint16)
78      N×8    Record offsets (4 bytes BE uint32 + 4 bytes attributes each)
```

- **Record 0**: 25 bytes of metadata
  - Byte 0: compression type (always 4)
  - Byte 1: resolution flag (0 or 1, internally always 640px)
  - Bytes 20-24: capture date (month, day, year as BE uint16)
- **Records 1..N-1**: compressed image data (4 rows each, ~1700-2400 bytes)
- **Last record**: RGB565 thumbnail (36×28 pixels, big-endian)

### Compression (Type 4)

**Bitstream codec — delta coding with 3 modes:**

| Bits | Meaning |
|------|---------|
| `1` | Repeat previous value |
| `0 00000 BBBBBBBB` | 8-bit literal (absolute value) |
| `0 SDDDD` | Delta: S=sign (1=minus), DDDD=magnitude 1-15 |

MSB-first bit order. Byte alignment (flush) between the passes.

**3-pass structure per 2 rows (from COMConduit.dll fcn.100029c0):**

1. **Pass 1**: row 0, odd columns [1,3,5,...,639]
   - Byte flush → 1 literal byte → horizontal delta prediction
2. **Pass 2**: row 1, even columns [0,2,4,...,638]
   - Byte flush → 1 literal byte → horizontal delta prediction
3. **Pass 3**: row 0 even + row 1 odd (zigzag)
   - Byte flush → literal for Z0[0]
   - Byte flush → literal for Z1[1]
   - Zigzag: Z0[2k] ← Δ(Z1[2k-1]), Z1[2k+1] ← Δ(Z0[2k])

Each PDB record contains 2 such calls = 4 rows.
120 records × 4 rows = 480 image rows.

### Sensor data

The decompressed data is **raw Bayer sensor output** (NOT UYVY/YCbCr).

**GBRG Bayer pattern:**
```
Row 0: Gb  B  Gb  B  Gb  B  ...   (even row)
Row 1: R   Gr R   Gr R   Gr ...   (odd row)
```

**Channel characteristics:**
| Channel | Position | Typical mean | Required gain |
|---------|----------|--------------|---------------|
| R (red) | odd row, even column | 80-130 | 1.0-1.7× |
| Gr (green) | odd row, odd column | 75-100 | 1.3-1.7× |
| Gb (green) | even row, even column | 75-100 | 1.3-1.7× |
| B (blue) | even row, odd column | 24-56 | 2.3-5.4× |

The high blue gain is typical for cameras under artificial light (incandescent/halogen).

### Processing pipeline

```
PDB records
  → Delta decompression (3 passes per 2 rows)
  → Raw Bayer 640×480 (8 bits per pixel)
  → White balance (gray world auto WB)
  → Bilinear Bayer demosaicing (GBRG)
  → RGB 640×480
  → JPEG
```

## Source: COMConduit.dll

The decoding was verified by disassembling the Windows HotSync conduit DLL
(COMConduit.dll, 147,456 bytes) from the Veo installer (VeoSetupEN.exe).

Relevant functions:
- `fcn.100029c0` (1076 bytes): core 3-pass delta decoder
- `fcn.10002e00` (71 bytes): row loop (H/2 iterations)
- `fcn.10002610` (182 bytes): Type 4 handler (buffer allocation, record reading)
- `fcn.10002890` (85 bytes): bitstream reader setup

## Tested files

The decoder was tested on four PDB files. Two of them are included in `data/`:

| File | Size | Subject | WB R/Gr/Gb/B |
|------|------|---------|--------------|
| 3840520404.pdb | 221 KB | iPhone on a table (indoor) | 0.99/1.64/1.69/5.35 |
| 3798526348.pdb | 319 KB | Watering can on a lawn | 1.50/1.33/1.35/2.77 |

All 4 files were decoded correctly and reproducibly.

## Files

- **`src/veo_pdb_decoder.py`** — final decoder (CLI tool, usable for any PDB file)
- **`docs/OVERVIEW.md`** — this file
- **`final_3840520404.jpg`** — example output (not included in the repository)
- **`src/veo_decoder_v6.py` to `v9*.py`** — development versions
- **`src/veo_trace_record1.py`** — byte-by-byte trace tool
- **`src/veo_analyze_passes.py`** — pass structure analysis
- **`src/investigate_320.py`** — analysis of the 320px metadata
- **`src/check_pdb_meta.py`** — metadata comparison across all PDB files
