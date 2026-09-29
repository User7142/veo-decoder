# Veo Palm Camera PDB → JPEG Decoder

A reverse-engineered decoder for the `.pdb` picture files of the Veo digital camera for
Palm OS devices from the early 2000s.

The camera stored every picture as a Palm Database (`.pdb`) with a proprietary
compression ("Type 4"). The Windows software that converted them to JPEG is long gone.
This decoder makes the pictures readable again.

The full story with pictures: [palm2000.com](https://palm2000.com/articles/47)

## Quality

Compared with a JPEG created by the original Veo software:

- **PSNR:** 24.46 dB
- **Correlation:** 0.984
- **Speed:** about 0.4 seconds per picture (640×480)

The result is not bit-identical (the original software uses its own demosaicing and JPEG
settings), but visually there is no difference.

## Usage

Requirements: Python 3.8+, numpy and Pillow.

```bash
pip install numpy Pillow
```

**Single file:**
```bash
python3 src/veo_pdb_decoder.py picture.pdb [output.jpg]
```

**Whole folder:**
```bash
python3 src/veo_pdb_decoder.py --batch path/to/pdbs/ path/to/output/
```

**With the embedded thumbnail:**
```bash
python3 src/veo_pdb_decoder.py picture.pdb --thumb
```

Two sample files are included in [`data/`](data/README.md):

```bash
python3 src/veo_pdb_decoder.py data/3840520404.pdb iphone.jpg
```

## How it works

- **Container:** Palm Database with a 78-byte header. Record 0 holds 25 bytes of
  metadata, the last record a 36×28 pixel RGB565 thumbnail, and the records in between
  the compressed image, four rows per record (120 records → 480 rows).
- **Raw Bayer data:** The camera stores raw sensor data in a GBRG pattern, not a
  processed picture:
  ```
  Row 0 (even): Gb  B  Gb  B  Gb  B  ...
  Row 1 (odd):  R   Gr R   Gr R   Gr ...
  ```
- **Type 4 compression:** Each record is its own bitstream (MSB first). Each call decodes
  two rows in three byte-aligned passes:
  1. Row 0, odd columns (blue)
  2. Row 1, even columns (red)
  3. Row 0 even and row 1 odd columns (green), alternating, predicted from the other row
- **Codes per value:** `1` = repeat the previous value · `0 00000` + 8 bits = literal ·
  `0 SDDDD` = difference ±1…15.
- **Picture:** gray-world white balance, then bilinear demosaicing to 640×480.

Typical white balance gains: R 1.0×, Gr 1.6×, Gb 1.7×, B 4.8× (depends on the light).

The format was reverse-engineered from the Veo HotSync conduit for Windows
(`COMConduit.dll`). The whole way there, including the dead ends, is documented in
[docs/REVERSE_ENGINEERING.md](docs/REVERSE_ENGINEERING.md); a short technical overview is
in [docs/OVERVIEW.md](docs/OVERVIEW.md).

## Files

| Path | Content |
|---|---|
| `src/veo_pdb_decoder.py` | The decoder (command line tool) |
| `src/veo_decoder_v3.py` … `v9*.py` | Earlier versions, kept to document the process |
| `src/veo_*` (other) | Analysis and debugging tools used during reverse engineering |
| `data/` | Two sample `.pdb` files |
| `docs/` | Overview and reverse engineering notes |

## Your pictures?

The decoder was developed with pictures from one camera. If you have Veo `.pdb` files
that do not decode correctly, please open an issue.

## License

MIT, see [LICENSE](LICENSE).
