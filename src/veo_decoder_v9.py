#!/usr/bin/env python3
"""VEO PDB→JPEG Decoder v9 — Full-resolution Bayer demosaicing.

Decodes Veo Palm Camera .pdb files (Type 4 compression) to JPEG images.

Format summary:
  - PDB header: 78 bytes, records at 8-byte intervals
  - Record 0: 25-byte metadata (width, height, type)
  - Records 1-N: Compressed image data (4 rows per record, 2 decoder calls each)
  - Last record: RGB565 thumbnail (36x28 or similar)
  - Compression: 3-pass delta coding per 2-row pair, GBRG Bayer raw output
  - White balance: Gray world auto-WB (adjustable)

Pipeline: PDB → delta decompress → raw Bayer (GBRG) → white balance → bilinear demosaic → RGB → JPEG
"""

import struct, sys, os
import numpy as np
from PIL import Image


# ============================================================================
# Bitstream reader (verified against COMConduit.dll disassembly)
# ============================================================================
class BitReader:
    def __init__(self, data):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 7  # MSB first

    def read_bit(self):
        if self.byte_pos >= len(self.data):
            return 0
        bit = (self.data[self.byte_pos] >> self.bit_pos) & 1
        self.bit_pos -= 1
        if self.bit_pos < 0:
            self.bit_pos = 7
            self.byte_pos += 1
        return bit

    def read_bits(self, n):
        value = 0
        for _ in range(n):
            value = (value << 1) | self.read_bit()
        return value

    def flush_to_byte(self):
        if self.bit_pos != 7:
            self.bit_pos = 7
            self.byte_pos += 1

    def read_raw_byte(self):
        if self.byte_pos >= len(self.data):
            return 0
        val = self.data[self.byte_pos]
        self.byte_pos += 1
        return val


# ============================================================================
# Delta decoder (from fcn.100029c0 in COMConduit.dll)
# ============================================================================
def decode_delta(reader, prev):
    """Decode one value: 1-bit repeat, 5-bit delta, or 8-bit literal."""
    flag = reader.read_bit()
    if flag == 1:
        return prev & 0xFF  # REPEAT previous
    code = reader.read_bits(5)
    if code == 0:
        return reader.read_bits(8) & 0xFF  # 8-bit LITERAL
    sign = (code >> 4) & 1
    mag = code & 0x0F
    return ((prev - mag) if sign else (prev + mag)) & 0xFF  # DELTA +-1..15


# ============================================================================
# 3-Pass record decoder (from fcn.100029c0 disassembly)
# ============================================================================
def decode_3pass_record(rec, output, base_row, W):
    """Decode one PDB record → 4 output rows (2 calls × 2 rows each).

    Each call decodes 2 rows using 3 interleaved passes:
      Pass 1: row0 odd columns [1,3,...], horizontal prediction
      Pass 2: row1 even columns [0,2,...], horizontal prediction
      Pass 3: row0 even + row1 odd (zigzag cross-row prediction)
    """
    reader = BitReader(rec)
    H = output.shape[0]

    for call in range(2):
        r0 = base_row + call * 2
        r1 = r0 + 1
        if r1 >= H:
            break

        # Pass 1: row0 odd positions [1, 3, 5, ..., W-1]
        reader.flush_to_byte()
        lit = reader.read_raw_byte()
        output[r0, 1] = lit
        prev = lit
        for c in range(1, W // 2):
            val = decode_delta(reader, prev)
            output[r0, c * 2 + 1] = val
            prev = val

        # Pass 2: row1 even positions [0, 2, 4, ..., W-2]
        reader.flush_to_byte()
        lit = reader.read_raw_byte()
        output[r1, 0] = lit
        prev = lit
        for c in range(1, W // 2):
            val = decode_delta(reader, prev)
            output[r1, c * 2] = val
            prev = val

        # Pass 3: row0 even + row1 odd (zigzag with cross-row prediction)
        reader.flush_to_byte()
        lit0 = reader.read_raw_byte()
        output[r0, 0] = lit0
        reader.flush_to_byte()
        lit1 = reader.read_raw_byte()
        output[r1, 1] = lit1

        for c in range(1, W // 2):
            col_even = c * 2
            col_odd = c * 2 + 1
            # row0[even] predicted from row1[odd-1]
            pred = int(output[r1, col_even - 1])
            val = decode_delta(reader, pred)
            output[r0, col_even] = val
            # row1[odd] predicted from row0[even]
            if col_odd < W:
                pred = int(output[r0, col_even])
                val = decode_delta(reader, pred)
                output[r1, col_odd] = val


# ============================================================================
# PDB parser
# ============================================================================
def parse_pdb(filepath):
    """Parse Palm Database file, return list of records."""
    with open(filepath, 'rb') as f:
        data = f.read()
    name = data[:32].split(b'\x00')[0].decode('latin-1', errors='replace')
    num_records = struct.unpack('>H', data[76:78])[0]
    offsets = []
    for i in range(num_records):
        off = 78 + i * 8
        offsets.append(struct.unpack('>I', data[off:off+4])[0])
    offsets.append(len(data))
    records = [data[offsets[i]:offsets[i+1]] for i in range(num_records)]
    return name, records


def parse_metadata(rec0):
    """Parse 25-byte metadata record. Returns (width, height, type)."""
    if len(rec0) < 25:
        return 640, 480, 4
    meta = struct.unpack('>5I', rec0[:20])
    width = meta[1]
    height_code = meta[2]
    img_type = meta[3]
    # height_code: 0=496, 1=480 (effective 480 rows used)
    height = 496 if height_code == 0 else 480
    return width, height, img_type


# ============================================================================
# Bayer demosaicing — vectorized bilinear interpolation
# ============================================================================
def demosaic_gbrg_fast(raw_wb):
    """Fast vectorized bilinear demosaicing for GBRG Bayer pattern.

    GBRG layout (half-res channel mapping):
      Row 0: Gb[i,j]  B[i,j]   Gb[i,j+1]  B[i,j+1]  ...   (even row)
      Row 1: R[i,j]   Gr[i,j]  R[i,j+1]   Gr[i,j+1] ...   (odd row)

    Each color channel is extracted at half resolution, padded, then
    interpolated to missing positions using bilinear averaging of actual
    same-color neighbors (not raw mixed-color neighbors).
    """
    H, W = raw_wb.shape
    hH, hW = H // 2, W // 2
    raw_f = raw_wb.astype(np.float32)

    # Extract half-resolution channels
    Gb = raw_f[0::2, 0::2]   # (hH, hW)
    B  = raw_f[0::2, 1::2]   # (hH, hW)
    R  = raw_f[1::2, 0::2]   # (hH, hW)
    Gr = raw_f[1::2, 1::2]   # (hH, hW)

    # Pad each channel for safe border access
    Gb_p = np.pad(Gb, 1, mode='edge')  # (hH+2, hW+2)
    B_p  = np.pad(B,  1, mode='edge')
    R_p  = np.pad(R,  1, mode='edge')
    Gr_p = np.pad(Gr, 1, mode='edge')

    # Helper: slice (hH, hW) block at offset (di, dj) from half-res padded array
    def at(P_p, di, dj):
        return P_p[1+di:hH+1+di, 1+dj:hW+1+dj]

    rgb = np.zeros((H, W, 3), dtype=np.float32)

    # === Gb at full-res (2i, 2j): Green known ===
    rgb[0::2, 0::2, 1] = at(Gb_p, 0, 0)                           # G = Gb[i,j]
    rgb[0::2, 0::2, 0] = (at(R_p, -1, 0) + at(R_p, 0, 0)) / 2    # R: vertical avg
    rgb[0::2, 0::2, 2] = (at(B_p, 0, -1) + at(B_p, 0, 0)) / 2    # B: horizontal avg

    # === B at full-res (2i, 2j+1): Blue known ===
    rgb[0::2, 1::2, 2] = at(B_p, 0, 0)                            # B = B[i,j]
    rgb[0::2, 1::2, 1] = (at(Gr_p, -1, 0) + at(Gr_p, 0, 0) +     # G: 4 cardinal
                           at(Gb_p, 0, 0) + at(Gb_p, 0, 1)) / 4
    rgb[0::2, 1::2, 0] = (at(R_p, -1, 0) + at(R_p, -1, 1) +      # R: 4 diagonal
                           at(R_p, 0, 0) + at(R_p, 0, 1)) / 4

    # === R at full-res (2i+1, 2j): Red known ===
    rgb[1::2, 0::2, 0] = at(R_p, 0, 0)                            # R = R[i,j]
    rgb[1::2, 0::2, 1] = (at(Gb_p, 0, 0) + at(Gb_p, 1, 0) +      # G: 4 cardinal
                           at(Gr_p, 0, -1) + at(Gr_p, 0, 0)) / 4
    rgb[1::2, 0::2, 2] = (at(B_p, 0, -1) + at(B_p, 0, 0) +       # B: 4 diagonal
                           at(B_p, 1, -1) + at(B_p, 1, 0)) / 4

    # === Gr at full-res (2i+1, 2j+1): Green known ===
    rgb[1::2, 1::2, 1] = at(Gr_p, 0, 0)                           # G = Gr[i,j]
    rgb[1::2, 1::2, 0] = (at(R_p, 0, 0) + at(R_p, 0, 1)) / 2     # R: horizontal avg
    rgb[1::2, 1::2, 2] = (at(B_p, 0, 0) + at(B_p, 1, 0)) / 2     # B: vertical avg

    return np.clip(rgb, 0, 255).astype(np.uint8)


# ============================================================================
# White balance
# ============================================================================
def apply_white_balance(raw, gains):
    """Apply per-channel white balance gains to raw Bayer data.

    Args:
        raw: H×W uint8 raw Bayer array (GBRG pattern)
        gains: dict with keys 'R', 'Gr', 'Gb', 'B'
    Returns:
        H×W float32 white-balanced raw array
    """
    raw_f = raw.astype(np.float32)
    wb = raw_f.copy()
    wb[0::2, 0::2] *= gains['Gb']   # Even row, even col = Gb
    wb[0::2, 1::2] *= gains['B']    # Even row, odd col = B
    wb[1::2, 0::2] *= gains['R']    # Odd row, even col = R
    wb[1::2, 1::2] *= gains['Gr']   # Odd row, odd col = Gr
    return np.clip(wb, 0, 255)


def auto_white_balance(raw, target_mean=128.0):
    """Gray world auto white balance — scale each channel to same mean."""
    channels = {
        'Gb': raw[0::2, 0::2].astype(float).mean(),
        'B':  raw[0::2, 1::2].astype(float).mean(),
        'R':  raw[1::2, 0::2].astype(float).mean(),
        'Gr': raw[1::2, 1::2].astype(float).mean(),
    }
    gains = {}
    for ch, mean_val in channels.items():
        gains[ch] = target_mean / mean_val if mean_val > 0 else 1.0
    return gains, channels


# ============================================================================
# Thumbnail decoder (RGB565, last record)
# ============================================================================
def decode_thumbnail(rec):
    """Decode RGB565 thumbnail from last PDB record."""
    # Try common thumbnail sizes
    for th, tw in [(28, 36), (24, 32), (30, 40)]:
        expected = th * tw * 2
        if len(rec) >= expected:
            pixels = np.zeros((th, tw, 3), dtype=np.uint8)
            for y in range(th):
                for x in range(tw):
                    off = (y * tw + x) * 2
                    if off + 1 < len(rec):
                        val = struct.unpack('>H', rec[off:off+2])[0]
                        r = ((val >> 11) & 0x1F) << 3
                        g = ((val >> 5) & 0x3F) << 2
                        b = (val & 0x1F) << 3
                        pixels[y, x] = [r, g, b]
            return Image.fromarray(pixels, 'RGB')
    return None


# ============================================================================
# Main decoder pipeline
# ============================================================================
def decode_pdb(pdb_path, output_path=None, wb_target=128.0, jpeg_quality=85):
    """Decode a Veo PDB file to JPEG.

    Args:
        pdb_path: Path to .pdb file
        output_path: Output JPEG path (default: same name with .jpg extension)
        wb_target: Target mean for gray world white balance (default 128)
        jpeg_quality: JPEG quality (default 85)

    Returns:
        PIL Image object
    """
    if output_path is None:
        base = os.path.splitext(pdb_path)[0]
        output_path = base + '_decoded.jpg'

    # Parse PDB
    name, records = parse_pdb(pdb_path)
    print(f"PDB: {name}, {len(records)} records")

    # Parse metadata
    meta_rec = records[0]
    W, H_meta, img_type = parse_metadata(meta_rec)
    print(f"Metadata: {W}x{H_meta}, type={img_type}")

    # Determine actual height from data records
    data_records = records[1:-1]  # Skip metadata and thumbnail
    H = min(H_meta, len(data_records) * 4)  # 4 rows per record
    # Round down to multiple of 4
    H = (H // 4) * 4
    print(f"Decoding: {W}x{H} ({len(data_records)} data records)")

    # Step 1: Decompress all records
    raw = np.zeros((H, W), dtype=np.uint8)
    for i, rec in enumerate(data_records):
        base_row = i * 4
        if base_row + 3 >= H:
            break
        decode_3pass_record(rec, raw, base_row, W)

    # Step 2: White balance
    gains, ch_means = auto_white_balance(raw, target_mean=wb_target)
    print(f"Channel means — R:{ch_means['R']:.1f} Gr:{ch_means['Gr']:.1f} "
          f"Gb:{ch_means['Gb']:.1f} B:{ch_means['B']:.1f}")
    print(f"WB gains — R:{gains['R']:.2f} Gr:{gains['Gr']:.2f} "
          f"Gb:{gains['Gb']:.2f} B:{gains['B']:.2f}")

    raw_wb = apply_white_balance(raw, gains)

    # Step 3: Demosaic
    print("Demosaicing (bilinear)...")
    rgb = demosaic_gbrg_fast(raw_wb.astype(np.uint8))

    # Step 4: Save
    img = Image.fromarray(rgb, 'RGB')
    img.save(output_path, quality=jpeg_quality)
    print(f"Saved: {output_path}")

    # Also save thumbnail if available
    thumb = decode_thumbnail(records[-1])
    if thumb:
        thumb_path = os.path.splitext(output_path)[0] + '_thumb.jpg'
        thumb_save = thumb.resize((W, H), Image.NEAREST)
        # Don't save upscaled thumb, just the small one
        thumb.save(thumb_path, quality=jpeg_quality)
        print(f"Thumbnail: {thumb_path}")

    return img


# ============================================================================
# CLI + batch mode
# ============================================================================
if __name__ == '__main__':
    out_dir = "output/"

    pdb_files = [
        "data/3840520404.pdb",
        "data/3798526348.pdb",
    ]

    ref_path = "data/Thom_091225_001.JPG"

    for pdb_path in pdb_files:
        if not os.path.exists(pdb_path):
            print(f"SKIP: {pdb_path} not found")
            continue

        basename = os.path.splitext(os.path.basename(pdb_path))[0]
        out_path = os.path.join(out_dir, f"v9_{basename}.jpg")

        print(f"\n{'='*60}")
        print(f"Decoding: {os.path.basename(pdb_path)}")
        print(f"{'='*60}")

        img = decode_pdb(pdb_path, out_path)

    # Compare first image with reference
    print(f"\n{'='*60}")
    print("Quality comparison with reference")
    print(f"{'='*60}")

    decoded = np.array(Image.open(os.path.join(out_dir, "v9_3840520404.jpg"))).astype(float)
    ref_rgb = np.array(Image.open(ref_path).convert('RGB')).astype(float)

    # Crop to same size if needed
    h = min(decoded.shape[0], ref_rgb.shape[0])
    w = min(decoded.shape[1], ref_rgb.shape[1])
    decoded = decoded[:h, :w]
    ref_rgb = ref_rgb[:h, :w]

    mse = np.mean((decoded - ref_rgb) ** 2)
    psnr = 10 * np.log10(255**2 / mse)
    corr = np.corrcoef(decoded.flatten(), ref_rgb.flatten())[0, 1]

    # Per-channel
    for ch, name in enumerate(['Red', 'Green', 'Blue']):
        ch_mse = np.mean((decoded[:,:,ch] - ref_rgb[:,:,ch]) ** 2)
        ch_psnr = 10 * np.log10(255**2 / ch_mse) if ch_mse > 0 else float('inf')
        ch_corr = np.corrcoef(decoded[:,:,ch].flatten(), ref_rgb[:,:,ch].flatten())[0, 1]
        print(f"  {name}: PSNR={ch_psnr:.2f}dB, corr={ch_corr:.4f}")

    print(f"  Overall: PSNR={psnr:.2f}dB, corr={corr:.4f}")
    print("\nDone!")
