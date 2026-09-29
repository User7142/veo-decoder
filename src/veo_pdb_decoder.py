#!/usr/bin/env python3
"""
VEO Palm Camera PDB → JPEG Decoder
===================================

Decodes .pdb image files from the Veo Stingray/Connect Palm OS camera
(Type 4 delta compression) into standard JPEG images.

Usage:
    python3 veo_pdb_decoder.py <input.pdb> [output.jpg]
    python3 veo_pdb_decoder.py --batch <dir_with_pdbs> [output_dir]

Requirements:
    Python 3.8+, numpy, Pillow

Format Details:
    - Palm Database (.pdb) container with 78-byte header
    - Record 0: 25-byte metadata
    - Records 1..N-1: Delta-compressed raw Bayer sensor data
    - Record N (last): RGB565 thumbnail
    - Compression: 3-pass interleaved delta coding (Type 4)
    - Sensor: GBRG Bayer color filter array, 640x480 pixels
    - Each record contains 4 rows of image data (2 decoder calls × 2 rows)

Reverse-engineered from COMConduit.dll (Veo HotSync conduit for Windows).
"""

import struct, sys, os, glob, time
import numpy as np
from PIL import Image


# ==========================================================================
# Bitstream reader — MSB-first, verified against x86 disassembly
# ==========================================================================
class BitReader:
    __slots__ = ('data', 'byte_pos', 'bit_pos', 'length')

    def __init__(self, data):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 7
        self.length = len(data)

    def read_bit(self):
        if self.byte_pos >= self.length:
            return 0
        bit = (self.data[self.byte_pos] >> self.bit_pos) & 1
        self.bit_pos -= 1
        if self.bit_pos < 0:
            self.bit_pos = 7
            self.byte_pos += 1
        return bit

    def read_bits(self, n):
        val = 0
        for _ in range(n):
            val = (val << 1) | self.read_bit()
        return val

    def flush_to_byte(self):
        if self.bit_pos != 7:
            self.bit_pos = 7
            self.byte_pos += 1

    def read_raw_byte(self):
        if self.byte_pos >= self.length:
            return 0
        val = self.data[self.byte_pos]
        self.byte_pos += 1
        return val


# ==========================================================================
# Delta decoder — from fcn.100029c0 in COMConduit.dll
#
# Encoding:
#   1          → repeat previous value
#   0 00000    → next 8 bits are literal value
#   0 SDDDD   → delta: S=sign (1=subtract), DDDD=magnitude (1..15)
# ==========================================================================
def decode_delta(reader, prev):
    if reader.read_bit() == 1:
        return prev & 0xFF
    code = reader.read_bits(5)
    if code == 0:
        return reader.read_bits(8) & 0xFF
    sign = (code >> 4) & 1
    mag = code & 0x0F
    return ((prev - mag) if sign else (prev + mag)) & 0xFF


# ==========================================================================
# 3-Pass record decoder — from fcn.100029c0 disassembly
#
# Each record → 2 calls → 4 output rows
# Each call decodes 2 rows (r0, r1) in 3 byte-aligned passes:
#   Pass 1: r0 odd columns  [1,3,5,...] — horizontal delta prediction
#   Pass 2: r1 even columns [0,2,4,...] — horizontal delta prediction
#   Pass 3: r0 even + r1 odd — zigzag cross-row prediction
# ==========================================================================
def decode_record(rec, output, base_row, W):
    reader = BitReader(rec)
    H = output.shape[0]

    for call in range(2):
        r0 = base_row + call * 2
        r1 = r0 + 1
        if r1 >= H:
            break

        # Pass 1: row0 odd columns
        reader.flush_to_byte()
        prev = reader.read_raw_byte()
        output[r0, 1] = prev
        for c in range(1, W // 2):
            prev = decode_delta(reader, prev)
            output[r0, c * 2 + 1] = prev

        # Pass 2: row1 even columns
        reader.flush_to_byte()
        prev = reader.read_raw_byte()
        output[r1, 0] = prev
        for c in range(1, W // 2):
            prev = decode_delta(reader, prev)
            output[r1, c * 2] = prev

        # Pass 3: row0 even + row1 odd (zigzag)
        reader.flush_to_byte()
        output[r0, 0] = reader.read_raw_byte()
        reader.flush_to_byte()
        output[r1, 1] = reader.read_raw_byte()

        for c in range(1, W // 2):
            ce = c * 2       # even column
            co = c * 2 + 1   # odd column
            output[r0, ce] = decode_delta(reader, int(output[r1, ce - 1]))
            if co < W:
                output[r1, co] = decode_delta(reader, int(output[r0, ce]))


# ==========================================================================
# PDB parser
# ==========================================================================
def parse_pdb(filepath):
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


# ==========================================================================
# Bayer demosaicing — fast vectorized bilinear for GBRG pattern
#
# GBRG layout per 2×2 block:
#   (even row, even col) = Gb   (even row, odd col)  = B
#   (odd row,  even col) = R    (odd row,  odd col)  = Gr
# ==========================================================================
def demosaic_gbrg(raw_wb):
    H, W = raw_wb.shape
    hH, hW = H // 2, W // 2
    raw_f = raw_wb.astype(np.float32)

    # Half-resolution channels
    Gb = raw_f[0::2, 0::2]
    B  = raw_f[0::2, 1::2]
    R  = raw_f[1::2, 0::2]
    Gr = raw_f[1::2, 1::2]

    # Edge-padded for safe neighbor access
    Gb_p = np.pad(Gb, 1, mode='edge')
    B_p  = np.pad(B,  1, mode='edge')
    R_p  = np.pad(R,  1, mode='edge')
    Gr_p = np.pad(Gr, 1, mode='edge')

    def at(P, di, dj):
        return P[1+di:hH+1+di, 1+dj:hW+1+dj]

    rgb = np.zeros((H, W, 3), dtype=np.float32)

    # Gb positions (even row, even col): G known
    rgb[0::2, 0::2, 1] = at(Gb_p, 0, 0)
    rgb[0::2, 0::2, 0] = (at(R_p, -1, 0) + at(R_p, 0, 0)) / 2
    rgb[0::2, 0::2, 2] = (at(B_p, 0, -1) + at(B_p, 0, 0)) / 2

    # B positions (even row, odd col): B known
    rgb[0::2, 1::2, 2] = at(B_p, 0, 0)
    rgb[0::2, 1::2, 1] = (at(Gr_p, -1, 0) + at(Gr_p, 0, 0) +
                           at(Gb_p, 0, 0) + at(Gb_p, 0, 1)) / 4
    rgb[0::2, 1::2, 0] = (at(R_p, -1, 0) + at(R_p, -1, 1) +
                           at(R_p, 0, 0) + at(R_p, 0, 1)) / 4

    # R positions (odd row, even col): R known
    rgb[1::2, 0::2, 0] = at(R_p, 0, 0)
    rgb[1::2, 0::2, 1] = (at(Gb_p, 0, 0) + at(Gb_p, 1, 0) +
                           at(Gr_p, 0, -1) + at(Gr_p, 0, 0)) / 4
    rgb[1::2, 0::2, 2] = (at(B_p, 0, -1) + at(B_p, 0, 0) +
                           at(B_p, 1, -1) + at(B_p, 1, 0)) / 4

    # Gr positions (odd row, odd col): G known
    rgb[1::2, 1::2, 1] = at(Gr_p, 0, 0)
    rgb[1::2, 1::2, 0] = (at(R_p, 0, 0) + at(R_p, 0, 1)) / 2
    rgb[1::2, 1::2, 2] = (at(B_p, 0, 0) + at(B_p, 1, 0)) / 2

    return np.clip(rgb, 0, 255).astype(np.uint8)


# ==========================================================================
# White balance — gray world assumption
# ==========================================================================
def auto_white_balance(raw, target=128.0):
    gains = {}
    for name, sl in [('Gb', (slice(0,None,2), slice(0,None,2))),
                      ('B',  (slice(0,None,2), slice(1,None,2))),
                      ('R',  (slice(1,None,2), slice(0,None,2))),
                      ('Gr', (slice(1,None,2), slice(1,None,2)))]:
        mean = raw[sl].astype(float).mean()
        gains[name] = target / mean if mean > 0 else 1.0
    return gains


def apply_white_balance(raw, gains):
    wb = raw.astype(np.float32)
    wb[0::2, 0::2] *= gains['Gb']
    wb[0::2, 1::2] *= gains['B']
    wb[1::2, 0::2] *= gains['R']
    wb[1::2, 1::2] *= gains['Gr']
    return np.clip(wb, 0, 255)


# ==========================================================================
# Thumbnail decoder (RGB565 big-endian, last PDB record)
# ==========================================================================
def decode_thumbnail(rec_data):
    for th, tw in [(28, 36), (24, 32), (30, 40)]:
        if len(rec_data) >= th * tw * 2:
            img = np.zeros((th, tw, 3), dtype=np.uint8)
            for y in range(th):
                for x in range(tw):
                    off = (y * tw + x) * 2
                    val = struct.unpack('>H', rec_data[off:off+2])[0]
                    img[y, x] = [((val >> 11) & 0x1F) << 3,
                                 ((val >> 5) & 0x3F) << 2,
                                 (val & 0x1F) << 3]
            return Image.fromarray(img, 'RGB')
    return None


# ==========================================================================
# Main decode pipeline
# ==========================================================================
def decode_pdb(pdb_path, output_path=None, quality=85, verbose=True):
    """Decode a Veo PDB file to JPEG.

    Args:
        pdb_path:    Path to .pdb file
        output_path: Output JPEG path (default: <name>_decoded.jpg)
        quality:     JPEG quality 1-100 (default 85)
        verbose:     Print progress info

    Returns:
        PIL.Image.Image — the decoded RGB image
    """
    if output_path is None:
        base = os.path.splitext(pdb_path)[0]
        output_path = base + '_decoded.jpg'

    name, records = parse_pdb(pdb_path)
    data_records = records[1:-1]

    # Image is always 640×480 (120 records × 4 rows)
    W, H = 640, len(data_records) * 4

    if verbose:
        print(f"  {name}: {W}x{H}, {len(data_records)} records")

    # Step 1: Decompress
    raw = np.zeros((H, W), dtype=np.uint8)
    for i, rec in enumerate(data_records):
        base_row = i * 4
        if base_row + 3 >= H:
            break
        decode_record(rec, raw, base_row, W)

    # Step 2: White balance
    gains = auto_white_balance(raw)
    if verbose:
        for ch in ['R', 'Gr', 'Gb', 'B']:
            print(f"    WB {ch}: gain={gains[ch]:.2f}")
    raw_wb = apply_white_balance(raw, gains)

    # Step 3: Demosaic
    rgb = demosaic_gbrg(raw_wb.astype(np.uint8))

    # Step 4: Save
    img = Image.fromarray(rgb, 'RGB')
    img.save(output_path, quality=quality)
    if verbose:
        print(f"    → {output_path}")

    return img


# ==========================================================================
# CLI
# ==========================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Decode Veo Palm Camera .pdb files to JPEG')
    parser.add_argument('input', help='.pdb file or directory')
    parser.add_argument('output', nargs='?', default=None,
                        help='Output .jpg path or directory')
    parser.add_argument('-q', '--quality', type=int, default=85,
                        help='JPEG quality (default: 85)')
    parser.add_argument('--batch', action='store_true',
                        help='Process all .pdb files in input directory')
    parser.add_argument('--thumb', action='store_true',
                        help='Also extract RGB565 thumbnail')
    args = parser.parse_args()

    if args.batch or os.path.isdir(args.input):
        # Batch mode
        pdb_files = sorted(glob.glob(os.path.join(args.input, '*.pdb')))
        if not pdb_files:
            print(f"No .pdb files found in {args.input}")
            sys.exit(1)
        out_dir = args.output or args.input
        os.makedirs(out_dir, exist_ok=True)
        print(f"Batch decoding {len(pdb_files)} files...")
        for pdb_path in pdb_files:
            basename = os.path.splitext(os.path.basename(pdb_path))[0]
            out_path = os.path.join(out_dir, basename + '.jpg')
            t0 = time.time()
            decode_pdb(pdb_path, out_path, quality=args.quality)
            print(f"    ({time.time()-t0:.1f}s)")
            if args.thumb:
                _, records = parse_pdb(pdb_path)
                thumb = decode_thumbnail(records[-1])
                if thumb:
                    thumb_path = os.path.join(out_dir, basename + '_thumb.jpg')
                    thumb.save(thumb_path, quality=args.quality)
                    print(f"    Thumbnail → {thumb_path}")
    else:
        # Single file
        if not os.path.exists(args.input):
            print(f"File not found: {args.input}")
            sys.exit(1)
        t0 = time.time()
        out_path = args.output or os.path.splitext(args.input)[0] + '_decoded.jpg'
        decode_pdb(args.input, out_path, quality=args.quality)
        print(f"    ({time.time()-t0:.1f}s)")
        if args.thumb:
            _, records = parse_pdb(args.input)
            thumb = decode_thumbnail(records[-1])
            if thumb:
                thumb_path = os.path.splitext(out_path)[0] + '_thumb.jpg'
                thumb.save(thumb_path, quality=args.quality)
                print(f"    Thumbnail → {thumb_path}")


if __name__ == '__main__':
    main()
