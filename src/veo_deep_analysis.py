#!/usr/bin/env python3
"""Deep analysis: understand the pixel mapping between decoded PDB and reference JPEG."""

import struct
import numpy as np
from PIL import Image
import os

class BitReader:
    def __init__(self, data):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 7
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

def decompress_linear(rec_data, count):
    reader = BitReader(rec_data)
    output = []
    prev = 128
    for _ in range(count):
        flag = reader.read_bit()
        if flag == 1:
            output.append(prev & 0xFF)
        else:
            code = reader.read_bits(5)
            if code == 0:
                val = reader.read_bits(8)
                prev = val
                output.append(val & 0xFF)
            else:
                sign = (code >> 4) & 1
                mag = code & 0x0F
                val = ((prev - mag) if sign else (prev + mag)) & 0xFF
                prev = val
                output.append(val)
    return output

def parse_pdb(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()
    num_records = struct.unpack('>H', data[76:78])[0]
    offsets = []
    for i in range(num_records):
        off = 78 + i * 8
        offsets.append(struct.unpack('>I', data[off:off+4])[0])
    offsets.append(len(data))
    return [data[offsets[i]:offsets[i+1]] for i in range(num_records)]

pdb_path = "data/3840520404.pdb"
out_dir = "output/"
ref_path = "data/Thom_091225_001.JPG"

records = parse_pdb(pdb_path)
data_records = records[1:-1]
W, H = 640, 480

ref_rgb = np.array(Image.open(ref_path).convert('RGB'))
ref_gray = np.array(Image.open(ref_path).convert('L'))

# Decode all records linearly
all_data = bytearray()
for rec in data_records:
    pixels = decompress_linear(rec, W * 4)
    all_data.extend(pixels)
raw = np.frombuffer(bytes(all_data), dtype=np.uint8).reshape(H, W)

print("=== Structural Analysis ===")
print(f"Decoded data shape: {raw.shape}")
print(f"Reference shape: {ref_gray.shape}")

# Row-by-row correlation to find which decoded row matches which reference row
print("\n=== Row-by-row correlation ===")
print("Finding best matching reference row for each decoded row...")

row_matches = []
for dec_row in range(0, H, 10):  # Sample every 10th row
    best_corr = -1
    best_ref_row = -1
    for ref_row in range(H):
        c = np.corrcoef(raw[dec_row].astype(float), ref_gray[ref_row].astype(float))[0,1]
        if c > best_corr:
            best_corr = c
            best_ref_row = ref_row
    row_matches.append((dec_row, best_ref_row, best_corr))
    print(f"  Decoded row {dec_row:3d} → Reference row {best_ref_row:3d} (corr={best_corr:.4f})")

# Column-by-column correlation to find spatial mapping
print("\n=== Column-by-column correlation ===")
for dec_col in range(0, W, 80):  # Sample every 80th column
    best_corr = -1
    best_ref_col = -1
    for ref_col in range(W):
        c = np.corrcoef(raw[:, dec_col].astype(float), ref_gray[:, ref_col].astype(float))[0,1]
        if c > best_corr:
            best_corr = c
            best_ref_col = ref_col
    print(f"  Decoded col {dec_col:3d} → Reference col {best_ref_col:3d} (corr={best_corr:.4f})")

# Check if rows within a 4-row group are scrambled
print("\n=== 4-row group analysis ===")
for g in range(3):  # First 3 groups
    print(f"\n  Group {g} (rows {g*4}-{g*4+3}):")
    for local_row in range(4):
        dec_row = g * 4 + local_row
        # Try matching against nearby reference rows
        for ref_row_offset in range(-2, 8):
            ref_row = g * 4 + ref_row_offset
            if 0 <= ref_row < H:
                c = np.corrcoef(raw[dec_row].astype(float), 
                               ref_gray[ref_row].astype(float))[0,1]
                if c > 0.3:
                    print(f"    Dec row {dec_row} ↔ Ref row {ref_row}: corr={c:.4f}")

# Check: maybe the first N bytes of each record are a header
print("\n=== Per-record header analysis ===")
for i in range(5):
    rec = data_records[i]
    print(f"  Record {i+1}: first 8 bytes = {[f'0x{b:02x}' for b in rec[:8]]}")
    # Check if first byte could be a row count or header size
    print(f"    First byte as decimal: {rec[0]}")
    # Check if removing first 1-2 bytes improves decompression
    for skip in [0, 1, 2, 4]:
        pixels = decompress_linear(rec[skip:], W * 4 - skip * 8)
        # Check how many bits are left
        reader = BitReader(rec[skip:])
        px = []
        prev = 128
        for _ in range(W * 4):
            flag = reader.read_bit()
            if flag == 1:
                px.append(prev)
            else:
                code = reader.read_bits(5)
                if code == 0:
                    val = reader.read_bits(8)
                    prev = val
                    px.append(val)
                else:
                    sign = (code >> 4) & 1
                    mag = code & 0x0F
                    val = ((prev - mag) if sign else (prev + mag)) & 0xFF
                    prev = val
                    px.append(val)
        bits_used = reader.byte_pos * 8 + (7 - reader.bit_pos)
        bits_total = (len(rec) - skip) * 8
        print(f"    Skip {skip}: bits {bits_used}/{bits_total} ({bits_total-bits_used} left)")

# Check: what if consecutive rows in output alternate between two different "channels"?
print("\n=== Even/odd row analysis ===")
even_rows = raw[0::2, :]  # rows 0,2,4,...
odd_rows = raw[1::2, :]   # rows 1,3,5,...
ref_even = ref_gray[0::2, :]
ref_odd = ref_gray[1::2, :]
print(f"Even rows mean: decoded={even_rows.mean():.1f} ref={ref_even.mean():.1f}")
print(f"Odd rows mean:  decoded={odd_rows.mean():.1f} ref={ref_odd.mean():.1f}")
c_ee = np.corrcoef(even_rows.flatten().astype(float), ref_even.flatten().astype(float))[0,1]
c_oo = np.corrcoef(odd_rows.flatten().astype(float), ref_odd.flatten().astype(float))[0,1]
c_eo = np.corrcoef(even_rows.flatten().astype(float), ref_odd.flatten().astype(float))[0,1]
c_oe = np.corrcoef(odd_rows.flatten().astype(float), ref_even.flatten().astype(float))[0,1]
print(f"Corr even→even: {c_ee:.4f}")
print(f"Corr odd→odd:   {c_oo:.4f}")
print(f"Corr even→odd:  {c_eo:.4f}")
print(f"Corr odd→even:  {c_oe:.4f}")

# Check: are even/odd COLUMNS different channels?
print("\n=== Even/odd column analysis ===")
even_cols = raw[:, 0::2]  # cols 0,2,4,...
odd_cols = raw[:, 1::2]   # cols 1,3,5,...
print(f"Even cols mean: {even_cols.mean():.1f}, std: {even_cols.std():.1f}")
print(f"Odd cols mean:  {odd_cols.mean():.1f}, std: {odd_cols.std():.1f}")

# If data is YCbCr 4:2:2, even cols = Cb/Cr (centered ~128) and odd cols = Y
# Or vice versa
print(f"Even cols histogram peak: {np.argmax(np.bincount(even_cols.flatten()))}")
print(f"Odd cols histogram peak: {np.argmax(np.bincount(odd_cols.flatten()))}")

# Save even and odd columns separately
Image.fromarray(even_cols, 'L').save(os.path.join(out_dir, "analysis_even_cols.png"))
Image.fromarray(odd_cols, 'L').save(os.path.join(out_dir, "analysis_odd_cols.png"))

# Also check: raw pixel value distribution
print("\n=== Value distribution ===")
hist = np.bincount(raw.flatten(), minlength=256)
top10 = np.argsort(hist)[-10:][::-1]
for v in top10:
    print(f"  Value {v:3d} (0x{v:02x}): {hist[v]:6d} ({hist[v]/raw.size*100:.2f}%)")

print("\nDone!")
