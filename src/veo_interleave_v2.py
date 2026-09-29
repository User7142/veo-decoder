#!/usr/bin/env python3
"""Test interleaving based on finding: odd rows = image data, even rows = different data."""

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

ref_gray = np.array(Image.open(ref_path).convert('L'))
ref_rgb = np.array(Image.open(ref_path).convert('RGB'))

# Decode all records linearly
all_data = bytearray()
for rec in data_records:
    all_data.extend(decompress_linear(rec, W * 4))
raw = np.frombuffer(bytes(all_data), dtype=np.uint8).reshape(H, W)

# ====================================================================
# Hypothesis 1: Even rows = chroma plane, odd rows = luma plane
# Each record gives: [chroma_row0, luma_row0, chroma_row1, luma_row1]
# ====================================================================
print("=== Hypothesis 1: Even=chroma, Odd=luma ===")
luma = raw[1::2, :]    # 240 rows of luma
chroma = raw[0::2, :]  # 240 rows of chroma

# Upscale luma to 480 rows
luma_full = np.repeat(luma, 2, axis=0)  # Simple doubling
corr_luma = np.corrcoef(luma_full.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  Luma (doubled): corr={corr_luma:.4f}")
Image.fromarray(luma_full, 'L').save(os.path.join(out_dir, "h1_luma_doubled.png"))

# Also try interpolated upscale
luma_img = Image.fromarray(luma, 'L')
luma_interp = np.array(luma_img.resize((W, H), Image.BILINEAR))
corr_interp = np.corrcoef(luma_interp.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  Luma (interpolated): corr={corr_interp:.4f}")
Image.fromarray(luma_interp, 'L').save(os.path.join(out_dir, "h1_luma_interp.png"))

# ====================================================================
# Hypothesis 2: Data is column-interleaved within each record
# Linearly decoded values map to:
# First 320 values = odd col positions of row 0
# Next 320 values = odd col positions of row 1
# Next 320 values = even col positions of row 0
# Next 320 values = even col positions of row 1
# (repeat for rows 2-3)
# ====================================================================
print("\n=== Hypothesis 2: Column-interleaved decode ===")
h2 = np.zeros((H, W), dtype=np.uint8)
for rec_idx in range(120):
    base = rec_idx * 2560
    for row_pair in range(2):
        # Odd column positions
        for r in range(2):
            row = rec_idx * 4 + row_pair * 2 + r
            if row >= H:
                break
            for c in range(320):
                src_idx = base + row_pair * 1280 + r * 320 + c
                h2[row, c * 2 + 1] = all_data[src_idx]
        # Even column positions
        for r in range(2):
            row = rec_idx * 4 + row_pair * 2 + r
            if row >= H:
                break
            for c in range(320):
                src_idx = base + row_pair * 1280 + 640 + r * 320 + c
                h2[row, c * 2] = all_data[src_idx]

corr_h2 = np.corrcoef(h2.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  Column-interleaved: corr={corr_h2:.4f}")
Image.fromarray(h2, 'L').save(os.path.join(out_dir, "h2_col_interleaved.png"))

# ====================================================================
# Hypothesis 3: The data is ACTUALLY decoded in interleaved passes,
# but the predictor should reset between passes.
# Pass 1: decode 1280 values (640+640 for 2 rows) → these go to even columns
# Pass 2: decode 1280 values → these go to odd columns
# ====================================================================
print("\n=== Hypothesis 3: Two-pass decode with predictor reset ===")
h3 = np.zeros((H, W), dtype=np.uint8)
for rec_idx, rec in enumerate(data_records):
    reader = BitReader(rec)
    base_row = rec_idx * 4

    # Try: decode 2 rows worth of data (2*640=1280 pixels) in first pass
    # Then 2 more rows in second pass
    # First pass = rows 0,1; Second pass = rows 2,3
    for pass_num in range(2):
        prev = 128
        for local_row in range(2):
            row = base_row + pass_num * 2 + local_row
            if row >= H:
                break
            for col in range(W):
                flag = reader.read_bit()
                if flag == 1:
                    h3[row, col] = prev & 0xFF
                else:
                    code = reader.read_bits(5)
                    if code == 0:
                        val = reader.read_bits(8)
                        prev = val
                        h3[row, col] = val & 0xFF
                    else:
                        sign = (code >> 4) & 1
                        mag = code & 0x0F
                        val = ((prev - mag) if sign else (prev + mag)) & 0xFF
                        prev = val
                        h3[row, col] = val

corr_h3 = np.corrcoef(h3.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  Two-pass (predictor reset per 2 rows): corr={corr_h3:.4f}")
Image.fromarray(h3, 'L').save(os.path.join(out_dir, "h3_two_pass.png"))

# ====================================================================
# Hypothesis 4: The even rows contain DELTA data relative to odd rows
# Reconstruct: actual_row[n] = odd_row[n/2] + even_row[n/2] or similar
# ====================================================================
print("\n=== Hypothesis 4: Even rows are deltas ===")
# Try: reconstructed[row] = luma[row//2] + chroma[row//2] for each row
h4a = np.zeros((H, W), dtype=np.uint8)
for row in range(H):
    luma_row = raw[row | 1, :]  # Nearest odd row
    delta_row = raw[row & ~1, :].astype(np.int16) - 128  # Even row as signed delta
    h4a[row, :] = np.clip(luma_row.astype(np.int16) + delta_row, 0, 255).astype(np.uint8)
corr_h4a = np.corrcoef(h4a.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  Luma + delta (centered): corr={corr_h4a:.4f}")

# ====================================================================
# Hypothesis 5: Interleave using ACTUAL COMConduit stride pattern
# "gerade und ungerade Pixel getrennt"
# What if per record: first half = EVERY OTHER PIXEL of 4 rows,
# second half = the remaining pixels of 4 rows?
# ====================================================================
print("\n=== Hypothesis 5: Pixel-interleaved (every other pixel) ===")
h5 = np.zeros((H, W), dtype=np.uint8)
for rec_idx in range(120):
    base = rec_idx * 2560
    base_row = rec_idx * 4
    # First 1280 values = pixels at even positions (0,2,4,...) of all 4 rows
    # Next 1280 values = pixels at odd positions (1,3,5,...) of all 4 rows
    for i in range(1280):
        row = base_row + (i // 320)
        col = (i % 320) * 2  # Even position
        if row < H:
            h5[row, col] = all_data[base + i]
    for i in range(1280):
        row = base_row + (i // 320)
        col = (i % 320) * 2 + 1  # Odd position
        if row < H:
            h5[row, col] = all_data[base + 1280 + i]

corr_h5 = np.corrcoef(h5.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  Pixel-interleaved (even/odd cols): corr={corr_h5:.4f}")
Image.fromarray(h5, 'L').save(os.path.join(out_dir, "h5_pixel_interleaved.png"))

# ====================================================================
# Hypothesis 6: Maybe decoder needs previous row as predictor reference
# instead of just the previous pixel
# ====================================================================
print("\n=== Hypothesis 6: Decode with row-above predictor ===")
h6 = np.zeros((H, W), dtype=np.uint8)
prev_row = np.full(W, 128, dtype=np.uint8)

for rec_idx, rec in enumerate(data_records):
    reader = BitReader(rec)
    base_row = rec_idx * 4
    
    for local_row in range(4):
        row = base_row + local_row
        if row >= H:
            break
        for col in range(W):
            # Predictor = pixel directly above (from previous row)
            if row == 0 and col == 0:
                prev = 128
            elif col == 0:
                prev = h6[row - 1, col]
            else:
                prev = h6[row, col - 1]
            
            flag = reader.read_bit()
            if flag == 1:
                h6[row, col] = prev & 0xFF
            else:
                code = reader.read_bits(5)
                if code == 0:
                    val = reader.read_bits(8)
                    h6[row, col] = val & 0xFF
                else:
                    sign = (code >> 4) & 1
                    mag = code & 0x0F
                    if sign:
                        val = (prev - mag) & 0xFF
                    else:
                        val = (prev + mag) & 0xFF
                    h6[row, col] = val

corr_h6 = np.corrcoef(h6.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  Row-above predictor: corr={corr_h6:.4f}")
Image.fromarray(h6, 'L').save(os.path.join(out_dir, "h6_row_predictor.png"))

# ====================================================================
# Summary
# ====================================================================
print("\n=== Summary ===")
results = [
    ("Baseline (linear raw)", 0.3156),
    ("H1: Luma doubled", corr_luma),
    ("H1: Luma interpolated", corr_interp),
    ("H2: Column-interleaved", corr_h2),
    ("H3: Two-pass predictor reset", corr_h3),
    ("H4: Luma + delta", corr_h4a),
    ("H5: Pixel-interleaved", corr_h5),
    ("H6: Row-above predictor", corr_h6),
]
results.sort(key=lambda x: -x[1])
for name, corr in results:
    print(f"  {name:40s}: {corr:.4f}")

print("\nDone!")
