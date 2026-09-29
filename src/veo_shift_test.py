#!/usr/bin/env python3
"""Test circular column shifts and row mappings to maximize correlation."""

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

def decode_linear(rec_data, count, prev=128):
    reader = BitReader(rec_data)
    out = []
    for _ in range(count):
        flag = reader.read_bit()
        if flag == 1:
            out.append(prev & 0xFF)
        else:
            code = reader.read_bits(5)
            if code == 0:
                val = reader.read_bits(8)
                prev = val
                out.append(val & 0xFF)
            else:
                sign = (code >> 4) & 1
                mag = code & 0x0F
                val = ((prev - mag) if sign else (prev + mag)) & 0xFF
                prev = val
                out.append(val)
    return out

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
    all_data.extend(decode_linear(rec, W * 4))
raw = np.frombuffer(bytes(all_data), dtype=np.uint8).reshape(H, W)

# Extract odd rows (which have 0.78 correlation)
odd_rows = raw[1::2, :]  # 240 rows

# ====================================================================
# Test 1: Find optimal circular shift per row
# ====================================================================
print("=== Optimal per-row circular shift ===")
shifts = []
for row_idx in range(0, 240, 10):
    dec_row = odd_rows[row_idx]
    best_corr = -1
    best_shift = 0
    # Compare with nearest reference row
    ref_row_idx = min(row_idx * 2, H - 1)
    ref_row = ref_gray[ref_row_idx]
    
    for shift in range(W):
        shifted = np.roll(dec_row, shift)
        c = np.corrcoef(shifted.astype(float), ref_row.astype(float))[0,1]
        if c > best_corr:
            best_corr = c
            best_shift = shift
    shifts.append(best_shift)
    print(f"  Row {row_idx:3d} → shift {best_shift:3d} (corr={best_corr:.4f})")

# ====================================================================
# Test 2: Apply global best shift
# ====================================================================
from collections import Counter
shift_counts = Counter(shifts)
print(f"\nShift distribution: {shift_counts.most_common(5)}")

# Try the most common shift values
print("\n=== Global shift tests ===")
for shift in set(shifts):
    shifted_odd = np.roll(odd_rows, shift, axis=1)
    shifted_full = np.array(Image.fromarray(shifted_odd, 'L').resize((W, H), Image.BILINEAR))
    c = np.corrcoef(shifted_full.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
    if c > 0.7:
        print(f"  Shift {shift:3d}: corr={c:.4f}")

# Also try small shifts around the most common
for shift in range(max(0, shifts[0] - 10), shifts[0] + 10):
    shifted_odd = np.roll(odd_rows, shift, axis=1)
    shifted_full = np.array(Image.fromarray(shifted_odd, 'L').resize((W, H), Image.BILINEAR))
    c = np.corrcoef(shifted_full.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
    print(f"  Shift {shift:3d}: corr={c:.4f}")

# ====================================================================
# Test 3: Apply per-row optimal shift
# ====================================================================
print("\n=== Per-row optimal shift ===")
shifted_per_row = np.zeros_like(odd_rows)
for row_idx in range(240):
    dec_row = odd_rows[row_idx]
    ref_row_idx = min(row_idx * 2, H - 1)
    ref_row = ref_gray[ref_row_idx]
    best_corr = -1
    best_shift = 0
    for shift in range(W):
        shifted = np.roll(dec_row, shift)
        c = np.corrcoef(shifted.astype(float), ref_row.astype(float))[0,1]
        if c > best_corr:
            best_corr = c
            best_shift = shift
    shifted_per_row[row_idx] = np.roll(dec_row, best_shift)

shifted_full = np.array(Image.fromarray(shifted_per_row, 'L').resize((W, H), Image.BILINEAR))
c = np.corrcoef(shifted_full.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  Per-row optimal: corr={c:.4f}")
Image.fromarray(shifted_full, 'L').save(os.path.join(out_dir, "v5_per_row_shifted.png"))

# ====================================================================
# Test 4: Try treating ALL 4 rows per record as image data
# (not just odd rows) with correct rearrangement
# ====================================================================
print("\n=== All-rows tests ===")

# What if rows within a record need to be reordered as [1,0,3,2]?
# (odd rows first, then even rows)
img_reorder = np.zeros((H, W), dtype=np.uint8)
for g in range(H // 4):
    for new, old in enumerate([1, 0, 3, 2]):
        img_reorder[g*4 + new] = raw[g*4 + old]
c = np.corrcoef(img_reorder.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  Reorder [1,0,3,2]: corr={c:.4f}")

# What if the 4 rows per record should interleave: odd rows go first, even rows after?
# All odd rows (240), then all even rows (240)
img_split = np.zeros((H, W), dtype=np.uint8)
img_split[:240, :] = odd_rows
img_split[240:, :] = raw[0::2, :]
c = np.corrcoef(img_split.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  Odd rows then even rows: corr={c:.4f}")

# What if odd rows = one Bayer color and even rows = another?
# Try simple green channel extraction
# Assume: odd decoded rows = actual rows (all colors), just need scaling
# Reference image has gamma applied, our data might be linear
# Try: output = input^(1/2.2) * 255 (gamma correction)
gamma = odd_rows.astype(float) / 255.0
gamma_corrected = (np.power(gamma + 0.001, 1/2.2) * 255).astype(np.uint8)
gc_full = np.array(Image.fromarray(gamma_corrected, 'L').resize((W, H), Image.BILINEAR))
c = np.corrcoef(gc_full.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  Gamma corrected odd rows: corr={c:.4f}")
Image.fromarray(gc_full, 'L').save(os.path.join(out_dir, "v5_gamma_corrected.png"))

# Try using the GAMM lookup table from Veo.PRC
# Approximate: input 0→0, 1→27, 128→193, 255→255
# This is an encoding gamma (linear sensor → sRGB-like)
gamm_lut = np.zeros(256, dtype=np.uint8)
for i in range(256):
    # Approximate gamma 0.45 encoding
    gamm_lut[i] = int(np.clip(np.power(i / 255.0, 0.45) * 255, 0, 255))

gamma_lut_applied = gamm_lut[odd_rows]
glut_full = np.array(Image.fromarray(gamma_lut_applied, 'L').resize((W, H), Image.BILINEAR))
c = np.corrcoef(glut_full.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  GAMM LUT applied odd rows: corr={c:.4f}")
Image.fromarray(glut_full, 'L').save(os.path.join(out_dir, "v5_gamm_lut.png"))

print("\nDone!")
