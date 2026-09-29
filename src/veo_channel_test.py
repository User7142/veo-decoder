#!/usr/bin/env python3
"""Test YCbCr channel separation and pixel rearrangement."""

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
    bits_used = reader.byte_pos * 8 + (7 - reader.bit_pos)
    bits_avail = len(rec_data) * 8
    return output, bits_used, bits_avail

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

ref = np.array(Image.open(ref_path).convert('L'))

# First: analyze bitstream usage per record
print("=== Bitstream analysis ===")
total_bits_used = 0
total_bits_avail = 0
all_linear = bytearray()

for i, rec in enumerate(data_records):
    pixels, used, avail = decompress_linear(rec, W * 4)
    all_linear.extend(pixels)
    total_bits_used += used
    total_bits_avail += avail
    if i < 3:
        print(f"Rec {i+1}: {len(rec)} bytes, decoded {len(pixels)} px, "
              f"bits: {used}/{avail} ({avail-used} left)")

print(f"\nTotal: {total_bits_used} / {total_bits_avail} bits "
      f"({total_bits_avail - total_bits_used} unused)")

raw = np.frombuffer(bytes(all_linear), dtype=np.uint8).reshape(H, W)

# Now test: maybe the data is already luma-only (one byte per pixel)
# and the 640-byte rows represent 640 actual pixels
print("\n=== Approach 1: Raw grayscale (640x480, 1 byte/pixel) ===")
mse = np.mean((raw.astype(float) - ref.astype(float))**2)
corr = np.corrcoef(raw.flatten().astype(float), ref.flatten().astype(float))[0,1]
print(f"  MSE={mse:.1f} PSNR={10*np.log10(255**2/mse):.2f}dB Corr={corr:.4f}")

# Approach 2: UYVY - extract Y channel only (bytes 1,3,5,7...)
print("\n=== Approach 2: UYVY Y-channel extraction ===")
y_only = raw[:, 1::2]  # Every other byte starting at 1 (Y0, Y1)
y_img = Image.fromarray(y_only, 'L').resize((W, H), Image.BILINEAR)
y_arr = np.array(y_img)
mse = np.mean((y_arr.astype(float) - ref.astype(float))**2)
corr = np.corrcoef(y_arr.flatten().astype(float), ref.flatten().astype(float))[0,1]
print(f"  Y only (320 wide): MSE={mse:.1f} Corr={corr:.4f}")
Image.fromarray(y_only, 'L').save(os.path.join(out_dir, "test_y_channel.png"))

# Approach 3: Extract even bytes only
print("\n=== Approach 3: Even bytes only ===")
even = raw[:, 0::2]
even_img = Image.fromarray(even, 'L')
even_img.save(os.path.join(out_dir, "test_even_bytes.png"))
even_r = np.array(even_img.resize((W, H), Image.BILINEAR))
mse = np.mean((even_r.astype(float) - ref.astype(float))**2)
corr = np.corrcoef(even_r.flatten().astype(float), ref.flatten().astype(float))[0,1]
print(f"  Even bytes: MSE={mse:.1f} Corr={corr:.4f}")

# Approach 4: Look at record structure - maybe each record contains
# TWO separate bitstreams (one for Y, one for C)?
# Let's try decoding only half the pixels per record
print("\n=== Approach 4: Half pixels per record ===")
half_data = bytearray()
for rec in data_records:
    pixels, _, _ = decompress_linear(rec, W * 2)  # Only 2 rows worth
    half_data.extend(pixels)
half_raw = np.frombuffer(bytes(half_data[:W*H]), dtype=np.uint8).reshape(H, W)
mse = np.mean((half_raw.astype(float) - ref.astype(float))**2)
corr = np.corrcoef(half_raw.flatten().astype(float), ref.flatten().astype(float))[0,1]
print(f"  Half: MSE={mse:.1f} Corr={corr:.4f}")
Image.fromarray(half_raw, 'L').save(os.path.join(out_dir, "test_half.png"))

# Approach 5: Linear but reorganize rows within each 4-row group
# Try: [row0, row2, row1, row3] or [row0, row1, row2, row3] etc
print("\n=== Approach 5: Row reordering within 4-row groups ===")
import itertools
best_corr = -1
best_perm = None
for perm in itertools.permutations(range(4)):
    reordered = np.zeros_like(raw)
    for g in range(H // 4):
        for i, j in enumerate(perm):
            reordered[g*4 + i] = raw[g*4 + j]
    c = np.corrcoef(reordered.flatten().astype(float), ref.flatten().astype(float))[0,1]
    if c > best_corr:
        best_corr = c
        best_perm = perm
print(f"  Best row permutation: {best_perm} Corr={best_corr:.4f}")
reordered = np.zeros_like(raw)
for g in range(H // 4):
    for i, j in enumerate(best_perm):
        reordered[g*4 + i] = raw[g*4 + j]
Image.fromarray(reordered, 'L').save(os.path.join(out_dir, "test_reordered.png"))

# Approach 6: What if the 4 rows per record represent:
# [Y_even_row0, Y_odd_row0, Y_even_row1, Y_odd_row1] ?
# i.e., 320 Y values + 320 Y values per row?
print("\n=== Approach 6: De-interleave within rows ===")
deint = np.zeros((H, W), dtype=np.uint8)
for g in range(H // 2):
    # Row g*2: take from decoded rows g*4+0 (even) and g*4+1 (odd)
    row_a = raw[g*2, :]  # In the linear output
    # Actually let me just re-map from the linear flat data
    rec_start = g * 2 * W  # Start of 4 decoded rows for this record group
    # First 640 values = even pixels of row 0
    # Next 640 values = odd pixels of row 0?
    for col in range(320):
        deint[g, col*2] = all_linear[rec_start + col]
        deint[g, col*2+1] = all_linear[rec_start + 320 + col]
# This doesn't work well for general case, skip...

# Approach 7: Try pure grayscale without any per-row interleaving,
# but scale the values properly
print("\n=== Approach 7: Scaled grayscale ===")
# The reference image has mean ~108, our data has mean ~88
# Try linear scaling
raw_f = raw.astype(float)
ref_f = ref.astype(float)
# Optimal linear transform: a*raw + b ≈ ref
a = np.sum((raw_f - raw_f.mean()) * (ref_f - ref_f.mean())) / np.sum((raw_f - raw_f.mean())**2)
b = ref_f.mean() - a * raw_f.mean()
scaled = np.clip(a * raw_f + b, 0, 255).astype(np.uint8)
mse = np.mean((scaled.astype(float) - ref_f)**2)
corr = np.corrcoef(scaled.flatten().astype(float), ref_f.flatten().astype(float))[0,1]
print(f"  Scaled (a={a:.3f}, b={b:.1f}): MSE={mse:.1f} Corr={corr:.4f}")
Image.fromarray(scaled, 'L').save(os.path.join(out_dir, "test_scaled.png"))

# Approach 8: Try different numbers of pixels per record to see which
# gives the cleanest decompression (matches bitstream size best)
print("\n=== Approach 8: Optimal pixels per record ===")
rec = data_records[0]
for n_pixels in [1280, 1920, 2560, 3200, 3840, 5120]:
    reader = BitReader(rec)
    try:
        pixels, used, avail = decompress_linear(rec, n_pixels)
        ratio = used / avail
        print(f"  {n_pixels} px: {used}/{avail} bits ({ratio:.4f}), "
              f"last 5 vals: {pixels[-5:]}")
    except:
        print(f"  {n_pixels} px: ERROR")

print("\nDone!")
