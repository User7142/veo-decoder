#!/usr/bin/env python3
"""Test different interleaving patterns for Veo decoder."""

import struct
import sys
import os
import numpy as np
from PIL import Image
from pathlib import Path


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


def decompress_delta(reader, count, prev=128):
    """Decompress 'count' pixels using delta encoding."""
    output = []
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
                magnitude = code & 0x0F
                if sign:
                    val = (prev - magnitude) & 0xFF
                else:
                    val = (prev + magnitude) & 0xFF
                prev = val
                output.append(val)
    return output, prev


def parse_pdb(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()
    num_records = struct.unpack('>H', data[76:78])[0]
    record_offsets = []
    for i in range(num_records):
        off = 78 + i * 8
        rec_offset = struct.unpack('>I', data[off:off+4])[0]
        record_offsets.append(rec_offset)
    record_offsets.append(len(data))
    records = []
    for i in range(num_records):
        records.append(data[record_offsets[i]:record_offsets[i+1]])
    return records


def try_pattern(records, width, height, pattern_name, fill_func):
    """Try a specific interleaving pattern."""
    data_records = records[1:-1]  # Skip metadata and thumbnail
    output = np.zeros((height, width), dtype=np.uint8)
    
    try:
        fill_func(data_records, output, width, height)
    except Exception as e:
        print(f"  {pattern_name}: ERROR - {e}")
        return None
    
    return output


pdb_path = "data/3840520404.pdb"
out_dir = "output/"
ref_path = "data/Thom_091225_001.JPG"

records = parse_pdb(pdb_path)
width, height = 640, 480
data_records = records[1:-1]

ref_img = Image.open(ref_path).convert('L')  # Grayscale reference
ref_arr = np.array(ref_img)

# ========================================================================
# Pattern A: Linear decode, then rearrange output
# First decode everything linearly
# ========================================================================
print("Decoding all records linearly first...")

all_linear = bytearray()
for rec in data_records:
    reader = BitReader(rec)
    pixels, _ = decompress_delta(reader, width * 4)
    all_linear.extend(pixels)

print(f"  Total decoded: {len(all_linear)} bytes")

patterns = {}

# Pattern 1: Linear as-is (640 wide)
patterns['p01_linear'] = np.frombuffer(bytes(all_linear), dtype=np.uint8).reshape(height, width)

# Pattern 2: Each record = 4 rows, but rows are interleaved (row0_even, row0_odd, row1_even, row1_odd)
def deinterleave_2x(data, w, h):
    out = np.zeros((h, w), dtype=np.uint8)
    arr = np.frombuffer(bytes(data), dtype=np.uint8)
    for rec_idx in range(h // 4):
        rec_start = rec_idx * w * 4
        rec_data = arr[rec_start:rec_start + w * 4]
        # First w pixels = row0_even + row0_odd interleaved
        # Next w pixels = row1_even + row1_odd interleaved
        for local_row in range(4):
            row = rec_idx * 4 + local_row
            row_data = rec_data[local_row * w:(local_row + 1) * w]
            out[row, :] = row_data
    return out
patterns['p02_4rows_sequential'] = deinterleave_2x(all_linear, width, height)

# Pattern 3: Halve horizontal, double vertical (maybe 320 wide?)
# Actually the image might be 320 pixels wide in UYVY (640 bytes = 320 pixel pairs)
p3 = np.frombuffer(bytes(all_linear[:320*960]), dtype=np.uint8).reshape(960, 320) if len(all_linear) >= 320*960 else None
if p3 is not None:
    patterns['p03_320wide'] = p3

# Pattern 4: Interpret as columns instead of rows
p4 = np.frombuffer(bytes(all_linear), dtype=np.uint8).reshape(height, width)
patterns['p04_transposed'] = p4.T[:height, :width] if p4.T.shape[0] >= height and p4.T.shape[1] >= width else p4

# ========================================================================
# Now try different DECOMPRESSION interleaving
# ========================================================================

# Pattern 5: Per-record, decompress even positions then odd positions (reset predictor)
def decode_interleaved_reset(records, w, h):
    out = np.zeros((h, w), dtype=np.uint8)
    for rec_idx, rec in enumerate(records):
        reader = BitReader(rec)
        base_row = rec_idx * 4
        # Pass 1: even positions of 4 rows
        prev = 128
        for local_row in range(4):
            for col in range(0, w, 2):
                row = base_row + local_row
                flag = reader.read_bit()
                if flag == 1:
                    out[row, col] = prev & 0xFF
                else:
                    code = reader.read_bits(5)
                    if code == 0:
                        val = reader.read_bits(8)
                        prev = val
                        out[row, col] = val & 0xFF
                    else:
                        sign = (code >> 4) & 1
                        mag = code & 0x0F
                        val = (prev - mag) & 0xFF if sign else (prev + mag) & 0xFF
                        prev = val
                        out[row, col] = val
        # Pass 2: odd positions of 4 rows
        prev = 128
        for local_row in range(4):
            for col in range(1, w, 2):
                row = base_row + local_row
                flag = reader.read_bit()
                if flag == 1:
                    out[row, col] = prev & 0xFF
                else:
                    code = reader.read_bits(5)
                    if code == 0:
                        val = reader.read_bits(8)
                        prev = val
                        out[row, col] = val & 0xFF
                    else:
                        sign = (code >> 4) & 1
                        mag = code & 0x0F
                        val = (prev - mag) & 0xFF if sign else (prev + mag) & 0xFF
                        prev = val
                        out[row, col] = val
    return out

patterns['p05_even_odd_4rows'] = decode_interleaved_reset(data_records, width, height)

# Pattern 6: Per-record, 2 rows at a time, even then odd
def decode_2rows_even_odd(records, w, h):
    out = np.zeros((h, w), dtype=np.uint8)
    for rec_idx, rec in enumerate(records):
        reader = BitReader(rec)
        base_row = rec_idx * 4
        for row_pair in range(2):  # 2 pairs of rows per record
            # Even positions of 2 rows
            prev = 128
            for local_row in range(2):
                row = base_row + row_pair * 2 + local_row
                for col in range(0, w, 2):
                    flag = reader.read_bit()
                    if flag == 1:
                        out[row, col] = prev & 0xFF
                    else:
                        code = reader.read_bits(5)
                        if code == 0:
                            val = reader.read_bits(8)
                            prev = val
                            out[row, col] = val & 0xFF
                        else:
                            sign = (code >> 4) & 1
                            mag = code & 0x0F
                            val = (prev - mag) & 0xFF if sign else (prev + mag) & 0xFF
                            prev = val
                            out[row, col] = val
            # Odd positions of 2 rows
            prev = 128
            for local_row in range(2):
                row = base_row + row_pair * 2 + local_row
                for col in range(1, w, 2):
                    flag = reader.read_bit()
                    if flag == 1:
                        out[row, col] = prev & 0xFF
                    else:
                        code = reader.read_bits(5)
                        if code == 0:
                            val = reader.read_bits(8)
                            prev = val
                            out[row, col] = val & 0xFF
                        else:
                            sign = (code >> 4) & 1
                            mag = code & 0x0F
                            val = (prev - mag) & 0xFF if sign else (prev + mag) & 0xFF
                            prev = val
                            out[row, col] = val
    return out

patterns['p06_2rows_even_odd'] = decode_2rows_even_odd(data_records, width, height)

# Pattern 7: ARMC exact - odd first (dest+1), then even (dest+0)
def decode_armc_odd_first(records, w, h):
    out = np.zeros((h, w), dtype=np.uint8)
    for rec_idx, rec in enumerate(records):
        reader = BitReader(rec)
        base_row = rec_idx * 4
        for row_pair in range(2):
            # Odd positions first (dest+1, stride 2)
            prev = 128
            for local_row in range(2):
                row = base_row + row_pair * 2 + local_row
                for col in range(1, w, 2):
                    flag = reader.read_bit()
                    if flag == 1:
                        out[row, col] = prev & 0xFF
                    else:
                        code = reader.read_bits(5)
                        if code == 0:
                            val = reader.read_bits(8)
                            prev = val
                            out[row, col] = val & 0xFF
                        else:
                            sign = (code >> 4) & 1
                            mag = code & 0x0F
                            val = (prev - mag) & 0xFF if sign else (prev + mag) & 0xFF
                            prev = val
                            out[row, col] = val
            # Even positions (dest+0, stride 2)
            prev = 128
            for local_row in range(2):
                row = base_row + row_pair * 2 + local_row
                for col in range(0, w, 2):
                    flag = reader.read_bit()
                    if flag == 1:
                        out[row, col] = prev & 0xFF
                    else:
                        code = reader.read_bits(5)
                        if code == 0:
                            val = reader.read_bits(8)
                            prev = val
                            out[row, col] = val & 0xFF
                        else:
                            sign = (code >> 4) & 1
                            mag = code & 0x0F
                            val = (prev - mag) & 0xFF if sign else (prev + mag) & 0xFF
                            prev = val
                            out[row, col] = val
    return out

patterns['p07_armc_odd_first'] = decode_armc_odd_first(data_records, width, height)

# Pattern 8: 2 rows per record (not 4), 240 records for 480 rows
def decode_2rows_per_record(records, w, h):
    out = np.zeros((h, w), dtype=np.uint8)
    for rec_idx, rec in enumerate(records):
        reader = BitReader(rec)
        base_row = rec_idx * 2
        if base_row >= h:
            break
        # Even positions of 2 rows
        prev = 128
        for local_row in range(2):
            row = base_row + local_row
            if row >= h:
                break
            for col in range(0, w, 2):
                flag = reader.read_bit()
                if flag == 1:
                    out[row, col] = prev & 0xFF
                else:
                    code = reader.read_bits(5)
                    if code == 0:
                        val = reader.read_bits(8)
                        prev = val
                        out[row, col] = val & 0xFF
                    else:
                        sign = (code >> 4) & 1
                        mag = code & 0x0F
                        val = (prev - mag) & 0xFF if sign else (prev + mag) & 0xFF
                        prev = val
                        out[row, col] = val
        # Odd positions of 2 rows
        prev = 128
        for local_row in range(2):
            row = base_row + local_row
            if row >= h:
                break
            for col in range(1, w, 2):
                flag = reader.read_bit()
                if flag == 1:
                    out[row, col] = prev & 0xFF
                else:
                    code = reader.read_bits(5)
                    if code == 0:
                        val = reader.read_bits(8)
                        prev = val
                        out[row, col] = val & 0xFF
                    else:
                        sign = (code >> 4) & 1
                        mag = code & 0x0F
                        val = (prev - mag) & 0xFF if sign else (prev + mag) & 0xFF
                        prev = val
                        out[row, col] = val
    return out

patterns['p08_2rows_per_record'] = decode_2rows_per_record(data_records, width, height)

# ========================================================================
# Save and compare all patterns
# ========================================================================
print(f"\nComparing {len(patterns)} patterns:")
results = []

for name, arr in patterns.items():
    if arr is None:
        continue
    
    # Save grayscale
    img = Image.fromarray(arr if arr.shape == (height, width) else arr[:height, :width], 'L')
    path = os.path.join(out_dir, f"{name}.png")
    img.save(path)
    
    # Compare with reference (grayscale)
    if arr.shape[0] != height or arr.shape[1] != width:
        comp = np.array(img.resize((width, height), Image.NEAREST))
    else:
        comp = arr
    
    mse = np.mean((comp.astype(float) - ref_arr.astype(float)) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    
    # Also check structural similarity (correlation)
    corr = np.corrcoef(comp.flatten().astype(float), ref_arr.flatten().astype(float))[0, 1]
    
    results.append((name, mse, psnr, corr))
    print(f"  {name:35s}: MSE={mse:8.1f}  PSNR={psnr:5.2f}dB  Corr={corr:+.4f}")

# Sort by correlation (best first)
results.sort(key=lambda x: -x[3])
print(f"\nBest by correlation:")
for name, mse, psnr, corr in results[:5]:
    print(f"  {name:35s}: Corr={corr:+.4f}  PSNR={psnr:.2f}dB")

print("\nDone!")
