#!/usr/bin/env python3
"""Test byte alignment at various points in the bitstream."""

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
    def flush_to_byte(self):
        if self.bit_pos != 7:
            self.bit_pos = 7
            self.byte_pos += 1
    @property
    def pos(self):
        return self.byte_pos * 8 + (7 - self.bit_pos)

def decode_delta(reader, prev):
    flag = reader.read_bit()
    if flag == 1:
        return prev & 0xFF, prev
    code = reader.read_bits(5)
    if code == 0:
        val = reader.read_bits(8)
        return val & 0xFF, val
    sign = (code >> 4) & 1
    mag = code & 0x0F
    val = ((prev - mag) if sign else (prev + mag)) & 0xFF
    return val, val

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

# Concatenate ALL record data
all_data = bytearray()
for rec in data_records:
    all_data.extend(rec)
print(f"Total compressed data: {len(all_data)} bytes = {len(all_data)*8} bits")

# ====================================================================
# Test: Flush to byte boundary after EVERY row of 640 pixels
# ====================================================================
print("\n=== Test: Byte flush after every 640 pixels ===")
reader = BitReader(bytes(all_data))
img = np.zeros((H, W), dtype=np.uint8)
prev = 128
for row in range(H):
    for col in range(W):
        val, prev = decode_delta(reader, prev)
        img[row, col] = val
    reader.flush_to_byte()
    prev = 128  # Reset predictor per row

corr = np.corrcoef(img.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  Flush every row, reset pred: corr={corr:.4f}")
Image.fromarray(img, 'L').save(os.path.join(out_dir, "ba_flush_every_row_reset.png"))

# Same but don't reset predictor
reader = BitReader(bytes(all_data))
img2 = np.zeros((H, W), dtype=np.uint8)
prev = 128
for row in range(H):
    for col in range(W):
        val, prev = decode_delta(reader, prev)
        img2[row, col] = val
    reader.flush_to_byte()

corr2 = np.corrcoef(img2.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  Flush every row, keep pred: corr={corr2:.4f}")
Image.fromarray(img2, 'L').save(os.path.join(out_dir, "ba_flush_every_row_keep.png"))

# ====================================================================
# Test: Flush every 2 rows (to match the 2-row block structure)
# ====================================================================
print("\n=== Test: Byte flush every 2 rows ===")
reader = BitReader(bytes(all_data))
img3 = np.zeros((H, W), dtype=np.uint8)
prev = 128
for row_pair in range(H // 2):
    for row_offset in range(2):
        row = row_pair * 2 + row_offset
        for col in range(W):
            val, prev = decode_delta(reader, prev)
            img3[row, col] = val
    reader.flush_to_byte()
    prev = 128

corr3 = np.corrcoef(img3.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  Flush every 2 rows: corr={corr3:.4f}")
Image.fromarray(img3, 'L').save(os.path.join(out_dir, "ba_flush_2rows.png"))

# ====================================================================
# Test: Flush every 320 pixels (half row = one pass of even/odd)
# ====================================================================
print("\n=== Test: Byte flush every 320 pixels ===")
reader = BitReader(bytes(all_data))
img4 = np.zeros((H, W), dtype=np.uint8)
prev = 128
for row in range(H):
    # First 320 pixels
    for col in range(320):
        val, prev = decode_delta(reader, prev)
        img4[row, col] = val
    reader.flush_to_byte()
    # Next 320 pixels
    for col in range(320, W):
        val, prev = decode_delta(reader, prev)
        img4[row, col] = val
    reader.flush_to_byte()

corr4 = np.corrcoef(img4.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  Flush every 320 pixels: corr={corr4:.4f}")
Image.fromarray(img4, 'L').save(os.path.join(out_dir, "ba_flush_320.png"))

# ====================================================================
# Test: Per-record processing (as before) but with flush at end of each record
# ====================================================================
print("\n=== Test: Per-record flush, 640px rows ===")
img5 = np.zeros((H, W), dtype=np.uint8)
for rec_idx, rec in enumerate(data_records):
    reader = BitReader(rec)
    base_row = rec_idx * 4
    prev = 128
    for local_row in range(4):
        row = base_row + local_row
        if row >= H: break
        for col in range(W):
            val, prev = decode_delta(reader, prev)
            img5[row, col] = val
        reader.flush_to_byte()
        prev = 128

corr5 = np.corrcoef(img5.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  Per-record, flush per row, reset: corr={corr5:.4f}")
Image.fromarray(img5, 'L').save(os.path.join(out_dir, "ba_per_record_flush_row.png"))

# ====================================================================
# Test: 240 rows from concatenated stream (skip every other "row")
# ====================================================================
print("\n=== Test: 240 rows from concatenated stream ===")
reader = BitReader(bytes(all_data))
img6 = np.zeros((240, W), dtype=np.uint8)
prev = 128
for row in range(240):
    for col in range(W):
        val, prev = decode_delta(reader, prev)
        img6[row, col] = val
    reader.flush_to_byte()
    prev = 128

img6_full = np.array(Image.fromarray(img6, 'L').resize((W, H), Image.BILINEAR))
corr6 = np.corrcoef(img6_full.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  240 rows concat, flush per row: corr={corr6:.4f}")
Image.fromarray(img6, 'L').save(os.path.join(out_dir, "ba_240rows_concat.png"))

# Try with 320-pixel half rows
reader = BitReader(bytes(all_data))
img7 = np.zeros((240, W), dtype=np.uint8)
for row in range(240):
    prev = 128
    for col in range(320):
        val, prev = decode_delta(reader, prev)
        img7[row, col] = val
    reader.flush_to_byte()
    prev = 128
    for col in range(320, W):
        val, prev = decode_delta(reader, prev)
        img7[row, col] = val
    reader.flush_to_byte()

img7_full = np.array(Image.fromarray(img7, 'L').resize((W, H), Image.BILINEAR))
corr7 = np.corrcoef(img7_full.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  240 rows concat, flush per 320: corr={corr7:.4f}")
Image.fromarray(img7, 'L').save(os.path.join(out_dir, "ba_240rows_320flush.png"))

# ====================================================================
# Summary
# ====================================================================
print("\n=== Summary ===")
results = [
    ("Baseline (linear odd rows)", 0.7823),
    ("Flush every row (reset)", corr),
    ("Flush every row (keep pred)", corr2),
    ("Flush every 2 rows", corr3),
    ("Flush every 320 pixels", corr4),
    ("Per-record flush per row", corr5),
    ("240 rows concat flush/row", corr6),
    ("240 rows 320-flush", corr7),
]
results.sort(key=lambda x: -x[1])
for name, c in results:
    print(f"  {name:40s}: {c:.4f}")

print("\nDone!")
