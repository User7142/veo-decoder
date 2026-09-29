#!/usr/bin/env python3
"""Investigate 320px PDB structure."""

import struct, os
import numpy as np
from PIL import Image


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
        value = 0
        for _ in range(n):
            value = (value << 1) | self.read_bit()
        return value
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
    def pos_info(self):
        return f"byte={self.byte_pos}, bit={self.bit_pos}"


def decode_delta(reader, prev):
    flag = reader.read_bit()
    if flag == 1:
        return prev & 0xFF
    code = reader.read_bits(5)
    if code == 0:
        return reader.read_bits(8) & 0xFF
    sign = (code >> 4) & 1
    mag = code & 0x0F
    return ((prev - mag) if sign else (prev + mag)) & 0xFF


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


# Load 320px PDB
pdb_path = "data/3798526348.pdb"
records = parse_pdb(pdb_path)
data_records = records[1:-1]

print(f"320px PDB: {len(data_records)} data records")
for i in range(min(5, len(data_records))):
    print(f"  Record {i+1}: {len(data_records[i])} bytes, first 20: {data_records[i][:20].hex()}")

# The 640px decoder uses W=640, 2 calls per record = 4 rows
# For 320px, the DLL code uses W from metadata.
# fcn.10002610 allocates W*H*2 buffer, and fcn.10002e00 loops H/2 times
# calling fcn.100029c0 which processes 2 rows at a time.
# So for 320x480: still 120 records, 2 calls per record?
# But W=320 means each call decodes 2 rows of 320 pixels.
# With 120 records × 2 calls × 2 rows = 480 rows. Same as 640px.

# HOWEVER: the summary says the DLL handler reads records via
# fcn.10002ef0 with height/4 iterations. 480/4 = 120.
# And fcn.10002e00 does numRows/2 iterations.
# So it's: 120 records, each producing 4 rows (2 calls × 2 rows).
# For 320px that's 120 records × 4 rows = 480 rows.

# Let's try: maybe the 320px images have a different internal format.
# Perhaps they're NOT type 4, or the metadata byte 0 is different?

meta = records[0]
print(f"\nMetadata: {meta.hex()}")
print(f"  Byte 0 (type): {meta[0]}")
print(f"  Byte 1 (res):  {meta[1]}")
print(f"  Byte 18 (seq?): {meta[18]}")

# Test: decode first record with W=320
W = 320
rec = data_records[0]
reader = BitReader(rec)

print(f"\n=== Testing W=320 decode of record 0 ===")
print(f"Record size: {len(rec)} bytes")

# Call 1, Pass 1: row0 odd [1,3,...,319]
reader.flush_to_byte()
lit = reader.read_raw_byte()
print(f"Pass 1 literal: {lit} ({reader.pos_info()})")
prev = lit
p1_vals = [lit]
for c in range(1, W//2):
    val = decode_delta(reader, prev)
    p1_vals.append(val)
    prev = val
print(f"After Pass 1: {reader.pos_info()}")
print(f"Pass 1 stats: mean={np.mean(p1_vals):.1f}, min={min(p1_vals)}, max={max(p1_vals)}")
print(f"Pass 1 first 20: {p1_vals[:20]}")

# Call 1, Pass 2: row1 even
reader.flush_to_byte()
lit = reader.read_raw_byte()
print(f"\nPass 2 literal: {lit} ({reader.pos_info()})")
prev = lit
p2_vals = [lit]
for c in range(1, W//2):
    val = decode_delta(reader, prev)
    p2_vals.append(val)
    prev = val
print(f"After Pass 2: {reader.pos_info()}")
print(f"Pass 2 stats: mean={np.mean(p2_vals):.1f}, min={min(p2_vals)}, max={max(p2_vals)}")

# Call 1, Pass 3: row0 even + row1 odd zigzag
reader.flush_to_byte()
lit0 = reader.read_raw_byte()
print(f"\nPass 3 lit0 (row0[0]): {lit0} ({reader.pos_info()})")
reader.flush_to_byte()
lit1 = reader.read_raw_byte()
print(f"Pass 3 lit1 (row1[1]): {lit1} ({reader.pos_info()})")

# Build row buffers
row0 = np.zeros(W, dtype=np.uint8)
row1 = np.zeros(W, dtype=np.uint8)
for i, v in enumerate(p1_vals):
    row0[i*2+1] = v
for i, v in enumerate(p2_vals):
    row1[i*2] = v
row0[0] = lit0
row1[1] = lit1

p3_r0 = [lit0]
p3_r1 = [lit1]
for c in range(1, W//2):
    col_even = c * 2
    col_odd = c * 2 + 1
    pred = int(row1[col_even - 1])
    val = decode_delta(reader, pred)
    row0[col_even] = val
    p3_r0.append(val)
    if col_odd < W:
        pred = int(row0[col_even])
        val = decode_delta(reader, pred)
        row1[col_odd] = val
        p3_r1.append(val)

print(f"After Pass 3: {reader.pos_info()}")
print(f"Pass 3 row0 even: mean={np.mean(p3_r0):.1f}")
print(f"Pass 3 row1 odd:  mean={np.mean(p3_r1):.1f}")

bits_used = reader.byte_pos * 8 + (7 - reader.bit_pos)
total_bits = len(rec) * 8
print(f"\nAfter call 1: {bits_used} bits used of {total_bits} ({bits_used/total_bits*100:.1f}%)")
print(f"  = {reader.byte_pos} of {len(rec)} bytes")

# Call 2
print(f"\n=== Call 2 ===")
reader.flush_to_byte()
lit = reader.read_raw_byte()
print(f"Call 2 Pass 1 literal: {lit} ({reader.pos_info()})")
prev = lit
c2p1 = [lit]
for c in range(1, W//2):
    val = decode_delta(reader, prev)
    c2p1.append(val)
    prev = val
print(f"After Call 2 Pass 1: {reader.pos_info()}")
print(f"Call 2 Pass 1 stats: mean={np.mean(c2p1):.1f}")

reader.flush_to_byte()
lit = reader.read_raw_byte()
print(f"Call 2 Pass 2 literal: {lit} ({reader.pos_info()})")
prev = lit
c2p2 = [lit]
for c in range(1, W//2):
    val = decode_delta(reader, prev)
    c2p2.append(val)
    prev = val
print(f"After Call 2 Pass 2: {reader.pos_info()}")
print(f"Call 2 Pass 2 stats: mean={np.mean(c2p2):.1f}")

reader.flush_to_byte()
lit0 = reader.read_raw_byte()
reader.flush_to_byte()
lit1 = reader.read_raw_byte()
print(f"Call 2 Pass 3 lits: {lit0}, {lit1} ({reader.pos_info()})")

# Build rows for call 2
row2 = np.zeros(W, dtype=np.uint8)
row3 = np.zeros(W, dtype=np.uint8)
for i, v in enumerate(c2p1):
    row2[i*2+1] = v
for i, v in enumerate(c2p2):
    row3[i*2] = v
row2[0] = lit0
row3[1] = lit1

for c in range(1, W//2):
    col_even = c * 2
    col_odd = c * 2 + 1
    pred = int(row3[col_even - 1])
    val = decode_delta(reader, pred)
    row2[col_even] = val
    if col_odd < W:
        pred = int(row2[col_even])
        val = decode_delta(reader, pred)
        row3[col_odd] = val

print(f"After Call 2 complete: {reader.pos_info()}")
bits_used2 = reader.byte_pos * 8 + (7 - reader.bit_pos)
print(f"Total: {bits_used2} bits of {total_bits} ({bits_used2/total_bits*100:.1f}%)")

# Now try full decode with W=320
print(f"\n{'='*60}")
print("Full decode with W=320...")
H = 480
raw = np.zeros((H, W), dtype=np.uint8)

def decode_3pass_record(rec, output, base_row, W):
    reader = BitReader(rec)
    H = output.shape[0]
    for call in range(2):
        r0 = base_row + call * 2
        r1 = r0 + 1
        if r1 >= H:
            break
        reader.flush_to_byte()
        lit = reader.read_raw_byte()
        output[r0, 1] = lit
        prev = lit
        for c in range(1, W // 2):
            val = decode_delta(reader, prev)
            output[r0, c * 2 + 1] = val
            prev = val
        reader.flush_to_byte()
        lit = reader.read_raw_byte()
        output[r1, 0] = lit
        prev = lit
        for c in range(1, W // 2):
            val = decode_delta(reader, prev)
            output[r1, c * 2] = val
            prev = val
        reader.flush_to_byte()
        lit0 = reader.read_raw_byte()
        output[r0, 0] = lit0
        reader.flush_to_byte()
        lit1 = reader.read_raw_byte()
        output[r1, 1] = lit1
        for c in range(1, W // 2):
            col_even = c * 2
            col_odd = c * 2 + 1
            pred = int(output[r1, col_even - 1])
            val = decode_delta(reader, pred)
            output[r0, col_even] = val
            if col_odd < W:
                pred = int(output[r0, col_even])
                val = decode_delta(reader, pred)
                output[r1, col_odd] = val

for i, rec in enumerate(data_records):
    base = i * 4
    if base + 3 >= H:
        break
    decode_3pass_record(rec, raw, base, W)

# Check channel statistics
print(f"Gb (r0 even): mean={raw[0::2, 0::2].astype(float).mean():.1f}")
print(f"B  (r0 odd):  mean={raw[0::2, 1::2].astype(float).mean():.1f}")
print(f"R  (r1 even): mean={raw[1::2, 0::2].astype(float).mean():.1f}")
print(f"Gr (r1 odd):  mean={raw[1::2, 1::2].astype(float).mean():.1f}")

# Save raw grayscale to see if the structure is correct
Image.fromarray(raw, 'L').save("output/v9_320_raw.png")
print("Saved raw grayscale")

# Also try: maybe 320px uses 4 calls per record (8 rows)?
# That would give 120 * 8 = 960 rows, which is 2x too much for 480 height.
# Unless it's 60 records of 8 rows each? But we have 120 records.

# Try: maybe 320px has different pass structure
# Maybe it's NOT interleaved Bayer at 320px but full-resolution at 320?
# Or maybe records contain 8 rows (4 calls)?

# Alternative: what if 320px means the image was captured at 640 but
# records contain data for 640px width?
print(f"\n=== Try W=640 on 320px PDB ===")
raw640 = np.zeros((H, 640), dtype=np.uint8)
for i, rec in enumerate(data_records):
    base = i * 4
    if base + 3 >= H:
        break
    decode_3pass_record(rec, raw640, base, 640)

print(f"640 Gb: mean={raw640[0::2, 0::2].astype(float).mean():.1f}")
print(f"640 B:  mean={raw640[0::2, 1::2].astype(float).mean():.1f}")
print(f"640 R:  mean={raw640[1::2, 0::2].astype(float).mean():.1f}")
print(f"640 Gr: mean={raw640[1::2, 1::2].astype(float).mean():.1f}")
Image.fromarray(raw640, 'L').save("output/v9_320_as640_raw.png")
print("Saved 320-as-640 raw")

# Compare data utilization
print(f"\n=== Data utilization comparison ===")
for i in range(3):
    rec = data_records[i]
    # Test with W=320
    reader = BitReader(rec)
    for call in range(2):
        r0 = call * 2
        r1 = r0 + 1
        reader.flush_to_byte()
        reader.read_raw_byte()
        prev = 128
        for c in range(1, 160):
            decode_delta(reader, prev)
        reader.flush_to_byte()
        reader.read_raw_byte()
        for c in range(1, 160):
            decode_delta(reader, prev)
        reader.flush_to_byte()
        reader.read_raw_byte()
        reader.flush_to_byte()
        reader.read_raw_byte()
        for c in range(1, 160):
            decode_delta(reader, prev)
            decode_delta(reader, prev)
    pct320 = reader.byte_pos / len(rec) * 100

    reader2 = BitReader(rec)
    for call in range(2):
        r0 = call * 2
        r1 = r0 + 1
        reader2.flush_to_byte()
        reader2.read_raw_byte()
        prev = 128
        for c in range(1, 320):
            decode_delta(reader2, prev)
        reader2.flush_to_byte()
        reader2.read_raw_byte()
        for c in range(1, 320):
            decode_delta(reader2, prev)
        reader2.flush_to_byte()
        reader2.read_raw_byte()
        reader2.flush_to_byte()
        reader2.read_raw_byte()
        for c in range(1, 320):
            decode_delta(reader2, prev)
            decode_delta(reader2, prev)
    pct640 = reader2.byte_pos / len(rec) * 100

    print(f"Record {i}: {len(rec)} bytes → W=320: {pct320:.1f}%, W=640: {pct640:.1f}%")
