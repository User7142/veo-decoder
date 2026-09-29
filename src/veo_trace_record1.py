#!/usr/bin/env python3
"""Trace first record byte-by-byte to understand the pass structure."""

import struct, numpy as np
from PIL import Image

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
    def read_raw_byte(self):
        if self.byte_pos >= len(self.data):
            return 0
        val = self.data[self.byte_pos]
        self.byte_pos += 1
        return val
    def pos_info(self):
        return f"byte={self.byte_pos}, bit={self.bit_pos}"

def decode_delta(reader, prev):
    flag = reader.read_bit()
    if flag == 1:
        return prev & 0xFF, "RPT"
    code = reader.read_bits(5)
    if code == 0:
        val = reader.read_bits(8) & 0xFF
        return val, f"LIT({val})"
    sign = (code >> 4) & 1
    mag = code & 0x0F
    if sign:
        val = (prev - mag) & 0xFF
        return val, f"D-{mag}"
    else:
        val = (prev + mag) & 0xFF
        return val, f"D+{mag}"

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
ref_path = "data/Thom_091225_001.JPG"
records = parse_pdb(pdb_path)
rec = records[1]  # First data record
ref_gray = np.array(Image.open(ref_path).convert('L'))

print(f"Record 1 size: {len(rec)} bytes")
print(f"First 20 bytes: {' '.join(f'{b:02x}' for b in rec[:20])}")

# ============================================================================
# Trace the 3-pass decode of first record
# ============================================================================
W = 640
reader = BitReader(rec)

# === Call 1: rows 0-1 ===
print("\n" + "="*60)
print("=== Call 1: rows 0-1 ===")
print("="*60)

# PASS 1: byte-align + literal + odd positions of row 0
print(f"\n--- Pass 1: row0 odd positions ---")
print(f"Before flush: {reader.pos_info()}")
reader.flush_to_byte()
print(f"After flush: {reader.pos_info()}")

lit = reader.read_raw_byte()
print(f"Literal start (row0[1]): {lit} (0x{lit:02x})")
print(f"After literal: {reader.pos_info()}")

pass1_vals = [lit]
prev = lit
for c in range(1, W//2):
    val, desc = decode_delta(reader, prev)
    pass1_vals.append(val)
    prev = val

print(f"After pass 1: {reader.pos_info()}")
print(f"Pass 1 values (first 20): {pass1_vals[:20]}")
print(f"Pass 1 stats: mean={np.mean(pass1_vals):.1f}, min={min(pass1_vals)}, max={max(pass1_vals)}")
print(f"Reference row 0 odd pixels (first 20): {list(ref_gray[0, 1::2][:20])}")

# PASS 2: byte-align + literal + even positions of row 1
print(f"\n--- Pass 2: row1 even positions ---")
print(f"Before flush: {reader.pos_info()}")
reader.flush_to_byte()
print(f"After flush: {reader.pos_info()}")

lit = reader.read_raw_byte()
print(f"Literal start (row1[0]): {lit} (0x{lit:02x})")

pass2_vals = [lit]
prev = lit
for c in range(1, W//2):
    val, desc = decode_delta(reader, prev)
    pass2_vals.append(val)
    prev = val

print(f"After pass 2: {reader.pos_info()}")
print(f"Pass 2 values (first 20): {pass2_vals[:20]}")
print(f"Pass 2 stats: mean={np.mean(pass2_vals):.1f}, min={min(pass2_vals)}, max={max(pass2_vals)}")
print(f"Reference row 1 even pixels (first 20): {list(ref_gray[1, 0::2][:20])}")

# PASS 3: two literals + zigzag
print(f"\n--- Pass 3: row0 even + row1 odd (zigzag) ---")
print(f"Before flush: {reader.pos_info()}")
reader.flush_to_byte()
print(f"After flush: {reader.pos_info()}")

lit0 = reader.read_raw_byte()
print(f"Literal row0[0]: {lit0} (0x{lit0:02x})")

print(f"Before 2nd flush: {reader.pos_info()}")
reader.flush_to_byte()
print(f"After 2nd flush: {reader.pos_info()}")

lit1 = reader.read_raw_byte()
print(f"Literal row1[1]: {lit1} (0x{lit1:02x})")

pass3_r0_vals = [lit0]  # row0 even
pass3_r1_vals = [lit1]  # row1 odd

# Zigzag: need the output buffer for predictors
row0 = np.zeros(W, dtype=np.uint8)
row1 = np.zeros(W, dtype=np.uint8)
# Fill in from passes 1 and 2
for i, v in enumerate(pass1_vals):
    row0[i*2+1] = v
for i, v in enumerate(pass2_vals):
    row1[i*2] = v
row0[0] = lit0
row1[1] = lit1

for c in range(1, W//2):
    col_even = c * 2
    col_odd = c * 2 + 1

    # row0[even] predicted from row1[odd-1]
    pred = int(row1[col_even - 1])
    val, desc = decode_delta(reader, pred)
    row0[col_even] = val
    pass3_r0_vals.append(val)

    if col_odd < W:
        pred = int(row0[col_even])
        val, desc = decode_delta(reader, pred)
        row1[col_odd] = val
        pass3_r1_vals.append(val)

print(f"After pass 3: {reader.pos_info()}")
print(f"Pass 3 row0 even (first 20): {pass3_r0_vals[:20]}")
print(f"Pass 3 row1 odd (first 20): {pass3_r1_vals[:20]}")
print(f"Pass 3 row0 stats: mean={np.mean(pass3_r0_vals):.1f}, min={min(pass3_r0_vals)}, max={max(pass3_r0_vals)}")
print(f"Pass 3 row1 stats: mean={np.mean(pass3_r1_vals):.1f}, min={min(pass3_r1_vals)}, max={max(pass3_r1_vals)}")
print(f"Reference row 0 even (first 20): {list(ref_gray[0, 0::2][:20])}")
print(f"Reference row 1 odd (first 20): {list(ref_gray[1, 1::2][:20])}")

# ============================================================================
# Compare the full rows
# ============================================================================
print(f"\n--- Full row comparison ---")
print(f"Row 0 (decoded): {list(row0[:20])}")
print(f"Row 0 (ref):     {list(ref_gray[0, :20])}")
print(f"Row 1 (decoded): {list(row1[:20])}")
print(f"Row 1 (ref):     {list(ref_gray[1, :20])}")

corr_r0 = np.corrcoef(row0.astype(float), ref_gray[0].astype(float))[0,1]
corr_r1 = np.corrcoef(row1.astype(float), ref_gray[1].astype(float))[0,1]
print(f"Row 0 correlation: {corr_r0:.4f}")
print(f"Row 1 correlation: {corr_r1:.4f}")

# ============================================================================
# Now try: what if we DON'T do the 3-pass structure?
# Just linear decode and see what we get
# ============================================================================
print(f"\n{'='*60}")
print("=== Linear decode comparison ===")
print(f"{'='*60}")

reader2 = BitReader(rec)
linear_vals = []
prev = 128
for _ in range(W * 4):
    val, desc = decode_delta(reader2, prev)
    linear_vals.append(val)
    prev = val

# Linear decode as 4 rows of 640
lin_rows = [linear_vals[i*W:(i+1)*W] for i in range(4)]
for i, r in enumerate(lin_rows):
    corr = np.corrcoef(np.array(r, dtype=float), ref_gray[i].astype(float))[0,1]
    print(f"Linear row {i}: mean={np.mean(r):.1f}, first 20: {r[:20]}")
    print(f"  Reference:    first 20: {list(ref_gray[i, :20])}")
    print(f"  Correlation: {corr:.4f}")

# ============================================================================
# Critical question: are the 3-pass decoded values correct or shifted?
# Let's find the best shift for each pass
# ============================================================================
print(f"\n{'='*60}")
print("=== Shift analysis per pass ===")
for pass_name, vals, ref_row, ref_cols in [
    ("Pass 1 (r0 odd)", pass1_vals, 0, slice(1, None, 2)),
    ("Pass 2 (r1 even)", pass2_vals, 1, slice(0, None, 2)),
    ("Pass 3 r0 even", pass3_r0_vals, 0, slice(0, None, 2)),
    ("Pass 3 r1 odd", pass3_r1_vals, 1, slice(1, None, 2)),
]:
    dec = np.array(vals, dtype=float)
    ref_subset = ref_gray[ref_row, ref_cols][:len(vals)].astype(float)

    best_shift = 0
    best_corr = -1
    for shift in range(len(vals)):
        c = np.corrcoef(np.roll(dec, shift), ref_subset)[0,1]
        if c > best_corr:
            best_corr = c
            best_shift = shift

    # Also find best scale
    slope = np.polyfit(dec, ref_subset, 1)
    print(f"{pass_name}: best_shift={best_shift}, corr@shift={best_corr:.4f}, "
          f"scale={slope[0]:.3f}, offset={slope[1]:.1f}")

# ============================================================================
# Remaining data after call 1
# ============================================================================
print(f"\n--- Data consumption ---")
bits_consumed = reader.byte_pos * 8 + (7 - reader.bit_pos)
total_bits = len(rec) * 8
print(f"After call 1: consumed {bits_consumed} bits of {total_bits} total")
print(f"  = {reader.byte_pos} bytes, remaining: {len(rec) - reader.byte_pos} bytes")
print(f"  Utilization: {bits_consumed/total_bits*100:.1f}%")

# Try call 2
print(f"\n--- Call 2: rows 2-3 ---")
reader.flush_to_byte()
lit = reader.read_raw_byte()
print(f"Pass 1 literal: {lit}")
prev = lit
call2_p1 = [lit]
for c in range(1, W//2):
    val, _ = decode_delta(reader, prev)
    call2_p1.append(val)
    prev = val
print(f"Call 2 Pass 1 stats: mean={np.mean(call2_p1):.1f}")
print(f"After call 2 pass 1: {reader.pos_info()}")

bits_consumed2 = reader.byte_pos * 8 + (7 - reader.bit_pos)
print(f"Total consumed: {bits_consumed2} of {total_bits} ({bits_consumed2/total_bits*100:.1f}%)")

print("\nDone!")
