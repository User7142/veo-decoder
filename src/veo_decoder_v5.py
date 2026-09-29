#!/usr/bin/env python3
"""V5: Rearrange linear decode output based on disassembly pass structure."""

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

def decode_linear(rec_data, count, prev=128):
    reader = BitReader(rec_data)
    out = []
    for _ in range(count):
        val, prev = decode_delta(reader, prev)
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

# ==========================================================================
# KEY INSIGHT from disassembly: per 2-row block within each record:
# Values 0-319: odd cols of row 0 (with continuous predictor)
# Values 320-639: odd cols of row 1 (predictor continues)
# Values 640-641: literals for col 0 of rows 0 and 1
# Values 642-960: even cols (2,4,...,638) of row 0
# Values 961-1279: even cols of row 1
#
# In linear decode, these become:
# Row 0: values 0-639 (= pass1 + pass2 = odd cols of both rows)
# Row 1: values 640-1279 (= pass3 + pass4 = even cols of both rows)
# Row 2: values 1280-1919 (= same pattern for row pair 2)
# Row 3: values 1920-2559
# ==========================================================================

# Strategy A: Rearrange linear output based on pass structure
# No byte flush (linear decode ignores it)
def rearrange_a(linear_vals, W):
    """Rearrange 2560 linear values into 4 rows of 640 bytes."""
    out = np.zeros((4, W), dtype=np.uint8)
    idx = 0
    
    for pair in range(2):  # 2 row-pairs
        r0 = pair * 2
        r1 = pair * 2 + 1
        
        # First 320 values → odd cols of row 0
        for c in range(W // 2):
            out[r0, c * 2 + 1] = linear_vals[idx]
            idx += 1
        
        # Next 320 values → odd cols of row 1
        for c in range(W // 2):
            out[r1, c * 2 + 1] = linear_vals[idx]
            idx += 1
        
        # Next 2 values → col 0 of rows 0 and 1
        out[r0, 0] = linear_vals[idx]; idx += 1
        out[r1, 0] = linear_vals[idx]; idx += 1
        
        # Next 319 values → even cols (2,4,...,638) of row 0
        for c in range(1, W // 2):
            out[r0, c * 2] = linear_vals[idx]
            idx += 1
        
        # Next 319 values → even cols of row 1
        for c in range(1, W // 2):
            out[r1, c * 2] = linear_vals[idx]
            idx += 1
    
    return out

# Strategy B: Same but with predictor running per pass (decode differently)
def decode_structured(rec_data, W):
    """Decode with correct per-pass predictor structure."""
    reader = BitReader(rec_data)
    out = np.zeros((4, W), dtype=np.uint8)
    
    for pair in range(2):
        r0 = pair * 2
        r1 = pair * 2 + 1
        
        # Pass 1: odd cols of row 0, predictor starts at 128
        prev = 128
        for c in range(W // 2):
            val, prev = decode_delta(reader, prev)
            out[r0, c * 2 + 1] = val
        
        # Pass 2: odd cols of row 1, predictor continues from pass 1
        for c in range(W // 2):
            val, prev = decode_delta(reader, prev)
            out[r1, c * 2 + 1] = val
        
        # Byte alignment flush
        reader.flush_to_byte()
        
        # Pass 3: literal bytes for col 0
        out[r0, 0] = reader.data[reader.byte_pos] if reader.byte_pos < len(reader.data) else 0
        reader.byte_pos += 1
        
        reader.flush_to_byte()
        
        out[r1, 0] = reader.data[reader.byte_pos] if reader.byte_pos < len(reader.data) else 0
        reader.byte_pos += 1
        
        # Pass 4a: even cols of row 0, horizontal prediction from col 0
        prev = int(out[r0, 0])
        for c in range(1, W // 2):
            val, prev = decode_delta(reader, prev)
            out[r0, c * 2] = val
        
        # Pass 4b: even cols of row 1, vertical prediction from row 0
        prev = int(out[r1, 0])
        for c in range(1, W // 2):
            above = int(out[r0, c * 2])
            val, _ = decode_delta(reader, above)
            out[r1, c * 2] = val
    
    return out

# Strategy C: Same structure but without second flush between literals
def decode_structured_v2(rec_data, W):
    reader = BitReader(rec_data)
    out = np.zeros((4, W), dtype=np.uint8)
    
    for pair in range(2):
        r0 = pair * 2
        r1 = pair * 2 + 1
        
        prev = 128
        for c in range(W // 2):
            val, prev = decode_delta(reader, prev)
            out[r0, c * 2 + 1] = val
        
        for c in range(W // 2):
            val, prev = decode_delta(reader, prev)
            out[r1, c * 2 + 1] = val
        
        reader.flush_to_byte()
        
        # Read 2 literal bytes consecutively
        b0 = reader.data[reader.byte_pos] if reader.byte_pos < len(reader.data) else 0
        reader.byte_pos += 1
        b1 = reader.data[reader.byte_pos] if reader.byte_pos < len(reader.data) else 0
        reader.byte_pos += 1
        out[r0, 0] = b0
        out[r1, 0] = b1
        
        prev = int(b0)
        for c in range(1, W // 2):
            val, prev = decode_delta(reader, prev)
            out[r0, c * 2] = val
        
        for c in range(1, W // 2):
            above = int(out[r0, c * 2])
            val, _ = decode_delta(reader, above)
            out[r1, c * 2] = val
    
    return out

# Strategy D: Horizontal prediction for pass 4b too (not vertical)
def decode_structured_v3(rec_data, W):
    reader = BitReader(rec_data)
    out = np.zeros((4, W), dtype=np.uint8)
    
    for pair in range(2):
        r0 = pair * 2
        r1 = pair * 2 + 1
        
        prev = 128
        for c in range(W // 2):
            val, prev = decode_delta(reader, prev)
            out[r0, c * 2 + 1] = val
        
        for c in range(W // 2):
            val, prev = decode_delta(reader, prev)
            out[r1, c * 2 + 1] = val
        
        reader.flush_to_byte()
        
        b0 = reader.data[reader.byte_pos] if reader.byte_pos < len(reader.data) else 0
        reader.byte_pos += 1
        b1 = reader.data[reader.byte_pos] if reader.byte_pos < len(reader.data) else 0
        reader.byte_pos += 1
        out[r0, 0] = b0
        out[r1, 0] = b1
        
        prev = int(b0)
        for c in range(1, W // 2):
            val, prev = decode_delta(reader, prev)
            out[r0, c * 2] = val
        
        prev = int(b1)
        for c in range(1, W // 2):
            val, prev = decode_delta(reader, prev)
            out[r1, c * 2] = val
    
    return out

# Strategy E: Only 1 row-pair per record (2 rows, not 4)
def decode_structured_1pair(rec_data, W):
    reader = BitReader(rec_data)
    out = np.zeros((2, W), dtype=np.uint8)
    
    prev = 128
    for c in range(W // 2):
        val, prev = decode_delta(reader, prev)
        out[0, c * 2 + 1] = val
    
    for c in range(W // 2):
        val, prev = decode_delta(reader, prev)
        out[1, c * 2 + 1] = val
    
    reader.flush_to_byte()
    
    b0 = reader.data[reader.byte_pos] if reader.byte_pos < len(reader.data) else 0
    reader.byte_pos += 1
    b1 = reader.data[reader.byte_pos] if reader.byte_pos < len(reader.data) else 0
    reader.byte_pos += 1
    out[0, 0] = b0
    out[1, 0] = b1
    
    prev = int(b0)
    for c in range(1, W // 2):
        val, prev = decode_delta(reader, prev)
        out[0, c * 2] = val
    
    prev = int(b1)
    for c in range(1, W // 2):
        val, prev = decode_delta(reader, prev)
        out[1, c * 2] = val
    
    return out

# ==========================================================================
# Test all strategies
# ==========================================================================

# First, decode all records linearly
all_linear = []
for rec in data_records:
    all_linear.append(decode_linear(rec, W * 4))

def test_strategy(name, img):
    corr = np.corrcoef(img.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
    mse = np.mean((img.astype(float) - ref_gray.astype(float))**2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 0
    print(f"  {name:40s}: corr={corr:.4f} PSNR={psnr:.2f}dB")
    Image.fromarray(img, 'L').save(os.path.join(out_dir, f"v5_{name}.png"))
    return corr

print("=== Rearrangement tests ===")

# A: Rearrange linear output
img_a = np.zeros((H, W), dtype=np.uint8)
for i, vals in enumerate(all_linear):
    rows = rearrange_a(vals, W)
    base = i * 4
    for r in range(4):
        if base + r < H:
            img_a[base + r] = rows[r]
test_strategy("a_rearranged_linear", img_a)

# B: Structured decode with byte flush + vertical prediction
img_b = np.zeros((H, W), dtype=np.uint8)
for i, rec in enumerate(data_records):
    rows = decode_structured(rec, W)
    base = i * 4
    for r in range(4):
        if base + r < H:
            img_b[base + r] = rows[r]
test_strategy("b_structured_flush_vpred", img_b)

# C: Structured without second flush
img_c = np.zeros((H, W), dtype=np.uint8)
for i, rec in enumerate(data_records):
    rows = decode_structured_v2(rec, W)
    base = i * 4
    for r in range(4):
        if base + r < H:
            img_c[base + r] = rows[r]
test_strategy("c_structured_1flush", img_c)

# D: Horizontal prediction for pass 4b
img_d = np.zeros((H, W), dtype=np.uint8)
for i, rec in enumerate(data_records):
    rows = decode_structured_v3(rec, W)
    base = i * 4
    for r in range(4):
        if base + r < H:
            img_d[base + r] = rows[r]
test_strategy("d_structured_hpred", img_d)

# E: 1 row-pair per record (240 records for 480 rows) - but we only have 120
img_e = np.zeros((H, W), dtype=np.uint8)
for i, rec in enumerate(data_records):
    rows = decode_structured_1pair(rec, W)
    base = i * 2
    for r in range(2):
        if base + r < H:
            img_e[base + r] = rows[r]
test_strategy("e_1pair_per_record", img_e)

# ==========================================================================
# Now take the best and compare visually
# ==========================================================================
print("\n=== Comparing best result with reference ===")
best_img = max([(img_a, 'a'), (img_b, 'b'), (img_c, 'c'), (img_d, 'd'), (img_e, 'e')],
               key=lambda x: np.corrcoef(x[0].flatten().astype(float), ref_gray.flatten().astype(float))[0,1])
print(f"  Best: strategy {best_img[1]}")

# Also check: extract just odd rows from EACH strategy and compare
print("\n=== Odd-row extraction ===")
for img, name in [(img_a,'a'), (img_b,'b'), (img_c,'c'), (img_d,'d'), (img_e,'e')]:
    odd = img[1::2, :]
    odd_full = np.array(Image.fromarray(odd, 'L').resize((W, H), Image.BILINEAR))
    corr = np.corrcoef(odd_full.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
    print(f"  {name} odd rows: corr={corr:.4f}")
    
    even = img[0::2, :]
    even_full = np.array(Image.fromarray(even, 'L').resize((W, H), Image.BILINEAR))
    corr_even = np.corrcoef(even_full.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
    print(f"  {name} even rows: corr={corr_even:.4f}")

print("\nDone!")
