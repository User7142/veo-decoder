#!/usr/bin/env python3
"""V7: Per-record 3-pass decoder based on COMConduit.dll disassembly.

Each PDB record is an independent bitstream, containing data for either:
  A) 1 call with width=1280 (2 rows of 1280 bytes → 640 UYVY pixels)
  B) 2 calls with width=640 (4 rows of 640 bytes each)

We test both interpretations.
"""

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

    def read_raw_byte(self):
        if self.byte_pos >= len(self.data):
            return 0
        val = self.data[self.byte_pos]
        self.byte_pos += 1
        return val

    def bits_remaining(self):
        return (len(self.data) - self.byte_pos) * 8 + self.bit_pos - 6

    def bits_consumed(self):
        return self.byte_pos * 8 + (7 - self.bit_pos)


def decode_delta(reader, prev):
    flag = reader.read_bit()
    if flag == 1:
        return prev & 0xFF
    code = reader.read_bits(5)
    if code == 0:
        return reader.read_bits(8) & 0xFF
    sign = (code >> 4) & 1
    mag = code & 0x0F
    if sign:
        return (prev - mag) & 0xFF
    else:
        return (prev + mag) & 0xFF


def decode_linear(data, count, prev=128):
    """Simple linear decode (no pass structure) for comparison."""
    reader = BitReader(data)
    out = []
    for _ in range(count):
        val = decode_delta(reader, prev)
        out.append(val)
        prev = val
    return out, reader.bits_consumed()


def decode_3pass(reader, output, r0, r1, width):
    """Exact 3-pass structure from fcn.100029c0 disassembly.

    Pass 1: row0 odd positions [1,3,...,width-1] — horizontal prediction
    Pass 2: row1 even positions [0,2,...,width-2] — horizontal prediction
    Pass 3: row0 even [0,2,...] + row1 odd [1,3,...] — zigzag cross-row prediction
    """
    # PASS 1: byte-align, literal at row0[1], then delta for row0[3,5,...,width-1]
    reader.flush_to_byte()
    lit = reader.read_raw_byte()
    output[r0, 1] = lit
    prev = lit
    for col in range(3, width, 2):
        val = decode_delta(reader, prev)
        output[r0, col] = val
        prev = val

    # PASS 2: byte-align, literal at row1[0], then delta for row1[2,4,...,width-2]
    reader.flush_to_byte()
    lit = reader.read_raw_byte()
    output[r1, 0] = lit
    prev = lit
    for col in range(2, width, 2):
        val = decode_delta(reader, prev)
        output[r1, col] = val
        prev = val

    # PASS 3: two literals + zigzag
    reader.flush_to_byte()
    lit0 = reader.read_raw_byte()
    output[r0, 0] = lit0

    reader.flush_to_byte()
    lit1 = reader.read_raw_byte()
    output[r1, 1] = lit1

    # Zigzag: row0[2] from row1[1], row1[3] from row0[2], row0[4] from row1[3], ...
    for col in range(2, width, 2):
        pred = int(output[r1, col - 1])
        val = decode_delta(reader, pred)
        output[r0, col] = val
        if col + 1 < width:
            pred = int(output[r0, col])
            val = decode_delta(reader, pred)
            output[r1, col + 1] = val


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


def uyvy_to_rgb_row(row_data, pixel_width):
    """Convert one UYVY row to RGB."""
    rgb = np.zeros((pixel_width, 3), dtype=np.uint8)
    for x in range(0, pixel_width, 2):
        bo = x * 2
        if bo + 3 >= len(row_data):
            break
        cb = int(row_data[bo]) - 128
        y0 = int(row_data[bo + 1]) - 16
        cr = int(row_data[bo + 2]) - 128
        y1 = int(row_data[bo + 3]) - 16
        rgb[x] = [max(0, min(255, (y0*1192 + cr*1634) >> 10)),
                   max(0, min(255, (y0*1192 - cr*832 - cb*400) >> 10)),
                   max(0, min(255, (y0*1192 + cb*2066) >> 10))]
        rgb[x+1] = [max(0, min(255, (y1*1192 + cr*1634) >> 10)),
                     max(0, min(255, (y1*1192 - cr*832 - cb*400) >> 10)),
                     max(0, min(255, (y1*1192 + cb*2066) >> 10))]
    return rgb


# ============================================================================
pdb_path = "data/3840520404.pdb"
out_dir = "output/"
ref_path = "data/Thom_091225_001.JPG"
records = parse_pdb(pdb_path)
data_records = records[1:-1]
ref_gray = np.array(Image.open(ref_path).convert('L'))
ref_rgb = np.array(Image.open(ref_path).convert('RGB'))

print("=" * 70)
print("VEO Decoder V7 - Per-Record 3-Pass Structure Tests")
print("=" * 70)
print(f"Records: {len(data_records)}, Ref: {ref_gray.shape}")

# ============================================================================
# Strategy A: Per-record, 2 calls per record, width=640, 4 output rows
# ============================================================================
print("\n=== Strategy A: 2 calls/record, width=640, 4 rows/record ===")
H, W = 480, 640
output_a = np.zeros((H, W), dtype=np.uint8)
total_bits_a = 0

for i, rec in enumerate(data_records):
    reader = BitReader(rec)
    base_row = i * 4
    if base_row + 3 >= H:
        break
    decode_3pass(reader, output_a, base_row, base_row + 1, W)
    decode_3pass(reader, output_a, base_row + 2, base_row + 3, W)
    total_bits_a += reader.bits_consumed()

# Analyze
y_a = output_a[:, 1::2]  # Y channel (odd positions)
y_a_r = np.array(Image.fromarray(y_a, 'L').resize((640, 480), Image.BILINEAR))
corr_a = np.corrcoef(y_a_r.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  Y channel corr: {corr_a:.4f}")
print(f"  Y mean={y_a.mean():.1f}, std={y_a.std():.1f}")
Image.fromarray(output_a, 'L').save(os.path.join(out_dir, "v7_a_raw.png"))

# Full output correlation (grayscale comparison at 640x480)
full_a = np.array(Image.fromarray(output_a, 'L').resize((640, 480), Image.BILINEAR))
corr_a_full = np.corrcoef(full_a.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  Full output corr: {corr_a_full:.4f}")

# Check row-type correlations
for t in range(4):
    rows_t = output_a[t::4, :]
    rows_r = np.array(Image.fromarray(rows_t, 'L').resize((640, 480), Image.BILINEAR))
    c = np.corrcoef(rows_r.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
    print(f"  Row type {t}: corr={c:.4f}, mean={rows_t.mean():.1f}, std={rows_t.std():.1f}")


# ============================================================================
# Strategy B: Per-record, 1 call per record, width=640, 2 output rows → 240 rows
# ============================================================================
print("\n=== Strategy B: 1 call/record, width=640, 2 rows/record (240 rows total) ===")
output_b = np.zeros((240, W), dtype=np.uint8)

for i, rec in enumerate(data_records):
    reader = BitReader(rec)
    r0 = i * 2
    r1 = i * 2 + 1
    if r1 >= 240:
        break
    decode_3pass(reader, output_b, r0, r1, W)

y_b = output_b[:, 1::2]
y_b_r = np.array(Image.fromarray(y_b, 'L').resize((640, 480), Image.BILINEAR))
corr_b = np.corrcoef(y_b_r.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  Y channel corr: {corr_b:.4f}")
Image.fromarray(output_b, 'L').save(os.path.join(out_dir, "v7_b_raw.png"))

for t in range(2):
    rows_t = output_b[t::2, :]
    rows_r = np.array(Image.fromarray(rows_t, 'L').resize((640, 480), Image.BILINEAR))
    c = np.corrcoef(rows_r.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
    print(f"  Row type {t}: corr={c:.4f}, mean={rows_t.mean():.1f}, std={rows_t.std():.1f}")


# ============================================================================
# Strategy C: Per-record, linear decode (baseline), 4 rows of 640
# ============================================================================
print("\n=== Strategy C: Linear decode (baseline), 4 rows/record ===")
output_c = np.zeros((H, W), dtype=np.uint8)

for i, rec in enumerate(data_records):
    vals, _ = decode_linear(rec, W * 4)
    base = i * 4
    for r in range(4):
        row_start = r * W
        if base + r < H:
            for c in range(W):
                output_c[base + r, c] = vals[row_start + c]

for t in range(4):
    rows_t = output_c[t::4, :]
    rows_r = np.array(Image.fromarray(rows_t, 'L').resize((640, 480), Image.BILINEAR))
    c = np.corrcoef(rows_r.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
    print(f"  Row type {t}: corr={c:.4f}, mean={rows_t.mean():.1f}, std={rows_t.std():.1f}")


# ============================================================================
# Strategy D: Per-record 3-pass but treating row0=odd, row1=even differently
# What if the pass structure is actually:
#   Pass 1: odd cols (Y), row 0 → writes at output+1, stride 2
#   Pass 2: odd cols (Y), row 1 → writes at output+stride+1, stride 2
#   Pass 3: even cols (CbCr), with literals and zigzag
# The key difference: Passes 1+2 are both "Y" data, Pass 3 is "CbCr"
# ============================================================================
print("\n=== Strategy D: Swapped pass order (Y first for both rows, then CbCr) ===")

def decode_3pass_v2(reader, output, r0, r1, width):
    """Alternative: Pass 1 = Y row0, Pass 2 = Y row1, Pass 3 = CbCr both"""
    # PASS 1: row0 odd positions (Y of row 0)
    reader.flush_to_byte()
    lit = reader.read_raw_byte()
    output[r0, 1] = lit
    prev = lit
    for col in range(3, width, 2):
        val = decode_delta(reader, prev)
        output[r0, col] = val
        prev = val

    # PASS 2: row1 odd positions (Y of row 1) — NOT even!
    reader.flush_to_byte()
    lit = reader.read_raw_byte()
    output[r1, 1] = lit
    prev = lit
    for col in range(3, width, 2):
        val = decode_delta(reader, prev)
        output[r1, col] = val
        prev = val

    # PASS 3: CbCr for both rows (even positions)
    reader.flush_to_byte()
    lit0 = reader.read_raw_byte()
    output[r0, 0] = lit0

    reader.flush_to_byte()
    lit1 = reader.read_raw_byte()
    output[r1, 0] = lit1

    # Zigzag: row0[2] from row1[0], row1[2] from row0[2], ...
    for col in range(2, width, 2):
        pred = int(output[r1, col - 2])
        val = decode_delta(reader, pred)
        output[r0, col] = val
        pred = int(output[r0, col])
        val = decode_delta(reader, pred)
        output[r1, col] = val

output_d = np.zeros((H, W), dtype=np.uint8)
for i, rec in enumerate(data_records):
    reader = BitReader(rec)
    base_row = i * 4
    if base_row + 3 >= H:
        break
    decode_3pass_v2(reader, output_d, base_row, base_row + 1, W)
    decode_3pass_v2(reader, output_d, base_row + 2, base_row + 3, W)

for t in range(4):
    rows_t = output_d[t::4, :]
    rows_r = np.array(Image.fromarray(rows_t, 'L').resize((640, 480), Image.BILINEAR))
    c = np.corrcoef(rows_r.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
    print(f"  Row type {t}: corr={c:.4f}, mean={rows_t.mean():.1f}, std={rows_t.std():.1f}")


# ============================================================================
# Strategy E: What if the disassembly actually shows:
# Pass 1: row0 odd → Y of row 0
# Pass 2: row1 even → CbCr of row 1
# Pass 3: row0 even + row1 odd → CbCr of row 0 + Y of row 1
# Then: Y is in row0[odd] and row1[odd], CbCr in row0[even] and row1[even]
# So the Y values for the full image at 640 bytes width = 320 Y per row
# But we have data for 480 rows → 480 × 320 = 153,600 Y values
# Upscale by 2 → 640×480
# ============================================================================
print("\n=== Strategy E: Extract Y from rows with 3-pass data ===")

# Use strategy A output (3-pass per-record with 2 calls)
# Even rows (r0 from each pair): Y at odd positions
# Odd rows (r1 from each pair): Y at odd positions
y_all = output_a[:, 1::2]  # All Y values: 480 rows × 320 cols
y_all_r = np.array(Image.fromarray(y_all, 'L').resize((640, 480), Image.BILINEAR))
corr_y = np.corrcoef(y_all_r.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  All rows Y (320→640): corr={corr_y:.4f}")
Image.fromarray(y_all_r, 'L').save(os.path.join(out_dir, "v7_y_all.png"))

# Even positions (CbCr)
cbcr = output_a[:, 0::2]
print(f"  CbCr: mean={cbcr.mean():.1f}, std={cbcr.std():.1f}")

# Check if CbCr centers around 128 (expected for chrominance)
cb = cbcr[:, 0::2]  # Every other even position → Cb
cr = cbcr[:, 1::2]  # Every other even position → Cr
print(f"  Cb channel: mean={cb.mean():.1f}, std={cb.std():.1f}")
print(f"  Cr channel: mean={cr.mean():.1f}, std={cr.std():.1f}")


# ============================================================================
# Now try full UYVY → RGB with strategy A output
# ============================================================================
print("\n=== Full UYVY → RGB from Strategy A ===")

# UYVY: [Cb, Y0, Cr, Y1, Cb, Y0, Cr, Y1, ...]
rgb_uyvy = np.zeros((480, 320, 3), dtype=np.uint8)
for y in range(480):
    for x in range(0, 320, 2):
        bo = x * 2
        if bo + 3 >= 640:
            break
        cb_val = int(output_a[y, bo]) - 128
        y0_val = int(output_a[y, bo+1]) - 16
        cr_val = int(output_a[y, bo+2]) - 128
        y1_val = int(output_a[y, bo+3]) - 16

        rgb_uyvy[y, x] = [max(0,min(255,(y0_val*1192+cr_val*1634)>>10)),
                           max(0,min(255,(y0_val*1192-cr_val*832-cb_val*400)>>10)),
                           max(0,min(255,(y0_val*1192+cb_val*2066)>>10))]
        rgb_uyvy[y, x+1] = [max(0,min(255,(y1_val*1192+cr_val*1634)>>10)),
                              max(0,min(255,(y1_val*1192-cr_val*832-cb_val*400)>>10)),
                              max(0,min(255,(y1_val*1192+cb_val*2066)>>10))]

rgb_uyvy_640 = np.array(Image.fromarray(rgb_uyvy, 'RGB').resize((640, 480), Image.BILINEAR))
mse = np.mean((rgb_uyvy_640.astype(float) - ref_rgb.astype(float))**2)
psnr = 10 * np.log10(255**2/mse)
corr = np.corrcoef(rgb_uyvy_640.flatten().astype(float), ref_rgb.flatten().astype(float))[0,1]
print(f"  UYVY→RGB (320→640): PSNR={psnr:.2f}dB, corr={corr:.4f}")
Image.fromarray(rgb_uyvy_640, 'RGB').save(os.path.join(out_dir, "v7_uyvy_rgb.jpg"), quality=62)


# ============================================================================
# Compare all strategies visually
# ============================================================================
print("\n=== Summary ===")
print("Strategy A (2 calls/rec, 3-pass): see v7_a_raw.png")
print("Strategy B (1 call/rec, 3-pass): see v7_b_raw.png")
print("Strategy C (linear baseline): known corr ~0.85 for odd rows")

# Save the best grayscale result
best_y = y_all_r
Image.fromarray(best_y, 'L').save(os.path.join(out_dir, "v7_best_gray.png"))

print("\nDone!")
