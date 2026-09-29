#!/usr/bin/env python3
"""Identify which Bayer color channel each decoded row represents."""

import struct, numpy as np
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

ref_rgb = np.array(Image.open(ref_path).convert('RGB'))
ref_r = ref_rgb[:,:,0].astype(float)
ref_g = ref_rgb[:,:,1].astype(float)
ref_b = ref_rgb[:,:,2].astype(float)
ref_gray = np.array(Image.open(ref_path).convert('L')).astype(float)

# Decode per-record, get 4 half-rows per record
all_data = bytearray()
for rec in data_records:
    all_data.extend(decode_linear(rec, W * 4))
raw = np.frombuffer(bytes(all_data), dtype=np.uint8).reshape(H, W)

# Separate 4 types of rows (within each 4-row group)
row_types = [[], [], [], []]
for g in range(H // 4):
    for t in range(4):
        row_types[t].append(raw[g*4 + t])

for t in range(4):
    row_types[t] = np.array(row_types[t])  # Shape: (120, 640)

# Compare each row type against each reference color channel
print("=== Row type vs reference channel correlation ===")
print(f"{'':20s} {'R':>8s} {'G':>8s} {'B':>8s} {'Gray':>8s} {'Mean':>8s} {'Std':>8s}")

for t in range(4):
    rows = row_types[t]  # 120 rows of 640
    # Downsample reference to 120 rows for comparison
    ref_r_half = ref_r[t::4, :][:120, :]
    ref_g_half = ref_g[t::4, :][:120, :]
    ref_b_half = ref_b[t::4, :][:120, :]
    ref_gray_half = ref_gray[t::4, :][:120, :]
    
    cr = np.corrcoef(rows.flatten(), ref_r_half.flatten())[0,1]
    cg = np.corrcoef(rows.flatten(), ref_g_half.flatten())[0,1]
    cb = np.corrcoef(rows.flatten(), ref_b_half.flatten())[0,1]
    cgray = np.corrcoef(rows.flatten(), ref_gray_half.flatten())[0,1]
    
    print(f"  Row type {t} (rows {t},+4,...): {cr:+.4f}  {cg:+.4f}  {cb:+.4f}  {cgray:+.4f}  "
          f"mean={rows.mean():.1f}  std={rows.std():.1f}")

# Now try: each row type compared against all rows of each reference channel
print("\n=== Row type vs ALL reference rows ===")
for t in range(4):
    rows = row_types[t]
    rows_full = np.array(Image.fromarray(rows.astype(np.uint8), 'L').resize((W, H), Image.BILINEAR)).astype(float)
    
    cr = np.corrcoef(rows_full.flatten(), ref_r.flatten())[0,1]
    cg = np.corrcoef(rows_full.flatten(), ref_g.flatten())[0,1]
    cb = np.corrcoef(rows_full.flatten(), ref_b.flatten())[0,1]
    cgray = np.corrcoef(rows_full.flatten(), ref_gray.flatten())[0,1]
    
    print(f"  Row type {t}: R={cr:+.4f}  G={cg:+.4f}  B={cb:+.4f}  Gray={cgray:+.4f}")

# With global shift 633
print("\n=== With shift 633 applied ===")
for t in range(4):
    rows = np.roll(row_types[t], 633, axis=1)
    rows_full = np.array(Image.fromarray(rows.astype(np.uint8), 'L').resize((W, H), Image.BILINEAR)).astype(float)
    
    cr = np.corrcoef(rows_full.flatten(), ref_r.flatten())[0,1]
    cg = np.corrcoef(rows_full.flatten(), ref_g.flatten())[0,1]
    cb = np.corrcoef(rows_full.flatten(), ref_b.flatten())[0,1]
    cgray = np.corrcoef(rows_full.flatten(), ref_gray.flatten())[0,1]
    
    print(f"  Row type {t} (shift 633): R={cr:+.4f}  G={cg:+.4f}  B={cb:+.4f}  Gray={cgray:+.4f}")

# Try splitting each 640-value row into two 320-value halves
# and see which half correlates with which channel
print("\n=== First/second half of each row type ===")
for t in range(4):
    first_half = row_types[t][:, :320]   # 120 x 320
    second_half = row_types[t][:, 320:]  # 120 x 320
    
    # Compare with 320-wide reference subsets
    ref_r_320a = ref_r[t::4, :320][:120, :]
    ref_g_320a = ref_g[t::4, :320][:120, :]
    ref_b_320a = ref_b[t::4, :320][:120, :]
    
    ref_r_320b = ref_r[t::4, 320:][:120, :]
    ref_g_320b = ref_g[t::4, 320:][:120, :]
    ref_b_320b = ref_b[t::4, 320:][:120, :]
    
    print(f"  Row type {t}, first 320:")
    print(f"    mean={first_half.mean():.1f}  "
          f"R={np.corrcoef(first_half.flatten(), ref_r_320a.flatten())[0,1]:+.4f}  "
          f"G={np.corrcoef(first_half.flatten(), ref_g_320a.flatten())[0,1]:+.4f}  "
          f"B={np.corrcoef(first_half.flatten(), ref_b_320a.flatten())[0,1]:+.4f}")
    print(f"  Row type {t}, second 320:")
    print(f"    mean={second_half.mean():.1f}  "
          f"R={np.corrcoef(second_half.flatten(), ref_r_320b.flatten())[0,1]:+.4f}  "
          f"G={np.corrcoef(second_half.flatten(), ref_g_320b.flatten())[0,1]:+.4f}  "
          f"B={np.corrcoef(second_half.flatten(), ref_b_320b.flatten())[0,1]:+.4f}")

# Build color image using row types as different channels
print("\n=== Color reconstruction attempts ===")
# Attempt: row_type 1 = G (highest gray correlation),
# split row types 0 and 2 into R and B based on correlation
for r_type, b_type in [(0, 2), (2, 0)]:
    g_rows = np.roll(row_types[1], 633, axis=1)
    r_rows = np.roll(row_types[r_type], 633, axis=1)
    b_rows = np.roll(row_types[b_type], 633, axis=1)
    
    rgb = np.zeros((120, W, 3), dtype=np.uint8)
    rgb[:,:,0] = r_rows
    rgb[:,:,1] = g_rows
    rgb[:,:,2] = b_rows
    
    rgb_full = np.array(Image.fromarray(rgb, 'RGB').resize((W, H), Image.BILINEAR))
    
    mse = np.mean((rgb_full.astype(float) - ref_rgb.astype(float))**2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 0
    
    fname = f"color_R{r_type}_G1_B{b_type}.jpg"
    Image.fromarray(rgb_full, 'RGB').save(os.path.join(out_dir, fname), quality=62)
    print(f"  R=type{r_type}, G=type1, B=type{b_type}: PSNR={psnr:.2f}dB")

# Also try with gamma correction on all channels
gamm_lut = np.zeros(256, dtype=np.uint8)
for i in range(256):
    gamm_lut[i] = int(np.clip(np.power(i / 255.0, 0.45) * 255, 0, 255))

g_rows = gamm_lut[np.roll(row_types[1], 633, axis=1)]
r_rows = gamm_lut[np.roll(row_types[0], 633, axis=1)]
b_rows = gamm_lut[np.roll(row_types[2], 633, axis=1)]

rgb_gamma = np.zeros((120, W, 3), dtype=np.uint8)
rgb_gamma[:,:,0] = r_rows
rgb_gamma[:,:,1] = g_rows
rgb_gamma[:,:,2] = b_rows

rgb_gamma_full = np.array(Image.fromarray(rgb_gamma, 'RGB').resize((W, H), Image.BILINEAR))
mse = np.mean((rgb_gamma_full.astype(float) - ref_rgb.astype(float))**2)
psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 0
Image.fromarray(rgb_gamma_full, 'RGB').save(os.path.join(out_dir, "color_gamma_R0_G1_B2.jpg"), quality=62)
print(f"  Gamma R=type0, G=type1, B=type2: PSNR={psnr:.2f}dB")

print("\nDone!")
