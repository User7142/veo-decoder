#!/usr/bin/env python3
"""Detailed analysis of 3-pass decoder output to understand data layout."""

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

pdb_path = "data/3840520404.pdb"
ref_path = "data/Thom_091225_001.JPG"
out_dir = "output/"
records = parse_pdb(pdb_path)
data_records = records[1:-1]
ref_gray = np.array(Image.open(ref_path).convert('L')).astype(float)
ref_rgb = np.array(Image.open(ref_path).convert('RGB')).astype(float)
H, W = 480, 640

# ============================================================================
# Decode using 3-pass structure, but store each pass's data separately
# ============================================================================
pass1_data = np.zeros((H, W//2), dtype=np.uint8)  # row0 odd positions
pass2_data = np.zeros((H, W//2), dtype=np.uint8)  # row1 even positions
pass3_r0_data = np.zeros((H, W//2), dtype=np.uint8)  # row0 even positions
pass3_r1_data = np.zeros((H, W//2), dtype=np.uint8)  # row1 odd positions

# Also store the combined output
output = np.zeros((H, W), dtype=np.uint8)

for rec_idx, rec in enumerate(data_records):
    reader = BitReader(rec)
    for call in range(2):
        r0 = rec_idx * 4 + call * 2
        r1 = r0 + 1
        if r1 >= H:
            break

        # PASS 1: row0 odd [1,3,...,639]
        reader.flush_to_byte()
        lit = reader.read_raw_byte()
        output[r0, 1] = lit
        pass1_data[r0, 0] = lit
        prev = lit
        for c in range(1, W//2):
            col = c * 2 + 1
            val = decode_delta(reader, prev)
            output[r0, col] = val
            pass1_data[r0, c] = val
            prev = val

        # PASS 2: row1 even [0,2,...,638]
        reader.flush_to_byte()
        lit = reader.read_raw_byte()
        output[r1, 0] = lit
        pass2_data[r1, 0] = lit
        prev = lit
        for c in range(1, W//2):
            col = c * 2
            val = decode_delta(reader, prev)
            output[r1, col] = val
            pass2_data[r1, c] = val
            prev = val

        # PASS 3: row0 even + row1 odd (zigzag)
        reader.flush_to_byte()
        lit0 = reader.read_raw_byte()
        output[r0, 0] = lit0
        pass3_r0_data[r0, 0] = lit0

        reader.flush_to_byte()
        lit1 = reader.read_raw_byte()
        output[r1, 1] = lit1
        pass3_r1_data[r1, 0] = lit1

        for c in range(1, W//2):
            col_even = c * 2
            col_odd = c * 2 + 1

            # row0[even] predicted from row1[odd-1]
            pred = int(output[r1, col_even - 1])
            val = decode_delta(reader, pred)
            output[r0, col_even] = val
            pass3_r0_data[r0, c] = val

            # row1[odd] predicted from row0[even]
            if col_odd < W:
                pred = int(output[r0, col_even])
                val = decode_delta(reader, pred)
                output[r1, col_odd] = val
                pass3_r1_data[r1, c] = val

print("=== Pass data statistics ===")
print(f"Pass 1 (row0 odd):  mean={pass1_data.mean():.1f}, std={pass1_data.std():.1f}")
print(f"Pass 2 (row1 even): mean={pass2_data.mean():.1f}, std={pass2_data.std():.1f}")
print(f"Pass 3 row0 even:   mean={pass3_r0_data.mean():.1f}, std={pass3_r0_data.std():.1f}")
print(f"Pass 3 row1 odd:    mean={pass3_r1_data.mean():.1f}, std={pass3_r1_data.std():.1f}")

# ============================================================================
# Key test: which pass data correlates with luminance?
# ============================================================================
print("\n=== Correlation of each pass with reference ===")

# We need to compare 320-wide decoded data against 640-wide reference.
# Two approaches: compare at same pixel positions, or resize.

# Approach: compare pass data against reference at CORRESPONDING pixel positions
# Pass 1 = row0 odd positions = pixels at x=1,3,5,...,639 of row 0
# These correspond to reference pixels at x=1,3,5,...,639

# For fair comparison, extract reference pixels at the same positions
ref_odd_pixels = ref_gray[:, 1::2]  # 480 x 320 - pixels at x=1,3,...
ref_even_pixels = ref_gray[:, 0::2]  # 480 x 320 - pixels at x=0,2,...

# Pass 1: row0 data correlated with ref at same rows (even rows) and odd pixel positions
p1_rows = pass1_data[0::2, :]  # Only even rows of pass1 data (these are r0 from each pair)
ref_p1 = ref_odd_pixels[0::2, :][:p1_rows.shape[0], :]
# But this compares every-other-row of decoded with every-other-row of ref
# Better: compare all rows since we have row assignments

# Pass 1 writes to rows r0 = 0, 2, 4, 6, ... (first row of each pair from each call)
# For call 0 of record i: r0 = i*4, call 1: r0 = i*4+2
# So pass1 rows are: 0, 2, 4, 6, ..., 478
# These should be compared with reference rows 0, 2, 4, ..., 478

p1_valid = pass1_data[0::2, :]  # rows 0,2,4,...
ref_for_p1 = ref_odd_pixels[0::2, :][:p1_valid.shape[0], :]
corr_p1 = np.corrcoef(p1_valid.flatten(), ref_for_p1.flatten())[0, 1]
print(f"Pass 1 (row0 odd) vs ref odd pixels, even rows: corr={corr_p1:.4f}")

# Pass 2 writes to rows r1 = 1, 3, 5, ... (second row of each pair)
p2_valid = pass2_data[1::2, :]
ref_for_p2_even = ref_even_pixels[1::2, :][:p2_valid.shape[0], :]
ref_for_p2_odd = ref_odd_pixels[1::2, :][:p2_valid.shape[0], :]
corr_p2_even = np.corrcoef(p2_valid.flatten(), ref_for_p2_even.flatten())[0, 1]
corr_p2_odd = np.corrcoef(p2_valid.flatten(), ref_for_p2_odd.flatten())[0, 1]
print(f"Pass 2 (row1 even) vs ref even pixels, odd rows: corr={corr_p2_even:.4f}")
print(f"Pass 2 (row1 even) vs ref odd pixels, odd rows: corr={corr_p2_odd:.4f}")

# Pass 3 row0 even
p3r0_valid = pass3_r0_data[0::2, :]
ref_for_p3r0_even = ref_even_pixels[0::2, :][:p3r0_valid.shape[0], :]
ref_for_p3r0_odd = ref_odd_pixels[0::2, :][:p3r0_valid.shape[0], :]
corr_p3r0_even = np.corrcoef(p3r0_valid.flatten(), ref_for_p3r0_even.flatten())[0, 1]
corr_p3r0_odd = np.corrcoef(p3r0_valid.flatten(), ref_for_p3r0_odd.flatten())[0, 1]
print(f"Pass 3 r0 even vs ref even pixels, even rows: corr={corr_p3r0_even:.4f}")
print(f"Pass 3 r0 even vs ref odd pixels, even rows: corr={corr_p3r0_odd:.4f}")

# Pass 3 row1 odd
p3r1_valid = pass3_r1_data[1::2, :]
ref_for_p3r1_even = ref_even_pixels[1::2, :][:p3r1_valid.shape[0], :]
ref_for_p3r1_odd = ref_odd_pixels[1::2, :][:p3r1_valid.shape[0], :]
corr_p3r1_even = np.corrcoef(p3r1_valid.flatten(), ref_for_p3r1_even.flatten())[0, 1]
corr_p3r1_odd = np.corrcoef(p3r1_valid.flatten(), ref_for_p3r1_odd.flatten())[0, 1]
print(f"Pass 3 r1 odd vs ref even pixels, odd rows: corr={corr_p3r1_even:.4f}")
print(f"Pass 3 r1 odd vs ref odd pixels, odd rows: corr={corr_p3r1_odd:.4f}")

# ============================================================================
# The key question: is Pass 1 data luminance or chrominance?
# If luminance: high corr with gray reference
# If chrominance: low corr with gray, might correlate with R-B difference
# ============================================================================
print("\n=== Is the data Y or CbCr? ===")

# If UYVY: odd positions = Y, even positions = CbCr
# In our 3-pass: Pass 1 = row0 odd, so should be Y
# Let's check: do Pass 1 values look like luminance?
print(f"Pass 1 value range: [{pass1_data.min()}, {pass1_data.max()}], "
      f"mean={pass1_data.mean():.1f}")
print(f"Reference gray range: [{ref_gray.min():.0f}, {ref_gray.max():.0f}], "
      f"mean={ref_gray.mean():.1f}")

# Try linear regression: ref_gray = a * pass1 + b
# Use only valid rows
x = p1_valid.flatten().astype(float)
y = ref_for_p1.flatten().astype(float)
A = np.vstack([x, np.ones(len(x))]).T
result = np.linalg.lstsq(A, y, rcond=None)
slope, intercept = result[0]
print(f"\nLinear fit: ref_gray = {slope:.3f} * pass1 + {intercept:.1f}")
print(f"  (For Y→gray: expect slope≈1.164, intercept≈-18.6)")

# Also check Pass 2
x2 = p2_valid.flatten().astype(float)
y2 = ref_for_p2_even.flatten().astype(float)
A2 = np.vstack([x2, np.ones(len(x2))]).T
result2 = np.linalg.lstsq(A2, y2, rcond=None)
slope2, intercept2 = result2[0]
print(f"Linear fit: ref_gray = {slope2:.3f} * pass2 + {intercept2:.1f}")

# ============================================================================
# Try building an image from just the Y values (interpreting as UYVY)
# Y values are at odd positions: pass1 (r0) and pass3_r1 (r1)
# ============================================================================
print("\n=== Y-only image (UYVY interpretation) ===")

y_image = np.zeros((H, W//2), dtype=np.uint8)
for row in range(H):
    if row % 2 == 0:
        # Even row (r0): Y from Pass 1 (odd positions)
        y_image[row, :] = pass1_data[row, :]
    else:
        # Odd row (r1): Y from Pass 3 (odd positions)
        y_image[row, :] = pass3_r1_data[row, :]

y_resized = np.array(Image.fromarray(y_image, 'L').resize((640, 480), Image.BILINEAR))
corr_y = np.corrcoef(y_resized.flatten(), ref_gray.flatten())[0, 1]
mse_y = np.mean((y_resized - ref_gray)**2)
psnr_y = 10 * np.log10(255**2 / mse_y)
print(f"Y-only (UYVY odd pos, 320→640): corr={corr_y:.4f}, PSNR={psnr_y:.2f}dB")
Image.fromarray(y_resized.astype(np.uint8), 'L').save(
    out_dir + "v7_y_only_uyvy.png")

# Also try Y with BT.601 scaling
y_scaled = np.clip((y_image.astype(float) - 16) * (255.0/219.0), 0, 255).astype(np.uint8)
y_scaled_resized = np.array(Image.fromarray(y_scaled, 'L').resize((640, 480), Image.BILINEAR))
corr_ys = np.corrcoef(y_scaled_resized.flatten(), ref_gray.flatten())[0, 1]
mse_ys = np.mean((y_scaled_resized.astype(float) - ref_gray)**2)
psnr_ys = 10 * np.log10(255**2 / mse_ys)
print(f"Y-only (BT.601 scaled): corr={corr_ys:.4f}, PSNR={psnr_ys:.2f}dB")

# Try using the linear regression result for optimal scaling
y_opt = np.clip(y_image.astype(float) * slope + intercept, 0, 255).astype(np.uint8)
y_opt_r = np.array(Image.fromarray(y_opt, 'L').resize((640, 480), Image.BILINEAR))
corr_yo = np.corrcoef(y_opt_r.flatten(), ref_gray.flatten())[0, 1]
mse_yo = np.mean((y_opt_r.astype(float) - ref_gray)**2)
psnr_yo = 10 * np.log10(255**2 / mse_yo)
print(f"Y-only (optimal scaled): corr={corr_yo:.4f}, PSNR={psnr_yo:.2f}dB")
Image.fromarray(y_opt_r.astype(np.uint8), 'L').save(
    out_dir + "v7_y_optimal_scaled.png")

# ============================================================================
# Check if the data layout might be different from what we assumed
# What if Pass 1 is NOT at odd positions but at specific UYVY byte positions?
# In UYVY: byte 0=Cb, 1=Y0, 2=Cr, 3=Y1
# So Y values are at positions 1, 3, 5, 7, ... → these ARE odd positions
# CbCr values are at positions 0, 2, 4, 6, ... → these ARE even positions
# Our interpretation seems correct.
# ============================================================================

# ============================================================================
# Try the alternative: treat the data as raw grayscale (not UYVY)
# If each byte is a pixel, we have 640x480 grayscale at 640 pixels wide
# But then what explains the different statistics at even/odd positions?
# ============================================================================
print("\n=== Raw grayscale interpretation ===")
output_resized = np.array(Image.fromarray(output, 'L').resize((640, 480), Image.NEAREST))
corr_raw = np.corrcoef(output_resized.flatten(), ref_gray.flatten())[0, 1]
print(f"Raw output as grayscale: corr={corr_raw:.4f}")

# Check: what if the width is actually 1280 and we're only getting half the data?
# Then our 640-byte rows are actually half-rows, and we need to interleave them
# For now, just look at the visual

# ============================================================================
# FINAL: Build UYVY image and convert to RGB
# ============================================================================
print("\n=== UYVY → RGB Conversion ===")

# Scale Y values using the regression slope to better match reference
rgb_final = np.zeros((H, W//2, 3), dtype=np.uint8)
for y in range(H):
    for x in range(0, W//2, 2):
        bo = x * 2
        if bo + 3 >= W:
            break
        cb = int(output[y, bo]) - 128
        y0 = int(output[y, bo + 1]) - 16
        cr = int(output[y, bo + 2]) - 128
        y1 = int(output[y, bo + 3]) - 16

        r0 = max(0, min(255, (y0 * 1192 + cr * 1634) >> 10))
        g0 = max(0, min(255, (y0 * 1192 - cr * 832 - cb * 400) >> 10))
        b0 = max(0, min(255, (y0 * 1192 + cb * 2066) >> 10))

        r1 = max(0, min(255, (y1 * 1192 + cr * 1634) >> 10))
        g1 = max(0, min(255, (y1 * 1192 - cr * 832 - cb * 400) >> 10))
        b1 = max(0, min(255, (y1 * 1192 + cb * 2066) >> 10))

        rgb_final[y, x] = [r0, g0, b0]
        rgb_final[y, x+1] = [r1, g1, b1]

rgb_640 = np.array(Image.fromarray(rgb_final, 'RGB').resize((640, 480), Image.BILINEAR))
mse_rgb = np.mean((rgb_640.astype(float) - ref_rgb)**2)
psnr_rgb = 10 * np.log10(255**2 / mse_rgb)
corr_rgb = np.corrcoef(rgb_640.flatten(), ref_rgb.flatten())[0, 1]
print(f"UYVY→RGB: PSNR={psnr_rgb:.2f}dB, corr={corr_rgb:.4f}")
Image.fromarray(rgb_640, 'RGB').save(out_dir + "v7_uyvy_final.jpg", quality=62)

# Also try with gamma correction
gamma_lut = np.array([int(np.clip(np.power(i/255.0, 0.45)*255, 0, 255))
                       for i in range(256)], dtype=np.uint8)
rgb_gamma = gamma_lut[rgb_640]
mse_rg = np.mean((rgb_gamma.astype(float) - ref_rgb)**2)
psnr_rg = 10 * np.log10(255**2 / mse_rg)
corr_rg = np.corrcoef(rgb_gamma.flatten(), ref_rgb.flatten())[0, 1]
print(f"UYVY→RGB→Gamma: PSNR={psnr_rg:.2f}dB, corr={corr_rg:.4f}")
Image.fromarray(rgb_gamma, 'RGB').save(out_dir + "v7_uyvy_gamma.jpg", quality=62)

print("\nDone!")
