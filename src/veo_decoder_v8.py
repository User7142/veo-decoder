#!/usr/bin/env python3
"""V8: Bayer raw decoder — treats 3-pass output as raw Bayer sensor data.

Discovery: The 3 passes produce data at very different scales:
  Pass 1 (row0 odd): mean=20, needs ~4.8x gain → likely Blue pixels
  Pass 2 (row1 even): mean=101, needs ~1.0x gain → likely Red pixels
  Pass 3 r0 even: mean=64, needs ~1.7x gain → likely Green (Gb)
  Pass 3 r1 odd: mean=64, needs ~1.7x gain → likely Green (Gr)

This implies GBRG Bayer pattern:
  Row 0: Gb B Gb B Gb B ...  (even=Gb, odd=B)
  Row 1: R  Gr R  Gr R  Gr ...(even=R, odd=Gr)
"""

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


def decode_3pass_record(rec, output, base_row, W):
    """Decode one record (2 calls = 4 rows) using 3-pass structure."""
    reader = BitReader(rec)

    for call in range(2):
        r0 = base_row + call * 2
        r1 = r0 + 1
        if r1 >= output.shape[0]:
            break

        # Pass 1: row0 odd positions
        reader.flush_to_byte()
        lit = reader.read_raw_byte()
        output[r0, 1] = lit
        prev = lit
        for c in range(1, W // 2):
            val = decode_delta(reader, prev)
            output[r0, c * 2 + 1] = val
            prev = val

        # Pass 2: row1 even positions
        reader.flush_to_byte()
        lit = reader.read_raw_byte()
        output[r1, 0] = lit
        prev = lit
        for c in range(1, W // 2):
            val = decode_delta(reader, prev)
            output[r1, c * 2] = val
            prev = val

        # Pass 3: row0 even + row1 odd (zigzag)
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


def bayer_demosaic_simple(raw, pattern='GBRG'):
    """Simple bilinear Bayer demosaicing."""
    H, W = raw.shape
    rgb = np.zeros((H, W, 3), dtype=np.float64)
    raw_f = raw.astype(np.float64)

    # Pad for border handling
    padded = np.pad(raw_f, 1, mode='edge')

    for y in range(H):
        for x in range(W):
            py, px = y + 1, x + 1  # padded coordinates

            # Determine which color this pixel is
            if pattern == 'GBRG':
                if y % 2 == 0 and x % 2 == 0:
                    color = 'Gb'  # Green in blue row
                elif y % 2 == 0 and x % 2 == 1:
                    color = 'B'
                elif y % 2 == 1 and x % 2 == 0:
                    color = 'R'
                else:
                    color = 'Gr'  # Green in red row

            if color == 'Gb' or color == 'Gr':
                # Green pixel
                rgb[y, x, 1] = raw_f[y, x]  # Green is known
                # Red: average of neighbors
                if color == 'Gb':
                    # Gb: R is at (y+1,x) and (y-1,x) for vertical
                    rgb[y, x, 0] = (padded[py-1, px] + padded[py+1, px]) / 2
                    # B is at (y,x+1) and (y,x-1) for horizontal
                    rgb[y, x, 2] = (padded[py, px-1] + padded[py, px+1]) / 2
                else:  # Gr
                    # R is at (y,x-1) and (y,x+1)
                    rgb[y, x, 0] = (padded[py, px-1] + padded[py, px+1]) / 2
                    # B is at (y-1,x) and (y+1,x)
                    rgb[y, x, 2] = (padded[py-1, px] + padded[py+1, px]) / 2

            elif color == 'R':
                rgb[y, x, 0] = raw_f[y, x]  # Red is known
                # Green: average of 4 neighbors
                rgb[y, x, 1] = (padded[py-1, px] + padded[py+1, px] +
                                padded[py, px-1] + padded[py, px+1]) / 4
                # Blue: average of 4 diagonal neighbors
                rgb[y, x, 2] = (padded[py-1, px-1] + padded[py-1, px+1] +
                                padded[py+1, px-1] + padded[py+1, px+1]) / 4

            elif color == 'B':
                rgb[y, x, 2] = raw_f[y, x]  # Blue is known
                # Green: average of 4 neighbors
                rgb[y, x, 1] = (padded[py-1, px] + padded[py+1, px] +
                                padded[py, px-1] + padded[py, px+1]) / 4
                # Red: average of 4 diagonal neighbors
                rgb[y, x, 0] = (padded[py-1, px-1] + padded[py-1, px+1] +
                                padded[py+1, px-1] + padded[py+1, px+1]) / 4

    return rgb


# ============================================================================
# Main
# ============================================================================
pdb_path = "data/3840520404.pdb"
out_dir = "output/"
ref_path = "data/Thom_091225_001.JPG"
records = parse_pdb(pdb_path)
data_records = records[1:-1]
ref_gray = np.array(Image.open(ref_path).convert('L')).astype(float)
ref_rgb = np.array(Image.open(ref_path).convert('RGB')).astype(float)
H, W = 480, 640

print("=" * 60)
print("VEO Decoder V8 - Bayer Raw Interpretation")
print("=" * 60)

# Step 1: Decode all records using 3-pass structure
raw = np.zeros((H, W), dtype=np.uint8)
for i, rec in enumerate(data_records):
    base = i * 4
    if base + 3 >= H:
        break
    decode_3pass_record(rec, raw, base, W)

Image.fromarray(raw, 'L').save(os.path.join(out_dir, "v8_raw_bayer.png"))
print(f"Raw decoded: {H}x{W}")

# Step 2: Identify optimal white balance gains
# From regression analysis:
# Pass 1 (row0 odd = Blue):   scale=4.78, offset=12.4
# Pass 2 (row1 even = Red):   scale=1.02, offset=2.9
# Pass 3 r0 even (Gb):        scale=1.71, offset=-2.2
# Pass 3 r1 odd (Gr):         scale=1.66, offset=0.7

# Test multiple Bayer patterns and white balance gains
patterns = ['GBRG', 'RGGB', 'BGGR', 'GRBG']
gains_by_pattern = {
    'GBRG': {  # Row0: Gb B, Row1: R Gr
        'R_gain': 1.02, 'R_off': 2.9,
        'Gr_gain': 1.66, 'Gr_off': 0.7,
        'Gb_gain': 1.71, 'Gb_off': -2.2,
        'B_gain': 4.78, 'B_off': 12.4,
    },
}

# For other patterns, just try without pattern-specific gains
# and use automatic gain estimation

print("\n=== Testing Bayer patterns with auto white balance ===")

for pattern in patterns:
    print(f"\n--- Pattern: {pattern} ---")

    # Identify which pass data goes where based on pattern
    # Row0 even, Row0 odd, Row1 even, Row1 odd
    if pattern == 'RGGB':
        # R0e=R, R0o=Gr, R1e=Gb, R1o=B
        pass_names = ['R', 'Gr', 'Gb', 'B']
    elif pattern == 'BGGR':
        # R0e=B, R0o=Gb, R1e=Gr, R1o=R
        pass_names = ['B', 'Gb', 'Gr', 'R']
    elif pattern == 'GBRG':
        # R0e=Gb, R0o=B, R1e=R, R1o=Gr
        pass_names = ['Gb', 'B', 'R', 'Gr']
    elif pattern == 'GRBG':
        # R0e=Gr, R0o=R, R1e=B, R1o=Gb
        pass_names = ['Gr', 'R', 'B', 'Gb']

    # Extract channels
    channels = {}
    channels[pass_names[0]] = raw[0::2, 0::2].astype(float)  # Row0 even
    channels[pass_names[1]] = raw[0::2, 1::2].astype(float)  # Row0 odd
    channels[pass_names[2]] = raw[1::2, 0::2].astype(float)  # Row1 even
    channels[pass_names[3]] = raw[1::2, 1::2].astype(float)  # Row1 odd

    print(f"  R: mean={channels['R'].mean():.1f}, Gr: mean={channels['Gr'].mean():.1f}, "
          f"Gb: mean={channels['Gb'].mean():.1f}, B: mean={channels['B'].mean():.1f}")

    # Auto white balance: scale each channel so gray world assumption holds
    # Target: all channels same mean (gray world)
    target_mean = 128.0
    gains = {}
    for ch_name in ['R', 'Gr', 'Gb', 'B']:
        ch_mean = channels[ch_name].mean()
        if ch_mean > 0:
            gains[ch_name] = target_mean / ch_mean
        else:
            gains[ch_name] = 1.0
        print(f"  Auto gain {ch_name}: {gains[ch_name]:.3f}")

    # Apply gains to raw data
    raw_wb = raw.astype(float).copy()
    for y in range(H):
        for x in range(W):
            row_type = y % 2
            col_type = x % 2
            if row_type == 0 and col_type == 0:
                ch = pass_names[0]
            elif row_type == 0 and col_type == 1:
                ch = pass_names[1]
            elif row_type == 1 and col_type == 0:
                ch = pass_names[2]
            else:
                ch = pass_names[3]
            raw_wb[y, x] = np.clip(raw[y, x] * gains[ch], 0, 255)

    raw_wb_u8 = raw_wb.astype(np.uint8)

    # Simple demosaicing using numpy operations (much faster than per-pixel)
    # Extract half-resolution channels
    r_half = np.clip(channels['R'] * gains['R'], 0, 255)
    g_half = np.clip((channels['Gr'] * gains['Gr'] + channels['Gb'] * gains['Gb']) / 2, 0, 255)
    b_half = np.clip(channels['B'] * gains['B'], 0, 255)

    # Create half-resolution RGB (240x320)
    rgb_half = np.stack([r_half, g_half, b_half], axis=2).astype(np.uint8)
    rgb_full = np.array(Image.fromarray(rgb_half, 'RGB').resize((W, H), Image.BILINEAR))

    # Compare with reference
    mse = np.mean((rgb_full.astype(float) - ref_rgb) ** 2)
    psnr = 10 * np.log10(255**2 / mse)
    corr = np.corrcoef(rgb_full.flatten().astype(float), ref_rgb.flatten())[0, 1]
    print(f"  Half-res RGB (auto WB): PSNR={psnr:.2f}dB, corr={corr:.4f}")

    fname = f"v8_{pattern}_auto_wb.jpg"
    Image.fromarray(rgb_full, 'RGB').save(os.path.join(out_dir, fname), quality=62)

    # Also try with optimal per-channel linear regression gains
    ref_r_half = ref_rgb[0::2, 0::2, 0] if pass_names[0] in ['R','Gr','Gb','B'] else ref_rgb[0::2, 0::2, 0]
    # Actually let's just find optimal gains via regression against reference
    for ch_name, ch_data in channels.items():
        ch_flat = ch_data.flatten()
        if ch_name in ['R']:
            ref_ch = ref_rgb[::2, ::2, 0].flatten()[:len(ch_flat)]  # Approximate
        elif ch_name in ['Gr', 'Gb']:
            ref_ch = ref_rgb[::2, ::2, 1].flatten()[:len(ch_flat)]
        elif ch_name in ['B']:
            ref_ch = ref_rgb[::2, ::2, 2].flatten()[:len(ch_flat)]
        slope, intercept = np.polyfit(ch_flat, ref_ch, 1)
        print(f"  Optimal {ch_name}: slope={slope:.3f}, intercept={intercept:.1f}")

    # Apply optimal gains via regression
    r_opt = np.clip(channels['R'] * np.polyfit(channels['R'].flatten(),
        ref_rgb[::2,::2,0].flatten()[:channels['R'].size], 1)[0] +
        np.polyfit(channels['R'].flatten(),
        ref_rgb[::2,::2,0].flatten()[:channels['R'].size], 1)[1], 0, 255)
    g_opt = np.clip((channels['Gr'] + channels['Gb']) / 2 *
        np.polyfit(((channels['Gr']+channels['Gb'])/2).flatten(),
        ref_rgb[::2,::2,1].flatten()[:channels['Gr'].size], 1)[0] +
        np.polyfit(((channels['Gr']+channels['Gb'])/2).flatten(),
        ref_rgb[::2,::2,1].flatten()[:channels['Gr'].size], 1)[1], 0, 255)
    b_opt = np.clip(channels['B'] * np.polyfit(channels['B'].flatten(),
        ref_rgb[::2,::2,2].flatten()[:channels['B'].size], 1)[0] +
        np.polyfit(channels['B'].flatten(),
        ref_rgb[::2,::2,2].flatten()[:channels['B'].size], 1)[1], 0, 255)

    rgb_opt_half = np.stack([r_opt, g_opt, b_opt], axis=2).astype(np.uint8)
    rgb_opt = np.array(Image.fromarray(rgb_opt_half, 'RGB').resize((W, H), Image.BILINEAR))
    mse_opt = np.mean((rgb_opt.astype(float) - ref_rgb) ** 2)
    psnr_opt = 10 * np.log10(255**2 / mse_opt)
    corr_opt = np.corrcoef(rgb_opt.flatten().astype(float), ref_rgb.flatten())[0, 1]
    print(f"  Optimal WB RGB: PSNR={psnr_opt:.2f}dB, corr={corr_opt:.4f}")
    Image.fromarray(rgb_opt, 'RGB').save(
        os.path.join(out_dir, f"v8_{pattern}_optimal_wb.jpg"), quality=62)

    # Also save Y-only (grayscale from Green channels)
    g_only = np.clip((channels['Gr'] * gains['Gr'] + channels['Gb'] * gains['Gb']) / 2,
                     0, 255).astype(np.uint8)
    g_full = np.array(Image.fromarray(g_only, 'L').resize((W, H), Image.BILINEAR))
    corr_g = np.corrcoef(g_full.flatten().astype(float), ref_gray.flatten())[0, 1]
    print(f"  Green-only gray: corr={corr_g:.4f}")


print("\n" + "=" * 60)
print("Testing with gamma correction on best result...")

# Take the best pattern and apply gamma
# We'll use gamma ≈ 0.45 (as documented in the GAMM LUT)
best_fname = "v8_GBRG_optimal_wb.jpg"
best_img = np.array(Image.open(os.path.join(out_dir, best_fname)))
gamma_img = np.clip(np.power(best_img.astype(float) / 255.0, 0.45) * 255, 0, 255).astype(np.uint8)
mse_g = np.mean((gamma_img.astype(float) - ref_rgb) ** 2)
psnr_g = 10 * np.log10(255**2 / mse_g)
corr_g = np.corrcoef(gamma_img.flatten().astype(float), ref_rgb.flatten())[0, 1]
print(f"GBRG optimal + gamma 0.45: PSNR={psnr_g:.2f}dB, corr={corr_g:.4f}")
Image.fromarray(gamma_img, 'RGB').save(
    os.path.join(out_dir, "v8_GBRG_gamma.jpg"), quality=62)

print("\nDone!")
