#!/usr/bin/env python3
"""V6: Correct 3-pass decoder based on precise COMConduit.dll disassembly.

Key insights from fcn.100029c0:
- All PDB records are concatenated into ONE continuous bitstream
- The decoder is called 240 times (height/2), each producing 2 rows of 640 bytes
- Each call has 3 passes with byte-alignment flushes between them:
  Pass 1: row0 odd positions [1,3,5,...,639] - horizontal delta prediction
  Pass 2: row1 even positions [0,2,4,...,638] - horizontal delta prediction
  Pass 3: row0 even [0,2,...,638] + row1 odd [1,3,...,639] - zigzag cross-row prediction
- The output is UYVY format at 640 bytes per row (320 pixels)
- The JPEG upscales to 640 pixels wide
"""

import struct
import numpy as np
from PIL import Image
import os
import sys

class BitReader:
    def __init__(self, data):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 7  # MSB first

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
        """Align to next byte boundary (skip remaining bits in current byte)."""
        if self.bit_pos != 7:
            self.bit_pos = 7
            self.byte_pos += 1

    def read_raw_byte(self):
        """Read a raw byte directly from the stream (after flush)."""
        if self.byte_pos >= len(self.data):
            return 0
        val = self.data[self.byte_pos]
        self.byte_pos += 1
        return val

    def bits_consumed(self):
        return self.byte_pos * 8 + (7 - self.bit_pos)


def decode_delta(reader, prev):
    """Decode one delta-encoded pixel value."""
    flag = reader.read_bit()
    if flag == 1:
        return prev & 0xFF  # REPEAT
    code = reader.read_bits(5)
    if code == 0:
        return reader.read_bits(8) & 0xFF  # LITERAL (8 raw bits)
    sign = (code >> 4) & 1
    mag = code & 0x0F
    if sign:
        return (prev - mag) & 0xFF  # NEGATIVE delta
    else:
        return (prev + mag) & 0xFF  # POSITIVE delta


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


def decode_row_pair(reader, output, r0, r1, width):
    """Decode one pair of rows using the 3-pass structure from fcn.100029c0.

    Pass 1: row0 odd positions - horizontal prediction
    Pass 2: row1 even positions - horizontal prediction
    Pass 3: row0 even + row1 odd - zigzag cross-row prediction
    """
    # ==================== PASS 1 ====================
    # Byte-align the bitstream
    reader.flush_to_byte()

    # Read literal start byte → row0[1]
    lit = reader.read_raw_byte()
    output[r0, 1] = lit
    prev = lit

    # Delta-decode remaining odd positions: 3, 5, 7, ..., width-1
    for col in range(3, width, 2):
        val = decode_delta(reader, prev)
        output[r0, col] = val
        prev = val

    # ==================== PASS 2 ====================
    # Byte-align
    reader.flush_to_byte()

    # Read literal start byte → row1[0]
    lit = reader.read_raw_byte()
    output[r1, 0] = lit
    prev = lit

    # Delta-decode remaining even positions: 2, 4, 6, ..., width-2
    for col in range(2, width, 2):
        val = decode_delta(reader, prev)
        output[r1, col] = val
        prev = val

    # ==================== PASS 3 ====================
    # Byte-align, read literal → row0[0]
    reader.flush_to_byte()
    lit0 = reader.read_raw_byte()
    output[r0, 0] = lit0

    # Byte-align, read literal → row1[1]
    reader.flush_to_byte()
    lit1 = reader.read_raw_byte()
    output[r1, 1] = lit1

    # Zigzag decode: alternating between row0 even and row1 odd
    # row0[2] predicted from row1[1]
    # row1[3] predicted from row0[2]
    # row0[4] predicted from row1[3]
    # row1[5] predicted from row0[4]
    # ...
    for col in range(2, width, 2):
        # row0[col] (even) - predictor is row1[col-1]
        pred = int(output[r1, col - 1])
        val = decode_delta(reader, pred)
        output[r0, col] = val

        # row1[col+1] (odd) - predictor is row0[col]
        if col + 1 < width:
            pred = int(output[r0, col])
            val = decode_delta(reader, pred)
            output[r1, col + 1] = val


def uyvy_to_rgb(uyvy_data, pixel_height, byte_width):
    """Convert UYVY data (640 bytes/row = 320 pixels) to RGB."""
    pixel_width = byte_width // 2  # 320 pixels for 640 bytes
    rgb = np.zeros((pixel_height, pixel_width, 3), dtype=np.uint8)

    for y in range(pixel_height):
        for x in range(0, pixel_width, 2):
            byte_off = x * 2
            cb = int(uyvy_data[y, byte_off]) - 128
            y0 = int(uyvy_data[y, byte_off + 1]) - 16
            cr = int(uyvy_data[y, byte_off + 2]) - 128
            y1 = int(uyvy_data[y, byte_off + 3]) - 16

            # BT.601 conversion (10-bit fixed point)
            r0 = max(0, min(255, (y0 * 1192 + cr * 1634) >> 10))
            g0 = max(0, min(255, (y0 * 1192 - cr * 832 - cb * 400) >> 10))
            b0 = max(0, min(255, (y0 * 1192 + cb * 2066) >> 10))

            r1 = max(0, min(255, (y1 * 1192 + cr * 1634) >> 10))
            g1 = max(0, min(255, (y1 * 1192 - cr * 832 - cb * 400) >> 10))
            b1 = max(0, min(255, (y1 * 1192 + cb * 2066) >> 10))

            rgb[y, x] = [r0, g0, b0]
            rgb[y, x + 1] = [r1, g1, b1]

    return rgb


def decode_veo_pdb(pdb_path):
    """Full VEO PDB to image decoder."""
    records = parse_pdb(pdb_path)
    meta = records[0]
    data_records = records[1:-1]  # Skip record 0 (metadata) and last (thumbnail)

    # Parse metadata
    image_type = meta[0]
    width_flag = meta[1]
    height_flag = meta[2]

    pixel_width = 640 if width_flag != 0 else 320
    pixel_height = 480 if height_flag != 0 else 496
    byte_width = pixel_width  # Bytes per row in the decompressed output

    print(f"Image type: {image_type}, pixel dimensions: {pixel_width}x{pixel_height}")
    print(f"Byte width per row: {byte_width}")
    print(f"Data records: {len(data_records)}")

    # Concatenate all record data into one continuous bitstream
    all_data = bytearray()
    for rec in data_records:
        all_data.extend(rec)
    print(f"Total compressed data: {len(all_data)} bytes ({len(all_data)*8} bits)")

    reader = BitReader(bytes(all_data))

    # Allocate output buffer
    output = np.zeros((pixel_height, byte_width), dtype=np.uint8)

    # Decode: height/2 row pairs
    num_pairs = pixel_height // 2
    print(f"Decoding {num_pairs} row pairs...")

    for pair in range(num_pairs):
        r0 = pair * 2
        r1 = pair * 2 + 1
        decode_row_pair(reader, output, r0, r1, byte_width)

        if pair % 60 == 0:
            consumed = reader.bits_consumed()
            print(f"  Pair {pair}/{num_pairs}: consumed {consumed} bits "
                  f"({consumed//8} bytes of {len(all_data)})")

    consumed = reader.bits_consumed()
    print(f"Total bits consumed: {consumed} ({consumed//8} of {len(all_data)} bytes)")

    return output, pixel_width, pixel_height, byte_width


# ============================================================================
# Main
# ============================================================================

pdb_path = "data/3840520404.pdb"
out_dir = "output/"
ref_path = "data/Thom_091225_001.JPG"

print("=" * 70)
print("VEO Decoder V6 - Correct 3-Pass Structure")
print("=" * 70)

output, pixel_width, pixel_height, byte_width = decode_veo_pdb(pdb_path)

# Save raw decompressed data as grayscale for inspection
Image.fromarray(output, 'L').save(os.path.join(out_dir, "v6_raw_decompressed.png"))
print(f"\nRaw decompressed data saved ({pixel_height}x{byte_width})")

# Load reference
ref_rgb = np.array(Image.open(ref_path).convert('RGB'))
ref_gray = np.array(Image.open(ref_path).convert('L'))
print(f"Reference: {ref_rgb.shape}")

# ============================================================================
# Analysis: Check correlation of different channels
# ============================================================================
print("\n=== Channel Analysis ===")

# Extract Y channel (odd byte positions in UYVY)
y_channel = output[:, 1::2]  # positions 1,3,5,...
print(f"Y channel shape: {y_channel.shape}, mean={y_channel.mean():.1f}, std={y_channel.std():.1f}")

# Extract CbCr channel (even byte positions)
cbcr_channel = output[:, 0::2]
print(f"CbCr channel shape: {cbcr_channel.shape}, mean={cbcr_channel.mean():.1f}, std={cbcr_channel.std():.1f}")

# Resize Y channel to match reference for comparison
y_resized = np.array(Image.fromarray(y_channel, 'L').resize((640, 480), Image.BILINEAR))
corr = np.corrcoef(y_resized.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"Y channel correlation with reference gray: {corr:.4f}")
Image.fromarray(y_channel, 'L').save(os.path.join(out_dir, "v6_y_channel.png"))

# Also check full output as grayscale
output_resized = np.array(Image.fromarray(output, 'L').resize((640, 480), Image.BILINEAR))
corr_full = np.corrcoef(output_resized.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"Full output correlation with reference gray: {corr_full:.4f}")

# Check even rows vs odd rows
even_rows = output[0::2, :]
odd_rows = output[1::2, :]
print(f"\nEven rows: mean={even_rows.mean():.1f}, std={even_rows.std():.1f}")
print(f"Odd rows: mean={odd_rows.mean():.1f}, std={odd_rows.std():.1f}")

# Y from even rows only
y_even = even_rows[:, 1::2]
y_even_resized = np.array(Image.fromarray(y_even, 'L').resize((640, 480), Image.BILINEAR))
corr_y_even = np.corrcoef(y_even_resized.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"Y channel (even rows only): corr={corr_y_even:.4f}")

y_odd = odd_rows[:, 1::2]
y_odd_resized = np.array(Image.fromarray(y_odd, 'L').resize((640, 480), Image.BILINEAR))
corr_y_odd = np.corrcoef(y_odd_resized.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"Y channel (odd rows only): corr={corr_y_odd:.4f}")

# ============================================================================
# Try UYVY → RGB conversion
# ============================================================================
print("\n=== UYVY → RGB Conversion ===")

# Method 1: UYVY at 320 pixels wide, then upscale to 640
print("Converting UYVY (320px wide)...")
uyvy_pixel_width = byte_width // 2
rgb_320 = np.zeros((pixel_height, uyvy_pixel_width, 3), dtype=np.uint8)

for y in range(pixel_height):
    for x in range(0, uyvy_pixel_width, 2):
        bo = x * 2
        if bo + 3 >= byte_width:
            break
        cb = int(output[y, bo]) - 128
        y0_val = int(output[y, bo + 1]) - 16
        cr = int(output[y, bo + 2]) - 128
        y1_val = int(output[y, bo + 3]) - 16

        r0 = max(0, min(255, (y0_val * 1192 + cr * 1634) >> 10))
        g0 = max(0, min(255, (y0_val * 1192 - cr * 832 - cb * 400) >> 10))
        b0 = max(0, min(255, (y0_val * 1192 + cb * 2066) >> 10))

        r1 = max(0, min(255, (y1_val * 1192 + cr * 1634) >> 10))
        g1 = max(0, min(255, (y1_val * 1192 - cr * 832 - cb * 400) >> 10))
        b1 = max(0, min(255, (y1_val * 1192 + cb * 2066) >> 10))

        rgb_320[y, x] = [r0, g0, b0]
        rgb_320[y, x + 1] = [r1, g1, b1]

Image.fromarray(rgb_320, 'RGB').save(os.path.join(out_dir, "v6_rgb_320.png"))
rgb_640 = np.array(Image.fromarray(rgb_320, 'RGB').resize((640, 480), Image.BILINEAR))
Image.fromarray(rgb_640, 'RGB').save(os.path.join(out_dir, "v6_rgb_640.jpg"), quality=62)

# Compare with reference
mse = np.mean((rgb_640.astype(float) - ref_rgb.astype(float))**2)
psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 0
corr_rgb = np.corrcoef(rgb_640.flatten().astype(float), ref_rgb.flatten().astype(float))[0,1]
print(f"UYVY→RGB (upscaled to 640): PSNR={psnr:.2f}dB, corr={corr_rgb:.4f}")

# ============================================================================
# Alternative: maybe it's NOT UYVY but pure Y-Cb-Cr interleaved differently
# Try: treat odd positions as Y, convert just that to grayscale
# ============================================================================
print("\n=== Alternative interpretations ===")

# What if the format is YUYV (not UYVY)?
rgb_yuyv = np.zeros((pixel_height, uyvy_pixel_width, 3), dtype=np.uint8)
for y in range(pixel_height):
    for x in range(0, uyvy_pixel_width, 2):
        bo = x * 2
        if bo + 3 >= byte_width:
            break
        y0_val = int(output[y, bo]) - 16
        cb = int(output[y, bo + 1]) - 128
        y1_val = int(output[y, bo + 2]) - 16
        cr = int(output[y, bo + 3]) - 128

        r0 = max(0, min(255, (y0_val * 1192 + cr * 1634) >> 10))
        g0 = max(0, min(255, (y0_val * 1192 - cr * 832 - cb * 400) >> 10))
        b0 = max(0, min(255, (y0_val * 1192 + cb * 2066) >> 10))

        r1 = max(0, min(255, (y1_val * 1192 + cr * 1634) >> 10))
        g1 = max(0, min(255, (y1_val * 1192 - cr * 832 - cb * 400) >> 10))
        b1 = max(0, min(255, (y1_val * 1192 + cb * 2066) >> 10))

        rgb_yuyv[y, x] = [r0, g0, b0]
        rgb_yuyv[y, x + 1] = [r1, g1, b1]

rgb_yuyv_640 = np.array(Image.fromarray(rgb_yuyv, 'RGB').resize((640, 480), Image.BILINEAR))
mse_yuyv = np.mean((rgb_yuyv_640.astype(float) - ref_rgb.astype(float))**2)
psnr_yuyv = 10 * np.log10(255**2 / mse_yuyv) if mse_yuyv > 0 else 0
corr_yuyv = np.corrcoef(rgb_yuyv_640.flatten().astype(float), ref_rgb.flatten().astype(float))[0,1]
Image.fromarray(rgb_yuyv_640, 'RGB').save(os.path.join(out_dir, "v6_rgb_yuyv.jpg"), quality=62)
print(f"YUYV→RGB: PSNR={psnr_yuyv:.2f}dB, corr={corr_yuyv:.4f}")

# Save Y-only image at 640x480 for comparison
y_full = np.array(Image.fromarray(y_channel, 'L').resize((640, 480), Image.BILINEAR))
mse_y = np.mean((y_full.astype(float) - ref_gray.astype(float))**2)
psnr_y = 10 * np.log10(255**2 / mse_y) if mse_y > 0 else 0
print(f"Y-only grayscale: PSNR={psnr_y:.2f}dB, corr={corr:.4f}")
Image.fromarray(y_full, 'L').save(os.path.join(out_dir, "v6_y_only.png"))

# Also try per-row shift optimization on Y channel
print("\n=== Shift analysis on Y channel ===")
best_shift = 0
best_corr = -1
for shift in range(y_channel.shape[1]):
    shifted = np.roll(y_channel, shift, axis=1)
    shifted_resized = np.array(Image.fromarray(shifted, 'L').resize((640, 480), Image.BILINEAR))
    c = np.corrcoef(shifted_resized.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
    if c > best_corr:
        best_corr = c
        best_shift = shift

print(f"Best global shift for Y channel: {best_shift} (corr={best_corr:.4f})")

# Apply best shift and save
y_shifted = np.roll(y_channel, best_shift, axis=1)
y_shifted_resized = np.array(Image.fromarray(y_shifted, 'L').resize((640, 480), Image.BILINEAR))
Image.fromarray(y_shifted_resized, 'L').save(os.path.join(out_dir, "v6_y_shifted.png"))

print("\nDone!")
