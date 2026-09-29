#!/usr/bin/env python3
"""
Veo Palm Camera PDB to JPEG Decoder v3
Based on reverse engineering of COMConduit.dll and Veo.PRC ARMC resource.

Algorithm: Predictive bitstream with delta encoding (Type 4)
Pipeline: PDB → Decompress → YCbCr 4:2:2 → RGB → JPEG (quality 62)

Author: Claude Opus 4.6 (Reverse Engineering)
Date: 2026-03-12
"""

import struct
import sys
import os
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Installing required packages...")
    os.system("pip3 install Pillow numpy")
    from PIL import Image
    import numpy as np


class BitReader:
    """Reads bits from a byte stream, MSB first."""
    def __init__(self, data):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 7  # MSB first, counts down 7→0

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

    @property
    def bits_consumed(self):
        return self.byte_pos * 8 + (7 - self.bit_pos)


def decompress_record_linear(record_data, num_pixels):
    """Decompress a record using Type 4 delta encoding (linear order)."""
    reader = BitReader(record_data)
    output = bytearray(num_pixels)
    prev = 128  # Initial predictor value

    for i in range(num_pixels):
        flag = reader.read_bit()
        if flag == 1:
            # REPEAT: copy previous value
            output[i] = prev & 0xFF
        else:
            code = reader.read_bits(5)
            if code == 0:
                # LITERAL: 8 raw bits
                val = reader.read_bits(8)
                output[i] = val & 0xFF
                prev = val
            else:
                # DELTA: sign bit + 4-bit magnitude
                sign = (code >> 4) & 1
                magnitude = code & 0x0F
                if sign:
                    val = (prev - magnitude) & 0xFF
                else:
                    val = (prev + magnitude) & 0xFF
                output[i] = val
                prev = val

    return output


def decompress_record_interleaved(record_data, width, num_rows):
    """Decompress a record with interleaved even/odd byte processing.

    Based on ARMC code: processes even byte positions first (stride 2),
    then odd byte positions, for each pair of rows.
    """
    reader = BitReader(record_data)
    output = bytearray(width * num_rows)

    # Process in the order the ARMC code does it:
    # For each pair of rows:
    #   Pass 1: even positions (0, 2, 4, ...) of row 0, then row 1
    #   Pass 2: odd positions (1, 3, 5, ...) of row 0, then row 1

    for row_pair in range(num_rows // 2):
        for parity in range(2):  # 0=even positions, 1=odd positions
            prev = 128
            for row_in_pair in range(2):
                row = row_pair * 2 + row_in_pair
                for col in range(parity, width, 2):
                    idx = row * width + col
                    flag = reader.read_bit()
                    if flag == 1:
                        output[idx] = prev & 0xFF
                    else:
                        code = reader.read_bits(5)
                        if code == 0:
                            val = reader.read_bits(8)
                            output[idx] = val & 0xFF
                            prev = val
                        else:
                            sign = (code >> 4) & 1
                            magnitude = code & 0x0F
                            if sign:
                                val = (prev - magnitude) & 0xFF
                            else:
                                val = (prev + magnitude) & 0xFF
                            output[idx] = val
                            prev = val

    return output


def decompress_record_armc_exact(record_data, width, num_rows):
    """Decompress matching the exact ARMC code flow.

    ARMC processes:
    - dest+1 with stride 2 for 640 iterations (even bytes of row 0)
    - dest+0x280 with stride 2 for 640 iterations (even bytes of row 1)
    - Then odd bytes similarly
    Wait - re-reading: "dest+1 with stride 2" means starting at byte 1, stepping by 2
    That would be ODD positions. Let me try both orderings.
    """
    reader = BitReader(record_data)
    output = bytearray(width * num_rows)

    # Interpretation: each record has 4 rows
    # The ARMC code does 2 passes per row-pair:
    # Pass 1 processes positions 1,3,5,... of row 0, then 1,3,5,... of row 1
    # Pass 2 processes positions 0,2,4,... of row 0, then 0,2,4,... of row 1
    # (Or vice versa)

    for row_pair in range(num_rows // 2):
        for start_pos in [1, 0]:  # Try odd first, then even (dest+1 = odd start)
            prev = 128
            for row_offset in [0, 1]:
                row = row_pair * 2 + row_offset
                for col in range(start_pos, width, 2):
                    idx = row * width + col
                    flag = reader.read_bit()
                    if flag == 1:
                        output[idx] = prev & 0xFF
                    else:
                        code = reader.read_bits(5)
                        if code == 0:
                            val = reader.read_bits(8)
                            output[idx] = val & 0xFF
                            prev = val
                        else:
                            sign = (code >> 4) & 1
                            magnitude = code & 0x0F
                            if sign:
                                val = (prev - magnitude) & 0xFF
                            else:
                                val = (prev + magnitude) & 0xFF
                            output[idx] = val
                            prev = val

    return output


def ycbcr422_to_rgb(ycbcr_data, width, height):
    """Convert YCbCr 4:2:2 (UYVY) to RGB using BT.601 coefficients.

    UYVY format: [Cb, Y0, Cr, Y1, Cb, Y0, Cr, Y1, ...]
    Each 4 bytes = 2 pixels
    """
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    for row in range(height):
        row_offset = row * width  # width bytes per row in YCbCr 4:2:2
        # In UYVY: width bytes = width/2 pixel pairs
        for col_pair in range(width // 4):  # Process 4 bytes at a time
            base = row_offset + col_pair * 4
            if base + 3 >= len(ycbcr_data):
                break

            cb = ycbcr_data[base + 0] - 128
            y0 = ycbcr_data[base + 1] - 16
            cr = ycbcr_data[base + 2] - 128
            y1 = ycbcr_data[base + 3] - 16

            # BT.601 conversion with 10-bit fixed point
            for y_val, pixel_col in [(y0, col_pair * 2), (y1, col_pair * 2 + 1)]:
                r = (y_val * 1192 + cr * 1634) >> 10
                g = (y_val * 1192 - cr * 832 - cb * 400) >> 10
                b = (y_val * 1192 + cb * 2066) >> 10

                rgb[row, pixel_col, 0] = max(0, min(255, r))
                rgb[row, pixel_col, 1] = max(0, min(255, g))
                rgb[row, pixel_col, 2] = max(0, min(255, b))

    return rgb


def raw_to_rgb_direct(raw_data, width, height):
    """Interpret raw decompressed data directly as grayscale or interleaved."""
    # Just show what we got
    gray = np.zeros((height, width), dtype=np.uint8)
    for row in range(height):
        for col in range(width):
            idx = row * width + col
            if idx < len(raw_data):
                gray[row, col] = raw_data[idx]
    return gray


def parse_pdb(filepath):
    """Parse a Palm PDB file and extract records."""
    with open(filepath, 'rb') as f:
        data = f.read()

    name = data[0:32].split(b'\x00')[0].decode('ascii', errors='replace')
    num_records = struct.unpack('>H', data[76:78])[0]

    # Parse record entries
    record_offsets = []
    for i in range(num_records):
        off = 78 + i * 8
        rec_offset = struct.unpack('>I', data[off:off+4])[0]
        record_offsets.append(rec_offset)
    record_offsets.append(len(data))  # Sentinel for last record size

    # Extract record data
    records = []
    for i in range(num_records):
        rec_data = data[record_offsets[i]:record_offsets[i+1]]
        records.append(rec_data)

    return name, records


def decode_metadata(record0):
    """Parse metadata from Record 0 (25 bytes)."""
    img_type = record0[0]
    width = 640 if record0[1] != 0 else 320
    height_flag = record0[2]
    # From COMConduit: height = 480 + (16 if height_flag == 0 else 0)
    height = 480 if height_flag != 0 else 496

    # Date from bytes 20, 22, 23-24
    day = record0[20] if len(record0) > 20 else 0
    month = record0[22] if len(record0) > 22 else 0
    year = struct.unpack('>H', record0[23:25])[0] if len(record0) > 24 else 0

    return {
        'type': img_type,
        'width': width,
        'height': height,
        'height_flag': height_flag,
        'day': day,
        'month': month,
        'year': year,
    }


def decode_veo_image(pdb_path, output_path=None):
    """Main decoder: PDB → JPEG."""
    print(f"Decoding: {pdb_path}")

    name, records = parse_pdb(pdb_path)
    print(f"  PDB name: {name}, {len(records)} records")

    meta = decode_metadata(records[0])
    print(f"  Type: {meta['type']}, Size: {meta['width']}x{meta['height']}")
    print(f"  Date: {meta['day']:02d}.{meta['month']:02d}.{meta['year']}")

    width = meta['width']
    height = meta['height']

    # Actual image height for processing (480 rows for 120 data records × 4 rows/record)
    data_records = records[1:-1]  # Skip metadata (0) and thumbnail (last)
    num_data_records = len(data_records)
    rows_per_record = 4  # Type 4: 4 rows per record
    actual_height = num_data_records * rows_per_record
    print(f"  Data records: {num_data_records}, rows/record: {rows_per_record}")
    print(f"  Actual image height: {actual_height}")

    pixels_per_record = width * rows_per_record

    # Try multiple decompression strategies
    strategies = {
        'linear': decompress_record_linear,
        'interleaved_even_first': lambda d, n: decompress_record_interleaved(d, width, rows_per_record),
        'interleaved_armc': lambda d, n: decompress_record_armc_exact(d, width, rows_per_record),
    }

    for strategy_name, decompress_fn in strategies.items():
        print(f"\n  Strategy: {strategy_name}")

        # Decompress all records
        all_data = bytearray()
        total_bits = 0
        total_pixels = 0

        for i, rec_data in enumerate(data_records):
            if strategy_name == 'linear':
                decompressed = decompress_fn(rec_data, pixels_per_record)
            else:
                decompressed = decompress_fn(rec_data, pixels_per_record)
            all_data.extend(decompressed)
            total_pixels += pixels_per_record

        print(f"    Decompressed: {len(all_data)} bytes ({total_pixels} pixels)")
        print(f"    Byte range: {min(all_data)}-{max(all_data)}, mean: {sum(all_data)/len(all_data):.1f}")

        base_name = Path(pdb_path).stem
        if output_path:
            out_dir = Path(output_path)
        else:
            out_dir = Path(pdb_path).parent

        # Save raw grayscale
        gray = raw_to_rgb_direct(all_data, width, actual_height)
        gray_img = Image.fromarray(gray, 'L')
        gray_path = out_dir / f"{base_name}_{strategy_name}_gray.png"
        gray_img.save(str(gray_path))
        print(f"    Saved grayscale: {gray_path}")

        # Save as YCbCr 4:2:2 → RGB
        # In UYVY format, 640 bytes per row = 320 pixel pairs = 320 pixels wide? No...
        # 640 bytes / 4 bytes per pixel pair * 2 pixels = 320 pixels
        # But image should be 640 wide...
        #
        # Alternative: Maybe 640 bytes per row where each byte is one component
        # and the format is NOT UYVY but rather planar or line-interleaved?
        #
        # Or maybe 640 bytes = 640 grayscale pixels (Bayer pattern)?

        # Try interpretation 1: UYVY (320 pixel pairs = 320 pixels wide)
        try:
            rgb_uyvy = ycbcr422_to_rgb(all_data, width, actual_height)
            rgb_img = Image.fromarray(rgb_uyvy, 'RGB')
            rgb_path = out_dir / f"{base_name}_{strategy_name}_uyvy.jpg"
            rgb_img.save(str(rgb_path), quality=62)
            print(f"    Saved UYVY→RGB: {rgb_path}")
        except Exception as e:
            print(f"    UYVY failed: {e}")

        # Try interpretation 2: Raw Bayer pattern (640x480 grayscale, then demosaic)
        try:
            # Simple Bayer demosaicing (RGGB pattern)
            bayer = np.frombuffer(bytes(all_data[:width*actual_height]), dtype=np.uint8).reshape(actual_height, width)
            rgb_bayer = np.zeros((actual_height, width, 3), dtype=np.uint8)

            # Simple nearest-neighbor demosaicing
            # RGGB pattern
            rgb_bayer[0::2, 0::2, 0] = bayer[0::2, 0::2]  # R
            rgb_bayer[0::2, 1::2, 1] = bayer[0::2, 1::2]  # G
            rgb_bayer[1::2, 0::2, 1] = bayer[1::2, 0::2]  # G
            rgb_bayer[1::2, 1::2, 2] = bayer[1::2, 1::2]  # B

            # Fill in missing channels with neighbors
            # Simple bilinear for R channel
            rgb_bayer[0::2, 1::2, 0] = bayer[0::2, 0::2][:, :bayer[0::2, 1::2].shape[1]]  # Approx
            rgb_bayer[1::2, :, 0] = rgb_bayer[0::2, :, 0][:rgb_bayer[1::2, :, 0].shape[0]]  # Approx
            # Simple bilinear for B channel
            rgb_bayer[1::2, 0::2, 2] = bayer[1::2, 1::2][:, :bayer[1::2, 0::2].shape[1]]  # Approx
            rgb_bayer[0::2, :, 2] = rgb_bayer[1::2, :, 2][:rgb_bayer[0::2, :, 2].shape[0]]  # Approx
            # G channel - average of two G positions
            rgb_bayer[0::2, 0::2, 1] = ((bayer[0::2, 1::2].astype(int)[:, :bayer[0::2, 0::2].shape[1]] +
                                          bayer[1::2, 0::2].astype(int)[:bayer[0::2, 0::2].shape[0]]) // 2).astype(np.uint8)
            rgb_bayer[1::2, 1::2, 1] = ((bayer[0::2, 1::2].astype(int)[:bayer[1::2, 1::2].shape[0]] +
                                          bayer[1::2, 0::2].astype(int)[:, :bayer[1::2, 1::2].shape[1]]) // 2).astype(np.uint8)

            bayer_img = Image.fromarray(rgb_bayer, 'RGB')
            bayer_path = out_dir / f"{base_name}_{strategy_name}_bayer.jpg"
            bayer_img.save(str(bayer_path), quality=62)
            print(f"    Saved Bayer→RGB: {bayer_path}")
        except Exception as e:
            print(f"    Bayer failed: {e}")

    # Also save the thumbnail from last record
    thumb_data = records[-1]
    print(f"\n  Thumbnail record: {len(thumb_data)} bytes")
    if len(thumb_data) == 2016:
        # RGB565 Big-Endian, 36x28
        thumb_rgb = np.zeros((28, 36, 3), dtype=np.uint8)
        for i in range(1008):
            pixel = struct.unpack('>H', thumb_data[i*2:i*2+2])[0]
            r = ((pixel >> 11) & 0x1F) << 3
            g = ((pixel >> 5) & 0x3F) << 2
            b = (pixel & 0x1F) << 3
            row = i // 36
            col = i % 36
            if row < 28:
                thumb_rgb[row, col] = [r, g, b]

        thumb_img = Image.fromarray(thumb_rgb, 'RGB')
        thumb_path = out_dir / f"{base_name}_thumbnail.png"
        thumb_img.save(str(thumb_path))
        print(f"  Saved thumbnail: {thumb_path}")

    print("\nDone!")


def compare_with_reference(decoded_path, reference_path):
    """Compare decoded image with reference JPEG."""
    try:
        decoded = Image.open(decoded_path).convert('RGB')
        reference = Image.open(reference_path).convert('RGB')

        # Resize if needed
        if decoded.size != reference.size:
            decoded = decoded.resize(reference.size, Image.NEAREST)

        dec_arr = np.array(decoded)
        ref_arr = np.array(reference)

        # Pixel-perfect match
        matches = np.sum(dec_arr == ref_arr)
        total = ref_arr.size
        accuracy = matches / total * 100

        # PSNR
        mse = np.mean((dec_arr.astype(float) - ref_arr.astype(float)) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(255**2 / mse)
        else:
            psnr = float('inf')

        print(f"  Accuracy: {accuracy:.4f}% ({matches}/{total})")
        print(f"  MSE: {mse:.2f}, PSNR: {psnr:.2f} dB")

        return accuracy, psnr
    except Exception as e:
        print(f"  Comparison failed: {e}")
        return 0, 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Default: decode the test file
        pdb_path = "data/3840520404.pdb"
        output_dir = "output/"
    else:
        pdb_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(pdb_path)

    decode_veo_image(pdb_path, output_dir)

    # Compare with reference if available
    ref_path = "data/Thom_091225_001.JPG"
    if os.path.exists(ref_path):
        print("\nComparing with reference JPEG:")
        base = Path(pdb_path).stem
        for suffix in ['linear_uyvy', 'linear_bayer', 'linear_gray',
                       'interleaved_even_first_uyvy', 'interleaved_even_first_bayer',
                       'interleaved_armc_uyvy', 'interleaved_armc_bayer']:
            test_path = os.path.join(output_dir, f"{base}_{suffix}.jpg")
            if os.path.exists(test_path):
                print(f"\n  {suffix}:")
                compare_with_reference(test_path, ref_path)
