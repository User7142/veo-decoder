#!/usr/bin/env python3
"""
Veo Palm Camera PDB Decoder v4 - Based on EXACT COMConduit.dll disassembly.

Key insights from disassembly:
1. ALL records are concatenated into one contiguous bitstream
2. Per 2-row block, 4 passes with interleaved column positions
3. Byte-boundary alignment between pass 2 and pass 3
4. Vertical prediction in pass 4 for even columns of row 1

Author: Claude Opus 4.6
Date: 2026-03-12
"""

import struct
import numpy as np
from PIL import Image
import os, sys

class BitReader:
    """Bitstream reader with byte-boundary alignment support."""
    def __init__(self, data):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 7  # MSB first, counts 7→0
    
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
    
    def read_byte_literal(self):
        """Read a full byte from the bitstream (8 bits)."""
        return self.read_bits(8)
    
    def flush_to_byte(self):
        """Align to next byte boundary (skip remaining bits in current byte)."""
        if self.bit_pos != 7:
            self.bit_pos = 7
            self.byte_pos += 1
    
    def read_raw_byte(self):
        """Read a raw byte at current position (assumes byte-aligned)."""
        if self.byte_pos >= len(self.data):
            return 0
        val = self.data[self.byte_pos]
        self.byte_pos += 1
        return val

    @property
    def bits_consumed(self):
        return self.byte_pos * 8 + (7 - self.bit_pos)


def decode_delta_pixel(reader, prev):
    """Decode one pixel using delta encoding."""
    flag = reader.read_bit()
    if flag == 1:
        return prev & 0xFF, prev
    code = reader.read_bits(5)
    if code == 0:
        val = reader.read_bits(8)
        return val & 0xFF, val
    sign = (code >> 4) & 1
    mag = code & 0x0F
    if sign:
        val = (prev - mag) & 0xFF
    else:
        val = (prev + mag) & 0xFF
    return val, val


def parse_pdb(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()
    name = data[0:32].split(b'\x00')[0].decode('ascii', errors='replace')
    num_records = struct.unpack('>H', data[76:78])[0]
    offsets = []
    for i in range(num_records):
        off = 78 + i * 8
        offsets.append(struct.unpack('>I', data[off:off+4])[0])
    offsets.append(len(data))
    records = [data[offsets[i]:offsets[i+1]] for i in range(num_records)]
    return name, records


def decode_type4(data_records, width, height):
    """Decode Type 4 compressed Bayer data.
    
    Based on COMConduit.dll disassembly:
    - fcn.10002610: Concatenates all records into one buffer
    - fcn.10002e00: Calls fcn.100029c0 for each 2-row block
    - fcn.100029c0: 4-pass interleaved delta decoder
    """
    # Step 1: Concatenate all record data into one bitstream
    all_data = bytearray()
    for rec in data_records:
        all_data.extend(rec)
    
    reader = BitReader(bytes(all_data))
    output = np.zeros((height, width), dtype=np.uint8)
    
    num_row_pairs = height // 2
    
    for row_pair in range(num_row_pairs):
        r0 = row_pair * 2      # First row of pair
        r1 = row_pair * 2 + 1  # Second row of pair
        
        # === PASS 1: Odd columns of row 0 (positions 1, 3, 5, ..., width-1) ===
        # Read first literal byte
        first_val = reader.read_raw_byte() if reader.bit_pos == 7 else reader.read_byte_literal()
        output[r0, 1] = first_val
        prev = first_val
        
        # Decode remaining odd columns (3, 5, 7, ..., width-1)
        for col in range(3, width, 2):
            val, prev = decode_delta_pixel(reader, prev)
            output[r0, col] = val
        
        # === PASS 2: Odd columns of row 1 (positions 1, 3, 5, ..., width-1) ===
        first_val = reader.read_byte_literal()
        output[r1, 1] = first_val
        prev = first_val
        
        for col in range(3, width, 2):
            val, prev = decode_delta_pixel(reader, prev)
            output[r1, col] = val
        
        # === BYTE ALIGNMENT between pass 2 and pass 3 ===
        reader.flush_to_byte()
        
        # === PASS 3: Literal start values for even column 0 ===
        output[r0, 0] = reader.read_raw_byte()
        
        reader.flush_to_byte()
        
        output[r1, 0] = reader.read_raw_byte()
        
        # === PASS 4: Even columns of both rows ===
        # Pass 4a: Even columns of row 0 (2, 4, 6, ..., width-2)
        # Horizontal prediction from previous even column
        prev = int(output[r0, 0])
        for col in range(2, width, 2):
            val, prev = decode_delta_pixel(reader, prev)
            output[r0, col] = val
        
        # Pass 4b: Even columns of row 1 (2, 4, 6, ..., width-2)
        # Vertical prediction from row 0 same column
        prev = int(output[r1, 0])
        for col in range(2, width, 2):
            # Predictor is pixel directly above (row 0, same column)
            above = int(output[r0, col])
            val, prev_new = decode_delta_pixel(reader, above)
            output[r1, col] = val
            prev = val  # or prev_new?
    
    return output


def decode_type4_v2(data_records, width, height):
    """Alternative: try without byte alignment, different pass order."""
    all_data = bytearray()
    for rec in data_records:
        all_data.extend(rec)
    
    reader = BitReader(bytes(all_data))
    output = np.zeros((height, width), dtype=np.uint8)
    
    for row_pair in range(height // 2):
        r0 = row_pair * 2
        r1 = r0 + 1
        
        # Pass 1: All of row 0, odd cols (1,3,5,...,639)
        prev = 128
        for col in range(1, width, 2):
            val, prev = decode_delta_pixel(reader, prev)
            output[r0, col] = val
        
        # Pass 2: All of row 1, odd cols  
        prev = 128
        for col in range(1, width, 2):
            val, prev = decode_delta_pixel(reader, prev)
            output[r1, col] = val
        
        # Flush to byte boundary
        reader.flush_to_byte()
        
        # Pass 3: Literal bytes for col 0
        output[r0, 0] = reader.read_raw_byte()
        output[r1, 0] = reader.read_raw_byte()
        
        # Pass 4a: Row 0 even cols (2,4,...,638), horizontal prediction
        prev = int(output[r0, 0])
        for col in range(2, width, 2):
            val, prev = decode_delta_pixel(reader, prev)
            output[r0, col] = val
        
        # Pass 4b: Row 1 even cols, vertical prediction from row 0
        for col in range(2, width, 2):
            above = int(output[r0, col])
            val, _ = decode_delta_pixel(reader, above)
            output[r1, col] = val
    
    return output


def decode_type4_v3(data_records, width, height):
    """V3: Try per-record processing (not concatenated)."""
    output = np.zeros((height, width), dtype=np.uint8)
    
    rows_per_record = 4
    for rec_idx, rec in enumerate(data_records):
        reader = BitReader(rec)
        base_row = rec_idx * rows_per_record
        
        for sub_pair in range(2):  # 2 row-pairs per record
            r0 = base_row + sub_pair * 2
            r1 = r0 + 1
            if r1 >= height:
                break
            
            # Pass 1: odd cols row 0
            prev = 128
            for col in range(1, width, 2):
                val, prev = decode_delta_pixel(reader, prev)
                output[r0, col] = val
            
            # Pass 2: odd cols row 1
            prev = 128
            for col in range(1, width, 2):
                val, prev = decode_delta_pixel(reader, prev)
                output[r1, col] = val
            
            # Flush
            reader.flush_to_byte()
            
            # Pass 3: literals for col 0
            output[r0, 0] = reader.read_raw_byte()
            output[r1, 0] = reader.read_raw_byte()
            
            # Pass 4a: even cols row 0
            prev = int(output[r0, 0])
            for col in range(2, width, 2):
                val, prev = decode_delta_pixel(reader, prev)
                output[r0, col] = val
            
            # Pass 4b: even cols row 1, vertical prediction
            for col in range(2, width, 2):
                above = int(output[r0, col])
                val, _ = decode_delta_pixel(reader, above)
                output[r1, col] = val
    
    return output


def decode_type4_v4(data_records, width, height):
    """V4: Concatenated, no alignment flush, first literal then delta."""
    all_data = bytearray()
    for rec in data_records:
        all_data.extend(rec)
    
    reader = BitReader(bytes(all_data))
    output = np.zeros((height, width), dtype=np.uint8)
    
    for row_pair in range(height // 2):
        r0 = row_pair * 2
        r1 = r0 + 1
        
        # Pass 1: odd cols row 0 - first is literal, rest delta
        first = reader.read_bits(8)
        output[r0, 1] = first
        prev = first
        for col in range(3, width, 2):
            val, prev = decode_delta_pixel(reader, prev)
            output[r0, col] = val
        
        # Pass 2: odd cols row 1
        first = reader.read_bits(8)
        output[r1, 1] = first
        prev = first
        for col in range(3, width, 2):
            val, prev = decode_delta_pixel(reader, prev)
            output[r1, col] = val
        
        # Flush to byte boundary
        reader.flush_to_byte()
        
        # Pass 3+4: even cols
        output[r0, 0] = reader.read_raw_byte()
        reader.flush_to_byte()
        output[r1, 0] = reader.read_raw_byte()
        
        prev = int(output[r0, 0])
        for col in range(2, width, 2):
            val, prev = decode_delta_pixel(reader, prev)
            output[r0, col] = val
        
        for col in range(2, width, 2):
            above = int(output[r0, col])
            val, _ = decode_delta_pixel(reader, above)
            output[r1, col] = val
    
    return output


def bayer_demosaic_simple(raw, pattern='GRBG'):
    """Simple bilinear Bayer demosaicing."""
    h, w = raw.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    raw_f = raw.astype(np.float32)
    
    # For each color, create a mask and interpolate
    r = np.zeros_like(raw_f)
    g = np.zeros_like(raw_f)
    b = np.zeros_like(raw_f)
    
    if pattern == 'GRBG':
        # Row 0: G R G R ...
        # Row 1: B G B G ...
        g[0::2, 0::2] = raw_f[0::2, 0::2]
        r[0::2, 1::2] = raw_f[0::2, 1::2]
        b[1::2, 0::2] = raw_f[1::2, 0::2]
        g[1::2, 1::2] = raw_f[1::2, 1::2]
        
        # Interpolate R
        r[0::2, 0::2] = raw_f[0::2, 1::2][:, :r[0::2, 0::2].shape[1]]
        r[1::2, 1::2] = raw_f[0::2, 1::2][:r[1::2, 1::2].shape[0]]
        r[1::2, 0::2] = raw_f[0::2, 1::2][:r[1::2, 0::2].shape[0], :r[1::2, 0::2].shape[1]]
        
        # Interpolate B
        b[1::2, 1::2] = raw_f[1::2, 0::2][:, :b[1::2, 1::2].shape[1]]
        b[0::2, 0::2] = raw_f[1::2, 0::2][:b[0::2, 0::2].shape[0]]
        b[0::2, 1::2] = raw_f[1::2, 0::2][:b[0::2, 1::2].shape[0], :b[0::2, 1::2].shape[1]]
        
        # Interpolate G
        g[0::2, 1::2] = (raw_f[0::2, 0::2][:, :g[0::2, 1::2].shape[1]] + 
                          raw_f[1::2, 1::2][:g[0::2, 1::2].shape[0]]) / 2
        g[1::2, 0::2] = (raw_f[0::2, 0::2][:g[1::2, 0::2].shape[0]] + 
                          raw_f[1::2, 1::2][:, :g[1::2, 0::2].shape[1]]) / 2
    elif pattern == 'RGGB':
        r[0::2, 0::2] = raw_f[0::2, 0::2]
        g[0::2, 1::2] = raw_f[0::2, 1::2]
        g[1::2, 0::2] = raw_f[1::2, 0::2]
        b[1::2, 1::2] = raw_f[1::2, 1::2]
        
        r[0::2, 1::2] = raw_f[0::2, 0::2][:, :r[0::2, 1::2].shape[1]]
        r[1::2, :] = r[0::2, :][:r[1::2, :].shape[0]]
        
        b[1::2, 0::2] = raw_f[1::2, 1::2][:, :b[1::2, 0::2].shape[1]]
        b[0::2, :] = b[1::2, :][:b[0::2, :].shape[0]]
        
        g[0::2, 0::2] = (raw_f[0::2, 1::2][:, :g[0::2, 0::2].shape[1]] +
                          raw_f[1::2, 0::2][:g[0::2, 0::2].shape[0]]) / 2
        g[1::2, 1::2] = (raw_f[0::2, 1::2][:g[1::2, 1::2].shape[0]] +
                          raw_f[1::2, 0::2][:, :g[1::2, 1::2].shape[1]]) / 2
    
    rgb[:,:,0] = np.clip(r, 0, 255).astype(np.uint8)
    rgb[:,:,1] = np.clip(g, 0, 255).astype(np.uint8)
    rgb[:,:,2] = np.clip(b, 0, 255).astype(np.uint8)
    return rgb


def main():
    pdb_path = sys.argv[1] if len(sys.argv) > 1 else "data/3840520404.pdb"
    out_dir = "output/"
    ref_path = "data/Thom_091225_001.JPG"
    
    name, records = parse_pdb(pdb_path)
    meta = records[0]
    img_type = meta[0]
    width = 640 if meta[1] != 0 else 320
    height = 480 if meta[2] != 0 else 496
    
    print(f"PDB: {name}, Type: {img_type}, Size: {width}x{height}")
    
    data_records = records[1:-1]
    actual_height = min(height, len(data_records) * 4)
    # For type 4, height/4 records are expected but we have 120 records
    # 120 * 4 = 480 rows
    print(f"Data records: {len(data_records)}, Estimated rows: {actual_height}")
    
    ref_gray = np.array(Image.open(ref_path).convert('L'))
    ref_rgb = np.array(Image.open(ref_path).convert('RGB'))
    
    strategies = {
        'v1_concat_4pass': lambda: decode_type4(data_records, width, actual_height),
        'v2_concat_noliterals': lambda: decode_type4_v2(data_records, width, actual_height),
        'v3_per_record': lambda: decode_type4_v3(data_records, width, actual_height),
        'v4_concat_literal_start': lambda: decode_type4_v4(data_records, width, actual_height),
    }
    
    for strat_name, decode_func in strategies.items():
        print(f"\n=== Strategy: {strat_name} ===")
        try:
            raw = decode_func()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        print(f"  Output: mean={raw.mean():.1f}, std={raw.std():.1f}, "
              f"min={raw.min()}, max={raw.max()}")
        
        # Save grayscale
        gray_path = os.path.join(out_dir, f"v4_{strat_name}_gray.png")
        Image.fromarray(raw, 'L').save(gray_path)
        
        # Compare with reference grayscale
        corr_gray = np.corrcoef(raw.flatten().astype(float), 
                                ref_gray.flatten().astype(float))[0,1]
        mse_gray = np.mean((raw.astype(float) - ref_gray.astype(float))**2)
        psnr_gray = 10 * np.log10(255**2 / mse_gray) if mse_gray > 0 else 0
        print(f"  Grayscale: corr={corr_gray:.4f} PSNR={psnr_gray:.2f}dB")
        
        # Try Bayer demosaicing with different patterns
        for pattern in ['GRBG', 'RGGB']:
            try:
                rgb = bayer_demosaic_simple(raw, pattern)
                rgb_path = os.path.join(out_dir, f"v4_{strat_name}_{pattern}.jpg")
                Image.fromarray(rgb, 'RGB').save(rgb_path, quality=62)
                
                # Compare with reference RGB
                mse_rgb = np.mean((rgb.astype(float) - ref_rgb.astype(float))**2)
                psnr_rgb = 10 * np.log10(255**2 / mse_rgb) if mse_rgb > 0 else 0
                matches = np.sum(rgb == ref_rgb)
                total = ref_rgb.size
                accuracy = matches / total * 100
                print(f"  Bayer {pattern}: PSNR={psnr_rgb:.2f}dB accuracy={accuracy:.4f}%")
            except Exception as e:
                print(f"  Bayer {pattern}: ERROR: {e}")
    
    print("\nDone!")

if __name__ == '__main__':
    main()
