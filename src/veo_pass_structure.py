#!/usr/bin/env python3
"""Test different pass structures with predictor resets."""

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

def decode_n_pixels(reader, n, prev=128):
    """Decode n pixels from the bitstream. Returns (pixels, final_prev)."""
    output = []
    for _ in range(n):
        flag = reader.read_bit()
        if flag == 1:
            output.append(prev & 0xFF)
        else:
            code = reader.read_bits(5)
            if code == 0:
                val = reader.read_bits(8)
                prev = val
                output.append(val & 0xFF)
            else:
                sign = (code >> 4) & 1
                mag = code & 0x0F
                val = ((prev - mag) if sign else (prev + mag)) & 0xFF
                prev = val
                output.append(val)
    return output, prev

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

def test_arrangement(name, decode_func):
    """Test a specific decode/arrangement strategy."""
    img = np.zeros((H, W), dtype=np.uint8)
    try:
        for rec_idx, rec in enumerate(data_records):
            decode_func(rec, rec_idx, img)
    except Exception as e:
        print(f"  {name}: ERROR at rec {rec_idx}: {e}")
        return 0
    
    corr = np.corrcoef(img.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
    mse = np.mean((img.astype(float) - ref_gray.astype(float))**2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 0
    
    fn = os.path.join(out_dir, f"pass_{name}.png")
    Image.fromarray(img, 'L').save(fn)
    print(f"  {name:45s}: corr={corr:.4f} PSNR={psnr:.2f}dB")
    return corr

# ====================================================================
# Strategy A: 2 passes per record, each pass = 1280 pixels
# Pass 1 → even cols, Pass 2 → odd cols
# ====================================================================
def strat_a(rec, rec_idx, img):
    reader = BitReader(rec)
    base_row = rec_idx * 4
    pass1, _ = decode_n_pixels(reader, 1280, 128)
    pass2, _ = decode_n_pixels(reader, 1280, 128)
    for i in range(1280):
        row = base_row + i // 320
        col = (i % 320) * 2
        if row < H: img[row, col] = pass1[i]
    for i in range(1280):
        row = base_row + i // 320
        col = (i % 320) * 2 + 1
        if row < H: img[row, col] = pass2[i]

test_arrangement("a_2pass_even_odd", strat_a)

# ====================================================================
# Strategy B: Same but odd first
# ====================================================================
def strat_b(rec, rec_idx, img):
    reader = BitReader(rec)
    base_row = rec_idx * 4
    pass1, _ = decode_n_pixels(reader, 1280, 128)
    pass2, _ = decode_n_pixels(reader, 1280, 128)
    for i in range(1280):
        row = base_row + i // 320
        col = (i % 320) * 2 + 1
        if row < H: img[row, col] = pass1[i]
    for i in range(1280):
        row = base_row + i // 320
        col = (i % 320) * 2
        if row < H: img[row, col] = pass2[i]

test_arrangement("b_2pass_odd_even", strat_b)

# ====================================================================
# Strategy C: 4 passes, each 640 pixels
# P1: row0 even, P2: row1 even, P3: row0 odd, P4: row1 odd
# (then repeat for rows 2-3)
# ====================================================================
def strat_c(rec, rec_idx, img):
    reader = BitReader(rec)
    base_row = rec_idx * 4
    for row_pair in range(2):
        # Even pixels first
        p_r0_even, _ = decode_n_pixels(reader, 320, 128)
        p_r1_even, _ = decode_n_pixels(reader, 320, 128)
        # Odd pixels
        p_r0_odd, _ = decode_n_pixels(reader, 320, 128)
        p_r1_odd, _ = decode_n_pixels(reader, 320, 128)
        r0 = base_row + row_pair * 2
        r1 = r0 + 1
        for c in range(320):
            if r0 < H:
                img[r0, c*2] = p_r0_even[c]
                img[r0, c*2+1] = p_r0_odd[c]
            if r1 < H:
                img[r1, c*2] = p_r1_even[c]
                img[r1, c*2+1] = p_r1_odd[c]

test_arrangement("c_4pass_320_even_odd_per_row", strat_c)

# ====================================================================
# Strategy D: 2 passes of 1280 each, but interleave per ROW, not per col
# Pass 1 → rows 0,2 of 4-row group, Pass 2 → rows 1,3
# ====================================================================
def strat_d(rec, rec_idx, img):
    reader = BitReader(rec)
    base_row = rec_idx * 4
    pass1, _ = decode_n_pixels(reader, 1280, 128)
    pass2, _ = decode_n_pixels(reader, 1280, 128)
    # Pass 1 → rows 0,2
    for i in range(640):
        row = base_row + 0
        if row < H: img[row, i] = pass1[i]
    for i in range(640):
        row = base_row + 2
        if row < H: img[row, i] = pass1[640 + i]
    # Pass 2 → rows 1,3
    for i in range(640):
        row = base_row + 1
        if row < H: img[row, i] = pass2[i]
    for i in range(640):
        row = base_row + 3
        if row < H: img[row, i] = pass2[640 + i]

test_arrangement("d_2pass_rows_02_13", strat_d)

# ====================================================================
# Strategy E: Reverse of D
# ====================================================================
def strat_e(rec, rec_idx, img):
    reader = BitReader(rec)
    base_row = rec_idx * 4
    pass1, _ = decode_n_pixels(reader, 1280, 128)
    pass2, _ = decode_n_pixels(reader, 1280, 128)
    # Pass 1 → rows 1,3
    for i in range(640):
        row = base_row + 1
        if row < H: img[row, i] = pass1[i]
    for i in range(640):
        row = base_row + 3
        if row < H: img[row, i] = pass1[640 + i]
    # Pass 2 → rows 0,2
    for i in range(640):
        row = base_row + 0
        if row < H: img[row, i] = pass2[i]
    for i in range(640):
        row = base_row + 2
        if row < H: img[row, i] = pass2[640 + i]

test_arrangement("e_2pass_rows_13_02", strat_e)

# ====================================================================
# Strategy F: 1 continuous pass, no predictor reset, but put in
# column-interleaved order: value[n] goes to col = n%640 
# where first 320 values → even cols, next 320 → odd cols of same row
# ====================================================================
def strat_f(rec, rec_idx, img):
    reader = BitReader(rec)
    base_row = rec_idx * 4
    pixels, _ = decode_n_pixels(reader, 2560, 128)
    for row_in_rec in range(4):
        row = base_row + row_in_rec
        if row >= H: break
        row_start = row_in_rec * 640
        # First 320 pixels = even columns
        for c in range(320):
            img[row, c * 2] = pixels[row_start + c]
        # Next 320 pixels = odd columns
        for c in range(320):
            img[row, c * 2 + 1] = pixels[row_start + 320 + c]

test_arrangement("f_linear_col_deinterleave", strat_f)

# ====================================================================
# Strategy G: Same as F but odd first
# ====================================================================
def strat_g(rec, rec_idx, img):
    reader = BitReader(rec)
    base_row = rec_idx * 4
    pixels, _ = decode_n_pixels(reader, 2560, 128)
    for row_in_rec in range(4):
        row = base_row + row_in_rec
        if row >= H: break
        row_start = row_in_rec * 640
        # First 320 pixels = odd columns
        for c in range(320):
            img[row, c * 2 + 1] = pixels[row_start + c]
        # Next 320 pixels = even columns
        for c in range(320):
            img[row, c * 2] = pixels[row_start + 320 + c]

test_arrangement("g_linear_col_deinterleave_odd_first", strat_g)

# ====================================================================
# Strategy H: Decode record as 2 passes of 1280 each
# Pass 1: continuous predictor → rows 0,1 as-is (full rows)
# Pass 2: predictor reset → rows 2,3 as-is (full rows)
# ====================================================================
def strat_h(rec, rec_idx, img):
    reader = BitReader(rec)
    base_row = rec_idx * 4
    pass1, _ = decode_n_pixels(reader, 1280, 128)
    pass2, _ = decode_n_pixels(reader, 1280, 128)
    for i in range(1280):
        row = base_row + i // 640
        col = i % 640
        if row < H: img[row, col] = pass1[i]
    for i in range(1280):
        row = base_row + 2 + i // 640
        col = i % 640
        if row < H: img[row, col] = pass2[i]

test_arrangement("h_2pass_sequential_rows", strat_h)

# ====================================================================
# Strategy I: What if each record only has 2 actual rows, not 4?
# Decode 1280 pixels per record (not 2560), predictor reset per record
# ====================================================================
def strat_i(rec, rec_idx, img):
    reader = BitReader(rec)
    base_row = rec_idx * 2
    pixels, _ = decode_n_pixels(reader, 1280, 128)
    for i in range(1280):
        row = base_row + i // 640
        col = i % 640
        if row < H: img[row, col] = pixels[i]

test_arrangement("i_2rows_per_record_1280px", strat_i)

# ====================================================================
# Strategy J: Decode with previous pixel from left as predictor
# But also use the pixel from the row above when starting a new row
# ====================================================================
def strat_j(rec, rec_idx, img):
    reader = BitReader(rec)
    base_row = rec_idx * 4
    for local_row in range(4):
        row = base_row + local_row
        if row >= H: break
        # Start of new row: use pixel from above as initial predictor
        if row > 0:
            prev = int(img[row - 1, 0])
        else:
            prev = 128
        for col in range(W):
            flag = reader.read_bit()
            if flag == 1:
                img[row, col] = prev & 0xFF
            else:
                code = reader.read_bits(5)
                if code == 0:
                    val = reader.read_bits(8)
                    prev = val
                    img[row, col] = val & 0xFF
                else:
                    sign = (code >> 4) & 1
                    mag = code & 0x0F
                    val = ((prev - mag) if sign else (prev + mag)) & 0xFF
                    prev = val
                    img[row, col] = val

test_arrangement("j_prev_row_start_predictor", strat_j)

print("\nDone!")
