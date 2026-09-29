#!/usr/bin/env python3
"""Quick test of v9 decoder on single PDB file."""

import struct, os, time
import numpy as np
from PIL import Image


class BitReader:
    __slots__ = ('data', 'byte_pos', 'bit_pos', 'length')

    def __init__(self, data):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 7
        self.length = len(data)

    def read_bit(self):
        if self.byte_pos >= self.length:
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
        if self.byte_pos >= self.length:
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


def decode_3pass_record(rec, output, base_row, W):
    reader = BitReader(rec)
    H = output.shape[0]

    for call in range(2):
        r0 = base_row + call * 2
        r1 = r0 + 1
        if r1 >= H:
            break

        # Pass 1: row0 odd
        reader.flush_to_byte()
        lit = reader.read_raw_byte()
        output[r0, 1] = lit
        prev = lit
        for c in range(1, W // 2):
            val = decode_delta(reader, prev)
            output[r0, c * 2 + 1] = val
            prev = val

        # Pass 2: row1 even
        reader.flush_to_byte()
        lit = reader.read_raw_byte()
        output[r1, 0] = lit
        prev = lit
        for c in range(1, W // 2):
            val = decode_delta(reader, prev)
            output[r1, c * 2] = val
            prev = val

        # Pass 3: zigzag
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


def demosaic_gbrg_fast(raw_wb):
    H, W = raw_wb.shape
    hH, hW = H // 2, W // 2
    raw_f = raw_wb.astype(np.float32)

    Gb = raw_f[0::2, 0::2]
    B  = raw_f[0::2, 1::2]
    R  = raw_f[1::2, 0::2]
    Gr = raw_f[1::2, 1::2]

    Gb_p = np.pad(Gb, 1, mode='edge')
    B_p  = np.pad(B,  1, mode='edge')
    R_p  = np.pad(R,  1, mode='edge')
    Gr_p = np.pad(Gr, 1, mode='edge')

    def at(P_p, di, dj):
        return P_p[1+di:hH+1+di, 1+dj:hW+1+dj]

    rgb = np.zeros((H, W, 3), dtype=np.float32)

    # Gb at (2i, 2j)
    rgb[0::2, 0::2, 1] = at(Gb_p, 0, 0)
    rgb[0::2, 0::2, 0] = (at(R_p, -1, 0) + at(R_p, 0, 0)) / 2
    rgb[0::2, 0::2, 2] = (at(B_p, 0, -1) + at(B_p, 0, 0)) / 2

    # B at (2i, 2j+1)
    rgb[0::2, 1::2, 2] = at(B_p, 0, 0)
    rgb[0::2, 1::2, 1] = (at(Gr_p, -1, 0) + at(Gr_p, 0, 0) +
                           at(Gb_p, 0, 0) + at(Gb_p, 0, 1)) / 4
    rgb[0::2, 1::2, 0] = (at(R_p, -1, 0) + at(R_p, -1, 1) +
                           at(R_p, 0, 0) + at(R_p, 0, 1)) / 4

    # R at (2i+1, 2j)
    rgb[1::2, 0::2, 0] = at(R_p, 0, 0)
    rgb[1::2, 0::2, 1] = (at(Gb_p, 0, 0) + at(Gb_p, 1, 0) +
                           at(Gr_p, 0, -1) + at(Gr_p, 0, 0)) / 4
    rgb[1::2, 0::2, 2] = (at(B_p, 0, -1) + at(B_p, 0, 0) +
                           at(B_p, 1, -1) + at(B_p, 1, 0)) / 4

    # Gr at (2i+1, 2j+1)
    rgb[1::2, 1::2, 1] = at(Gr_p, 0, 0)
    rgb[1::2, 1::2, 0] = (at(R_p, 0, 0) + at(R_p, 0, 1)) / 2
    rgb[1::2, 1::2, 2] = (at(B_p, 0, 0) + at(B_p, 1, 0)) / 2

    return np.clip(rgb, 0, 255).astype(np.uint8)


# ============================================================================
pdb_path = "data/3840520404.pdb"
ref_path = "data/Thom_091225_001.JPG"
out_dir = "output/"

records = parse_pdb(pdb_path)
data_records = records[1:-1]
H, W = 480, 640

print(f"Decoding {len(data_records)} records → {W}x{H}...")
t0 = time.time()

raw = np.zeros((H, W), dtype=np.uint8)
for i, rec in enumerate(data_records):
    base = i * 4
    if base + 3 >= H:
        break
    decode_3pass_record(rec, raw, base, W)

t1 = time.time()
print(f"Decompression: {t1-t0:.1f}s")

# White balance
Gb_mean = raw[0::2, 0::2].astype(float).mean()
B_mean  = raw[0::2, 1::2].astype(float).mean()
R_mean  = raw[1::2, 0::2].astype(float).mean()
Gr_mean = raw[1::2, 1::2].astype(float).mean()
print(f"Channel means — R:{R_mean:.1f} Gr:{Gr_mean:.1f} Gb:{Gb_mean:.1f} B:{B_mean:.1f}")

target = 128.0
raw_wb = raw.astype(np.float32).copy()
raw_wb[0::2, 0::2] *= target / Gb_mean
raw_wb[0::2, 1::2] *= target / B_mean
raw_wb[1::2, 0::2] *= target / R_mean
raw_wb[1::2, 1::2] *= target / Gr_mean
raw_wb = np.clip(raw_wb, 0, 255)

# Demosaic
t2 = time.time()
rgb = demosaic_gbrg_fast(raw_wb.astype(np.uint8))
t3 = time.time()
print(f"Demosaic: {t3-t2:.1f}s")

# Save
out_path = os.path.join(out_dir, "v9_3840520404.jpg")
Image.fromarray(rgb, 'RGB').save(out_path, quality=85)
print(f"Saved: {out_path}")

# Compare with reference
ref_rgb = np.array(Image.open(ref_path).convert('RGB')).astype(float)
decoded = rgb.astype(float)
h = min(decoded.shape[0], ref_rgb.shape[0])
w = min(decoded.shape[1], ref_rgb.shape[1])
decoded = decoded[:h, :w]
ref_rgb = ref_rgb[:h, :w]

mse = np.mean((decoded - ref_rgb) ** 2)
psnr = 10 * np.log10(255**2 / mse)
corr = np.corrcoef(decoded.flatten(), ref_rgb.flatten())[0, 1]

for ch, name in enumerate(['Red', 'Green', 'Blue']):
    ch_mse = np.mean((decoded[:,:,ch] - ref_rgb[:,:,ch]) ** 2)
    ch_psnr = 10 * np.log10(255**2 / ch_mse) if ch_mse > 0 else float('inf')
    ch_corr = np.corrcoef(decoded[:,:,ch].flatten(), ref_rgb[:,:,ch].flatten())[0, 1]
    print(f"  {name}: PSNR={ch_psnr:.2f}dB, corr={ch_corr:.4f}")

print(f"  Overall: PSNR={psnr:.2f}dB, corr={corr:.4f}")
print(f"Total time: {time.time()-t0:.1f}s")
