#!/usr/bin/env python3
"""Batch decode all PDB files with v9 decoder."""

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
        reader.flush_to_byte()
        lit = reader.read_raw_byte()
        output[r0, 1] = lit
        prev = lit
        for c in range(1, W // 2):
            val = decode_delta(reader, prev)
            output[r0, c * 2 + 1] = val
            prev = val
        reader.flush_to_byte()
        lit = reader.read_raw_byte()
        output[r1, 0] = lit
        prev = lit
        for c in range(1, W // 2):
            val = decode_delta(reader, prev)
            output[r1, c * 2] = val
            prev = val
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
    name = data[:32].split(b'\x00')[0].decode('latin-1', errors='replace')
    num_records = struct.unpack('>H', data[76:78])[0]
    offsets = []
    for i in range(num_records):
        off = 78 + i * 8
        offsets.append(struct.unpack('>I', data[off:off+4])[0])
    offsets.append(len(data))
    records = [data[offsets[i]:offsets[i+1]] for i in range(num_records)]
    # Parse metadata
    meta = records[0]
    img_type = meta[0] if len(meta) > 0 else 4
    # Internal data is always 640px wide regardless of metadata flag
    W = 640
    return name, W, img_type, records


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
    rgb[0::2, 0::2, 1] = at(Gb_p, 0, 0)
    rgb[0::2, 0::2, 0] = (at(R_p, -1, 0) + at(R_p, 0, 0)) / 2
    rgb[0::2, 0::2, 2] = (at(B_p, 0, -1) + at(B_p, 0, 0)) / 2
    rgb[0::2, 1::2, 2] = at(B_p, 0, 0)
    rgb[0::2, 1::2, 1] = (at(Gr_p, -1, 0) + at(Gr_p, 0, 0) +
                           at(Gb_p, 0, 0) + at(Gb_p, 0, 1)) / 4
    rgb[0::2, 1::2, 0] = (at(R_p, -1, 0) + at(R_p, -1, 1) +
                           at(R_p, 0, 0) + at(R_p, 0, 1)) / 4
    rgb[1::2, 0::2, 0] = at(R_p, 0, 0)
    rgb[1::2, 0::2, 1] = (at(Gb_p, 0, 0) + at(Gb_p, 1, 0) +
                           at(Gr_p, 0, -1) + at(Gr_p, 0, 0)) / 4
    rgb[1::2, 0::2, 2] = (at(B_p, 0, -1) + at(B_p, 0, 0) +
                           at(B_p, 1, -1) + at(B_p, 1, 0)) / 4
    rgb[1::2, 1::2, 1] = at(Gr_p, 0, 0)
    rgb[1::2, 1::2, 0] = (at(R_p, 0, 0) + at(R_p, 0, 1)) / 2
    rgb[1::2, 1::2, 2] = (at(B_p, 0, 0) + at(B_p, 1, 0)) / 2
    return np.clip(rgb, 0, 255).astype(np.uint8)


def decode_pdb_to_jpeg(pdb_path, out_path, quality=85):
    name, W, img_type, records = parse_pdb(pdb_path)
    data_records = records[1:-1]
    H = len(data_records) * 4
    H = (H // 4) * 4

    print(f"  {name}: {W}x{H}, type={img_type}, {len(data_records)} records")

    raw = np.zeros((H, W), dtype=np.uint8)
    for i, rec in enumerate(data_records):
        base = i * 4
        if base + 3 >= H:
            break
        decode_3pass_record(rec, raw, base, W)

    # Auto white balance (gray world)
    target = 128.0
    raw_wb = raw.astype(np.float32)
    for (rs, cs), ch_name in [((slice(0,None,2), slice(0,None,2)), 'Gb'),
                               ((slice(0,None,2), slice(1,None,2)), 'B'),
                               ((slice(1,None,2), slice(0,None,2)), 'R'),
                               ((slice(1,None,2), slice(1,None,2)), 'Gr')]:
        ch_mean = raw[rs, cs].astype(float).mean()
        gain = target / ch_mean if ch_mean > 0 else 1.0
        raw_wb[rs, cs] *= gain
        print(f"    {ch_name}: mean={ch_mean:.1f}, gain={gain:.2f}")
    raw_wb = np.clip(raw_wb, 0, 255)

    rgb = demosaic_gbrg_fast(raw_wb.astype(np.uint8))
    Image.fromarray(rgb, 'RGB').save(out_path, quality=quality)
    print(f"    → {out_path}")
    return rgb


# ============================================================================
out_dir = "output/"

pdb_files = [
    "data/3840520404.pdb",
    "data/3798526348.pdb",
]

for pdb_path in pdb_files:
    if not os.path.exists(pdb_path):
        print(f"SKIP: {pdb_path}")
        continue
    basename = os.path.splitext(os.path.basename(pdb_path))[0]
    out_path = os.path.join(out_dir, f"v9_{basename}.jpg")
    print(f"\nDecoding {os.path.basename(pdb_path)}...")
    t0 = time.time()
    decode_pdb_to_jpeg(pdb_path, out_path)
    print(f"    Time: {time.time()-t0:.1f}s")

print("\nAll done!")
