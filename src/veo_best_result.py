#!/usr/bin/env python3
"""Generate best possible images from current decoder state."""

import struct
import numpy as np
from PIL import Image
import os, glob

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
    name = data[0:32].split(b'\x00')[0].decode('ascii', errors='replace')
    num_records = struct.unpack('>H', data[76:78])[0]
    offsets = []
    for i in range(num_records):
        off = 78 + i * 8
        offsets.append(struct.unpack('>I', data[off:off+4])[0])
    offsets.append(len(data))
    records = [data[offsets[i]:offsets[i+1]] for i in range(num_records)]
    meta = records[0]
    return name, records, meta

def decode_pdb_image(pdb_path, shift=633):
    """Decode a PDB file to the best grayscale image we can produce."""
    name, records, meta = parse_pdb(pdb_path)
    img_type = meta[0]
    width = 640 if meta[1] != 0 else 320
    
    data_records = records[1:-1]
    actual_height = len(data_records) * 4
    
    # Decode linearly
    all_data = bytearray()
    for rec in data_records:
        all_data.extend(decode_linear(rec, width * 4))
    
    raw = np.frombuffer(bytes(all_data[:width * actual_height]), 
                        dtype=np.uint8).reshape(actual_height, width)
    
    # Extract odd rows (luma)
    odd_rows = raw[1::2, :]
    
    # Apply global circular shift
    shifted = np.roll(odd_rows, shift, axis=1)
    
    # Apply gamma LUT (from Veo.PRC GAMM resource)
    gamm_lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        gamm_lut[i] = int(np.clip(np.power(i / 255.0, 0.45) * 255, 0, 255))
    gamma_applied = gamm_lut[shifted]
    
    # Upscale to full height
    img_half = Image.fromarray(gamma_applied, 'L')
    img_full = img_half.resize((width, actual_height // 2), Image.BILINEAR)
    
    # Also return thumbnail
    thumb_data = records[-1]
    thumb = None
    if len(thumb_data) == 2016:
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
        thumb = Image.fromarray(thumb_rgb, 'RGB')
    
    return name, img_full, shifted, odd_rows, thumb, meta

out_dir = "output/"

# Process main test file
print("=== Processing 3840520404.pdb ===")
pdb_path = "data/3840520404.pdb"
name, img, shifted, raw_odd, thumb, meta = decode_pdb_image(pdb_path)
img.save(os.path.join(out_dir, f"{name}_decoded_gray.png"))
print(f"  Saved: {name}_decoded_gray.png ({img.size})")
if thumb:
    thumb_up = thumb.resize((360, 280), Image.NEAREST)
    thumb_up.save(os.path.join(out_dir, f"{name}_thumbnail.png"))

# Save shifted with gamma for best grayscale
best_gray = Image.fromarray(shifted, 'L')
best_gray_full = np.array(best_gray.resize((640, 480), Image.BILINEAR))
ref_gray = np.array(Image.open("data/Thom_091225_001.JPG").convert('L'))
corr = np.corrcoef(best_gray_full.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  Correlation (shifted, no gamma): {corr:.4f}")

# Apply gamma
gamm_lut = np.zeros(256, dtype=np.uint8)
for i in range(256):
    gamm_lut[i] = int(np.clip(np.power(i / 255.0, 0.45) * 255, 0, 255))
gamma_full = gamm_lut[best_gray_full.astype(np.uint8)]
Image.fromarray(gamma_full, 'L').save(os.path.join(out_dir, f"{name}_best_gray_gamma.png"))
corr_g = np.corrcoef(gamma_full.flatten().astype(float), ref_gray.flatten().astype(float))[0,1]
print(f"  Correlation (shifted + gamma): {corr_g:.4f}")

# Find optimal linear transform
raw_f = best_gray_full.astype(float)
ref_f = ref_gray.astype(float)
a = np.sum((raw_f - raw_f.mean()) * (ref_f - ref_f.mean())) / np.sum((raw_f - raw_f.mean())**2)
b = ref_f.mean() - a * raw_f.mean()
scaled = np.clip(a * raw_f + b, 0, 255).astype(np.uint8)
Image.fromarray(scaled, 'L').save(os.path.join(out_dir, f"{name}_best_scaled.png"))
corr_s = np.corrcoef(scaled.flatten().astype(float), ref_f.flatten().astype(float))[0,1]
mse_s = np.mean((scaled.astype(float) - ref_f)**2)
psnr_s = 10 * np.log10(255**2 / mse_s)
print(f"  Correlation (scaled): {corr_s:.4f}, PSNR: {psnr_s:.2f}dB")

# Process additional PDB files
print("\n=== Processing additional PDB files ===")
for pdb_glob in ["data/*.pdb"]:
    for pdb_file in sorted(glob.glob(pdb_glob)):
        basename = os.path.basename(pdb_file)
        if "3840520404" in basename:
            continue  # Already processed
        try:
            name, img, shifted, raw_odd, thumb, meta = decode_pdb_image(pdb_file)
            out_path = os.path.join(out_dir, f"{name}_decoded.png")
            Image.fromarray(shifted, 'L').resize((640, 480), Image.BILINEAR).save(out_path)
            print(f"  {basename}: Type={meta[0]}, {img.size}, saved as {name}_decoded.png")
            if thumb:
                thumb.save(os.path.join(out_dir, f"{name}_thumb.png"))
                print(f"    Thumbnail saved")
        except Exception as e:
            print(f"  {basename}: ERROR - {e}")

print("\nDone!")
