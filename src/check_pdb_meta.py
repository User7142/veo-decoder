#!/usr/bin/env python3
"""Check metadata of all PDB files."""
import struct, os

def check_pdb(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()
    name = data[:32].split(b'\x00')[0].decode('latin-1', errors='replace')
    num_records = struct.unpack('>H', data[76:78])[0]
    offsets = []
    for i in range(num_records):
        off = 78 + i * 8
        offsets.append(struct.unpack('>I', data[off:off+4])[0])
    offsets.append(len(data))

    rec0 = data[offsets[0]:offsets[1]]
    print(f"\n{os.path.basename(filepath)}: {len(data)} bytes, {num_records} records")
    print(f"  Name: {name}")
    print(f"  Record 0 ({len(rec0)} bytes): {rec0.hex()}")
    if len(rec0) >= 20:
        meta = struct.unpack('>5I', rec0[:20])
        print(f"  Meta words: {meta}")
        print(f"  Width={meta[1]}, Height_code={meta[2]}, Type={meta[3]}")
    # Record sizes
    sizes = [offsets[i+1]-offsets[i] for i in range(num_records)]
    print(f"  Record sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.0f}")
    print(f"  Data records: {num_records-2} (excl. meta + thumbnail)")
    print(f"  Last record size: {sizes[-1]}")

for pdb in [
    "data/3840520404.pdb",
    "data/3798526348.pdb",
]:
    if os.path.exists(pdb):
        check_pdb(pdb)
