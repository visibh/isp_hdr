"""
Parse the DNG OpcodeList3
"""
import struct
import subprocess

import numpy as np

OPCODE_WARP_RECTILINEAR = 1
OPCODE_GAIN_MAP = 9

def parse_opcode_list3(dng_path: str) -> list[dict]:
    result = subprocess.run(
        ["exiftool", "-b", "-OpcodeList3", dng_path],
        capture_output=True,
    )
    raw = result.stdout
    if not raw:
        print("  OpcodeList3 not found hence, skipping correction.")
        return []

    def ru32(data: bytes, off: int) -> tuple[int, int]:
        return struct.unpack_from(">I", data, off)[0], off + 4

    def rf64(data: bytes, off: int) -> tuple[float, int]:
        return struct.unpack_from(">d", data, off)[0], off + 8

    offset = 0
    num_opcodes, offset = ru32(raw, offset)
    opcodes: list[dict] = []

    for _ in range(num_opcodes):
        opcode_id, offset = ru32(raw, offset)
        _dng_ver, offset = ru32(raw, offset)
        _flags, offset = ru32(raw, offset)
        param_len, offset = ru32(raw, offset)
        param = raw[offset: offset + param_len]
        offset += param_len

        if opcode_id == OPCODE_WARP_RECTILINEAR:
            poff = 0
            num_planes = struct.unpack_from(">I", param, poff)[0]; poff += 4
            planes = []
            for _ in range(num_planes):
                k = []
                for _ in range(6): # k0-k3 radial, k4-k5 tangential
                    v = struct.unpack_from(">d", param, poff)[0]; poff += 8
                    k.append(v)
                planes.append(k)
            cx = struct.unpack_from(">d", param, poff)[0]; poff += 8
            cy = struct.unpack_from(">d", param, poff)[0]; poff += 8
            opcodes.append({
                "type": "warp",
                "planes": planes,
                "cx": cx,
                "cy": cy,
            })

        elif opcode_id == OPCODE_GAIN_MAP:
            poff = 0
            top, poff = ru32(param, poff)
            left, poff = ru32(param, poff)
            bottom, poff = ru32(param, poff)
            right, poff = ru32(param, poff)
            _plane, poff = ru32(param, poff)
            _planes, poff = ru32(param, poff)
            _rp, poff = ru32(param, poff)
            _cp, poff = ru32(param, poff)
            map_pts_v, poff = ru32(param, poff)
            map_pts_h, poff = ru32(param, poff)
            spacing_v, poff = rf64(param, poff)
            spacing_h, poff = rf64(param, poff)
            origin_v, poff = rf64(param, poff)
            origin_h, poff = rf64(param, poff)
            map_planes, poff = ru32(param, poff)
            num_vals = map_pts_v * map_pts_h * map_planes
            gains_flat = np.array(
                struct.unpack_from(f">{num_vals}f", param, poff), dtype=np.float32
            )
            gains = gains_flat.reshape(map_pts_v, map_pts_h, map_planes)
            opcodes.append({
                "type": "gain_map",
                "top": top, "left": left, "bottom": bottom, "right": right,
                "map_pts_v": map_pts_v,
                "map_pts_h": map_pts_h,
                "map_planes": map_planes,
                "spacing_v": spacing_v,
                "spacing_h": spacing_h,
                "origin_v": origin_v,
                "origin_h": origin_h,
                "gains": gains,
            })

    return opcodes
