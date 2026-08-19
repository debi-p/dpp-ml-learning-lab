from __future__ import annotations

import shutil
import struct
import subprocess
import zlib
from pathlib import Path


BG = (247, 243, 234, 255)
ACCENT = (155, 90, 46, 255)
WHITE = (255, 255, 255, 255)


def create_icon(project_root: Path) -> Path:
    assets = project_root / "assets"
    iconset = assets / "Marg.iconset"
    icon_path = assets / "marg_icon.icns"
    assets.mkdir(exist_ok=True)
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir()

    sizes = {
        "icon_16x16.png": 16,
        "icon_16x16@2x.png": 32,
        "icon_32x32.png": 32,
        "icon_32x32@2x.png": 64,
        "icon_128x128.png": 128,
        "icon_128x128@2x.png": 256,
        "icon_256x256.png": 256,
        "icon_256x256@2x.png": 512,
        "icon_512x512.png": 512,
        "icon_512x512@2x.png": 1024,
    }
    for filename, size in sizes.items():
        write_png(iconset / filename, size)

    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icon_path)], check=True)
    return icon_path


def write_png(path: Path, size: int) -> None:
    rows = []
    radius = size * 0.34
    center = size / 2
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            dx = x + 0.5 - center
            dy = y + 0.5 - center
            distance = (dx * dx + dy * dy) ** 0.5
            color = BG
            if distance <= radius:
                color = ACCENT
            row.extend(draw_letter_m(x, y, size, color))
        rows.append(bytes(row))

    raw = b"".join(rows)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, level=9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def draw_letter_m(x: int, y: int, size: int, base_color: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    left = int(size * 0.33)
    right = int(size * 0.67)
    top = int(size * 0.35)
    bottom = int(size * 0.68)
    thickness = max(2, size // 18)
    on_vertical = (abs(x - left) <= thickness or abs(x - right) <= thickness) and top <= y <= bottom
    diagonal_left = abs((x - left) - (y - top) * 0.5) <= thickness and top <= y <= int(size * 0.55)
    diagonal_right = abs((right - x) - (y - top) * 0.5) <= thickness and top <= y <= int(size * 0.55)
    if base_color == ACCENT and (on_vertical or diagonal_left or diagonal_right):
        return WHITE
    return base_color


def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


if __name__ == "__main__":
    print(create_icon(Path(__file__).resolve().parents[1]))
