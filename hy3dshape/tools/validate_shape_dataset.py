#!/usr/bin/env python3
"""Validate the preprocessed dataset consumed by Shape DiT training.

The structural checks use only Python's standard library so this command can
run before the ML environment is installed. When NumPy is available, it also
checks that surface arrays contain finite numeric values.
"""

import argparse
import ast
import importlib.util
import json
import struct
import sys
import zipfile
import zlib
from pathlib import Path


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
REQUIRED_SURFACE_KEYS = ("random_surface", "sharp_surface")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Preprocessed root directory or JSON list")
    parser.add_argument("--views", type=int, default=24)
    parser.add_argument("--pc-size", type=int, default=81920)
    parser.add_argument("--pc-sharpedge-size", type=int, default=0)
    parser.add_argument("--max-errors", type=int, default=100)
    args = parser.parse_args()
    if args.views != 24:
        parser.error("--views must be 24 because the current dataset loader uses exactly 24 renders")
    if args.pc_size <= 0:
        parser.error("--pc-size must be greater than zero")
    if args.pc_sharpedge_size < 0:
        parser.error("--pc-sharpedge-size must be zero or greater")
    if args.max_errors <= 0:
        parser.error("--max-errors must be greater than zero")
    return args


def resolve_object_dirs(source):
    source = Path(source).expanduser().resolve()
    if source.suffix.lower() == ".json":
        with source.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)
        if not isinstance(entries, list):
            raise TypeError(f"Dataset JSON must contain a list: {source}")
        return [
            (Path(item) if Path(item).is_absolute() else source.parent / item).resolve()
            for item in entries
        ]

    if not source.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {source}")
    return sorted(path.resolve() for path in source.iterdir() if path.is_dir())


def read_png_header(path):
    with path.open("rb") as handle:
        if handle.read(8) != PNG_SIGNATURE:
            raise ValueError("not a valid PNG file")

        ihdr = None
        saw_idat = False
        saw_iend = False
        while not saw_iend:
            length_bytes = handle.read(4)
            if len(length_bytes) != 4:
                raise ValueError("truncated PNG chunk length")
            chunk_length = struct.unpack(">I", length_bytes)[0]
            chunk_type = handle.read(4)
            chunk_data = handle.read(chunk_length)
            crc_bytes = handle.read(4)
            if len(chunk_type) != 4 or len(chunk_data) != chunk_length or len(crc_bytes) != 4:
                raise ValueError("truncated PNG chunk")
            expected_crc = struct.unpack(">I", crc_bytes)[0]
            actual_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                raise ValueError(f"CRC mismatch in {chunk_type.decode('ascii', errors='replace')} chunk")

            if ihdr is None:
                if chunk_type != b"IHDR" or chunk_length != 13:
                    raise ValueError("first PNG chunk must be a 13-byte IHDR")
                ihdr = chunk_data
            elif chunk_type == b"IDAT":
                saw_idat = True
            elif chunk_type == b"IEND":
                if chunk_length != 0:
                    raise ValueError("IEND chunk must be empty")
                saw_iend = True

        if not saw_idat:
            raise ValueError("PNG contains no IDAT image data")

    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", ihdr
    )
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid dimensions {width}x{height}")
    if bit_depth != 8 or color_type != 6:
        raise ValueError(
            f"expected 8-bit RGBA PNG, got bit_depth={bit_depth}, color_type={color_type}"
        )
    if compression != 0 or filtering != 0 or interlace not in (0, 1):
        raise ValueError("unsupported PNG encoding")

    if importlib.util.find_spec("PIL") is not None:
        from PIL import Image

        with Image.open(path) as image:
            image.load()
            if image.mode != "RGBA":
                raise ValueError(f"expected decoded mode RGBA, got {image.mode}")
            if image.getchannel("A").getbbox() is None:
                raise ValueError("alpha channel is completely transparent")
    return width, height


def read_npy_header(handle):
    if handle.read(6) != b"\x93NUMPY":
        raise ValueError("invalid NPY magic")
    major, minor = struct.unpack("BB", handle.read(2))
    if major == 1:
        header_len = struct.unpack("<H", handle.read(2))[0]
        encoding = "latin1"
    elif major in (2, 3):
        header_len = struct.unpack("<I", handle.read(4))[0]
        encoding = "utf-8" if major == 3 else "latin1"
    else:
        raise ValueError(f"unsupported NPY version {major}.{minor}")

    metadata = ast.literal_eval(handle.read(header_len).decode(encoding).strip())
    shape = metadata.get("shape")
    dtype = metadata.get("descr")
    if not isinstance(shape, tuple) or not isinstance(dtype, str):
        raise ValueError("unsupported NPY metadata")
    return shape, dtype


def validate_npz_structure(path, pc_size, pc_sharpedge_size):
    minimum_rows = {
        "random_surface": pc_size,
        "sharp_surface": pc_sharpedge_size,
    }
    metadata = {}
    with zipfile.ZipFile(path, "r") as archive:
        members = set(archive.namelist())
        for key in REQUIRED_SURFACE_KEYS:
            member = f"{key}.npy"
            if member not in members:
                raise ValueError(f"missing array '{key}'")
            with archive.open(member, "r") as handle:
                shape, dtype = read_npy_header(handle)
            if len(shape) != 2 or shape[1] < 6:
                raise ValueError(f"array '{key}' must have shape (N, >=6), got {shape}")
            if shape[0] < minimum_rows[key]:
                raise ValueError(
                    f"array '{key}' has {shape[0]} rows; requires {minimum_rows[key]}"
                )
            dtype_kind = dtype.lstrip("<>=|")[:1]
            if dtype_kind not in {"f", "i", "u"}:
                raise ValueError(f"array '{key}' must be numeric, got dtype {dtype}")
            metadata[key] = (shape, dtype)
    return metadata


def validate_npz_values(path):
    if importlib.util.find_spec("numpy") is None:
        return []

    import numpy as np

    warnings = []
    with np.load(path, allow_pickle=False) as archive:
        for key in REQUIRED_SURFACE_KEYS:
            values = archive[key]
            if not np.issubdtype(values.dtype, np.number):
                raise ValueError(f"array '{key}' has non-numeric dtype {values.dtype}")
            if not np.isfinite(values).all():
                raise ValueError(f"array '{key}' contains NaN or infinity")
            if values.shape[0] and np.any(np.linalg.norm(values[:, 3:6], axis=1) == 0):
                warnings.append(f"array '{key}' contains zero-length normals")
    return warnings


def main():
    args = parse_args()
    errors = []
    warnings = []
    if importlib.util.find_spec("numpy") is None:
        warnings.append("NumPy is not installed; finite-value checks were skipped")
    if importlib.util.find_spec("PIL") is None:
        warnings.append("Pillow is not installed; decoded-pixel/alpha checks were skipped")

    try:
        object_dirs = resolve_object_dirs(args.source)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if not object_dirs:
        print("ERROR: no object directories found", file=sys.stderr)
        return 1

    seen_uids = {}
    checked_images = 0
    checked_npz = 0

    for object_dir in object_dirs:
        uid = object_dir.name
        uid_key = uid.casefold()
        if uid_key in seen_uids:
            warnings.append(f"duplicate UID basename: {seen_uids[uid_key]} and {uid}")
        seen_uids[uid_key] = uid

        if not object_dir.is_dir():
            errors.append(f"object directory does not exist: {object_dir}")
            continue

        render_dir = object_dir / "render_cond"
        surface_path = object_dir / "geo_data" / f"{uid}_surface.npz"
        longest_render_path = render_dir / f"{args.views - 1:03d}.png"
        for candidate in (longest_render_path, surface_path):
            if len(str(candidate)) >= 240:
                warnings.append(f"path is close to the legacy Windows limit: {candidate}")

        dimensions = set()
        for index in range(args.views):
            image_path = render_dir / f"{index:03d}.png"
            try:
                dimensions.add(read_png_header(image_path))
                checked_images += 1
            except Exception as error:
                errors.append(f"{uid}: {image_path.name}: {error}")
        if len(dimensions) > 1:
            errors.append(f"{uid}: render dimensions are inconsistent: {sorted(dimensions)}")

        try:
            validate_npz_structure(surface_path, args.pc_size, args.pc_sharpedge_size)
            checked_npz += 1
            warnings.extend(f"{uid}: {message}" for message in validate_npz_values(surface_path))
        except Exception as error:
            errors.append(f"{uid}: {surface_path.name}: {error}")

        if len(errors) >= args.max_errors:
            errors.append(f"stopped after reaching --max-errors={args.max_errors}")
            break

    print(
        f"Checked {len(object_dirs)} objects, {checked_images} RGBA renders, "
        f"and {checked_npz} surface archives."
    )
    for warning in dict.fromkeys(warnings):
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)

    if errors:
        print(f"Dataset validation FAILED with {len(errors)} error(s).", file=sys.stderr)
        return 1
    print("Dataset validation PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
