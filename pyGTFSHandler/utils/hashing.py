# -*- coding: utf-8 -*-
"""Content-hashing utilities for a GTFS file, folder, or zip archive.

Why this module exists: used to detect whether a previously-downloaded/
processed GTFS feed has actually changed before re-processing it (e.g. in
`downloaders`/caching workflows) -- independent of every other concern in
`utils/` (no polars, no GTFS parsing).
"""

import hashlib
import os
import zipfile


def hash_file(path: str, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a single file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_folder(folder_path: str, chunk_size: int = 8192) -> str:
    """Compute combined MD5 hash of all .txt files in a folder."""
    hashes = [hash_file(os.path.join(folder_path, f), chunk_size) for f in sorted(os.listdir(folder_path)) if f.endswith(".txt")]
    return hashlib.md5("".join(hashes).encode()).hexdigest()


def hash_zip(zip_path: str, chunk_size: int = 8192) -> str:
    """Compute combined MD5 hash of all .txt files in a zip archive."""
    hashes = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for f in sorted(z.namelist()):
            if f.endswith(".txt"):
                h = hashlib.md5()
                with z.open(f) as file:
                    for chunk in iter(lambda: file.read(chunk_size), b""):
                        h.update(chunk)
                hashes.append(h.hexdigest())
    return hashlib.md5("".join(hashes).encode()).hexdigest()


def hash_path(path: str, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file, folder, or zip archive."""
    if os.path.isdir(path):
        return hash_folder(path, chunk_size)
    if zipfile.is_zipfile(path):
        return hash_zip(path, chunk_size)
    if os.path.isfile(path):
        return hash_file(path, chunk_size)
    raise ValueError(f"{path} is not a file, folder, or zip archive")


def compare_paths(path1: str, path2: str) -> bool:
    """Compare any two paths by content hash."""
    return hash_path(path1) == hash_path(path2)
