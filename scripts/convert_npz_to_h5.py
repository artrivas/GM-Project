#!/usr/bin/env python3
"""Convert recorded CoinRun NPZ episodes into Orbis-compatible HDF5 videos."""

from __future__ import annotations

import argparse
import importlib.util
import io
import sys
import types
import zipfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


DEFAULT_FRAME_KEYS = (
    "observations",
    "frames",
    "obs",
    "observation",
    "rgb",
    "images",
    "video",
)


@dataclass(frozen=True)
class NpzSource:
    display_name: str
    relative_h5: Path
    filesystem_path: Path | None = None
    zip_path: Path | None = None
    zip_member: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Procgen CoinRun .npz recordings to one .h5 file per video "
            "for Orbis MultiHDF5Dataset. Handles nested folders and .npz files inside .zip archives."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, default=15.0)
    parser.add_argument("--target-fps", type=float, default=5.0)
    parser.add_argument("--frames-key", default="frames")
    parser.add_argument("--npz-frame-key", default=None)
    parser.add_argument(
        "--compression",
        default="lzf",
        choices=("lzf", "gzip", "none"),
        help="HDF5 compression. lzf is usually faster for training-time loading.",
    )
    parser.add_argument(
        "--gdrive-url",
        default=None,
        help="Optional Google Drive folder URL. Requires gdown and downloads into --input-dir.",
    )
    parser.add_argument(
        "--no-gdown-resume",
        action="store_true",
        help="Disable gdown resume mode. Resume is enabled by default.",
    )
    parser.add_argument(
        "--gdown-remaining-ok",
        action="store_true",
        help="Ask gdown to keep downloaded folder contents even if some files cannot be fetched.",
    )
    parser.add_argument(
        "--ignore-download-errors",
        action="store_true",
        help="Continue conversion with already downloaded files if gdown fails.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan files directly inside --input-dir.",
    )
    parser.add_argument(
        "--skip-zip",
        action="store_true",
        help="Ignore .zip archives instead of reading .npz files from them.",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write all .h5 files directly into --output-dir. By default, preserve team/session folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .h5 files.",
    )
    parser.add_argument(
        "--skip-existing-h5",
        action="store_true",
        help="Resume conversion by keeping valid existing .h5 files and adding them to the manifest.",
    )
    parser.add_argument(
        "--validate-orbis",
        action="store_true",
        help="Instantiate MultiHDF5Dataset from --orbis-custom-py after conversion.",
    )
    parser.add_argument(
        "--orbis-custom-py",
        type=Path,
        default=Path("custom.py"),
        help="Path to Orbis custom.py containing MultiHDF5Dataset.",
    )
    return parser.parse_args()


def maybe_download_gdrive(
    url: str | None,
    output_dir: Path,
    resume: bool,
    remaining_ok: bool,
    ignore_errors: bool,
) -> None:
    if not url:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit(
            "gdown is required for --gdrive-url. Install it with: pip install gdown"
        ) from exc
    print(f"Downloading Google Drive folder into {output_dir}...")
    try:
        gdown.download_folder(
            url=url,
            output=str(output_dir),
            quiet=False,
            use_cookies=False,
            resume=resume,
            remaining_ok=remaining_ok,
        )
    except Exception as exc:
        if not ignore_errors:
            raise
        print(f"Warning: gdown failed, continuing with files already present in {output_dir}: {exc}")


def safe_relative_h5(path: Path) -> Path:
    clean_parts = []
    for part in path.with_suffix(".h5").parts:
        if part in ("", ".", ".."): 
            continue
        clean_parts.append(part.replace(":", "_"))
    return Path(*clean_parts)


def collect_sources(input_dir: Path, recursive: bool, include_zip: bool, flat_output: bool) -> list[NpzSource]:
    pattern = "**/*" if recursive else "*"
    paths = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    sources: list[NpzSource] = []
    used_flat_names: set[str] = set()

    def unique_flat_name(rel: Path) -> Path:
        base = "__".join(rel.with_suffix(".h5").parts)
        candidate = base
        index = 2
        while candidate in used_flat_names:
            stem = Path(base).stem
            candidate = f"{stem}_{index}.h5"
            index += 1
        used_flat_names.add(candidate)
        return Path(candidate)

    def output_rel(rel: Path) -> Path:
        h5_rel = safe_relative_h5(rel)
        return unique_flat_name(h5_rel) if flat_output else h5_rel

    for path in paths:
        suffix = path.suffix.lower()
        rel = path.relative_to(input_dir)
        if suffix == ".npz":
            sources.append(
                NpzSource(
                    display_name=str(path),
                    relative_h5=output_rel(rel),
                    filesystem_path=path,
                )
            )
        elif suffix == ".zip" and include_zip:
            try:
                with zipfile.ZipFile(path) as archive:
                    members = sorted(
                        name for name in archive.namelist()
                        if not name.endswith("/") and name.lower().endswith(".npz")
                    )
            except zipfile.BadZipFile as exc:
                raise ValueError(f"{path} is not a valid zip file") from exc
            if not members:
                print(f"Warning: {path} contains no .npz files")
                continue
            for member in members:
                member_path = Path(member)
                if member_path.parts and member_path.parts[0] == rel.stem:
                    member_rel = rel.parent / member_path
                else:
                    member_rel = rel.with_suffix("") / member_path
                sources.append(
                    NpzSource(
                        display_name=f"{path}!{member}",
                        relative_h5=output_rel(member_rel),
                        zip_path=path,
                        zip_member=member,
                    )
                )

    if not sources:
        raise SystemExit(f"No .npz files found in {input_dir} or supported .zip archives")
    return sources


def open_npz_source(source: NpzSource) -> np.lib.npyio.NpzFile:
    if source.filesystem_path is not None:
        return np.load(source.filesystem_path)
    if source.zip_path is None or source.zip_member is None:
        raise ValueError(f"Invalid source: {source}")
    with zipfile.ZipFile(source.zip_path) as archive:
        data = archive.read(source.zip_member)
    return np.load(io.BytesIO(data))


def choose_frame_key(npz: np.lib.npyio.NpzFile, explicit_key: str | None, source_name: str) -> str:
    keys = set(npz.files)
    if explicit_key:
        if explicit_key not in keys:
            raise ValueError(f"{source_name}: requested key {explicit_key!r} not found. Keys: {npz.files}")
        return explicit_key
    for key in DEFAULT_FRAME_KEYS:
        if key in keys:
            return key
    candidates = []
    for key in npz.files:
        arr = npz[key]
        if arr.ndim in (3, 4):
            candidates.append(key)
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"{source_name}: could not auto-detect frame key. Keys: {npz.files}. "
        "Pass --npz-frame-key explicitly."
    )


def normalize_frames(frames: np.ndarray, source_name: str, key: str) -> np.ndarray:
    frames = np.asarray(frames)

    if frames.ndim != 4:
        raise ValueError(f"{source_name}:{key} must be rank 4 [T,H,W,C] or [T,C,H,W], got {frames.shape}")

    if frames.shape[-1] == 3:
        pass
    elif frames.shape[1] == 3:
        frames = np.transpose(frames, (0, 2, 3, 1))
    else:
        raise ValueError(
            f"{source_name}:{key} must have 3 color channels in channel-first or channel-last format, "
            f"got {frames.shape}"
        )

    if frames.shape[1:4] != (64, 64, 3):
        raise ValueError(f"{source_name}:{key} must be [T,64,64,3] after normalization, got {frames.shape}")

    if np.issubdtype(frames.dtype, np.floating):
        max_value = float(np.nanmax(frames)) if frames.size else 0.0
        if max_value <= 1.0:
            frames = frames * 255.0
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    elif frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    if len(frames) == 0:
        raise ValueError(f"{source_name}:{key} contains no frames")

    return np.ascontiguousarray(frames)


def frame_stride(source_fps: float, target_fps: float) -> int:
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("--source-fps and --target-fps must be positive")
    if target_fps > source_fps:
        raise ValueError("--target-fps cannot be greater than --source-fps for downsampling")
    return max(1, int(round(source_fps / target_fps)))


def write_h5(
    source_name: str,
    output_path: Path,
    frames: np.ndarray,
    frames_key: str,
    source_fps: float,
    target_fps: float,
    stride: int,
    compression: str,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h5_compression = None if compression == "none" else compression
    with h5py.File(output_path, "w") as h5:
        h5.create_dataset(
            frames_key,
            data=frames,
            dtype=np.uint8,
            compression=h5_compression,
            chunks=(min(len(frames), 64), 64, 64, 3),
        )
        h5.attrs["source"] = source_name
        h5.attrs["source_fps"] = float(source_fps)
        h5.attrs["target_fps"] = float(target_fps)
        h5.attrs["downsample_stride"] = int(stride)


def inspect_h5(path: Path, frames_key: str) -> int:
    with h5py.File(path, "r") as h5:
        if frames_key not in h5:
            raise AssertionError(f"{path}: missing key {frames_key!r}")
        frames = h5[frames_key]
        if frames.ndim != 4:
            raise AssertionError(f"{path}:{frames_key} must be rank 4, got {frames.shape}")
        if frames.shape[1:4] != (64, 64, 3):
            raise AssertionError(f"{path}:{frames_key} must have frame shape 64x64x3, got {frames.shape}")
        if frames.dtype != np.uint8:
            raise AssertionError(f"{path}:{frames_key} must be uint8, got {frames.dtype}")
        if len(frames) < 1:
            raise AssertionError(f"{path}:{frames_key} is empty")
        return len(frames)


def load_orbis_dataset_class(custom_py: Path):
    custom_py = custom_py.resolve()
    if not custom_py.exists():
        raise FileNotFoundError(f"Cannot find Orbis custom.py at {custom_py}")

    # Local custom.py imports ImagePaths even though MultiHDF5Dataset does not use it.
    # Provide a minimal stub if the full Orbis data package is not present.
    if "data" not in sys.modules:
        sys.modules["data"] = types.ModuleType("data")
    if "data.base" not in sys.modules:
        base_module = types.ModuleType("data.base")

        class ImagePaths:  # pragma: no cover - only for import compatibility.
            pass

        base_module.ImagePaths = ImagePaths
        sys.modules["data.base"] = base_module

    spec = importlib.util.spec_from_file_location("orbis_custom_local", custom_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {custom_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MultiHDF5Dataset


def validate_orbis_loader(manifest: Path, custom_py: Path) -> None:
    DatasetClass = load_orbis_dataset_class(custom_py)
    dataset = DatasetClass(size=64, hdf5_paths_file=str(manifest))
    try:
        total = len(dataset)
        if total < 1:
            raise AssertionError("MultiHDF5Dataset length is zero")
        sample = dataset[0]
        shape = tuple(sample.shape)
        if shape != (3, 64, 64):
            raise AssertionError(f"Expected sample shape [3,64,64], got {shape}")
        min_value = float(sample.min())
        max_value = float(sample.max())
        if min_value < -1.001 or max_value > 1.001:
            raise AssertionError(f"Expected normalized range [-1,1], got [{min_value}, {max_value}]")
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()


def convert(args: argparse.Namespace) -> list[Path]:
    maybe_download_gdrive(
        args.gdrive_url,
        args.input_dir,
        resume=not args.no_gdown_resume,
        remaining_ok=args.gdown_remaining_ok,
        ignore_errors=args.ignore_download_errors,
    )

    sources = collect_sources(
        args.input_dir,
        recursive=not args.no_recursive,
        include_zip=not args.skip_zip,
        flat_output=args.flat_output,
    )
    stride = frame_stride(args.source_fps, args.target_fps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    total_input_frames = 0
    total_output_frames = 0

    for source in sources:
        output_path = args.output_dir / source.relative_h5
        if output_path.exists() and args.skip_existing_h5 and not args.overwrite:
            written_frames = inspect_h5(output_path, args.frames_key)
            outputs.append(output_path.resolve())
            total_output_frames += written_frames
            print(f"Skipping existing {output_path} ({written_frames} frames)")
            continue

        with open_npz_source(source) as npz:
            frame_key = choose_frame_key(npz, args.npz_frame_key, source.display_name)
            frames = normalize_frames(npz[frame_key], source.display_name, frame_key)

        sampled = frames[::stride]
        if len(sampled) == 0:
            sampled = frames[:1]

        write_h5(
            source_name=source.display_name,
            output_path=output_path,
            frames=sampled,
            frames_key=args.frames_key,
            source_fps=args.source_fps,
            target_fps=args.target_fps,
            stride=stride,
            compression=args.compression,
            overwrite=args.overwrite,
        )
        written_frames = inspect_h5(output_path, args.frames_key)
        outputs.append(output_path.resolve())
        total_input_frames += len(frames)
        total_output_frames += written_frames
        print(f"{source.display_name} -> {output_path} ({len(frames)} frames -> {written_frames})")

    args.manifest.write_text(
        "".join(f"{path}\n" for path in outputs),
        encoding="utf-8",
    )

    print(f"Wrote manifest: {args.manifest.resolve()}")
    print(
        f"Converted {len(outputs)} videos, {total_input_frames} input frames, "
        f"{total_output_frames} output frames, stride={stride}."
    )
    return outputs


def main() -> None:
    args = parse_args()
    convert(args)
    if args.validate_orbis:
        validate_orbis_loader(args.manifest, args.orbis_custom_py)
        print("Orbis MultiHDF5Dataset validation passed.")


if __name__ == "__main__":
    main()


