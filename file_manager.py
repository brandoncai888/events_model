import argparse
import math
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = "data"
SOURCE_NOISE = "noise"
SOURCE_OBJECT = "object"
SOURCE_OTHER = "other"
SOURCES = (SOURCE_NOISE, SOURCE_OBJECT, SOURCE_OTHER)

ARTIFACT_EVENTS = "events"
ARTIFACT_IETS = "iets"
ARTIFACT_PICTURES = "pictures"
ARTIFACT_VIDEOS = "videos"
ARTIFACT_TRACKS = "tracks"
ARTIFACTS = (ARTIFACT_EVENTS, ARTIFACT_IETS, ARTIFACT_PICTURES, ARTIFACT_VIDEOS, ARTIFACT_TRACKS)


@dataclass(frozen=True)
class ManagedPathParts:
    source: str | None = None
    dataset: str | None = None
    slice_name: str | None = None
    artifact: str | None = None


def number_label(value):
    """Return the stable number text used in filenames."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip()
    return str(round(float(value), 12))


def hz_label(rate):
    if rate is None:
        raise ValueError("rate is required to build a noise dataset path.")
    return f"{number_label(rate)}Hz"


def seconds_label(duration):
    if duration is None:
        raise ValueError("duration is required to build a noise filename.")
    return f"{number_label(duration)}s"


def clean_part(value):
    """Make one path component safe without hiding the user's chosen name."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise ValueError("Path component cannot be empty.")
    for char in '<>:"/\\|?*':
        text = text.replace(char, "-")
    return text


def normalize_source(source):
    source = SOURCE_NOISE if source is None else source.lower()
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}; got {source!r}.")
    return source


def normalize_artifact(artifact):
    artifact = artifact.lower()
    if artifact not in ARTIFACTS:
        raise ValueError(f"artifact must be one of {ARTIFACTS}; got {artifact!r}.")
    return artifact


def default_dataset(source=None, dataset=None, rate=None, name=None, stem=None):
    source = normalize_source(source)
    if dataset is not None:
        return clean_part(dataset)
    if name is not None:
        return clean_part(Path(str(name)).stem)
    if source == SOURCE_NOISE:
        return clean_part(hz_label(rate))
    if stem is not None:
        return clean_part(str(stem).split("_")[0])
    raise ValueError("dataset or name is required for object data.")


def time_slice_name(start=None, end=None, slice_name=None):
    if slice_name:
        return clean_part(slice_name)
    if start is None and end is None:
        return None
    return clean_part(f"{number_label(start)}_{number_label(end)}")


def is_default_full_window(start, end):
    return float(start) == 0.0 and math.isinf(float(end))


def slice_from_window(start, end, slice_name=None):
    if slice_name:
        return clean_part(slice_name)
    if start is None or end is None:
        return None
    if is_default_full_window(start, end):
        return None
    return time_slice_name(start, end)


def stem_has_part(stem, part):
    stem = str(stem)
    part = clean_part(part)
    return stem == part or stem.endswith(f"_{part}") or f"_{part}_" in stem


def artifact_dir(
    artifact,
    *,
    data_root=DEFAULT_DATA_ROOT,
    source=None,
    dataset=None,
    rate=None,
    name=None,
    stem=None,
    slice_name=None,
    slice_start=None,
    slice_end=None,
    create=False,
):
    source = normalize_source(source)
    artifact = normalize_artifact(artifact)
    dataset = default_dataset(source, dataset=dataset, rate=rate, name=name, stem=stem)
    path = Path(data_root) / source / dataset

    time_slice = time_slice_name(slice_start, slice_end, slice_name=slice_name)
    if time_slice:
        path = path / time_slice

    path = path / artifact
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_stem(
    *,
    source=None,
    dataset=None,
    rate=None,
    duration=None,
    name=None,
    stem=None,
    slice_name=None,
    slice_start=None,
    slice_end=None,
    polarity=None,
    line=None,
):
    source = normalize_source(source)
    if stem is not None:
        base = str(stem)
    elif source == SOURCE_NOISE:
        base = f"poisson_noise_{hz_label(rate)}_{seconds_label(duration)}"
    else:
        base = default_dataset(source, dataset=dataset, name=name, rate=rate)

    time_slice = time_slice_name(slice_start, slice_end, slice_name=slice_name)
    if time_slice and not stem_has_part(base, time_slice):
        base = f"{base}_{time_slice}"

    if polarity:
        polarity = str(polarity).upper()
        if polarity not in {"ON", "OFF"}:
            raise ValueError("polarity must be ON or OFF.")
        if not stem_has_part(base, polarity):
            base = f"{base}_{polarity}"

    if line is not None:
        line_label = f"line{int(line)}"
        if not stem_has_part(base, line_label):
            base = f"{base}_{line_label}"

    return clean_part(base)


def managed_file(
    artifact,
    extension,
    *,
    data_root=DEFAULT_DATA_ROOT,
    source=None,
    dataset=None,
    rate=None,
    duration=None,
    name=None,
    stem=None,
    suffix=None,
    slice_name=None,
    slice_start=None,
    slice_end=None,
    polarity=None,
    line=None,
    create_parent=False,
):
    base = dataset_stem(
        source=source,
        dataset=dataset,
        rate=rate,
        duration=duration,
        name=name,
        stem=stem,
        slice_name=slice_name,
        slice_start=slice_start,
        slice_end=slice_end,
        polarity=polarity,
        line=line,
    )
    if suffix:
        base = f"{base}_{clean_part(suffix)}"
    if extension:
        extension = extension if extension.startswith(".") else f".{extension}"
    return artifact_dir(
        artifact,
        data_root=data_root,
        source=source,
        dataset=dataset,
        rate=rate,
        name=name,
        stem=stem,
        slice_name=slice_name,
        slice_start=slice_start,
        slice_end=slice_end,
        create=create_parent,
    ) / f"{base}{extension}"


def events_file(extension=".csv", **kwargs):
    return managed_file(ARTIFACT_EVENTS, extension, **kwargs)


def iet_file(**kwargs):
    return managed_file(ARTIFACT_IETS, ".pkl", suffix="iet", **kwargs)


def picture_base(picture_name, **kwargs):
    return managed_file(ARTIFACT_PICTURES, "", suffix=picture_name, **kwargs)


def picture_file(picture_name, extension=".png", **kwargs):
    return managed_file(ARTIFACT_PICTURES, extension, suffix=picture_name, **kwargs)


def video_file(video_name="animation", **kwargs):
    return managed_file(ARTIFACT_VIDEOS, ".mp4", suffix=video_name, **kwargs)


def track_file(track_name, extension=".csv", **kwargs):
    return managed_file(ARTIFACT_TRACKS, extension, suffix=track_name, **kwargs)


def center_of_mass_file(extension=".csv", **kwargs):
    return track_file("center_of_mass", extension=extension, **kwargs)


def center_of_mass_velocity_file(extension=".csv", **kwargs):
    return track_file("center_of_mass_velocity", extension=extension, **kwargs)


def find_track_file(track_name, extension=".csv", filename=None, **kwargs):
    if filename is not None:
        return Path(filename)
    preferred = track_file(track_name, extension=extension, **kwargs)
    stem = preferred.stem
    source = normalize_source(kwargs.get("source"))
    data_root = kwargs.get("data_root", DEFAULT_DATA_ROOT)
    return first_existing_or_preferred(
        preferred,
        Path(data_root) / f"{stem}{preferred.suffix}",
        Path(source) / f"{stem}{preferred.suffix}",
        Path(f"{stem}{preferred.suffix}"),
    )


def parse_managed_path(path, data_root=DEFAULT_DATA_ROOT):
    path = Path(path)
    parts = path.parts
    root_parts = Path(data_root).parts

    start = None
    if root_parts:
        for idx in range(0, len(parts) - len(root_parts) + 1):
            if parts[idx : idx + len(root_parts)] == root_parts:
                start = idx + len(root_parts)
                break

    if start is None:
        for idx, part in enumerate(parts):
            if part in SOURCES:
                start = idx
                break

    if start is None:
        return ManagedPathParts()

    source = parts[start] if parts[start] in SOURCES else None
    dataset = None
    if source and start + 1 < len(parts):
        raw_dataset = parts[start + 1]
        if start + 2 >= len(parts) and Path(raw_dataset).suffix:
            raw_dataset = Path(raw_dataset).stem
        dataset = clean_part(raw_dataset)
    if not source or start + 2 >= len(parts):
        return ManagedPathParts(source=source, dataset=dataset)

    third = parts[start + 2]
    if third in ARTIFACTS:
        return ManagedPathParts(source=source, dataset=dataset, artifact=third)

    if start + 3 < len(parts) and parts[start + 3] in ARTIFACTS:
        return ManagedPathParts(source=source, dataset=dataset, slice_name=third, artifact=parts[start + 3])

    return ManagedPathParts(source=source, dataset=dataset)


def context_from_path(path, *, data_root=DEFAULT_DATA_ROOT, source=None, dataset=None, slice_name=None):
    parsed = parse_managed_path(path, data_root=data_root)
    resolved_dataset = dataset or parsed.dataset
    inferred_slice = infer_slice_from_stem(Path(path).stem, resolved_dataset)
    return {
        "source": source or parsed.source or SOURCE_NOISE,
        "dataset": resolved_dataset,
        "slice_name": slice_name or parsed.slice_name or inferred_slice,
    }


def _looks_numeric(text):
    try:
        float(text)
    except ValueError:
        return False
    return True


def infer_slice_from_stem(stem, dataset=None):
    stem = str(stem)
    if stem.endswith("_iet"):
        stem = stem[:-4]
    if dataset and stem.startswith(f"{dataset}_"):
        stem = stem[len(dataset) + 1 :]
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    if _looks_numeric(parts[0]) and (_looks_numeric(parts[1]) or parts[1] == "inf"):
        return clean_part(f"{parts[0]}_{parts[1]}")
    return None


def first_existing_or_preferred(preferred, *fallbacks):
    preferred = Path(preferred)
    for candidate in (preferred, *fallbacks):
        candidate = Path(candidate)
        if candidate.exists():
            return candidate
    return preferred


def find_events_file(extension=".csv", filename=None, **kwargs):
    if filename is not None:
        return Path(filename)
    preferred = events_file(extension=extension, **kwargs)
    stem = preferred.stem
    source = normalize_source(kwargs.get("source"))
    data_root = kwargs.get("data_root", DEFAULT_DATA_ROOT)
    return first_existing_or_preferred(
        preferred,
        Path(data_root) / f"{stem}{preferred.suffix}",
        Path(source) / f"{stem}{preferred.suffix}",
        Path(f"{stem}{preferred.suffix}"),
    )


def find_iet_file(filename=None, **kwargs):
    if filename is not None:
        return Path(filename)
    preferred = iet_file(**kwargs)
    stem = preferred.stem
    source = normalize_source(kwargs.get("source"))
    data_root = kwargs.get("data_root", DEFAULT_DATA_ROOT)
    return first_existing_or_preferred(
        preferred,
        Path(data_root) / f"{stem}{preferred.suffix}",
        Path(source) / f"{stem}{preferred.suffix}",
        Path(f"{stem}{preferred.suffix}"),
    )


def _main():
    parser = argparse.ArgumentParser(description="Print a managed data path.")
    parser.add_argument("--data_root", "--folder", dest="data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source", choices=SOURCES, default=SOURCE_NOISE)
    parser.add_argument("--artifact", choices=ARTIFACTS, required=True)
    parser.add_argument("--rate", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--dataset", "--set", "--name", dest="dataset", default=None)
    parser.add_argument("--stem", default=None)
    parser.add_argument("--suffix", default=None)
    parser.add_argument("--extension", default=".csv")
    parser.add_argument("--slice", dest="slice_name", default=None)
    parser.add_argument("--slice_start", type=float, default=None)
    parser.add_argument("--slice_end", type=float, default=None)
    parser.add_argument("--polarity", choices=["ON", "OFF"], default=None)
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()

    path = managed_file(
        args.artifact,
        args.extension,
        data_root=args.data_root,
        source=args.source,
        dataset=args.dataset,
        rate=args.rate,
        duration=args.duration,
        stem=args.stem,
        suffix=args.suffix,
        slice_name=args.slice_name,
        slice_start=args.slice_start,
        slice_end=args.slice_end,
        polarity=args.polarity,
        create_parent=args.create,
    )
    print(path)


if __name__ == "__main__":
    _main()
