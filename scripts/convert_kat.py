"""Convert folded GBT Stokes I profiles into one-rotation WAVs and previews."""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile


# ============================================================
# USER SETTINGS
# ============================================================

SAMPLE_RATE = 48_000
PEAK_LEVEL = 0.95
EXACT_PREVIEW_SECONDS = 10.0

# Normalization:
#   "per_file" : normalize each KAt profile independently
#   "global"   : use one common scale for all profiles
#   "none"     : do not normalize
NORMALIZATION_MODE = "per_file"

# Circular phase alignment:
# The profile is circularly shifted so that a quiet, near-baseline region
# becomes the boundary between the end and beginning of the WAV.
ALIGN_PROFILE_PHASE = True

# Width of the circular window used to find an off-pulse/quiet region.
# 0.08 means 8 percent of the profile bins.
QUIET_WINDOW_FRACTION = 0.08

# Only bins within the quietest X percent are considered as boundary candidates.
QUIET_CANDIDATE_PERCENTILE = 15.0

# Among quiet boundary candidates, prefer the point opposite the pulse-energy
# center. Larger values place the pulse more strongly near the center.
CENTERING_WEIGHT = 1.0

# Prefer candidates whose exact boundary samples are close to the baseline.
BOUNDARY_SAMPLE_WEIGHT = 0.35

# False preserves the original Stokes I vertical offset.
# True subtracts the estimated off-pulse baseline before normalization,
# making the quiet region closer to digital amplitude 0.
SUBTRACT_OFFPULSE_BASELINE = False


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class RawProfile:
    source_path: Path
    pulsar_key: str
    pulsar_name: str
    band: str
    stokes_i: np.ndarray
    duplicate_rows_averaged: bool


@dataclass
class PreparedProfile:
    source_path: Path
    pulsar_key: str
    pulsar_name: str
    band: str
    audio_profile: np.ndarray
    input_bins: int
    duplicate_rows_averaged: bool
    estimated_baseline: float
    phase_cut_bin: int
    phase_shift_bins: int
    phase_shift_fraction: float
    quiet_window_bins: int
    pulse_center_bin: float
    boundary_start_value: float
    boundary_end_value: float


# ============================================================
# TEXT AND NAME UTILITIES
# ============================================================

MINUS_CHARS = "−–—﹣－"
FLOAT_PATTERN = (
    r"[+-]?"
    r"(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[Ee][+-]?\d+)?"
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for char in MINUS_CHARS:
        text = text.replace(char, "-")
    return text.upper().strip()


def extract_pulsar_key(text: str) -> str | None:
    """Return A, C, AD, AR, etc. from common Terzan 5 pulsar names."""

    normalized = normalize_text(text)

    patterns = [
        r"TER(?:ZAN)?[\s_\-]*5[\s_\-]*([A-Z]{1,3})(?![A-Z])",
        r"J[\s_\-]*1748[\s_\-]*2446[\s_\-]*([A-Z]{1,3})(?![A-Z])",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return match.group(1).upper()

    # Historical designation for Ter5A
    if re.search(r"B[\s_\-]*1744[\s_\-]*24[\s_\-]*A(?![A-Z])", normalized):
        return "A"

    return None


def canonical_pulsar_name(key: str) -> str:
    key = key.upper()
    return f"Ter5{key}" if len(key) == 1 else f"Ter5{key.lower()}"


def safe_filename_component(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._-") or "unnamed"


def format_number_for_filename(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}".replace(".", "p")


def strip_inline_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


# ============================================================
# SPIN FREQUENCY PARSING
# ============================================================


def parse_header_table(lines: list[str]) -> dict[str, float] | None:
    """Read a CSV/TSV/semicolon/whitespace table containing a frequency column."""

    nonempty = [strip_inline_comment(line) for line in lines]
    nonempty = [line for line in nonempty if line]

    if len(nonempty) < 2:
        return None

    header_normalized = normalize_text(nonempty[0])
    if "FREQ" not in header_normalized and "FREQUENCY" not in header_normalized:
        return None

    delimiter: str | None = None
    for candidate in ("\t", ",", ";"):
        if candidate in nonempty[0]:
            delimiter = candidate
            break

    if delimiter is None:
        rows = [re.split(r"\s+", line.strip()) for line in nonempty]
    else:
        rows = list(csv.reader(nonempty, delimiter=delimiter))
        rows = [[cell.strip() for cell in row] for row in rows]

    header_cells = [normalize_text(cell) for cell in rows[0]]

    frequency_index = next(
        (index for index, cell in enumerate(header_cells) if "FREQ" in cell),
        None,
    )

    name_index = next(
        (
            index
            for index, cell in enumerate(header_cells)
            if any(token in cell for token in ("PULSAR", "PSR", "NAME", "SOURCE"))
        ),
        0,
    )

    if frequency_index is None:
        return None

    frequencies: dict[str, float] = {}

    for row in rows[1:]:
        if max(name_index, frequency_index) >= len(row):
            continue

        key = extract_pulsar_key(row[name_index])
        if key is None:
            key = extract_pulsar_key(" ".join(row))
        if key is None:
            continue

        number_match = re.search(FLOAT_PATTERN, row[frequency_index])
        if not number_match:
            continue

        frequency = float(number_match.group(0))
        if frequency > 0 and np.isfinite(frequency):
            frequencies[key] = frequency

    return frequencies or None


def find_frequency_in_line(line: str) -> float | None:
    normalized = normalize_text(line)

    # Explicit Hz value
    hz_matches = re.findall(
        rf"({FLOAT_PATTERN})\s*(?:HZ|S\^?-1|1/S)\b",
        normalized,
    )
    if hz_matches:
        return float(hz_matches[-1])

    # Period in milliseconds
    ms_matches = re.findall(rf"({FLOAT_PATTERN})\s*MS\b", normalized)
    if ms_matches:
        period_ms = float(ms_matches[-1])
        return 1000.0 / period_ms if period_ms > 0 else None

    # Period in seconds
    second_matches = re.findall(
        rf"({FLOAT_PATTERN})\s*(?:SEC|SECS|SECOND|SECONDS)\b",
        normalized,
    )
    if second_matches:
        period_seconds = float(second_matches[-1])
        return 1.0 / period_seconds if period_seconds > 0 else None

    # Remove identifying numbers before finding a plain numeric frequency.
    cleaned = re.sub(
        r"TER(?:ZAN)?[\s_\-]*5[\s_\-]*[A-Z]{1,3}",
        " ",
        normalized,
    )
    cleaned = re.sub(
        r"J[\s_\-]*1748[\s_\-]*2446[\s_\-]*[A-Z]{1,3}",
        " ",
        cleaned,
    )
    cleaned = re.sub(
        r"B[\s_\-]*1744[\s_\-]*24[\s_\-]*A",
        " ",
        cleaned,
    )

    numbers = re.findall(FLOAT_PATTERN, cleaned)
    return float(numbers[0]) if numbers else None


def parse_spin_frequencies(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Spin-frequency file not found:\n{path}")

    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()

    table_result = parse_header_table(lines)
    if table_result is not None:
        return table_result

    frequencies: dict[str, float] = {}

    for line_number, raw_line in enumerate(lines, start=1):
        line = strip_inline_comment(raw_line)
        if not line:
            continue

        key = extract_pulsar_key(line)
        if key is None:
            continue

        frequency = find_frequency_in_line(line)
        if frequency is None:
            continue

        if frequency <= 0 or not np.isfinite(frequency):
            raise ValueError(f"Invalid frequency on line {line_number}: {raw_line}")

        if key in frequencies and not np.isclose(
            frequencies[key], frequency, rtol=1e-10, atol=1e-12
        ):
            raise ValueError(
                f"Conflicting frequencies for {canonical_pulsar_name(key)}: "
                f"{frequencies[key]} and {frequency} Hz"
            )

        frequencies[key] = frequency

    if not frequencies:
        raise ValueError(
            "No Terzan 5 pulsar frequencies were found.\n"
            "Recommended format:\n"
            "Pulsar,Frequency_Hz\n"
            "Ter5A,86.481639775"
        )

    return frequencies


# ============================================================
# KAt FILE READING AND BAND DETECTION
# ============================================================


def read_header_text(path: Path, max_lines: int = 30) -> str:
    lines: list[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for _ in range(max_lines):
            line = handle.readline()
            if not line:
                break
            if line.lstrip().startswith("#"):
                lines.append(line.strip())
            elif lines:
                break

    return "\n".join(lines)


def detect_band(path: Path, header_text: str = "") -> str:
    def classify(text: str) -> str | None:
        normalized = normalize_text(text).replace("_", " ")
        compact = normalized.replace(" ", "")

        if re.search(r"\bL\s*-?\s*BAND\b", normalized) or "LBAND" in compact:
            return "Lband"
        if re.search(r"\bS\s*-?\s*BAND\b", normalized) or "SBAND" in compact:
            return "Sband"
        return None

    return classify(path.name) or classify(header_text) or "UnknownBand"


def discover_kat_files(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Input profile folder not found:\n{folder}")

    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() == ".kat"
    )


def read_kat_profile(path: Path) -> RawProfile:
    header_text = read_header_text(path)

    key = extract_pulsar_key(f"{path.name} {header_text}")
    if key is None:
        raise ValueError("Could not identify a Terzan 5 pulsar name")

    data = np.loadtxt(path, comments="#", dtype=np.float64)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError("Expected at least 4 numeric columns")

    raw_bins = data[:, 2]
    raw_stokes_i = data[:, 3]

    if not np.all(np.isfinite(raw_bins)) or not np.all(np.isfinite(raw_stokes_i)):
        raise ValueError("Phase bins or Stokes I contain NaN or infinite values")

    rounded_bins = np.rint(raw_bins).astype(np.int64)
    if not np.allclose(raw_bins, rounded_bins, rtol=0.0, atol=1e-9):
        raise ValueError("Phase-bin column contains non-integer values")

    unique_bins, inverse = np.unique(rounded_bins, return_inverse=True)
    duplicate_rows_averaged = unique_bins.size != rounded_bins.size

    if duplicate_rows_averaged:
        sums = np.bincount(inverse, weights=raw_stokes_i)
        counts = np.bincount(inverse)
        bins = unique_bins
        stokes_i = sums / counts
    else:
        order = np.argsort(rounded_bins)
        bins = rounded_bins[order]
        stokes_i = raw_stokes_i[order]

    if bins.size < 2:
        raise ValueError("Profile contains fewer than two bins")

    expected_bins = np.arange(bins[0], bins[0] + bins.size)
    if not np.array_equal(bins, expected_bins):
        raise ValueError("Phase bins are not consecutive")

    return RawProfile(
        source_path=path,
        pulsar_key=key,
        pulsar_name=canonical_pulsar_name(key),
        band=detect_band(path, header_text),
        stokes_i=np.asarray(stokes_i, dtype=np.float64),
        duplicate_rows_averaged=duplicate_rows_averaged,
    )


# ============================================================
# CIRCULAR PHASE ALIGNMENT
# ============================================================


def make_odd_window_length(number_of_bins: int, fraction: float) -> int:
    if number_of_bins < 3:
        return 1

    window = max(3, int(round(number_of_bins * fraction)))
    if window % 2 == 0:
        window += 1

    # Circular smoothing works best with an odd window smaller than N.
    maximum = number_of_bins if number_of_bins % 2 == 1 else number_of_bins - 1
    return min(window, maximum)


def circular_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float64, copy=True)

    half = window // 2
    extended = np.concatenate((values[-half:], values, values[:half]))
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(extended, kernel, mode="valid")


def circular_distance(indices: np.ndarray, target: float, length: int) -> np.ndarray:
    direct = np.abs(indices.astype(np.float64) - target)
    return np.minimum(direct, length - direct)


def align_profile_to_quiet_boundary(
    profile: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """
    Circularly rotate a folded profile so its beginning/end fall in a quiet region.

    The values are not time-stretched and the pulse shape is not edited. Only the
    phase origin is changed.
    """

    profile = np.asarray(profile, dtype=np.float64)
    number_of_bins = profile.size

    baseline = float(np.median(profile))
    deviation = profile - baseline

    if not ALIGN_PROFILE_PHASE or number_of_bins < 3:
        aligned = profile.copy()
        if SUBTRACT_OFFPULSE_BASELINE:
            aligned -= baseline

        return aligned, {
            "estimated_baseline": baseline,
            "phase_cut_bin": 0,
            "phase_shift_bins": 0,
            "phase_shift_fraction": 0.0,
            "quiet_window_bins": 1,
            "pulse_center_bin": 0.0,
            "boundary_start_value": float(aligned[0]),
            "boundary_end_value": float(aligned[-1]),
        }

    quiet_window_bins = make_odd_window_length(
        number_of_bins,
        QUIET_WINDOW_FRACTION,
    )

    # Quiet-region score: local mean squared distance from the off-pulse baseline.
    total_energy = deviation**2
    quiet_score = circular_moving_average(total_energy, quiet_window_bins)

    # For Stokes I, positive deviation normally represents the pulse.
    positive_energy = np.maximum(deviation, 0.0) ** 2
    if float(np.sum(positive_energy)) <= np.finfo(np.float64).eps:
        pulse_energy = total_energy
    else:
        pulse_energy = positive_energy

    # Circular energy center of the pulse.
    phases = 2.0 * np.pi * np.arange(number_of_bins) / number_of_bins
    vector = np.sum(pulse_energy * np.exp(1j * phases))

    if abs(vector) > np.finfo(np.float64).eps:
        pulse_center_bin = (
            (np.angle(vector) % (2.0 * np.pi))
            * number_of_bins
            / (2.0 * np.pi)
        )
        ideal_cut_bin = (pulse_center_bin + number_of_bins / 2.0) % number_of_bins
    else:
        pulse_center_bin = float(np.argmax(circular_moving_average(total_energy, quiet_window_bins)))
        ideal_cut_bin = (pulse_center_bin + number_of_bins / 2.0) % number_of_bins

    percentile = float(np.clip(QUIET_CANDIDATE_PERCENTILE, 0.0, 100.0))
    threshold = float(np.percentile(quiet_score, percentile))
    candidates = np.flatnonzero(quiet_score <= threshold + np.finfo(np.float64).eps)

    if candidates.size == 0:
        candidates = np.array([int(np.argmin(quiet_score))], dtype=np.int64)

    distance_score = circular_distance(candidates, ideal_cut_bin, number_of_bins)
    distance_score /= max(number_of_bins / 2.0, 1.0)

    maximum_deviation = float(np.max(np.abs(deviation)))
    if maximum_deviation > 0.0:
        boundary_sample_score = (
            np.abs(deviation[candidates])
            + np.abs(deviation[(candidates - 1) % number_of_bins])
        ) / (2.0 * maximum_deviation)
    else:
        boundary_sample_score = np.zeros(candidates.size, dtype=np.float64)

    objective = (
        CENTERING_WEIGHT * distance_score
        + BOUNDARY_SAMPLE_WEIGHT * boundary_sample_score
    )

    phase_cut_bin = int(candidates[int(np.argmin(objective))])

    # np.roll(..., -cut) makes the selected cut bin the first sample.
    aligned = np.roll(profile, -phase_cut_bin)

    if SUBTRACT_OFFPULSE_BASELINE:
        aligned = aligned - baseline

    return aligned, {
        "estimated_baseline": baseline,
        "phase_cut_bin": phase_cut_bin,
        "phase_shift_bins": -phase_cut_bin,
        "phase_shift_fraction": phase_cut_bin / number_of_bins,
        "quiet_window_bins": quiet_window_bins,
        "pulse_center_bin": float(pulse_center_bin),
        "boundary_start_value": float(aligned[0]),
        "boundary_end_value": float(aligned[-1]),
    }


def prepare_profile(raw: RawProfile) -> PreparedProfile:
    aligned_profile, alignment = align_profile_to_quiet_boundary(raw.stokes_i)

    return PreparedProfile(
        source_path=raw.source_path,
        pulsar_key=raw.pulsar_key,
        pulsar_name=raw.pulsar_name,
        band=raw.band,
        audio_profile=aligned_profile,
        input_bins=raw.stokes_i.size,
        duplicate_rows_averaged=raw.duplicate_rows_averaged,
        estimated_baseline=float(alignment["estimated_baseline"]),
        phase_cut_bin=int(alignment["phase_cut_bin"]),
        phase_shift_bins=int(alignment["phase_shift_bins"]),
        phase_shift_fraction=float(alignment["phase_shift_fraction"]),
        quiet_window_bins=int(alignment["quiet_window_bins"]),
        pulse_center_bin=float(alignment["pulse_center_bin"]),
        boundary_start_value=float(alignment["boundary_start_value"]),
        boundary_end_value=float(alignment["boundary_end_value"]),
    )


# ============================================================
# AUDIO PROCESSING
# ============================================================


def periodic_resample(profile: np.ndarray, phases: np.ndarray) -> np.ndarray:
    source_phases = np.arange(profile.size, dtype=np.float64) / profile.size
    source_phases_extended = np.append(source_phases, 1.0)
    periodic_profile = np.append(profile, profile[0])

    return np.interp(phases, source_phases_extended, periodic_profile)


def normalize_profile(
    profile: np.ndarray,
    global_max_abs: float | None,
) -> np.ndarray:
    mode = NORMALIZATION_MODE.lower()

    if mode == "per_file":
        scale_reference = float(np.max(np.abs(profile)))
    elif mode == "global":
        if global_max_abs is None:
            raise RuntimeError("Global normalization scale was not calculated")
        scale_reference = global_max_abs
    elif mode == "none":
        return profile.astype(np.float64, copy=True)
    else:
        raise ValueError(
            "NORMALIZATION_MODE must be 'per_file', 'global', or 'none'"
        )

    if scale_reference == 0.0:
        raise ValueError("Stokes I is entirely zero")

    return profile * (PEAK_LEVEL / scale_reference)


def output_filename(
    record: PreparedProfile,
    spin_frequency_hz: float,
    spin_period_ms: float,
    output_type: str,
) -> str:
    source_stem = safe_filename_component(record.source_path.stem)
    frequency_text = format_number_for_filename(spin_frequency_hz, 9)
    period_text = format_number_for_filename(spin_period_ms, 9)

    return (
        f"{source_stem}_StokesI_"
        f"{frequency_text}Hz_"
        f"{period_text}ms_"
        f"phaseAligned_"
        f"{output_type}_"
        f"{SAMPLE_RATE // 1000}k.wav"
    )


def convert_profile(
    record: PreparedProfile,
    spin_frequency_hz: float,
    global_max_abs: float | None,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    spin_period_seconds = 1.0 / spin_frequency_hz
    spin_period_ms = spin_period_seconds * 1000.0

    normalized_profile = normalize_profile(record.audio_profile, global_max_abs)

    one_rotation_dir = output_dir / "one_rotation" / record.band
    exact_preview_dir = output_dir / "exact_preview" / record.band

    one_rotation_dir.mkdir(parents=True, exist_ok=True)
    exact_preview_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. One rotation
    # --------------------------------------------------------

    one_rotation_samples = max(
        2,
        int(round(SAMPLE_RATE / spin_frequency_hz)),
    )

    one_rotation_phases = (
        np.arange(one_rotation_samples, dtype=np.float64)
        / one_rotation_samples
    )

    one_rotation_audio = periodic_resample(
        normalized_profile,
        one_rotation_phases,
    )

    one_rotation_path = one_rotation_dir / output_filename(
        record,
        spin_frequency_hz,
        spin_period_ms,
        "one_rotation",
    )

    wavfile.write(one_rotation_path, SAMPLE_RATE, one_rotation_audio.astype(np.float32))

    rendered_period_seconds = one_rotation_samples / SAMPLE_RATE
    rendered_loop_frequency_hz = SAMPLE_RATE / one_rotation_samples

    # --------------------------------------------------------
    # 2. Exact preview
    # --------------------------------------------------------

    preview_samples = max(
        1,
        int(round(EXACT_PREVIEW_SECONDS * SAMPLE_RATE)),
    )

    preview_indices = np.arange(preview_samples, dtype=np.float64)
    preview_phases = (
        preview_indices * spin_frequency_hz / SAMPLE_RATE
    ) % 1.0

    preview_audio = periodic_resample(
        normalized_profile,
        preview_phases,
    )

    exact_preview_path = exact_preview_dir / output_filename(
        record,
        spin_frequency_hz,
        spin_period_ms,
        "exact_preview",
    )

    wavfile.write(exact_preview_path, SAMPLE_RATE, preview_audio.astype(np.float32))

    return {
        "source_file": str(record.source_path.relative_to(input_dir)),
        "pulsar": record.pulsar_name,
        "band": record.band,
        "input_bins": record.input_bins,
        "spin_frequency_hz": spin_frequency_hz,
        "spin_period_ms": spin_period_ms,
        "estimated_baseline": record.estimated_baseline,
        "phase_cut_bin": record.phase_cut_bin,
        "phase_shift_bins": record.phase_shift_bins,
        "phase_shift_fraction": record.phase_shift_fraction,
        "quiet_window_bins": record.quiet_window_bins,
        "pulse_center_bin_before_shift": record.pulse_center_bin,
        "boundary_start_value": record.boundary_start_value,
        "boundary_end_value": record.boundary_end_value,
        "one_rotation_samples": one_rotation_samples,
        "one_rotation_rendered_ms": rendered_period_seconds * 1000.0,
        "one_rotation_rendered_hz": rendered_loop_frequency_hz,
        "exact_preview_seconds": EXACT_PREVIEW_SECONDS,
        "duplicate_rows_averaged": record.duplicate_rows_averaged,
        "one_rotation_file": str(one_rotation_path.relative_to(output_dir)),
        "exact_preview_file": str(exact_preview_path.relative_to(output_dir)),
        "status": "OK",
        "error": "",
    }


# ============================================================
# REPORTING
# ============================================================

REPORT_FIELDS = [
    "source_file",
    "pulsar",
    "band",
    "input_bins",
    "spin_frequency_hz",
    "spin_period_ms",
    "estimated_baseline",
    "phase_cut_bin",
    "phase_shift_bins",
    "phase_shift_fraction",
    "quiet_window_bins",
    "pulse_center_bin_before_shift",
    "boundary_start_value",
    "boundary_end_value",
    "one_rotation_samples",
    "one_rotation_rendered_ms",
    "one_rotation_rendered_hz",
    "exact_preview_seconds",
    "duplicate_rows_averaged",
    "one_rotation_file",
    "exact_preview_file",
    "status",
    "error",
]


def error_row(
    source_file: str,
    error_message: str,
    record: PreparedProfile | None = None,
    frequency: float | None = None,
) -> dict[str, object]:
    row = {field: "" for field in REPORT_FIELDS}
    row["source_file"] = source_file
    row["status"] = "ERROR"
    row["error"] = error_message
    row["exact_preview_seconds"] = EXACT_PREVIEW_SECONDS

    if record is not None:
        row.update(
            {
                "pulsar": record.pulsar_name,
                "band": record.band,
                "input_bins": record.input_bins,
                "estimated_baseline": record.estimated_baseline,
                "phase_cut_bin": record.phase_cut_bin,
                "phase_shift_bins": record.phase_shift_bins,
                "phase_shift_fraction": record.phase_shift_fraction,
                "quiet_window_bins": record.quiet_window_bins,
                "pulse_center_bin_before_shift": record.pulse_center_bin,
                "boundary_start_value": record.boundary_start_value,
                "boundary_end_value": record.boundary_end_value,
                "duplicate_rows_averaged": record.duplicate_rows_averaged,
            }
        )

    if frequency is not None:
        row["spin_frequency_hz"] = frequency
        row["spin_period_ms"] = 1000.0 / frequency

    return row


def write_report(rows: list[dict[str, object]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "conversion_report.csv"

    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return report_path


# ============================================================
# MAIN
# ============================================================


def validate_settings() -> None:
    if SAMPLE_RATE <= 0:
        raise ValueError("SAMPLE_RATE must be greater than zero")
    if EXACT_PREVIEW_SECONDS <= 0:
        raise ValueError("EXACT_PREVIEW_SECONDS must be greater than zero")
    if not 0 < PEAK_LEVEL <= 1:
        raise ValueError("PEAK_LEVEL must be greater than 0 and <= 1")
    if not 0 < QUIET_WINDOW_FRACTION < 1:
        raise ValueError("QUIET_WINDOW_FRACTION must be between 0 and 1")
    if not 0 <= QUIET_CANDIDATE_PERCENTILE <= 100:
        raise ValueError("QUIET_CANDIDATE_PERCENTILE must be between 0 and 100")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True, help="Folder containing .KAt files")
    parser.add_argument("--spin-frequencies", type=Path, required=True, help="Spin-frequency table in Hz")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    input_dir = args.profiles
    output_dir = args.output
    validate_settings()

    print()
    print("Loading spin frequencies...")
    frequencies = parse_spin_frequencies(args.spin_frequencies)
    print(f"Loaded {len(frequencies)} spin frequencies.")

    print()
    print("Searching for KAt files...")
    kat_files = discover_kat_files(input_dir)

    if not kat_files:
        raise FileNotFoundError(
            f"No .KAt files were found inside:\n{input_dir}"
        )

    print(f"Found {len(kat_files)} KAt files.")

    prepared_records: list[PreparedProfile] = []
    report_rows: list[dict[str, object]] = []

    for path in kat_files:
        try:
            raw = read_kat_profile(path)
            prepared = prepare_profile(raw)
            prepared_records.append(prepared)

            print(
                f"Read: {prepared.pulsar_name} {prepared.band} | "
                f"cut bin {prepared.phase_cut_bin}/{prepared.input_bins}"
            )

        except Exception as exc:
            report_rows.append(
                error_row(
                    str(path.relative_to(input_dir)),
                    f"Profile read/preparation error: {exc}",
                )
            )
            print(f"ERROR reading {path.name}: {exc}")

    global_max_abs: float | None = None

    if NORMALIZATION_MODE.lower() == "global" and prepared_records:
        global_max_abs = max(
            float(np.max(np.abs(record.audio_profile)))
            for record in prepared_records
        )

        if global_max_abs == 0.0:
            raise ValueError("All prepared Stokes I profiles are entirely zero")

    print()
    print("Starting audio conversion...")
    print()

    converted_count = 0
    output_keys = set()

    for record in prepared_records:
        frequency = frequencies.get(record.pulsar_key)

        if frequency is None:
            report_rows.append(
                error_row(
                    str(record.source_path.relative_to(input_dir)),
                    "No matching spin frequency in Ter5_spin_freqs.txt",
                    record=record,
                )
            )
            print(
                f"SKIPPED: {record.source_path.name} — spin frequency not found"
            )
            continue

        try:
            name = output_filename(record, frequency, (1.0 / frequency) * 1000.0, "one_rotation")
            output_key = (record.band.casefold(), name.casefold())
            if output_key in output_keys:
                raise ValueError(f"Duplicate output filename within {record.band}: {name}")
            output_keys.add(output_key)
            result = convert_profile(record, frequency, global_max_abs, input_dir, output_dir)
            report_rows.append(result)
            converted_count += 1

            print(
                f"OK: {record.pulsar_name} {record.band} | "
                f"{frequency:.9f} Hz | cut bin {record.phase_cut_bin}"
            )

        except Exception as exc:
            report_rows.append(
                error_row(
                    str(record.source_path.relative_to(input_dir)),
                    f"Conversion error: {exc}",
                    record=record,
                    frequency=frequency,
                )
            )
            print(f"ERROR converting {record.source_path.name}: {exc}")

    report_path = write_report(report_rows, output_dir)
    error_count = sum(row["status"] == "ERROR" for row in report_rows)

    print()
    print("Terzan 5 batch conversion complete")
    print("----------------------------------")
    print(f"Spin frequencies loaded:  {len(frequencies)}")
    print(f"KAt files found:          {len(kat_files)}")
    print(f"Prepared profiles:        {len(prepared_records)}")
    print(f"Audio pairs created:      {converted_count}")
    print(f"Errors or unmatched:      {error_count}")
    print(f"Phase alignment:          {ALIGN_PROFILE_PHASE}")
    print(f"Subtract baseline:        {SUBTRACT_OFFPULSE_BASELINE}")
    print(f"Normalization mode:       {NORMALIZATION_MODE}")
    print(f"Output folder:            {output_dir}")
    print(f"Conversion report:        {report_path}")

    if error_count:
        print()
        print("Some files were not converted. Check conversion_report.csv.")

    return int(bool(error_count))


if __name__ == "__main__":
    raise SystemExit(main())
