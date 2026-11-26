"""Convert sampled and Gaussian-fit Terzan 5 profiles into one-rotation WAVs."""

import argparse
from pathlib import Path
import csv
import math
import re
import sys

import numpy as np
from scipy.io import wavfile


# ============================================================
# USER SETTINGS
# ============================================================

SAMPLE_RATE = 48_000

# Keep the profile unipolar, matching total intensity -> amplitude.
OUTPUT_PEAK = 0.98

# Rotate the profile so that a quiet/off-pulse region straddles
# the WAV boundary. This does NOT change the profile shape,
# only the phase origin.
ALIGN_QUIET_BOUNDARY = True

# Fraction of one rotation used to estimate the quietest region.
QUIET_WINDOW_FRACTION = 0.08

# High-resolution grid used only to locate the quiet region
# in Gaussian-model files.
GAUSSIAN_ANALYSIS_BINS = 4096


# ============================================================
# HELPERS
# ============================================================

def read_text(path: Path) -> str:
    """Read an ASCII-ish text file robustly."""
    for encoding in ("utf-8", "ascii", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            pass
    return path.read_text(encoding="utf-8", errors="replace")


def load_spin_frequencies(path: Path):
    """
    Read:
        Ter5ad   716.355561...
    and return a case-insensitive lookup dictionary.
    """
    freqs = {}

    for line in read_text(path).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        name = parts[0]
        try:
            freq = float(parts[1])
        except ValueError:
            continue

        if not math.isfinite(freq) or freq <= 0:
            continue

        freqs[name.lower()] = (name, freq)

    if not freqs:
        raise ValueError(f"No spin frequencies could be read from: {path}")

    return freqs


def identify_pulsar(path: Path, text: str, spin_freqs):
    """
    Identify Ter5 pulsar name primarily from filename.
    Falls back to PRESTO candidate text such as PSR_1748-24am.
    """
    # Example: Ter5ad(5).gaussians -> Ter5ad
    m = re.search(r"Ter5([A-Za-z]+)", path.name, flags=re.IGNORECASE)
    if m:
        candidate = "Ter5" + m.group(1)
        hit = spin_freqs.get(candidate.lower())
        if hit:
            return hit[0]

    # Example inside .bestprof:
    # # Candidate = PSR_1748-24am
    m = re.search(r"PSR[_\s-]*1748-24([A-Za-z]+)", text, flags=re.IGNORECASE)
    if m:
        candidate = "Ter5" + m.group(1)
        hit = spin_freqs.get(candidate.lower())
        if hit:
            return hit[0]

    return None


def looks_like_gaussian_fit(text: str) -> bool:
    """Detect pygaussfit output by its parameter names."""
    has_phase = re.search(r"^\s*phas\d+\s*=", text, flags=re.MULTILINE) is not None
    has_fwhm = re.search(r"^\s*fwhm\d+\s*=", text, flags=re.MULTILINE) is not None
    has_amp = re.search(r"^\s*ampl\d+\s*=", text, flags=re.MULTILINE) is not None
    return has_phase and has_fwhm and has_amp


def parse_gaussian_fit(text: str):
    """
    Parse pygaussfit parameters.

    PRESTO's pygaussfit stores each component as:
        phase (0..1), FWHM (rotation), amplitude (integrated flux scale)

    The additive 'const' is intentionally not needed for sonification:
    a later min/max normalization removes any additive baseline anyway.
    """
    phases = {}
    fwhms = {}
    amplitudes = {}

    for line in text.splitlines():
        m = re.match(
            r"^\s*(phas|fwhm|ampl)(\d+)\s*=\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            line,
        )
        if not m:
            continue

        kind = m.group(1)
        idx = int(m.group(2))
        value = float(m.group(3))

        if kind == "phas":
            phases[idx] = value
        elif kind == "fwhm":
            fwhms[idx] = value
        elif kind == "ampl":
            amplitudes[idx] = value

    indices = sorted(set(phases) & set(fwhms) & set(amplitudes))

    if not indices:
        raise ValueError("No complete phas/fwhm/ampl Gaussian components found.")

    if not (set(phases) == set(fwhms) == set(amplitudes)):
        raise ValueError("Gaussian phase/FWHM/amplitude component counts do not match.")

    return [(phases[i], fwhms[i], amplitudes[i]) for i in indices]


def presto_gaussian_profile_at_phases(phases, center, fwhm):
    """
    Reproduce PRESTO psr_utils.gaussian_profile() behavior on arbitrary phases.

    PRESTO uses:
        sigma = FWHM / 2.35482
    and wraps Gaussian distance across pulse phase 0/1.
    The Gaussian is normalized to unit integrated flux.
    """
    sigma = fwhm / 2.35482

    if sigma == 0:
        raise ValueError("Gaussian component has FWHM = 0.")

    mean = center % 1.0
    delta = np.asarray(phases, dtype=np.float64) - mean

    delta = np.where(delta > 0.5, delta - 1.0, delta)
    delta = np.where(delta < -0.5, delta + 1.0, delta)

    # Match PRESTO's signed-sigma behavior. Some unconstrained pygaussfit
    # outputs can contain both negative FWHM and negative amplitude.
    z = np.abs(delta) / sigma

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        y = np.where(
            z < 20.0,
            np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi)),
            0.0,
        )

    return y


def evaluate_gaussian_model(phases, components):
    """Sum all pygaussfit Gaussian components."""
    y = np.zeros(len(phases), dtype=np.float64)

    for center, fwhm, amplitude in components:
        y += amplitude * presto_gaussian_profile_at_phases(
            phases, center, fwhm
        )

    return y


def parse_sampled_profile(text: str):
    """
    Read a PRESTO .bestprof-style sampled profile.

    Non-comment data typically look like:
        0  3.278978e+10
        1  3.278989e+10
        ...

    The last numeric column is treated as total intensity/amplitude.
    """
    values = []

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        parts = stripped.replace(",", " ").split()

        if len(parts) < 2:
            continue

        try:
            nums = [float(x) for x in parts]
        except ValueError:
            continue

        values.append(nums[-1])

    if len(values) < 4:
        raise ValueError(
            "Could not find enough sampled profile rows "
            "(expected at least two numeric columns per row)."
        )

    y = np.asarray(values, dtype=np.float64)

    if not np.all(np.isfinite(y)):
        raise ValueError("Sampled profile contains NaN or infinite values.")

    return y


def circular_moving_average(y, window):
    """Centered circular moving average."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)

    if n == 0:
        raise ValueError("Empty profile.")

    window = max(1, min(int(window), n))

    if window == 1:
        return y.copy()

    left = window // 2
    right = window - 1 - left

    extended = np.concatenate((y[-left:] if left else np.empty(0), y,
                               y[:right] if right else np.empty(0)))
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(extended, kernel, mode="valid")


def find_quiet_phase(y):
    """
    Return phase (0..1) at the center of the quietest circular region.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)

    # Additive baseline and overall scale do not affect this decision.
    window = max(3, int(round(n * QUIET_WINDOW_FRACTION)))
    window = min(window, n)

    avg = circular_moving_average(y, window)
    idx = int(np.argmin(avg))
    return idx / n


def periodic_resample(y, target_samples, phase_offset=0.0):
    """
    Periodically resample one rotation using linear interpolation.

    phase_offset means:
        new phase 0 corresponds to old phase = phase_offset.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)

    old_phase = np.arange(n, dtype=np.float64) / n
    old_phase_ext = np.concatenate((old_phase, [1.0]))
    y_ext = np.concatenate((y, [y[0]]))

    new_phase = (
        np.arange(target_samples, dtype=np.float64) / target_samples
        + phase_offset
    ) % 1.0

    return np.interp(new_phase, old_phase_ext, y_ext)


def normalize_unipolar(y):
    """
    Map profile min -> 0.0 and max -> OUTPUT_PEAK.
    """
    y = np.asarray(y, dtype=np.float64)

    if not np.all(np.isfinite(y)):
        raise ValueError("Profile contains NaN or infinite values.")

    lo = float(np.min(y))
    hi = float(np.max(y))
    span = hi - lo

    if not math.isfinite(span) or span <= 0:
        raise ValueError("Profile has no usable amplitude variation.")

    y = (y - lo) / span
    y = np.clip(y, 0.0, 1.0)
    y *= OUTPUT_PEAK

    return y.astype(np.float32)


def target_sample_count(spin_freq_hz):
    """
    At fixed sample rate, one rotation is approximated by an integer
    number of samples:
        ideal samples = SAMPLE_RATE / spin frequency
        N = round(ideal samples)
    """
    n = int(round(SAMPLE_RATE / spin_freq_hz))

    if n < 2:
        raise ValueError(
            f"Spin frequency {spin_freq_hz} Hz is too high for "
            f"{SAMPLE_RATE} Hz one-cycle rendering."
        )

    return n


def convert_one(path: Path, spin_freqs, output_dir: Path):
    text = read_text(path)

    pulsar = identify_pulsar(path, text, spin_freqs)
    if pulsar is None:
        raise ValueError("Could not match filename/content to a pulsar in spin-frequency file.")

    spin_freq_hz = spin_freqs[pulsar.lower()][1]
    ideal_period_s = 1.0 / spin_freq_hz
    ideal_period_ms = ideal_period_s * 1000.0

    n_samples = target_sample_count(spin_freq_hz)
    actual_period_s = n_samples / SAMPLE_RATE
    actual_period_ms = actual_period_s * 1000.0
    effective_freq_hz = SAMPLE_RATE / n_samples

    if looks_like_gaussian_fit(text):
        source_type = "pygaussfit Gaussian model"
        components = parse_gaussian_fit(text)

        # Find a quiet phase on a dense grid.
        analysis_phase = (
            np.arange(GAUSSIAN_ANALYSIS_BINS, dtype=np.float64)
            / GAUSSIAN_ANALYSIS_BINS
        )
        analysis_y = evaluate_gaussian_model(analysis_phase, components)

        quiet_phase = find_quiet_phase(analysis_y) if ALIGN_QUIET_BOUNDARY else 0.0

        # Evaluate the Gaussian model directly at the final audio sample phases.
        output_phase = (
            np.arange(n_samples, dtype=np.float64) / n_samples
            + quiet_phase
        ) % 1.0
        audio = evaluate_gaussian_model(output_phase, components)

    else:
        source_type = "sampled profile"
        profile = parse_sampled_profile(text)

        quiet_phase = find_quiet_phase(profile) if ALIGN_QUIET_BOUNDARY else 0.0
        audio = periodic_resample(profile, n_samples, quiet_phase)

    audio = normalize_unipolar(audio)

    period_tag = f"{ideal_period_ms:.6f}".replace(".", "p")
    out_name = (
        f"{path.stem}_{path.suffix[1:].lower()}_one_rotation_"
        f"{period_tag}ms_48k.wav"
    )
    out_path = output_dir / out_name

    # scipy writes float32 as IEEE 32-bit floating-point WAV.
    wavfile.write(out_path, SAMPLE_RATE, audio)

    return {
        "source_file": path.name,
        "pulsar": pulsar,
        "source_type": source_type,
        "spin_freq_hz": f"{spin_freq_hz:.15g}",
        "ideal_period_ms": f"{ideal_period_ms:.9f}",
        "sample_rate_hz": SAMPLE_RATE,
        "samples": n_samples,
        "actual_wav_period_ms": f"{actual_period_ms:.9f}",
        "effective_cycle_freq_hz": f"{effective_freq_hz:.12f}",
        "period_error_us": f"{(actual_period_s - ideal_period_s) * 1e6:.6f}",
        "quiet_boundary_aligned": ALIGN_QUIET_BOUNDARY,
        "quiet_phase": quiet_phase,
        "output_file": out_name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True, help="Folder containing profile files")
    parser.add_argument("--spin-frequencies", type=Path, required=True, help="Spin-frequency table in Hz")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    input_dir = args.profiles
    output_dir = args.output
    spin_path = args.spin_frequencies
    spin_freqs = load_spin_frequencies(spin_path)
    supported_suffixes = {".bestprof", ".gaussians", ".summed"}
    files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
        and path.suffix.lower() in supported_suffixes
    )
    if not files:
        parser.error(f"No supported profile files in {input_dir}")

    # The output names include the stem and the source format. Detect case
    # collisions before writing so naming also works on macOS and Windows.
    source_keys = {(path.stem.casefold(), path.suffix.casefold()) for path in files}
    if len(source_keys) != len(files):
        parser.error("Profile names differ only by case. Rename duplicates before conversion.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Terzan 5 profile -> one-rotation WAV")
    print(f"Input : {input_dir}")
    print(f"Spin  : {spin_path}")
    print(f"Output: {output_dir}")
    print(f"Rate  : {SAMPLE_RATE} Hz")
    print("=" * 72)

    results = []
    failures = []

    for path in files:
        try:
            result = convert_one(path, spin_freqs, output_dir)
            results.append(result)

            print(
                f"[OK] {path.name} -> {result['output_file']}\n"
                f"     {result['pulsar']}: "
                f"{result['spin_freq_hz']} Hz, "
                f"ideal {result['ideal_period_ms']} ms, "
                f"{result['samples']} samples"
            )
        except Exception as exc:
            failures.append((path.name, str(exc)))
            print(f"[SKIP] {path.name}: {exc}")

    if results:
        summary_path = output_dir / "conversion_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

        print("-" * 72)
        print(f"Created {len(results)} WAV file(s).")
        print(f"Summary: {summary_path}")

    if failures:
        print("-" * 72)
        print(f"Skipped {len(failures)} file(s):")
        for name, reason in failures:
            print(f"  - {name}: {reason}")

    print("=" * 72)
    return int(bool(failures))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print("\nERROR:")
        print(exc)
        sys.exit(1)
