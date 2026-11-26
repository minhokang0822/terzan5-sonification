# Terzan 5 Sonification

Minho Kang’s Python workflow for converting folded radio pulse profiles from the globular cluster Terzan 5 into audio waveforms. The two converters use pulsar spin frequencies to render one rotation as a mono, 48 kHz, 32-bit floating-point WAV. These waveforms are source material for the audiovisual composition *Terzan 5*.

## Setup

Tested with Python 3.12.14, NumPy 2.3.5, and SciPy 1.17.0. From the repository root on macOS or Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/smoke_test.py
```

On Windows use `py -3.12 -m venv .venv`, then run the commands with `.venv\Scripts\python.exe` in place of `python`.

The smoke test generates its own profiles in a temporary folder. It exercises both converters, checks WAV format and timing, and verifies that two input formats with the same stem produce separate files.

## Run on the project data

Obtain the input files separately under their applicable access terms. Paths may point directly to the extracted original project folders. In the examples below, those folders have been placed under the repository’s ignored `data/` directory:

```text
data/
├── Ter5_Audio_Project 1/
│   ├── Ter5_spin_freqs.txt
│   └── Terzan5_PSRs_StokesI/
└── Ter5_Audio_Project 2/
    ├── Ter5_spin_freqs.txt
    └── other_Ter5_profs/
```

The files in each `--spin-frequencies` table are whitespace-delimited pulsar designations and spin frequencies in hertz. Use the timing table from the corresponding project folder. The second table covers the additional designations in Project 2.

```sh
python scripts/convert_kat.py \
  --profiles "data/Ter5_Audio_Project 1/Terzan5_PSRs_StokesI" \
  --spin-frequencies "data/Ter5_Audio_Project 1/Ter5_spin_freqs.txt" \
  --output outputs/kat
```

```sh
python scripts/convert_other_profiles.py \
  --profiles "data/Ter5_Audio_Project 2/other_Ter5_profs" \
  --spin-frequencies "data/Ter5_Audio_Project 2/Ter5_spin_freqs.txt" \
  --output outputs/other
```

Each script accepts `--help`. Use a new output directory for a clean run. Rerunning into an existing directory replaces outputs with the same filename and leaves unrelated files in place.

## Conversion methods and outputs

| Converter | Input and method | Output |
| --- | --- | --- |
| `convert_kat.py` | Reads the phase bin and Stokes I from columns 3 and 4 of each GBT `.KAt` profile. It averages duplicate bin entries when present, checks that bins are consecutive, rotates to a low-energy boundary, and scales each input profile to a peak absolute amplitude of 0.95 before periodic linear interpolation. The input offset is retained by default. | `one_rotation/<band>/`, `exact_preview/<band>/`, and `conversion_report.csv`. |
| `convert_other_profiles.py` | Accepts `.bestprof`, `.summed`, and `.gaussians`. It recognizes Gaussian parameters by content, since a `.summed` file can contain either a fitted model or sampled values. It shifts the phase origin to a quiet region, samples the circular Gaussian model or interpolates sampled values, and maps the resulting minimum and maximum to 0 and 0.98. | One WAV per input with the source format encoded in the name, plus `conversion_summary.csv`. |

One-rotation WAVs contain `round(48000 / spin_frequency_hz)` samples for the project’s frequency range. Looping one of these files at its native sample rate therefore gives an effective repetition rate of `48000 / sample_count`. The KAt converter also renders 10-second previews using the nominal spin frequency as a per-sample phase increment. Phase alignment changes the phase origin, and per-profile scaling changes the amplitude reference. The resulting WAVs should be used for waveform shape and timing. Use the original observations for absolute radio-intensity comparisons.

## Attribution and citation

- Scott M. Ransom supplied the spin-frequency tables and pulse-profile materials for the project.
- A. R. Martsen et al., “Radio Pulse Profiles and Polarization of the Terzan 5 Pulsars,” *The Astrophysical Journal*, 941(1), 22 (2022). [DOI](https://doi.org/10.3847/1538-4357/aca156).
- S. M. Ransom, *PRESTO: PulsaR Exploration and Search TOolkit* (2011). The supplementary converter follows the phase-wrapped Gaussian convention used by PRESTO. [ASCL record](https://ascl.net/1107.017) and [referenced implementation](https://github.com/scottransom/presto/blob/7b85205a2b3c2295a062f2d0544512e9d6d40340/python/presto/psr_utils.py#L2484).

Cite the archived release used for the paper once a version-specific DOI is assigned. See the [source review](docs/source-review.md) for verification results and the [publishing guide](docs/publishing.md) for GitHub and Zenodo steps.
