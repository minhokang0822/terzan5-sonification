# Source review

The review covered the two original Terzan5_Sonification project folders, including input profiles, spin-frequency tables, Python scripts, conversion logs, and generated WAVs.

| Contents reviewed | Project 1 | Project 2 |
| --- | ---: | ---: |
| Python converters | 1 | 1 |
| Spin-frequency entries | 32 | 49 |
| Profile files | 64 | 28 |
| Distinct pulsars represented by the profiles | 32 | 17 |
| Original WAVs | 128 | 17 |
| Original conversion logs | 1 | 1 |
| Regenerated WAVs | 128 | 28 |

The Projects 1 and 2 profile sets represent 49 distinct pulsars in total. Their shared spin-frequency entries agree at double precision. Project 1 includes 32 L-band and 32 S-band profiles. Across both bands, 12 profiles have 128 phase bins, 30 have 256, and 22 have 512. Project 2 has 17 `.gaussians`, 10 `.summed`, and one `.bestprof` input. Content-based parsing identified 18 Gaussian fits and 10 sampled profiles. Five macOS metadata files were omitted from the repository.

## Issue found and corrected

The original supplemental converter used the filename stem for the output. For example, `Ter5am.bestprof` and `Ter5am.gaussians` wrote to the same WAV name. Its CSV recorded 28 rows, but only 17 output paths were distinct. The revised converter appends the source extension to the stem and records the selected quiet phase. This preserves all 28 variants. The old WAV can be mapped to the new filename using the **last** row for that old filename in the original conversion summary.

Both converters accept explicit paths. The KAt converter now writes float WAV via SciPy, so both scripts share one audio-writing dependency. A batch returns a nonzero exit status if an input fails conversion. The original phase alignment, interpolation, normalization, and nominal timing settings were verified against the archived outputs.

## Verification

Both revised scripts were run on all 92 original profile files. All 156 new WAVs were read back as finite mono 48 kHz float32 data and checked against their expected sample counts. All 128 original KAt files and the 17 surviving supplemental files matched the regenerated counterparts sample for sample. The other 11 supplemental variants had no separate archived WAV for comparison. Matching samples can be stored in different WAV containers, so file hashes may differ.

The archive audit records source preparation. It does not identify which files appear in the final mix.
