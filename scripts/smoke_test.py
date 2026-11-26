"""Exercise both command-line converters with small synthetic inputs."""

import csv
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.io import wavfile


SCRIPTS = Path(__file__).resolve().parent


class ConversionSmokeTest(unittest.TestCase):
    def run_converter(self, script, profiles, timing, destination):
        return subprocess.run(
            [
                sys.executable, str(SCRIPTS / script),
                "--profiles", str(profiles),
                "--spin-frequencies", str(timing),
                "--output", str(destination),
            ],
            capture_output=True,
            text=True,
        )

    def assert_wav(self, path, samples):
        rate, audio = wavfile.read(path)
        self.assertEqual(rate, 48000)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.shape, (samples,))
        self.assertTrue(np.isfinite(audio).all())
        self.assertGreater(np.ptp(audio), 0)

    def test_profile_conversion(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            timing = root / "timing.txt"
            timing.write_text("Ter5J 12.5\n", encoding="utf-8")
            kat = root / "kat"
            kat.mkdir()
            intensities = [0.02, 0.01, 0.04, 0.75, 1.0, 0.25, 0.03, 0.02]
            rows = [f"0 0 {i} {value}" for i, value in enumerate(intensities)]
            (kat / "GBT_Ter5J_Lband_In.KAt").write_text("\n".join(rows), encoding="utf-8")

            sampled = root / "other"
            sampled.mkdir()
            rows = [f"{i} {value}" for i, value in enumerate(intensities)]
            (sampled / "Ter5J.summed").write_text("\n".join(rows), encoding="utf-8")
            (sampled / "Ter5J.gaussians").write_text(
                "phas1 = 0.5\nfwhm1 = 0.15\nampl1 = 1.0\n", encoding="utf-8"
            )

            kat_output = root / "kat-output"
            kat_result = self.run_converter("convert_kat.py", kat, timing, kat_output)
            self.assertEqual(kat_result.returncode, 0, kat_result.stdout + kat_result.stderr)
            one_rotation, = (kat_output / "one_rotation").rglob("*.wav")
            preview, = (kat_output / "exact_preview").rglob("*.wav")
            self.assert_wav(one_rotation, 3840)
            self.assert_wav(preview, 480000)

            other_output = root / "other-output"
            other_result = self.run_converter("convert_other_profiles.py", sampled, timing, other_output)
            self.assertEqual(other_result.returncode, 0, other_result.stdout + other_result.stderr)
            with (other_output / "conversion_summary.csv").open(newline="", encoding="utf-8") as stream:
                report = list(csv.DictReader(stream))
            self.assertEqual(len(report), 2)
            self.assertEqual(len({row["output_file"] for row in report}), 2)
            for row in report:
                self.assert_wav(other_output / row["output_file"], 3840)

            (sampled / "Ter5A.summed").write_text("\n".join(rows), encoding="utf-8")
            failed = self.run_converter("convert_other_profiles.py", sampled, timing, root / "invalid-output")
            self.assertEqual(failed.returncode, 1)
            self.assertIn("Ter5A.summed", failed.stdout)


if __name__ == "__main__":
    unittest.main()
