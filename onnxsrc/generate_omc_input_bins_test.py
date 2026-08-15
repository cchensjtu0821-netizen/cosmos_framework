from __future__ import annotations

import hashlib
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from cosmos_framework.onnxsrc.generate_omc_input_bins import INPUT_SPECS, generate_input_bins


class GenerateOmcInputBinsTest(unittest.TestCase):
    def test_generates_expected_raw_fp16_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            manifest_path = generate_input_bins(
                output_dir,
                seed=7,
                timestep=625.0,
                overwrite=False,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["input_order"], [spec.name for spec in INPUT_SPECS])
            self.assertEqual(len(manifest["inputs"]), len(INPUT_SPECS))
            for spec, metadata in zip(INPUT_SPECS, manifest["inputs"], strict=True):
                path = output_dir / f"{spec.name}.bin"
                data = path.read_bytes()
                values = [value for (value,) in struct.iter_unpack("<e", data)]
                self.assertEqual(len(values), math.prod(spec.shape))
                self.assertEqual(path.stat().st_size, math.prod(spec.shape) * 2)
                self.assertEqual(metadata["sha256"], hashlib.sha256(data).hexdigest())
                if "timestep" in spec.name:
                    self.assertEqual(values, [625.0] * math.prod(spec.shape))

    def test_seed_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_manifest = generate_input_bins(Path(first_dir), seed=11, timestep=500.0, overwrite=False)
            second_manifest = generate_input_bins(Path(second_dir), seed=11, timestep=500.0, overwrite=False)
            first = json.loads(first_manifest.read_text(encoding="utf-8"))
            second = json.loads(second_manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["sha256"] for item in first["inputs"]],
                [item["sha256"] for item in second["inputs"]],
            )

    def test_refuses_to_overwrite_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            generate_input_bins(output_dir, seed=0, timestep=500.0, overwrite=False)
            with self.assertRaises(FileExistsError):
                generate_input_bins(output_dir, seed=0, timestep=500.0, overwrite=False)


if __name__ == "__main__":
    unittest.main()
