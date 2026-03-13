#!/usr/bin/env python3
"""Tests for ONNX export metadata validation."""

import os
import sys
import tempfile
import unittest

import onnx
from onnx import helper


project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from scripts.validate_onnx_export import compare_name_maps, parse_label_map_pbtxt, validate_export


class TestExportValidation(unittest.TestCase):
    def test_parse_label_map_pbtxt_normalizes_one_based_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            label_map_path = os.path.join(tmpdir, 'labels.pbtxt')
            with open(label_map_path, 'w', encoding='utf-8') as handle:
                handle.write(
                    """
item {
  id: 1
  name: "Bee"
}
item {
  id: 2
  name: "Vespa-Crabro"
}
item {
  id: 3
  name: "Vespa-Velutina"
}
item {
  id: 4
  name: "Wasp"
}
""".strip()
                )

            self.assertEqual(
                parse_label_map_pbtxt(label_map_path),
                {0: 'Bee', 1: 'Vespa-Crabro', 2: 'Vespa-Velutina', 3: 'Wasp'},
            )

    def test_compare_name_maps_reports_mismatches(self):
        mismatches = compare_name_maps(
            {0: 'Bee', 1: 'Vespa-Crabro', 2: 'Vespa-Velutina', 3: 'Wasp'},
            {0: 'Bee', 1: 'Bee', 2: 'Vespa-Crabro', 3: 'Vespa-Crabro'},
        )
        self.assertEqual(
            mismatches,
            [
                "Class 1 mismatch: expected 'Vespa-Crabro', got 'Bee'",
                "Class 2 mismatch: expected 'Vespa-Velutina', got 'Vespa-Crabro'",
                "Class 3 mismatch: expected 'Wasp', got 'Vespa-Crabro'",
            ],
        )

    def test_validate_export_returns_nonzero_for_bad_embedded_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, 'bad.onnx')
            label_map_path = os.path.join(tmpdir, 'labels.pbtxt')

            with open(label_map_path, 'w', encoding='utf-8') as handle:
                handle.write(
                    """
item {
  id: 1
  name: "Bee"
}
item {
  id: 2
  name: "Vespa-Crabro"
}
item {
  id: 3
  name: "Vespa-Velutina"
}
item {
  id: 4
  name: "Wasp"
}
""".strip()
                )

            model = helper.make_model(helper.make_graph([], 'g', [], []))
            metadata = model.metadata_props.add()
            metadata.key = 'names'
            metadata.value = '{0: "Bee", 1: "Bee", 2: "Vespa-Crabro", 3: "Vespa-Crabro"}'
            onnx.save(model, onnx_path)

            self.assertEqual(validate_export(onnx_path, label_map_path), 1)

    def test_validate_export_returns_zero_for_matching_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, 'good.onnx')
            label_map_path = os.path.join(tmpdir, 'labels.pbtxt')

            with open(label_map_path, 'w', encoding='utf-8') as handle:
                handle.write(
                    """
item {
  id: 1
  name: "Bee"
}
item {
  id: 2
  name: "Vespa-Crabro"
}
item {
  id: 3
  name: "Vespa-Velutina"
}
item {
  id: 4
  name: "Wasp"
}
""".strip()
                )

            model = helper.make_model(helper.make_graph([], 'g', [], []))
            metadata = model.metadata_props.add()
            metadata.key = 'names'
            metadata.value = '{0: "Bee", 1: "Vespa-Crabro", 2: "Vespa-Velutina", 3: "Wasp"}'
            onnx.save(model, onnx_path)

            self.assertEqual(validate_export(onnx_path, label_map_path), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)