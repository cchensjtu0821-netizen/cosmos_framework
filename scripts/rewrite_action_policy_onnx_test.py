#!/usr/bin/env python3

from __future__ import annotations

import unittest
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
except ModuleNotFoundError:
    # This unit only exercises a graph-structure rewrite that does not use NumPy.
    sys.modules["numpy"] = SimpleNamespace()
    np = None

from cosmos_framework.scripts.rewrite_action_policy_onnx import (
    _decompose_gemm_nodes,
    _materialize_gemm_weight_transposes,
)


class _Helper:
    @staticmethod
    def get_attribute_value(attribute):
        return attribute.value

    @staticmethod
    def make_node(op_type, inputs, outputs, name="", **attributes):
        return SimpleNamespace(
            op_type=op_type,
            input=list(inputs),
            output=list(outputs),
            name=name,
            attribute=[
                SimpleNamespace(name=key, value=value) for key, value in attributes.items()
            ],
        )


class DecomposeGemmTest(unittest.TestCase):
    def _rewrite(self, inputs, materialized_transposed_weights=None, **attributes):
        onnx = SimpleNamespace(helper=_Helper())
        graph = SimpleNamespace(
            node=[_Helper.make_node("Gemm", inputs, ["output"], name="linear", **attributes)],
            input=[],
            output=[],
            value_info=[],
            initializer=[],
        )
        changes = []
        count = _decompose_gemm_nodes(
            graph,
            onnx,
            changes,
            materialized_transposed_weights=materialized_transposed_weights,
        )
        return graph, changes, count

    def test_biasless_gemm_becomes_only_matmul(self):
        graph, changes, count = self._rewrite(["input", "linear.weight"])

        self.assertEqual(count, 1)
        self.assertEqual([node.op_type for node in graph.node], ["MatMul"])
        self.assertEqual(graph.node[0].output, ["output"])
        self.assertFalse(changes[0]["has_bias"])

    def test_bias_add_is_emitted_only_for_present_bias(self):
        graph, changes, _ = self._rewrite(["input", "linear.weight", "linear.bias"])

        self.assertEqual([node.op_type for node in graph.node], ["MatMul", "Add"])
        self.assertEqual(graph.node[1].input[1], "linear.bias")
        self.assertTrue(changes[0]["has_bias"])

    def test_empty_optional_bias_does_not_emit_add(self):
        graph, _, _ = self._rewrite(["input", "linear.weight", ""])

        self.assertEqual([node.op_type for node in graph.node], ["MatMul"])

    def test_transposed_weight_gets_explicit_transpose(self):
        graph, _, _ = self._rewrite(["input", "linear.weight"], transB=1)

        self.assertEqual([node.op_type for node in graph.node], ["Transpose", "MatMul"])
        self.assertEqual(graph.node[0].attribute[0].value, [1, 0])

    def test_materialized_transposed_weight_connects_directly_to_matmul(self):
        graph, changes, _ = self._rewrite(
            ["input", "linear.weight"],
            materialized_transposed_weights={"linear.weight"},
            transB=1,
        )

        self.assertEqual([node.op_type for node in graph.node], ["MatMul"])
        self.assertEqual(graph.node[0].input, ["input", "linear.weight"])
        self.assertEqual(changes[0]["materialized_transposed_weight"], "linear.weight")


class _ExternalEntries(list):
    def add(self):
        entry = SimpleNamespace(key="", value="")
        self.append(entry)
        return entry


class _ExternalDataHelper:
    @staticmethod
    def _get_all_tensors(model):
        return list(model.graph.initializer)

    @staticmethod
    def uses_external_data(tensor):
        return bool(tensor.external_data)


class _NumpyHelper:
    @staticmethod
    def to_array(tensor, base_dir):
        metadata = {entry.key: entry.value for entry in tensor.external_data}
        path = Path(base_dir) / metadata["location"]
        with path.open("rb") as file:
            file.seek(int(metadata.get("offset", "0")))
            data = file.read(int(metadata["length"]))
        return np.frombuffer(data, dtype=np.float32).reshape(tuple(tensor.dims))


@unittest.skipIf(np is None, "NumPy is required for external-weight tests")
class MaterializeGemmWeightTransposeTest(unittest.TestCase):
    def test_external_weight_is_transposed_in_distinct_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "raw.onnx"
            model_path.touch()
            source_path = root / "raw.onnx.pb"
            target_path = root / "candidate.onnx.pb"
            weight_value = np.arange(6, dtype=np.float32).reshape(2, 3)
            source_path.write_bytes(weight_value.tobytes())

            external_data = _ExternalEntries(
                [
                    SimpleNamespace(key="location", value=source_path.name),
                    SimpleNamespace(key="offset", value="0"),
                    SimpleNamespace(key="length", value=str(weight_value.nbytes)),
                ]
            )
            weight = SimpleNamespace(
                name="linear.weight",
                dims=[2, 3],
                external_data=external_data,
                data_location=1,
            )
            graph = SimpleNamespace(
                node=[
                    _Helper.make_node(
                        "Gemm",
                        ["input", "linear.weight"],
                        ["output"],
                        name="linear",
                        transB=1,
                    )
                ],
                initializer=[weight],
            )
            model = SimpleNamespace(graph=graph)
            onnx = SimpleNamespace(
                helper=_Helper(),
                external_data_helper=_ExternalDataHelper(),
                numpy_helper=_NumpyHelper(),
                TensorProto=SimpleNamespace(EXTERNAL=1),
            )

            names, report = _materialize_gemm_weight_transposes(
                model_path,
                model,
                target_path,
                onnx,
            )

            self.assertEqual(names, {"linear.weight"})
            self.assertEqual(report["transposed_weight_count"], 1)
            self.assertEqual(weight.dims, [3, 2])
            self.assertEqual(
                {entry.key: entry.value for entry in weight.external_data}["location"],
                target_path.name,
            )
            transposed = np.frombuffer(target_path.read_bytes(), dtype=np.float32).reshape(3, 2)
            np.testing.assert_array_equal(transposed, weight_value.T)
            np.testing.assert_array_equal(
                np.frombuffer(source_path.read_bytes(), dtype=np.float32).reshape(2, 3),
                weight_value,
            )


if __name__ == "__main__":
    unittest.main()
