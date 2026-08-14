from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
from cosmos_framework.model.generator.mot.modeling_utils import TimestepEmbedder


class SeparateTimestepEmbeddersForOnnxTest(unittest.TestCase):
    @staticmethod
    def _network() -> Cosmos3VFMNetwork:
        network = Cosmos3VFMNetwork.__new__(Cosmos3VFMNetwork)
        nn.Module.__init__(network)
        network.config = SimpleNamespace(vision_gen=True, action_gen=True, sound_gen=False)
        network.time_embedder = TimestepEmbedder(hidden_size=8, frequency_embedding_size=4)
        network.time_embedder._init_weights()
        return network

    def test_clones_loaded_values_into_independent_modality_modules(self) -> None:
        network = self._network()
        shared = network.time_embedder
        expected = {name: value.detach().clone() for name, value in shared.state_dict().items()}

        separated = network.separate_timestep_embedders_for_onnx()

        self.assertEqual(separated, ("vision_time_embedder", "action_time_embedder"))
        self.assertFalse(hasattr(network, "time_embedder"))
        self.assertIs(network._timestep_embedder_for("vision"), network.vision_time_embedder)
        self.assertIs(network._timestep_embedder_for("action"), network.action_time_embedder)

        for embedder in (network.vision_time_embedder, network.action_time_embedder):
            for name, value in embedder.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

        for vision_parameter, action_parameter in zip(
            network.vision_time_embedder.parameters(),
            network.action_time_embedder.parameters(),
            strict=True,
        ):
            self.assertIsNot(vision_parameter, action_parameter)
            self.assertNotEqual(vision_parameter.data_ptr(), action_parameter.data_ptr())

        state_keys = set(network.state_dict())
        self.assertNotIn("time_embedder.mlp.0.weight", state_keys)
        self.assertIn("vision_time_embedder.mlp.0.weight", state_keys)
        self.assertIn("vision_time_embedder.mlp.2.weight", state_keys)
        self.assertIn("action_time_embedder.mlp.0.weight", state_keys)
        self.assertIn("action_time_embedder.mlp.2.weight", state_keys)

    def test_is_idempotent_after_separation(self) -> None:
        network = self._network()
        first = network.separate_timestep_embedders_for_onnx()
        second = network.separate_timestep_embedders_for_onnx()
        self.assertEqual(second, first)


if __name__ == "__main__":
    unittest.main()
