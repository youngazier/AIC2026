"""Unit tests for the Phase 2 embedding interface and CLIP adapter."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from data_processing.phase2_feature_extraction.base_embedding import BaseEmbedding
from data_processing.phase2_feature_extraction.clip_embedding import CLIPEmbedding


class FakeProcessor:
    @classmethod
    def from_pretrained(cls, model_name):
        return cls()

    def __call__(self, *, text=None, images=None, **kwargs):
        count = 1 if isinstance(text, str) else len(text or images)
        return {"values": torch.arange(count, dtype=torch.float32)[:, None]}


class FakeModel:
    @classmethod
    def from_pretrained(cls, model_name):
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self

    def get_text_features(self, values):
        return torch.cat((values + 1, values + 2), dim=1)

    def get_image_features(self, values):
        return torch.cat((values + 2, values + 1), dim=1)


@pytest.fixture
def embedder(monkeypatch):
    monkeypatch.setattr(
        "data_processing.phase2_feature_extraction.clip_embedding.CLIPProcessor", FakeProcessor
    )
    monkeypatch.setattr(
        "data_processing.phase2_feature_extraction.clip_embedding.CLIPModel", FakeModel
    )
    return CLIPEmbedding(device="cpu")


def test_base_embedding_is_abstract():
    with pytest.raises(TypeError):
        BaseEmbedding()


def test_encode_text_shapes_and_normalization(embedder):
    assert embedder.encode_text("query").shape == (2,)
    batch = embedder.encode_text(["one", "two"])
    assert batch.shape == (2, 2)
    torch.testing.assert_close(torch.linalg.vector_norm(batch, dim=1), torch.ones(2))


def test_transformers_5_model_output_is_supported(embedder):
    original_text = embedder.model.get_text_features
    original_image = embedder.model.get_image_features
    embedder.model.get_text_features = lambda **values: SimpleNamespace(
        pooler_output=original_text(**values)
    )
    embedder.model.get_image_features = lambda **values: SimpleNamespace(
        pooler_output=original_image(**values)
    )
    assert embedder.encode_text("query").shape == (2,)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert embedder.encode_image(image).shape == (2,)


def test_encode_image_inputs_and_batch(embedder, tmp_path: Path):
    array = np.zeros((8, 8, 3), dtype=np.uint8)
    path = tmp_path / "image.png"
    Image.fromarray(array).save(path)
    assert embedder.encode_image(path).shape == (2,)
    batch = embedder.encode_images_batch([array, Image.fromarray(array), str(path)])
    assert batch.shape == (3, 2)
    torch.testing.assert_close(torch.linalg.vector_norm(batch, dim=1), torch.ones(3))


def test_empty_batches_are_rejected(embedder):
    with pytest.raises(ValueError, match="text"):
        embedder.encode_text([])
    with pytest.raises(ValueError, match="image"):
        embedder.encode_images_batch([])
