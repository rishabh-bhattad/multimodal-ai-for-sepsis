"""SapBERT loading and device checks without external model downloads."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import config
from src import step2_extraction as step2


def test_cached_model_needs_no_download(monkeypatch):
    monkeypatch.setattr(config, "SAPBERT_OFFLINE", False)
    loader = Mock()
    assert step2.load_sapbert_component(loader) is loader.from_pretrained.return_value
    assert loader.from_pretrained.call_count == 1
    assert loader.from_pretrained.call_args.kwargs["local_files_only"] is True


def test_first_run_downloads_pinned_revision(monkeypatch):
    monkeypatch.setattr(config, "SAPBERT_OFFLINE", False)
    loader = Mock()
    loader.from_pretrained.side_effect = [OSError("missing cache"), "downloaded"]
    assert step2.load_sapbert_component(loader) == "downloaded"
    assert [c.kwargs["local_files_only"] for c in loader.from_pretrained.call_args_list] == [True, False]
    assert all(c.kwargs["revision"] == config.SAPBERT_MODEL_REVISION
               for c in loader.from_pretrained.call_args_list)


def test_offline_cache_miss_never_downloads(monkeypatch):
    monkeypatch.setattr(config, "SAPBERT_OFFLINE", True)
    loader = Mock()
    loader.from_pretrained.side_effect = OSError("missing cache")
    with pytest.raises(RuntimeError, match="not cached"):
        step2.load_sapbert_component(loader)
    assert loader.from_pretrained.call_count == 1


def test_auto_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(config, "SAPBERT_DEVICE", "auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert step2.sapbert_device().type == "cpu"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_embeddings_run_on_model_device_and_return_cpu(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(4, 3)
        def forward(self, input_ids):
            assert input_ids.device == self.embedding.weight.device
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
    model = Encoder().to(device).eval()
    tokenizer = lambda terms, **kwargs: {"input_ids": torch.ones((len(terms), 2), dtype=torch.long)}
    vectors = step2.embed_terms(["a", "b"], tokenizer, model)
    assert vectors.device.type == "cpu"
    assert vectors.shape == (2, 3)
    assert torch.allclose(vectors.norm(dim=1), torch.ones(2))
