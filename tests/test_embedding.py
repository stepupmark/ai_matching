"""Tests for the EmbeddingService."""

from app.services.embedding import EmbeddingService


def test_embed_single() -> None:
    svc = EmbeddingService()
    vec = svc.embed("Python developer with 5 years experience")
    assert vec.shape == (svc.dimension,)
    assert vec.dtype.name == "float32"
    # Normalized → norm ≈ 1.0
    norm = (vec ** 2).sum() ** 0.5
    assert abs(norm - 1.0) < 0.01


def test_embed_batch() -> None:
    svc = EmbeddingService()
    texts = ["Python developer", "Data scientist with ML skills"]
    vecs = svc.embed_batch(texts)
    assert vecs.shape == (2, svc.dimension)
    # Each row normalised
    for i in range(2):
        norm = (vecs[i] ** 2).sum() ** 0.5
        assert abs(norm - 1.0) < 0.01


def test_embed_consistency() -> None:
    svc = EmbeddingService()
    text = "Machine learning engineer expert in TensorFlow and PyTorch"
    v1 = svc.embed(text)
    v2 = svc.embed(text)
    # Deterministic
    assert (v1 == v2).all()
