from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

_model = None


def _get_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_model():
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel
        device = _get_device()
        logger.info(f"Loading BAAI/bge-m3 on {device}...")
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=device)
        logger.info("bge-m3 loaded.")
    return _model


def embed_documents(texts: List[str]) -> List[List[float]]:
    model = get_model()
    output = model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
    return output["dense_vecs"].tolist()


def embed_query(text: str) -> List[float]:
    return embed_documents([text])[0]
