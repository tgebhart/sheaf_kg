from typing import Any, Callable, Mapping, Optional, Sequence, Union

from pykeen.nn.representation import Embedding, Regularizer
from pykeen.typing import (
    Constrainer,
    Hint,
    HintType,
    Initializer,
    Normalizer,
    OneOrSequence,
)

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn


class ParameterizedEmbedding(Embedding):
    """Embeddings with an optional Torch parametrization.

    parametrization may be one of the two functions in
    torch.nn.utils.parametrizations, or a function that has the same
    behavior.
    """

    normalizer: Optional[Normalizer]
    constrainer: Optional[Constrainer]
    regularizer: Optional[Regularizer]
    dropout: Optional[nn.Dropout]

    def __init__(
        self,
        max_id: Optional[int] = None,
        num_embeddings: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        shape: Union[None, int, Sequence[int]] = None,
        initializer: Hint[Initializer] = None,
        initializer_kwargs: Optional[Mapping[str, Any]] = None,
        constrainer: Hint[Constrainer] = None,
        constrainer_kwargs: Optional[Mapping[str, Any]] = None,
        parametrization: Callable[[nn.Module], nn.Module] = None,
        trainable: bool = True,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__(
            max_id=max_id,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            shape=shape,
            initializer=initializer,
            initializer_kwargs=initializer_kwargs,
            constrainer=constrainer,
            constrainer_kwargs=constrainer_kwargs,
            trainable=trainable,
            dtype=dtype,
            **kwargs,
        )
        if parametrization is not None:
            # parametrize.register_parametrization(
            #     self._embeddings, "_weight", parametrization
            # )
            self._embeddings = parametrization(self._embeddings, name="weight")
