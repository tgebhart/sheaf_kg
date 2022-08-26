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
from torch.nn.utils.parametrizations import _Orthogonal, _OrthMaps
import torch.nn.utils.parametrize as parametrize
from torch import nn

class PykeenOrthogonal(_Orthogonal):

    def __init__(self,
                 shape,
                 weight,
                 orthogonal_map: _OrthMaps,
                 *,
                 use_trivialization=True) -> None:
        self.forward_shape = shape

        super().__init__(weight, orthogonal_map, use_trivialization=use_trivialization)

    def forward(self, X):
        return super(PykeenOrthogonal, self).forward(X.view(*self.forward_shape))

def pykeen_orthogonal(module: nn.Module,
               shape,
               name: str = 'weight',
               orthogonal_map: Optional[str] = None,
               *,
               use_trivialization: bool = True) -> nn.Module:
    
    weight = getattr(module, name, None).reshape(*shape)
    if not isinstance(weight, torch.Tensor):
        raise ValueError(
            "Module '{}' has no parameter ot buffer with name '{}'".format(module, name)
        )

    # We could implement this for 1-dim tensors as the maps on the sphere
    # but I believe it'd bite more people than it'd help
    if weight.ndim < 2:
        raise ValueError("Expected a matrix or batch of matrices. "
                         f"Got a tensor of {weight.ndim} dimensions.")

    if orthogonal_map is None:
        orthogonal_map = "matrix_exp" if weight.size(-2) == weight.size(-1) or weight.is_complex() else "householder"

    orth_enum = getattr(_OrthMaps, orthogonal_map, None)
    if orth_enum is None:
        raise ValueError('orthogonal_map has to be one of "matrix_exp", "cayley", "householder". '
                         f'Got: {orthogonal_map}')
    orth = _Orthogonal(
                        weight,
                       orth_enum,
                       use_trivialization=use_trivialization)
    parametrize.register_parametrization(module, name, orth, unsafe=True)
    return module

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
        parameterization_hint = None,
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
        if parameterization_hint is not None and parameterization_hint == 'orthogonal':
            self._embeddings = pykeen_orthogonal(self._embeddings, (self.num_embeddings, *self.shape), name="weight")