import warnings
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from pykeen.nn.representation import Regularizer, Representation, process_max_id, process_shape, constrainer_resolver
from pykeen.typing import (
    Constrainer,
    Hint,
    Initializer,
    Normalizer,
)
from pykeen.nn.init import initializer_resolver

import torch
from torch import nn

class ParameterizedEmbedding(Representation):
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
        parameterization: Callable[[nn.Module], nn.Module] = None,
        trainable: bool = True,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
       # normalize num_embeddings vs. max_id
        max_id = process_max_id(max_id, num_embeddings)

        # normalize embedding_dim vs. shape
        _embedding_dim, shape = process_shape(embedding_dim, shape)

        if dtype is None:
            dtype = torch.get_default_dtype()

        # work-around until full complex support (torch==1.10 still does not work)
        # TODO: verify that this is our understanding of complex!
        self.is_complex = dtype.is_complex
        _shape = shape
        if self.is_complex:
            _shape = tuple(shape[:-1]) + (shape[-1], 2)
            _embedding_dim = _embedding_dim * 2
            # note: this seems to work, as finfo returns the datatype of the underlying floating
            # point dtype, rather than the combined complex one
            dtype = getattr(torch, torch.finfo(dtype).dtype)
        self._shape = _shape

        super().__init__(max_id=max_id, shape=shape, **kwargs)

        # use make for initializer since there's a default, and make_safe
        # for the others to pass through None values
        self.initializer = initializer_resolver.make(initializer, initializer_kwargs)
        self.constrainer = constrainer_resolver.make_safe(constrainer, constrainer_kwargs)
        self._embeddings = nn.Parameter(torch.empty((self.num_embeddings, *self._shape), requires_grad=trainable, dtype=dtype))
        self.parameterization = parameterization

    @property
    def num_embeddings(self) -> int:  # noqa: D401
        """The total number of representations (i.e. the maximum ID)."""
        # wrapper around max_id, for backward compatibility
        warnings.warn(f"Directly use {self.__class__.__name__}.max_id instead of num_embeddings.")
        return self.max_id

    # docstr-coverage: inherited
    def reset_parameters(self) -> None:  # noqa: D102
        # initialize weights in-place
        self._embeddings.data = self.initializer(
            self._embeddings.data.view(self.num_embeddings, *self._shape),
        )
        # now add parameterization
        if self.parameterization is not None:
            self = self.parameterization(self, name='_embeddings', use_trivialization=False)

    # docstr-coverage: inherited
    def post_parameter_update(self):  # noqa: D102
        # apply constraints in-place
        if self.constrainer is not None:
            x = self._plain_forward()
            x = self.constrainer(x)
            # fixme: work-around until nn.Embedding supports complex
            if self.is_complex:
                x = torch.view_as_real(x)
            self._embeddings.data = x.view(*self._embeddings.data.shape)

    # docstr-coverage: inherited
    def _plain_forward(
        self,
        indices: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        if indices is None:
            prefix_shape = (self.max_id,)
            x = self._embeddings
        else:
            prefix_shape = indices.shape
            x = torch.index_select(self._embeddings, 0, indices.to(self.device))
        x = x.view(*prefix_shape, *self._shape)
        # fixme: work-around until nn.Embedding supports complex
        if self.is_complex:
            x = torch.view_as_complex(x)
        # verify that contiguity is preserved
        assert x.is_contiguous()
        return x