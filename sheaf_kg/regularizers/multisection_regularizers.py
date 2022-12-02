from typing import Any, ClassVar, Iterable, Mapping, Optional

import torch
from pykeen.regularizers import Regularizer

class OrthogonalSectionsRegularizer(Regularizer):
    """A simple L_p norm based regularizer."""

    #: The dimension along which to compute the vector-based regularization terms.
    dims: Optional[Iterable[int]]

    #: Whether to normalize the regularization term by the dimension of the vectors.
    #: This allows dimensionality-independent weight tuning.
    normalize: bool

    #: The default strategy for optimizing the LP regularizer's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        weight=dict(type=float, low=0.01, high=1.0, scale="log"),
    )

    def __init__(
        self,
        *,
        # could be moved into kwargs, but needs to stay for experiment integrity check
        weight: float = 1.0,
        # could be moved into kwargs, but needs to stay for experiment integrity check
        apply_only_once: bool = False,
        dims: Optional[Iterable[int]] = (-2, -1),
        normalize: bool = False,
        p: float = 2.0,
        **kwargs,
    ):
        """
        Initialize the regularizer.

        :param weight:
            The relative weight of the regularization
        :param apply_only_once:
            Should the regularization be applied more than once after reset?
        :param dim:
            the dimension along which to calculate the Lp norm, cf. :func:`lp_norm`
        :param normalize:
            whether to normalize the norm by the dimension, cf. :func:`lp_norm`
        :param p:
            the parameter $p$ of the Lp norm, cf. :func:`lp_norm`
        :param kwargs:
            additional keyword-based parameters passed to :meth:`Regularizer.__init__`
        """
        super().__init__(weight=weight, apply_only_once=apply_only_once, **kwargs)
        self.dims = dims
        self.normalize = normalize
        self.p = p

    # docstr-coverage: inherited
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # noqa: D102
        XtX = torch.transpose(x,-2,-1) @ x
        dshape = XtX.shape[-1]
        I = torch.eye(dshape, device=XtX.device).reshape((1, dshape, dshape)).repeat(x.shape[0], 1, 1)
        return torch.linalg.norm(XtX - I, ord=self.p, dim=self.dims).sum()