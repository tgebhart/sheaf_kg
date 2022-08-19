from collections import defaultdict
from typing import Optional, Tuple, List, Sequence, MutableMapping
import logging

import numpy as np
import torch

from class_resolver import HintOrType, OptionalKwargs
from pykeen.typing import Target, RankType, HintOrType, MappedTriples, cast
from pykeen.models import Model
from pykeen.evaluation import RankBasedEvaluator, RankBasedMetricResults, MetricResults
from pykeen.evaluation.rank_based_evaluator import _iter_ranks
from pykeen.evaluation.ranks import Ranks
from pykeen.metrics.ranking import HITS_METRICS, RankBasedMetric, rank_based_metric_resolver
from sheaf_kg.evaluation.evaluation_loop import evaluate

logger = logging.getLogger(__name__)

class BetaeEvaluator(RankBasedEvaluator):
    """A rank-based evaluator for KGE models."""

    num_entities: Optional[int]
    ranks: MutableMapping[Tuple[Target, RankType], List[np.ndarray]]
    num_candidates: MutableMapping[Target, List[np.ndarray]]

    def __init__(
        self,
        filtered: bool = False,
        metrics: Optional[Sequence[HintOrType[RankBasedMetric]]] = None,
        metrics_kwargs: OptionalKwargs = None,
        add_defaults: bool = True,
        **kwargs,
    ):
        """Initialize rank-based evaluator.

        :param filtered:
            Whether to use the filtered evaluation protocol. If enabled, ranking another true triple higher than the
            currently considered one will not decrease the score.
        :param metrics:
            the rank-based metrics to compute
        :param metrics_kwargs:
            additional keyword parameter
        :param add_defaults:
            whether to add all default metrics besides the ones specified by `metrics` / `metrics_kwargs`.
        :param kwargs: Additional keyword arguments that are passed to the base class.
        """
        super().__init__(
            filtered=filtered,
            metrics = metrics, 
            metrics_kwargs = metrics_kwargs,
            add_defaults = add_defaults,
            **kwargs,
        )

    def evaluate(
        self,
        model: Model,
        mapped_triples: MappedTriples,
        batch_size: Optional[int] = None,
        slice_size: Optional[int] = None,
        **kwargs,
    ) -> MetricResults:
        """
        Run :func:`pykeen.evaluation.evaluate` with this evaluator.

        This method will re-use the stored optimized batch and slice size, as well as the evaluator's inductive mode.

        :param model:
            the model to evaluate.
        :param mapped_triples: shape: (n, 3)
            the ID-based evaluation triples
        :param batch_size:
            the batch size to use, or `None` to trigger automatic memory optimization
        :param slice_size:
            the slice size to use
        :param kwargs:
            the keyword-based parameters passed to :func:`pykeen.evaluation.evaluate`

        :return:
            the evaluation results
        """
        # add mode parameter
        mode = kwargs.pop("mode", None)
        if mode is not None:
            logger.warning(f"Ignoring provided mode={mode}, and use the evaluator's mode={self.mode} instead")
        kwargs["mode"] = self.mode

        self.batch_size = 32 if batch_size is None else batch_size
        self.slice_size = 14 if slice_size is None else slice_size

        rv = evaluate(
            model=model,
            mapped_triples=mapped_triples,
            evaluator=self,
            batch_size=batch_size,
            slice_size=slice_size,
            **kwargs,
        )
        # Since squeeze is true, we can expect that evaluate returns a MetricResult, but we need to tell MyPy that
        return cast(MetricResults, rv)
