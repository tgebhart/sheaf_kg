from typing import Optional, Mapping, Collection, Union, List, cast, Iterable
from class_resolver import HintOrType, OptionalKwargs
from collections import defaultdict
import logging
from tqdm.autonotebook import tqdm

import torch
import numpy as np

from pykeen.typing import LABEL_HEAD, LABEL_TAIL, InductiveMode, MappedTriples, Target
from pykeen.constants import TARGET_TO_INDEX
from pykeen.evaluation.evaluator import MetricResults, optional_context_manager
from pykeen.models import Model
# from pykeen.evaluation import LCWAEvaluationLoop, LCWAEvaluationDataset
from pykeen.triples.triples_factory import CoreTriplesFactory
from pykeen.evaluation.evaluator import Evaluator, filter_scores_
from pykeen.utils import split_list_in_batches_iter

# from sheaf_kg.data_loader import TestDataset, flatten_query

logger = logging.getLogger(__name__)

def _evaluate_batch(
    batch: MappedTriples,
    model: Model,
    target: Target,
    evaluator: Evaluator,
    slice_size: Optional[int],
    all_pos_triples: Optional[MappedTriples],
    relation_filter: Optional[torch.BoolTensor],
    restrict_entities_to: Optional[torch.LongTensor],
    *,
    mode: Optional[InductiveMode],
) -> torch.BoolTensor:
    """
    Evaluate ranking for batch.

    :param batch: shape: (batch_size, 3)
        The batch of currently evaluated triples.
    :param model:
        The model to evaluate.
    :param target:
        The prediction target.
    :param evaluator:
        The evaluator
    :param slice_size:
        An optional slice size for computing the scores.
    :param all_pos_triples:
        All positive triples (required if filtering is necessary).
    :param relation_filter:
        The relation filter. Can be re-used.
    :param restrict_entities_to:
        Restriction to evaluate only for these entities.
    :param mode:
        the inductive mode, or None for transductive evaluation

    :raises ValueError:
        if all positive triples are required (either due to filtered evaluation, or requiring dense masks).

    :return:
        The relation filter, which can be re-used for the same batch.
    """
    scores = model.predict(hrt_batch=batch, target=target, slice_size=slice_size, mode=mode)

    true_idxs = [b['target'] for b in batch]

    if evaluator.filtered:
        # Select scores of true
        true_scores = scores[torch.arange(0, len(batch)), true_idxs]
        # the rank-based evaluators needs the true scores with trailing 1-dim
        true_scores = true_scores.unsqueeze(dim=-1)
    else:
        true_scores = None


    positive_mask = None

    # Restrict to entities of interest
    if restrict_entities_to is not None:
        scores = scores[:, restrict_entities_to]
        if positive_mask is not None:
            positive_mask = positive_mask[:, restrict_entities_to]

    # process scores
    evaluator.process_scores_(
        hrt_batch=batch,
        target=target,
        true_scores=true_scores,
        scores=scores,
        dense_positive_mask=positive_mask,
    )

    return relation_filter

def evaluate(
    model: Model,
    mapped_triples: MappedTriples,
    evaluator: Evaluator,
    only_size_probing: bool = False,
    batch_size: Optional[int] = None,
    slice_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    use_tqdm: bool = True,
    tqdm_kwargs: Optional[Mapping[str, str]] = None,
    restrict_entities_to: Optional[Collection[int]] = None,
    restrict_relations_to: Optional[Collection[int]] = None,
    do_time_consuming_checks: bool = True,
    additional_filter_triples: Union[None, MappedTriples, List[MappedTriples]] = None,
    pre_filtered_triples: bool = True,
    targets: Collection[Target] = (LABEL_HEAD, LABEL_TAIL),
    *,
    mode: Optional[InductiveMode],
) -> MetricResults:
    """Evaluate metrics for model on mapped triples.

    The model is used to predict scores for all tails and all heads for each triple. Subsequently, each abstract
    evaluator is applied to the scores, also receiving the batch itself (e.g. to compute entity-specific metrics).
    Thereby, the (potentially) expensive score computation against all entities is done only once. The metric evaluators
    are expected to maintain their own internal buffers. They are returned after running the evaluation, and should
    offer a possibility to extract some final metrics.

    :param model:
        The model to evaluate.
    :param mapped_triples:
        The triples on which to evaluate. The mapped triples should never contain inverse triples - these are created by
        the model class on the fly.
    :param evaluator:
        The evaluator.
    :param only_size_probing:
        The evaluation is only performed for two batches to test the memory footprint, especially on GPUs.
    :param batch_size: >0
        A positive integer used as batch size. Generally chosen as large as possible. Defaults to 1 if None.
    :param slice_size: >0
        The divisor for the scoring function when using slicing.
    :param device:
        The device on which the evaluation shall be run. If None is given, use the model's device.
    :param use_tqdm:
        Should a progress bar be displayed?
    :param tqdm_kwargs:
        Additional keyword based arguments passed to the progress bar.
    :param restrict_entities_to:
        Optionally restrict the evaluation to the given entity IDs. This may be useful if one is only interested in a
        part of the entities, e.g. due to type constraints, but wants to train on all available data. For ranking the
        entities, we still compute all scores for all possible replacement entities to avoid irregular access patterns
        which might decrease performance, but the scores will afterwards be filtered to only keep those of interest.
        If provided, we assume by default that the triples are already filtered, such that it only contains the
        entities of interest. To explicitly filter within this method, pass `pre_filtered_triples=False`.
    :param restrict_relations_to:
        Optionally restrict the evaluation to the given relation IDs. This may be useful if one is only interested in a
        part of the relations, e.g. due to relation types, but wants to train on all available data. If provided, we
        assume by default that the triples are already filtered, such that it only contains the relations of interest.
        To explicitly filter within this method, pass `pre_filtered_triples=False`.
    :param do_time_consuming_checks:
        Whether to perform some time consuming checks on the provided arguments. Currently, this encompasses:
        - If restrict_entities_to or restrict_relations_to is not None, check whether the triples have been filtered.
        Disabling this option can accelerate the method. Only effective if pre_filtered_triples is set to True.
    :param pre_filtered_triples:
        Whether the triples have been pre-filtered to adhere to restrict_entities_to / restrict_relations_to. When set
        to True, and the triples have *not* been filtered, the results may be invalid. Pre-filtering the triples
        accelerates this method, and is recommended when evaluating multiple times on the same set of triples.
    :param additional_filter_triples:
        additional true triples to filter out during filtered evaluation.
    :param targets:
        the prediction targets
    :param mode:
        the inductive mode, or None for transductive evaluation

    :raises NotImplementedError:
        if relation prediction evaluation is requested
    :raises ValueError:
        if the pre_filtered_triples contain unwanted entities (can only be detected with the time-consuming checks).

    :return:
        the evaluation results
    """
    print('inside sheaf_kg evaluation loop')
    
    # Send to device
    if device is not None:
        model = model.to(device)
    device = model.device

    # Ensure evaluation mode
    model.eval()

    all_pos_triples = None

    # Send tensors to device
    # mapped_triples = mapped_triples.to(device=device)

    # Prepare batches
    if batch_size is None:
        # This should be a reasonable default size that works on most setups while being faster than batch_size=1
        batch_size = 32
        logger.info(f"No evaluation batch_size provided. Setting batch_size to '{batch_size}'.")
    batches = cast(Iterable[np.ndarray], split_list_in_batches_iter(input_list=mapped_triples, batch_size=batch_size))

    # Show progressbar
    num_triples = len(mapped_triples)

    # Flag to check when to quit the size probing
    evaluated_once = False

    # Disable gradient tracking
    _tqdm_kwargs = dict(
        desc=f"Evaluating on {model.device}",
        total=num_triples,
        unit="triple",
        unit_scale=True,
        # Choosing no progress bar (use_tqdm=False) would still show the initial progress bar without disable=True
        disable=not use_tqdm,
    )
    if tqdm_kwargs:
        _tqdm_kwargs.update(tqdm_kwargs)
    with optional_context_manager(use_tqdm, tqdm(**_tqdm_kwargs)) as progress_bar, torch.inference_mode():
        # batch-wise processing
        for batch in batches:
            batch_size = len(batch)
            relation_filter = None
            for target in targets:
                relation_filter = _evaluate_batch(
                    batch=batch,
                    model=model,
                    target=target,
                    evaluator=evaluator,
                    slice_size=slice_size,
                    all_pos_triples=all_pos_triples,
                    relation_filter=relation_filter,
                    restrict_entities_to=restrict_entities_to,
                    mode=mode,
                )

            # If we only probe sizes we do not need more than one batch
            if only_size_probing and evaluated_once:
                break

            evaluated_once = True

            if use_tqdm:
                progress_bar.update(batch_size)

        # Finalize
        result = evaluator.finalize()

    return result