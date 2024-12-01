import logging
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
from match_maker.match_maker import Magneto
from valentine import valentine_match
from valentine.algorithms.matcher_results import MatcherResults

logger = logging.getLogger(__name__)


def matching(
    usecase: str,
    usecase_path: str,
    source: pd.DataFrame,
    target: pd.DataFrame,
    top_k: int = 20,
    use_gpu: bool = False,
    config: Optional[Dict[str, Any]] = dict(
        use_bp_reranker=False,
        use_gpt_reranker=True,
    ),
) -> Union[MatcherResults, Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]]:
    matcher = Magneto(topk=top_k, **config)
    matches = valentine_match(source, target, matcher)

    return matches
