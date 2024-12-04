from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
from valentine import valentine_match
from valentine.algorithms import Coma
from valentine.algorithms.matcher_results import MatcherResults


def matching(
    usecase: str,
    usecase_path: str,
    source: pd.DataFrame,
    target: pd.DataFrame,
    top_k: int = 10,
    use_gpu: bool = False,
    config: Optional[Dict[str, Any]] = dict(),
) -> Union[MatcherResults, Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]]:

    source = source.sample(min(500, source.shape[0]))
    target = target.sample(min(500, target.shape[0]))

    matcher = Coma(use_instances=True, java_xmx="10096m", max_n=top_k, **config)

    matches = valentine_match(source, target, matcher)

    return matches
