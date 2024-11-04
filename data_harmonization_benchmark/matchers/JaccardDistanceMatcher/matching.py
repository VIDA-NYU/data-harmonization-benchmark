from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
from valentine import valentine_match
from valentine.algorithms import JaccardDistanceMatcher
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
    matcher = JaccardDistanceMatcher(**config)

    matches = valentine_match(source, target, matcher)

    return matches
