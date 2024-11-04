import logging
from typing import Any, Dict, Optional

import pandas as pd
from valentine import valentine_match

from match_maker.match_maker import MatchMaker

logger = logging.getLogger(__name__)

def matching(
    usecase: str,
    usecase_path: str,
    source: pd.DataFrame,
    target: pd.DataFrame,
    top_k: int = 20,
    use_gpu: bool = False,
    config: Optional[Dict[str, Any]] = dict(
        use_instances = True,
        use_gpt = False
    ),
):
    


    matcher = MatchMaker(topk=top_k, **config)
    matches = valentine_match(source, target, matcher)

    return matches