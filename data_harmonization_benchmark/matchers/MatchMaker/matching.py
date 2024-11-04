
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

    ),
):
    


    matcher = MatchMaker()
    matches = matcher.match(source, target, top_k=top_k)

    return matches


