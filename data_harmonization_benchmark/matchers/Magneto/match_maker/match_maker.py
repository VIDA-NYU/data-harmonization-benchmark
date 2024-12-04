from typing import Dict, Tuple
import os

from valentine.algorithms.match import Match
from valentine.data_sources.base_table import BaseTable
from valentine.algorithms.base_matcher import BaseMatcher

from .embedding_matcher import DEFAULT_MODELS, EmbeddingMatcher
from .basic_matcher import get_str_similarity_candidates
from .utils import (
    clean_df,
    remove_invalid_characters,
    convert_simmap_to_valentine_format,
    convert_to_valentine_format,
    get_samples,
)

from .bp_reranker import arrange_bipartite_matches
from .llm_reranker import LLMReranker


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Magneto(BaseMatcher):
    ## attention
    ## for ablation experiments, make sure to have the default set correcly
    DEFAULT_PARAMS = {
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "encoding_mode": "header_values_repeat",
        "sampling_mode": "priority_sampling",
        "sampling_size": 10,
        "topk": 10,
        "include_strsim_matches": False,
        "include_embedding_matches": True,
        "embedding_threshold": 0.1,
        "include_equal_matches": True,
        "use_bp_reranker": True,
        "use_gpt_reranker": False,
    }

    def __init__(self, **kwargs):
        # Merge provided kwargs with defaults, use params in case you need more parameters: for ablation, etc
        self.params = {**self.DEFAULT_PARAMS, **kwargs}
        # print("Magneto Params:", self.params)

    def apply_strsim_matches(self):
        if self.params["include_strsim_matches"]:

            strsim_candidates = get_str_similarity_candidates(
                self.df_source.columns, self.df_target.columns
            )
            for (source_col, target_col), score in strsim_candidates.items():
                self.input_sim_map[source_col][target_col] = score

    def apply_embedding_matches(self):
        if not self.params["include_embedding_matches"]:
            return

        embeddingMatcher = EmbeddingMatcher(params=self.params)

        embedding_candidates = embeddingMatcher.get_embedding_similarity_candidates(
            self.df_source, self.df_target
        )
        for (col_source, col_target), score in embedding_candidates.items():
            self.input_sim_map[col_source][col_target] = score

        # if self.params['embedding_model'] in DEFAULT_MODELS:

        #     embeddingMatcher = EmbeddingMatcher(params=self.params)

        #     embedding_candidates = embeddingMatcher.get_embedding_similarity_candidates(
        #         self.df_source, self.df_target)
        #     for (col_source, col_target), score in embedding_candidates.items():
        #         self.input_sim_map[col_source][col_target] = score

        # else:

        #     retriever = Retriever(self.params['embedding_model'])

        #     source_values = {
        #         col: get_samples_2(self.df_source[col]) for col in self.df_source.columns
        #     }
        #     target_values = {
        #         col: get_samples_2(self.df_target[col]) for col in self.df_target.columns
        #     }
        #     matched_columns = retriever.find_matches(
        #         self.df_source, self.df_target, source_values, target_values, 20
        #     )
        #     # print("Initial Matches:", matched_columns)
        #     for key in matched_columns:
        #         for match in matched_columns[key]:
        #             self.input_sim_map[key][match[0]] = match[1]

    def apply_equal_matches(self):
        if self.params["include_equal_matches"]:

            source_cols_cleaned = {
                col: remove_invalid_characters(col.strip().lower())
                for col in self.df_source.columns
            }
            target_cols_cleaned = {
                col: remove_invalid_characters(col.strip().lower())
                for col in self.df_target.columns
            }

            for source_col, cand_source in source_cols_cleaned.items():
                for target_col, cand_target in target_cols_cleaned.items():
                    if cand_source == cand_target:
                        self.input_sim_map[source_col][target_col] = 1.0

    def get_top_k_matches(self, col_matches):
        sorted_matches = sorted(
            col_matches.items(), key=lambda item: item[1], reverse=True
        )
        top_k_matches = sorted_matches[: self.params["topk"]]
        return dict(top_k_matches)

    def call_llm_reranker(self, source_table, target_table, matches):

        orig_source_table, orig_target_table = source_table, target_table
        source_table = source_table.get_df()
        target_table = target_table.get_df()

        reranker = LLMReranker()

        source_values = {
            col: get_samples(source_table[col], 10) for col in source_table.columns
        }
        target_values = {
            col: get_samples(target_table[col], 10) for col in target_table.columns
        }

        matched_columns = {}
        for entry, score in matches.items():
            source_col = entry[0][1]
            target_col = entry[1][1]
            if source_col not in matched_columns:
                matched_columns[source_col] = [(target_col, score)]
            else:
                matched_columns[source_col].append((target_col, score))

        # print("Initial Matches:", matched_columns)

        matched_columns = reranker.rematch(
            source_table,
            target_table,
            source_values,
            target_values,
            matched_columns,
        )
        # print("Refined Matches:", matched_columns)

        converted_matches = convert_to_valentine_format(
            matched_columns,
            orig_source_table.name,
            orig_target_table.name,
        )

        return converted_matches

    def get_matches(
        self, source_table: BaseTable, target_table: BaseTable
    ) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:

        self.df_source = clean_df(source_table.get_df())
        self.df_target = clean_df(target_table.get_df())

        if len(self.df_source.columns) == 0 or len(self.df_target.columns) == 0:
            return {}

        # store similarity scores between columns
        # we replace the (col_src: col_tgt:score) entries with scores from "stronger" matchers as we progress
        self.input_sim_map = {col: {} for col in self.df_source.columns}

        if "strategy_order" in self.params:
            self.apply_strategies_in_order(self.params["strategy_order"])
        else:
            match_strategies = [
                self.apply_strsim_matches,
                self.apply_embedding_matches,
                self.apply_equal_matches,
            ]

            for strategy in match_strategies:
                strategy()  # runs the strategy and updates the input_sim_map

        # filter top-k matcher per column
        for col_source in self.input_sim_map:
            self.input_sim_map[col_source] = self.get_top_k_matches(
                self.input_sim_map[col_source]
            )

        matches = convert_simmap_to_valentine_format(
            self.input_sim_map, source_table.name, target_table.name
        )

        if self.params["use_bp_reranker"]:
            # print("Applying bipartite matching")
            matches = arrange_bipartite_matches(
                matches,
                self.df_source,
                source_table.name,
                self.df_target,
                target_table.name,
            )

        if self.params["use_gpt_reranker"]:
            print("Applying GPT reranker")
            matches = self.call_llm_reranker(source_table, target_table, matches)

        return matches

    ## only used in ablation experiments
    def apply_strategies_in_order(self, order):
        strategy_functions = {
            "strsim": self.apply_strsim_matches,
            "embedding": self.apply_embedding_matches,
            "equal": self.apply_equal_matches,
        }

        order = {k: v for k, v in order.items() if v != -1}
        sorted_strategies = sorted(order.items(), key=lambda item: item[1])

        for strategy, _ in sorted_strategies:
            strategy_functions[strategy]()
