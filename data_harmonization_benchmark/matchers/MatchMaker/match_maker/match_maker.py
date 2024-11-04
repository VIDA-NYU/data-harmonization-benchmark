import os
import pandas as pd
from typing import Dict, Tuple
from valentine.algorithms.base_matcher import BaseMatcher
from valentine.data_sources.base_table import BaseTable
from valentine.algorithms.match import Match
from .utils import clean_df, remove_invalid_characters, get_samples, convert_to_valentine_format
from .value_similarity_ranker import ValueSimilarityRanker
# from .value_similarity_ranker_tfidf import ValueSimilarityRankerTFIDF

from .similarity_ranker import SimilarityRanker

from .matcher import ColumnMatcher

from scipy.optimize import linear_sum_assignment
import numpy as np

import pprint as pp


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MatchMaker(BaseMatcher):

    def __init__(self, topk=20, use_instances=False, use_gpt=False):
        self.topk = topk
        self.use_instances = use_instances
        self.use_gpt = use_gpt

    def get_matches(self, source_table: BaseTable, target_table: BaseTable) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:

        self.df_source = clean_df(source_table.get_df())
        self.df_target = clean_df(target_table.get_df())

        if len(self.df_source.columns) == 0 or len(self.df_target.columns) == 0:
            return {}

        # Input similarity map
        self.input_sim_map = {col: {} for col in self.df_source.columns}

        schemaSimRanker = SimilarityRanker(self.topk)


        # SCHEMA-Based Matches with string similarity and alignment
        strBasicSimilarities = schemaSimRanker.get_str_similarity_candidates(
            self.df_source.columns,  self.df_target.columns)
        for (col_source, col_target), score in strBasicSimilarities.items():
            self.input_sim_map[col_source][col_target] = score

        # # # SCHEMA-Based Matches with LM embeddings
        strEmbeddingSimilarities = schemaSimRanker.get_embedding_similarity_candidates(self.df_source,  self.df_target)
        for (col_source, col_target), score in strEmbeddingSimilarities.items():
            self.input_sim_map[col_source][col_target] = score

        
        if self.use_instances:
            valueSimRanker = ValueSimilarityRanker()
            # valueSimRanker = ValueSimilarityRankerTFIDF()
            valueSimilarities = valueSimRanker.get_type_and_value_based_candidates(self.df_source, self.df_target)

            for (col_source, col_target), score in valueSimilarities.items():
                if col_source in self.input_sim_map and col_target in self.input_sim_map[col_source]:
                #     self.input_sim_map[col_source][col_target] = max(
                #         self.input_sim_map[col_source][col_target], score)
                # else:
                    self.input_sim_map[col_source][col_target] = score
                #self.input_sim_map[col_source][col_target] = score

        
       

    



        # # ## just add the exact matches on top
        for source_col in self.df_source.columns:
            cand_source = remove_invalid_characters(source_col.strip().lower())
            for target_col in self.df_target.columns:
                cand_target = remove_invalid_characters(
                    target_col.strip().lower())
                if cand_source == cand_target:
                    self.input_sim_map[source_col][target_col] = 1.0

        # # Keep only the top-k entries for each column in input_sim_map
        for col_source in self.input_sim_map:
            sorted_matches = sorted(
                self.input_sim_map[col_source].items(), key=lambda item: item[1], reverse=True)
            top_k_matches = sorted_matches[:self.topk]
            self.input_sim_map[col_source] = dict(top_k_matches)

        # Wraps the matches into Valentine format
        matches = {}
        for col_input, matches_dict in self.input_sim_map.items():
            for col_target, score in matches_dict.items():
                match = Match(target_table.name, col_target,
                              source_table.name, col_input, score).to_dict
                matches.update(match)
                # print(match)


        if not self.use_gpt:
            matches = self.arrange_bipartite_matches(matches, source_table, target_table)

  
        if self.use_gpt:
            matches = self.llm_matcher(source_table, target_table, matches)
            # print("Matches:", matches)
            # pp.pprint(matches)
            # normzlized_matches = self._min_max_normalize(matches)
            # print("Normalized Matches:", normzlized_matches)
            # matches = convert_to_valentine_format(normzlized_matches, source_table.name, target_table.name)

        


        return matches

        # return matches

    def _min_max_normalize(self, valentine_matched_columns):

        matched_columns = {}
        for entry, score in valentine_matched_columns.items():
            source_col = entry[0][1]
            target_col = entry[1][1]
            if source_col not in matched_columns:
                matched_columns[source_col] = [(target_col, score)]
            else:
                matched_columns[source_col].append((target_col, score))

        normalized_columns = {}
        for key, values in matched_columns.items():
            # Extract only the weights for min-max normalization
            weights = [weight for _, weight in values]
            min_w = min(weights)
            max_w = max(weights)
            range_w = max_w - min_w
            normalized_list = [(name, (weight - min_w) / range_w if range_w > 0 else 0) for name, weight in values]
            normalized_columns[key] = normalized_list
        return normalized_columns

    def llm_matcher(self, source_table, target_table, matches):

        orig_source_table, orig_target_table = source_table, target_table
        source_table = source_table.get_df()
        target_table = target_table.get_df()

        matcher = ColumnMatcher()

        source_values = {
            col: get_samples(source_table[col], 15, False) for col in source_table.columns
        }
        target_values = {
            col: get_samples(target_table[col],15, False) for col in target_table.columns
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

        matched_columns = matcher.rematch(
                source_table,
                target_table,
                source_values,
                target_values,
                self.topk,
                matched_columns,
                self.topk,
            )
            # print("Refined Matches:", matched_columns)
       

        converted_matches = convert_to_valentine_format(
            matched_columns,
            orig_source_table.name,
            orig_target_table.name,
        )

        return converted_matches
    
    def arrange_bipartite_matches(self, initial_matches, source_table, target_table):

        filtered_matches = self.bipartite_filtering(initial_matches, source_table, target_table)

        # Step 1: Remove all filtered_matches entries from initial_matches
        for key in filtered_matches.keys():
            initial_matches.pop(key, None)

        # Step 2: Find the minimum score in filtered_matches
        min_filtered_score = min(filtered_matches.values())

        # Step 3: Calculate the scaling factor to keep scores in initial_matches just below min_filtered_score
        initial_max_score = max(initial_matches.values())
        scaling_factor = (min_filtered_score - 0.01) / initial_max_score if initial_max_score > 0 else 1

        # Adjust scores in initial_matches, maintaining relative differences
        adjusted_initial_matches = {
            key: score * scaling_factor for key, score in initial_matches.items()
        }

        # Step 4: Combine filtered_matches and adjusted initial_matches
        # filtered_matches entries will remain at the beginning
        filtered_matches.update(adjusted_initial_matches)

        return filtered_matches


    def bipartite_filtering(self, initial_matches, source_table, target_table):

        source_cols = set()
        target_cols = set()
        for (col_source, col_target), score in initial_matches.items():
            col_source = col_source[1]
            col_target = col_target[1]
            source_cols.add(col_source)
            target_cols.add(col_target)

        # Map columns to indices
        source_col_to_num = {col: idx for idx, col in enumerate(source_cols)}
        target_col_to_num = {col: idx for idx, col in enumerate(target_cols)}

        # Initialize the score matrix with zeros
        score_matrix = np.zeros((len(source_cols), len(target_cols)))

        # Populate the matrix with scores from initial matches
        for (col_source, col_target), score in initial_matches.items():
            col_source = col_source[1]
            col_target = col_target[1]
            source_idx = source_col_to_num[col_source]
            target_idx = target_col_to_num[col_target]
            score_matrix[source_idx, target_idx] = score

        # print("Score Matrix:\n", score_matrix)

        row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)
        assignment = list(zip(row_ind, col_ind))

        # print("Assignment:", assignment)

        filtered_matches = {}

        source_idx_to_col = {idx: col for col,
                             idx in source_col_to_num.items()}
        target_idx_to_col = {idx: col for col,
                             idx in target_col_to_num.items()}

        for source_idx, target_idx in assignment:
            source_col = source_idx_to_col[source_idx]
            target_col = target_idx_to_col[target_idx]
            filtered_matches[((source_table.name, source_col), (target_table.name, target_col))] = score_matrix[source_idx, target_idx]
 

        # print("Filtered Matches:", filtered_matches)
        return filtered_matches
