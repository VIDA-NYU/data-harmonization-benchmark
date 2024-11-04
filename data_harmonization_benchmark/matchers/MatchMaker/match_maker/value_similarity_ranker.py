import concurrent.futures
from typing import Dict, List, Tuple

import numpy as np
import torch
import tqdm
from fuzzywuzzy import fuzz
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from .embedding_utils import compute_cosine_similarity, compute_cosine_similarity_simple
from .utils import (
    clean_column_name,
    common_prefix,
    get_samples,
    get_type2columns_map,
    preprocess_string,
)


class ValueSimilarityRanker:
    def __init__(
        self,
        model_name="sentence-transformers/all-mpnet-base-v2",
        topk=10,
        embedding_sim_threshold=0.5,
    ):
        if model_name is None:
            model_name = "sentence-transformers/all-mpnet-base-v2"

        self.model_name = model_name

        print(f"Loading model {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.topk = topk
        self.embedding_sim_threshold = embedding_sim_threshold

    def _get_embeddings(self, texts, batch_size=32):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts, padding=True, truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1))
        return torch.cat(embeddings)

    def _encode(self, col_name, col_values, mode="OnlyValues"):
        if mode == "OnlyValues":
            return col_values

        if mode == "ColumnNameAndValue":
            return [col_name + ": " + str(val) for val in col_values]

        if mode == "ColumnNameAndMultipleValues":
            # Sort the values first
            sorted_values = sorted(col_values)

            # Group them in sets of four and concatenate with the column name
            result = []
            for i in range(0, len(sorted_values), 4):
                group = sorted_values[
                    i : i + 4
                ]  # Get the next group (even if it's less than 4)
                group_str = ", ".join(map(str, group))  # Convert the group to string
                result.append(f"{col_name}: {group_str}")

            return result
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _create_column_value_embedding_dict(self, df):
        col2_val_emb = {}

        def process_column(column):
            col_values = df[column].dropna().unique().tolist()
            # col_values = get_samples(df[column], n=20, random=False)
            col_values = self._encode(column, col_values)
            return column, col_values, self._get_embeddings(col_values)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(process_column, column): column for column in df.columns
            }

            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                column, col_values, col_embeddings = future.result()
                col2_val_emb[column] = (col_values, col_embeddings)
        return col2_val_emb

    def get_type_and_value_based_candidates(
        self, source_df, target_df, include_key_as_matches=False
    ):
        candidates = {}

        type2cols_source = get_type2columns_map(source_df)
        type2cols_target = get_type2columns_map(target_df)

        # if include_key_as_matches:
        #     for key_source in type2cols_source['key']:
        #         for key_target in type2cols_target['key']:
        #             candidates[(key_source, key_target)] = (100 + fuzz.ratio(key_source, key_target)) / 200.0

        cat_cols_source = type2cols_source["categorical"]
        cat_cols_target = type2cols_target["categorical"]

        if len(cat_cols_source) == 0 or len(cat_cols_target) == 0:
            return {}

        source_df_cat = source_df[cat_cols_source]
        target_df_cat = target_df[cat_cols_target]

        col2valemb_input = self._create_column_value_embedding_dict(source_df_cat)
        col2valemb_target = self._create_column_value_embedding_dict(target_df_cat)

        # flatten version of the embeddings for faster computation of cosine similarity
        all_target_embeddings = []
        all_target_values = []
        target_column_mapping = []

        # Track the start and end positions for each target column, for retrieving the embeddings from all_target_embeddings for a specific column
        current_position = 0
        target_column_boundaries = {}

        for target_col, (target_values, target_embeddings) in col2valemb_target.items():
            num_values = len(target_values)

            all_target_embeddings.append(target_embeddings)
            all_target_values.extend(target_values)
            target_column_mapping.extend([target_col] * num_values)

            target_column_boundaries[target_col] = (
                current_position,
                current_position + num_values,
            )
            current_position += num_values

        # we flatten the embeddings for faster computation
        all_target_embeddings = torch.cat(all_target_embeddings)

        for input_col, (input_values, input_embeddings) in col2valemb_input.items():
            # for each column, compare to all values in target columns
            top_k_scores, top_k_indices, similarities = compute_cosine_similarity(
                input_embeddings, all_target_embeddings, self.topk
            )
            top_k_columns = {
                target_column_mapping[idx]
                for row_indices in top_k_indices
                for idx in row_indices
            }

            for target_col in list(top_k_columns):
                # TODO: save the value mappings
                (indx_begin, idx_end) = target_column_boundaries[target_col]

                similarities_col = similarities[:, indx_begin:idx_end]

                max_scores, max_positions = torch.max(similarities_col, dim=1)

                score_sum = torch.sum(max_scores).item() / len(max_scores)

                if score_sum >= self.embedding_sim_threshold:
                    candidates[(input_col, target_col)] = score_sum

        return candidates
