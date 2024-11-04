from typing import Dict, List, Tuple

import numpy as np
import torch
from fuzzywuzzy import fuzz
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from .embedding_utils import compute_cosine_similarity, compute_cosine_similarity_simple
from .utils import (
    clean_column_name,
    common_prefix,
    detect_column_type,
    get_samples,
    preprocess_string,
    remove_invalid_characters,
)


class SimilarityRanker:
    def __init__(
        self,
        topk=20,
        embedding_threshold=0.65,
        alignment_threshold=0.95,
        fuzzy_similarity_threshold=0.4,
    ):
        self.model_name = "sentence-transformers/all-mpnet-base-v2"
        print(f"Loading model {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        self.topk = topk

        self.embedding_threshold = embedding_threshold

        self.alignment_threshold = alignment_threshold
        self.fuzzy_similarity_threshold = fuzzy_similarity_threshold

    def alignment_score_consecutive(
        self, str1, str2, max_distance=2, size_ratio_threshold=2
    ):
        s1 = str1
        s2 = str2
        # Preprocess strings (assuming this function exists)
        str1 = preprocess_string(str1)
        str2 = preprocess_string(str2)

        # Determine shorter and longer strings
        if len(str1) <= len(str2):
            shorter, longer = str1, str2
        else:
            shorter, longer = str2, str1

        # Early exit if strings have disproportionate lengths
        if len(longer) > len(shorter) * size_ratio_threshold:
            return 0

        matches = 0
        last_index = -1

        # Find matches for each letter in the shorter string
        for char in shorter:
            for i in range(last_index + 1, len(longer)):
                if longer[i] == char:
                    # Check if the distance between the current match and the last one is <= max_distance
                    if last_index == -1 or (i - last_index) <= max_distance:
                        matches += 1
                        last_index = i
                        break
                    else:
                        # If the distance is greater than max_distance, stop the inner loop
                        break

        # Calculate score
        score = matches / len(shorter) if len(shorter) > 0 else 0

        return score

    def fuzzy_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity using fuzzy matching (Levenshtein distance)."""
        return fuzz.ratio(s1, s2) / 100.0  # Normalize the score to a range [0, 1]

    def get_str_similarity_candidates(self, source_column_names, target_column_names):
        prefix_source = common_prefix(list(source_column_names))
        prefix_target = common_prefix(list(target_column_names))

        candidates = {}
        for source_col in source_column_names:
            prep_source_col = source_col.replace(prefix_source, "")

            for target_col in target_column_names:
                prep_target_col = target_col.replace(prefix_target, "")

                alignment_score = self.alignment_score_consecutive(
                    prep_source_col, prep_target_col
                )

                if alignment_score >= self.alignment_threshold:
                    candidates[(source_col, target_col)] = alignment_score

                name_similarity = self.fuzzy_similarity(
                    prep_source_col, prep_target_col
                )

                if name_similarity >= self.fuzzy_similarity_threshold:
                    candidates[(source_col, target_col)] = name_similarity

        return candidates

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

    def encode(self, df, col):
        # header = clean_column_name(col)
        header = col
        data_type = detect_column_type(df[col])
        tokens = get_samples(df[col])

        text = (
            self.tokenizer.cls_token
            + "Column: "
            + header
            + self.tokenizer.sep_token
            + "Type: "
            + data_type
            + self.tokenizer.sep_token
            + "Values: "
            + self.tokenizer.sep_token.join(tokens)
            + self.tokenizer.sep_token
            + self.tokenizer.eos_token  # End-of-sequence token
        )

        # print(text)

        return text

    def get_embedding_similarity_candidates(self, source_df, target_df):
        # source_column_names = source_df.columns
        # target_column_names = target_df.columns

        # input_colnames_dict = {clean_column_name(
        #     col): col for col in source_column_names}
        # target_colnames_dict = {clean_column_name(
        #     col): col for col in target_column_names}

        input_colnames_dict = {
            self.encode(source_df, col): col for col in source_df.columns
        }
        target_colnames_dict = {
            self.encode(target_df, col): col for col in target_df.columns
        }

        # print(input_colnames_dict)
        # print(target_colnames_dict)

        cleaned_input_colnames = list(input_colnames_dict.keys())
        cleaned_target_colnames = list(target_colnames_dict.keys())

        embeddings_input = self._get_embeddings(cleaned_input_colnames)
        embeddings_target = self._get_embeddings(cleaned_target_colnames)

        top_k = min(self.topk, len(cleaned_target_colnames))
        topk_similarity, topk_indices = compute_cosine_similarity_simple(
            embeddings_input, embeddings_target, top_k
        )

        candidates = {}
        # avg_similarity = 0
        for i, cleaned_input_col in enumerate(cleaned_input_colnames):
            original_input_col = input_colnames_dict[cleaned_input_col]

            for j in range(top_k):
                cleaned_target_col = cleaned_target_colnames[topk_indices[i, j]]
                original_target_col = target_colnames_dict[cleaned_target_col]
                similarity = topk_similarity[i, j].item()
                # print(similarity)

                # avg_similarity += similarity

                if similarity >= self.embedding_threshold:
                    # print(original_input_col, original_target_col, similarity)

                    candidates[(original_input_col, original_target_col)] = similarity

        # avg_similarity /= len(cleaned_input_colnames) * top_k

        # print(f"Average similarity: {avg_similarity}")
        # print(candidates)
        return candidates

    # def get_embedding_similarity_candidates(self, source_column_names, target_column_names):

    #     input_colnames_dict = {clean_column_name(
    #         col): col for col in source_column_names}
    #     target_colnames_dict = {clean_column_name(
    #         col): col for col in target_column_names}

    #     cleaned_input_colnames = list(input_colnames_dict.keys())
    #     cleaned_target_colnames = list(target_colnames_dict.keys())

    #     embeddings_input = self._get_embeddings(cleaned_input_colnames)
    #     embeddings_target = self._get_embeddings(cleaned_target_colnames)

    #     top_k = min(self.topk, len(cleaned_target_colnames))
    #     topk_similarity, topk_indices = compute_cosine_similarity_simple(
    #         embeddings_input, embeddings_target, top_k)

    #     candidates = {}
    #     for i, cleaned_input_col in enumerate(cleaned_input_colnames):
    #         original_input_col = input_colnames_dict[cleaned_input_col]

    #         for j in range(top_k):
    #             cleaned_target_col = cleaned_target_colnames[topk_indices[i, j]]
    #             original_target_col = target_colnames_dict[cleaned_target_col]
    #             similarity = topk_similarity[i, j].item()

    #             print(similarity)
    #             if similarity >= self.embedding_threshold:

    #                 candidates[(original_input_col,
    #                             original_target_col)] = similarity
    #     return candidates
