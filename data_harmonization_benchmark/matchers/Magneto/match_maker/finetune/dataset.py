from algorithms.schema_matching.match_maker.utils import (
    detect_column_type,
    clean_element,
    get_samples,
)
from train_utils import sentence_transformer_map
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class CustomDataset(Dataset):
    def __init__(
        self,
        data,
        model_type="mpnet",
        serialization="header_values_verbose",
        augmentation="exact_semantic",
    ):
        self.serialization = serialization
        self.tokenizer = AutoTokenizer.from_pretrained(
            sentence_transformer_map[model_type]
        )
        self.labels = []
        self.items = self._initialize_items(data, augmentation)

        self._serialization_methods = {
            "header_values_default": self._serialize_header_values_default,
            "header_values_prefix": self._serialize_header_values_prefix,
            "header_values_repeat": self._serialize_header_values_repeat,
            "header_values_verbose": self._serialize_header_values_verbose,
            "header_only": self._serialize_header_only,
            "header_values_verbose_notype": self._serialize_header_values_verbose_notype,
            "header_values_columnvaluepair_notype": self._serialize_header_values_columnvaluepair_notype,
            "header_header_values_repeat_notype": self._serialize_header_values_repeat_notype,
            "header_values_default_notype": self._serialize_header_values_default,
        }
        self.cls_token = self.tokenizer.cls_token or ""
        self.sep_token = self.tokenizer.sep_token or ""
        self.eos_token = self.tokenizer.eos_token or ""

    def _initialize_items(self, data, augmentation):
        items = []
        class_id = 0

        for _, categories in data.items():
            for aug_type, columns in categories.items():
                if aug_type in augmentation or aug_type == "original":
                    for column_name, values in columns.items():
                        processed_column_name = (
                            column_name.rsplit("_", 1)[0]
                            if aug_type == "exact"
                            else column_name
                        )
                        values = [
                            (
                                clean_element(value)
                                if isinstance(value, str)
                                else str(value)
                            )
                            for value in values
                        ]
                        tokens = get_samples(
                            pd.Series(values), n=10, mode="priority_sampling"
                        )
                        items.append((processed_column_name, tokens, class_id))
                        self.labels.append(class_id)
            class_id += 1

        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        key, values, class_id = self.items[idx]
        text = self._serialize(key, values)
        return text, class_id

    # def _serialize(self, header, values):
    #     if values:
    #         col = pd.DataFrame({header: values})[header]
    #         data_type = detect_column_type(pd.DataFrame({header: values})[header])
    #     else:
    #         data_type = "unknown"
    #     serialization = {
    #         "header_values_default": f"{self.tokenizer.cls_token}{header}{self.tokenizer.sep_token}{data_type}{self.tokenizer.sep_token}{','.join(map(str, values))}",
    #         "header_values_prefix": f"{self.tokenizer.cls_token}header:{header}{self.tokenizer.sep_token}datatype:{data_type}{self.tokenizer.sep_token}values:{', '.join(map(str, values))}",
    #     }
    #     return serialization[self.serialization]
    def _serialize(self, header, values):
        if values:
            col = pd.DataFrame({header: values})[header]
            data_type = detect_column_type(col)
        else:
            data_type = "unknown"
            values = []

        tokens = [str(token) for token in values]

        return self._serialization_methods[self.serialization](
            header, data_type, tokens
        )

    def _serialize_header_values_verbose(self, header, data_type, tokens):
        """Serializes with detailed column header, type, and token values."""
        return (
            f"{self.cls_token}"
            f"Column: {header}{self.sep_token}"
            f"Type: {data_type}{self.sep_token}"
            f"Values: {self.sep_token.join(tokens)}{self.sep_token}"
        )

    def _serialize_header_values_default(self, header, data_type, tokens):
        """Serializes with default format including header, type, and tokens."""
        return (
            f"{self.cls_token}"
            f"{header}{self.sep_token}"
            f"{data_type}{self.sep_token}"
            f"{self.sep_token.join(tokens)}"
        )

    def _serialize_header_values_prefix(self, header, data_type, tokens):
        """Serializes with prefixed labels for header, datatype, and values."""
        return (
            f"{self.cls_token}"
            f"header:{header}{self.sep_token}"
            f"datatype:{data_type}{self.sep_token}"
            f"values:{', '.join(tokens)}"
        )

    def _serialize_header_values_repeat(self, header, data_type, tokens):
        """Serializes with repeated header for emphasis."""
        repeated_header = self.sep_token.join([header] * 5)
        return (
            f"{self.cls_token}"
            f"{repeated_header}{self.sep_token}"
            f"{data_type}{self.sep_token}"
            f"{self.sep_token.join(tokens)}"
        )

    def _serialize_header_only(self, header, data_type, tokens):
        """Serializes with header only."""
        return f"{self.cls_token}" f"{header}" f"{self.eos_token}"

    def _serialize_header_values_verbose_notype(self, header, data_type, tokens):
        """Serializes with simple format including header and tokens."""
        return (
            f"{self.cls_token}"
            f"Column: {header}{self.sep_token}"
            f"Values: {self.sep_token.join(tokens)}{self.sep_token}"
            f"{self.eos_token}"
        )

    def _serialize_header_values_columnvaluepair_notype(
        self, header, data_type, tokens
    ):

        tokens = [f"{header}:{token}" for token in tokens]
        return (
            f"{self.cls_token}"
            f"Column: {header}{self.sep_token}"
            f"Values: {self.sep_token.join(tokens)}{self.sep_token}"
            f"{self.eos_token}"
        )

    def _serialize_header_values_repeat_notype(self, header, data_type, tokens):
        """Serializes with repeated header for emphasis."""
        repeated_header = self.sep_token.join([header] * 5)
        return (
            f"{self.cls_token}"
            f"{repeated_header}{self.sep_token}"
            f"{data_type}{self.sep_token}"
            f"{self.sep_token.join(tokens)}"
        )

    def _serialize_header_values_default_notype(self, header, data_type, tokens):

        return (
            f"{self.cls_token}"
            f"{header}{self.sep_token}"
            f"{self.sep_token.join(tokens)}"
        )
