import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from fuzzywuzzy import fuzz

from .utils import get_samples, detect_column_type
from .embedding_utils import compute_cosine_similarity_simple
from .column_encoder import ColumnEncoder

import os

DEFAULT_MODELS = ["sentence-transformers/all-mpnet-base-v2"]


class EmbeddingMatcher:
    def __init__(self, params):
        self.params = params
        self.topk = params["topk"]
        self.embedding_threshold = params["embedding_threshold"]

        # Dynamically set device to GPU if available, else fallback to CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = params["embedding_model"]

        if self.model_name in DEFAULT_MODELS:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Load the model onto the selected device
            self.model = AutoModel.from_pretrained(
                self.model_name).to(self.device)
            print(f"Loaded ZeroShot Model on {self.device}")
        else:
            # Base model
            base_model = "sentence-transformers/all-mpnet-base-v2"
            self.model = SentenceTransformer(base_model)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)

            print(f"Loaded SentenceTransformer Model on {self.device}")

            # path to the trained model weights
            model_path = params["embedding_model"]
            if os.path.exists(model_path):
                print(f"Loading trained model from {model_path}")
                # Load state dict for the SentenceTransformer model
                state_dict = torch.load(
                    model_path, map_location=self.device, weights_only=True
                )
                # Assuming the state_dict contains the proper model weights and is compatible with SentenceTransformer
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.model.to(self.device)
            else:
                print(
                    f"Trained model not found at {
                        model_path}, loading default model."
                )

    def _get_embeddings(self, texts, batch_size=32):
        if self.model_name in DEFAULT_MODELS:
            return self._get_embeddings_zs(texts, batch_size)
        else:
            return self._get_embeddings_ft(texts, batch_size)

    def _get_embeddings_zs(self, texts, batch_size=32):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                # Move inputs to device
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1))
        return torch.cat(embeddings)

    def _get_embeddings_ft(self, texts, batch_size=32):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            with torch.no_grad():
                batch_embeddings = self.model.encode(
                    batch_texts, show_progress_bar=False, device=self.device
                )
            embeddings.append(torch.tensor(batch_embeddings))
        return torch.cat(embeddings)

    def get_embedding_similarity_candidates(self, source_df, target_df):
        encoder = ColumnEncoder(
            self.tokenizer,
            encoding_mode=self.params["encoding_mode"],
            sampling_mode=self.params["sampling_mode"],
            n_samples=self.params["sampling_size"],
        )

        input_col_repr_dict = {
            encoder.encode(source_df, col): col for col in source_df.columns
        }
        target_col_repr_dict = {
            encoder.encode(target_df, col): col for col in target_df.columns
        }

        cleaned_input_col_repr = list(input_col_repr_dict.keys())
        cleaned_target_col_repr = list(target_col_repr_dict.keys())

        embeddings_input = self._get_embeddings(cleaned_input_col_repr)
        embeddings_target = self._get_embeddings(cleaned_target_col_repr)

        top_k = min(self.topk, len(cleaned_target_col_repr))
        topk_similarity, topk_indices = compute_cosine_similarity_simple(
            embeddings_input, embeddings_target, top_k
        )

        candidates = {}

        for i, cleaned_input_col in enumerate(cleaned_input_col_repr):
            original_input_col = input_col_repr_dict[cleaned_input_col]

            for j in range(top_k):
                cleaned_target_col = cleaned_target_col_repr[topk_indices[i, j]]
                original_target_col = target_col_repr_dict[cleaned_target_col]
                similarity = topk_similarity[i, j].item()

                if similarity >= self.embedding_threshold:
                    candidates[(original_input_col,
                                original_target_col)] = similarity

        return candidates
