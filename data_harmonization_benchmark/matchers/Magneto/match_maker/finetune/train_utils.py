import numpy as np
import torch
from torch import nn
from torch.utils.data import Sampler
from sentence_transformers import SentenceTransformer

sentence_transformer_map = {
    "roberta": "sentence-transformers/all-roberta-large-v1",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, n_samples_per_class=2):
        self.labels = np.array(labels)
        self.labels_unique = np.unique(self.labels)
        self.label_to_indices = {
            label: np.where(self.labels == label)[0] for label in self.labels_unique
        }
        self.n_samples_per_class = n_samples_per_class
        self.n_classes_per_batch = batch_size // n_samples_per_class
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(len(self)):
            batch_indices = []
            classes = np.random.choice(
                self.labels_unique, self.n_classes_per_batch, replace=False
            )
            for label in classes:
                indices = np.random.choice(
                    self.label_to_indices[label],
                    self.n_samples_per_class,
                    replace=False,
                )
                batch_indices.extend(indices)
            # np.random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return len(self.labels) // self.batch_size


class SimCLRLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, temperature=0.5):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.sentence_embedder = model

    def forward(self, sentence_features, labels):
        embeddings = self.sentence_embedder(sentence_features[0])["sentence_embedding"]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        cosine_sim = torch.matmul(embeddings, embeddings.T)

        labels = labels.unsqueeze(0)
        mask = torch.eq(labels, labels.T).float()

        # Mask out the self-contrast cases (diagonal elements) without inplace operation
        diagonal_mask = torch.eye(mask.size(0), device=embeddings.device).bool()
        mask = mask * (~diagonal_mask)

        # Compute the contrastive loss
        exp_sim = torch.exp(cosine_sim / self.temperature)
        sum_exp_sim = exp_sim.sum(1, keepdim=True) - exp_sim.diag().unsqueeze(
            1
        )  # avoid self-contrast

        positive_sim = exp_sim * mask
        sum_positive_sim = positive_sim.sum(1, keepdim=True)

        log_prob = torch.log(sum_positive_sim / sum_exp_sim)
        loss = -log_prob.mean()  # Mean over the batch
        return loss
