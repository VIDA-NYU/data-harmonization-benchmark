import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses

from tqdm import tqdm

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

project_path = os.getcwd()
sys.path.append(os.path.join(project_path))

from train_utils import SimCLRLoss, BalancedBatchSampler, sentence_transformer_map
from eval import evaluate_metrics
from dataset import CustomDataset


def train_model(
    model,
    num_classes,
    data_loader,
    optimizer,
    model_path,
    loss_type="triplet",
    margin=1,
    epochs=100,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    best_accuracy = 0
    if loss_type == "triplet":
        loss_fn = losses.BatchHardTripletLoss(
            model=model,
            margin=margin,
            distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance,
        )
    elif loss_type == "simclr":
        loss_fn = SimCLRLoss(
            model=model,
            temperature=0.5,
        )

    for epoch in range(epochs):
        total_loss = 0
        # for batch in data_loader:
        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):

            texts, labels = batch
            labels = torch.tensor(labels, dtype=torch.float, device=device)

            optimizer.zero_grad()

            sentence_features = model.tokenize(texts)
            sentence_features = [
                {k: v.to(device) for k, v in sentence_features.items()}
            ]
            loss = loss_fn(sentence_features, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        # validation_accuracy = evaluate_recall_at_ground_truth(model, data_loader, device)
        # validation_accuracy = evaluate_top_k(model, data_loader, device, k=1)
        accuracy, recall_at_ground_truth, mrr = evaluate_metrics(
            model, data_loader, device, fixed_k=1
        )
        # print(f"Epoch {epoch+1}, Loss: {avg_loss}, Val Accuracy: {validation_accuracy}")
        print(
            f"Epoch {epoch+1}, Loss: {avg_loss}, Val Accuracy: {accuracy}, Recall at Ground Truth: {recall_at_ground_truth}, MRR: {mrr}"
        )
        # validation_accuracy = (accuracy + recall_at_ground_truth) / 2
        validation_accuracy = (recall_at_ground_truth + mrr) / 2
        # Save best model
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model with accuracy: {best_accuracy}")

    print("Training complete with best accuracy:", best_accuracy)


def main():
    parser = argparse.ArgumentParser(
        description="Match columns between source and target tables using pretrained models."
    )
    parser.add_argument(
        "--dataset",
        default="gdc",
        help="Name of the dataset for model customization",
    )
    parser.add_argument(
        "--model_type",
        default="mpnet",
        help="Type of model (roberta, distilbert, mpnet)",
    )
    parser.add_argument(
        "--serialization",
        default="header_values_repeat",
        help=(
            "Column serialization method. Choose from the following options: "
            "- header_values_default,"
            "- header_values_prefix,"
            "- header_values_repeat,"
            "- header_only,"
            "- header_values_simple,"
            "- header_values_verbose,"
        ),
    )
    parser.add_argument(
        "--augmentation",
        default="exact_semantic",
        help="Augmentation type (exact, semantic, exact_semantic)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--loss_type",
        default="triplet",
        choices=["triplet", "simclr"],
        help="Type of loss function to use (triplet or simclr)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Margin value for triplet loss",
    )

    args = parser.parse_args()

    file_path = f"data/synthetic/{args.dataset}_synthetic_matches.json"
    with open(file_path, "r") as file:
        data = json.load(file)

    dataset = CustomDataset(
        data,
        model_type=args.model_type,
        serialization="header_values_repeat",
        augmentation="exact_semantic",
    )

    labels = dataset.labels
    n_classes = len(np.unique(labels))
    balanced_sampler = BalancedBatchSampler(
        labels, batch_size=args.batch_size, n_samples_per_class=2
    )
    data_loader = DataLoader(
        dataset,
        batch_sampler=balanced_sampler,
        collate_fn=lambda x: ([d[0] for d in x], [d[1] for d in x]),
    )

    model = SentenceTransformer(sentence_transformer_map[args.model_type])
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.5)

    if not os.path.exists("models"):
        os.makedirs("models")

    loss_type = args.loss_type
    model_path = (
        f"models/{args.model_type}-{args.dataset}-{args.serialization}-{args.augmentation}-{str(args.batch_size)}-{str(args.margin)}.pth"
        if loss_type == "triplet"
        else f"models/{args.model_type}-{args.dataset}-{args.serialization}-{args.augmentation}-simclr.pth"
    )
    print(f"Training model using {args.model_type} on {args.dataset} dataset")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    train_model(
        model,
        n_classes,
        data_loader,
        optimizer,
        model_path,
        loss_type=loss_type,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
