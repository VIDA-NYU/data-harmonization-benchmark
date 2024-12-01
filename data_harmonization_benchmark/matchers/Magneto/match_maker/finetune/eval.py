import torch


def evaluate_top_k(model, validation_loader, device, k=1):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in validation_loader:
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            embeddings = model.encode(texts, convert_to_tensor=True, device=device)
            all_embeddings.append(embeddings)
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    normalized_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
    similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())

    correct_matches = 0
    total_matches = 0

    for i in range(len(all_labels)):
        similarity = similarity_matrix[i]
        similarity[i] = -1  # Set self-similarity to negative value to exclude it
        top_k_indices = torch.topk(similarity, k).indices
        correct_matches += 1 if all_labels[i] in all_labels[top_k_indices] else 0
        total_matches += 1

    accuracy = correct_matches / total_matches if total_matches > 0 else 0
    return accuracy


def evaluate_recall_at_ground_truth(model, validation_loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in validation_loader:
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            embeddings = model.encode(texts, convert_to_tensor=True, device=device)
            all_embeddings.append(embeddings)
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    normalized_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
    similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())

    correct_matches = 0
    total_matches = 0

    # Evaluate recall at ground truth
    for i in range(len(all_labels)):
        true_label = all_labels[i]
        k = torch.sum(all_labels == true_label).item() - 1

        similarity = similarity_matrix[i]
        similarity[i] = -1
        top_k_indices = torch.topk(similarity, k).indices

        correct_matches += torch.sum(all_labels[top_k_indices] == true_label).item()
        total_matches += k

    recall_at_ground_truth = correct_matches / total_matches if total_matches > 0 else 0
    return recall_at_ground_truth


def evaluate_metrics(model, validation_loader, device, fixed_k=1):
    model.eval()
    all_embeddings = []
    all_labels = []

    # Collect embeddings and labels
    with torch.no_grad():
        for texts, labels in validation_loader:
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            embeddings = model.encode(texts, convert_to_tensor=True, device=device)
            all_embeddings.append(embeddings)
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    normalized_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
    similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())

    correct_matches_fixed_k = 0
    total_matches_fixed_k = 0
    correct_matches_recall = 0
    total_matches_recall = 0
    mrr_total = 0

    for i in range(len(all_labels)):
        similarity = similarity_matrix[i]
        similarity[i] = -1  # Set self-similarity to a negative value to exclude it

        # Fixed k evaluation
        top_k_indices_fixed = torch.topk(similarity, fixed_k).indices
        correct_matches_fixed_k += (
            1 if all_labels[i] in all_labels[top_k_indices_fixed] else 0
        )
        total_matches_fixed_k += 1

        # Recall at ground truth evaluation
        true_label = all_labels[i]
        ground_truth_k = torch.sum(all_labels == true_label).item() - 1
        top_k_indices_recall = torch.topk(similarity, ground_truth_k).indices
        correct_matches_recall += torch.sum(
            all_labels[top_k_indices_recall] == true_label
        ).item()
        total_matches_recall += ground_truth_k

        # Mean Reciprocal Rank (MRR) evaluation
        correct_indices = (all_labels == true_label).nonzero(as_tuple=False).squeeze()
        ranks = torch.argsort(similarity, descending=True)
        reciprocal_rank = 0
        for idx in correct_indices:
            if idx == i:
                continue
            rank = (ranks == idx).nonzero(as_tuple=True)[0].item() + 1
            reciprocal_rank = 1 / rank
            break
        mrr_total += reciprocal_rank

    accuracy = (
        correct_matches_fixed_k / total_matches_fixed_k
        if total_matches_fixed_k > 0
        else 0
    )
    recall_at_ground_truth = (
        correct_matches_recall / total_matches_recall if total_matches_recall > 0 else 0
    )
    mean_reciprocal_rank = mrr_total / len(all_labels) if len(all_labels) > 0 else 0

    return accuracy, recall_at_ground_truth, mean_reciprocal_rank
