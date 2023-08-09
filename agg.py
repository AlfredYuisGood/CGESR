# This file includes the demonstration of aggregation over edge type
import numpy as np
import argparse

def compute_similarity(item_i, item_j):
    return np.dot(item_i, item_j)

def normalize_scores(scores):
    return scores / np.sum(scores)

def aggregate_items(item_i, item_neighbors, scores):
    aggregated_item = np.mean(item_neighbors, axis=0)
    weighted_aggregation = scores[:, np.newaxis] * np.concatenate((aggregated_item, item_i), axis=1)
    return np.maximum(0, weighted_aggregation)

def aggregate_user(user_neighbors, item_i, scores):
    aggregated_user = np.mean(user_neighbors, axis=0)
    weighted_aggregation = scores[:, np.newaxis] * np.concatenate((aggregated_user, item_i), axis=1)
    return np.maximum(0, weighted_aggregation)

def generate_session_preference(item_i, item_neighbors, user_neighbors, scores):
    item_aggregation = aggregate_items(item_i, item_neighbors, scores["out"])
    user_aggregation = aggregate_user(user_neighbors, item_i, scores["user"])
    input_aggregation = aggregate_items(item_i, item_neighbors, scores["in"])
    aggregated_representation = np.mean([item_aggregation, user_aggregation, input_aggregation], axis=0)
    return aggregated_representation

def recommend_items(session_preference, items):
    probabilities = softmax(np.dot(session_preference, items.T))
    return probabilities

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def evaluate(predictions, targets, k):
    top_k_items = np.argsort(predictions, axis=1)[:, -k:]

    num_sessions = targets.shape[0]
    recall_sum = 0
    precision_sum = 0
    mrr_sum = 0

    for i in range(num_sessions):
        true_positives = np.where(targets[i] == 1)[0]

        recall = len(set(true_positives) & set(top_k_items[i])) / len(true_positives)
        recall_sum += recall

        precision = len(set(true_positives) & set(top_k_items[i])) / k
        precision_sum += precision

        mrr = 0
        for j, item in enumerate(top_k_items[i]):
            if item in true_positives:
                mrr = 1 / (j + 1)
                break
        mrr_sum += mrr

    recall_at_k = recall_sum / num_sessions
    precision_at_k = precision_sum / num_sessions
    mrr_at_k = mrr_sum / num_sessions

    return recall_at_k, precision_at_k, mrr_at_k

def main():
    parser = argparse.ArgumentParser(description="Evaluate recommendation performance.")
    parser.add_argument("--k", type=int, default=10, help="Top-k for recall, precision, and MRR")
    args = parser.parse_args()

    item_i = np.random.rand(10)
    item_neighbors = np.random.rand(5, 10)
    user_neighbors = np.random.rand(3, 10)
    scores = {
        "out": normalize_scores(np.random.rand(5)),
        "user": normalize_scores(np.random.rand(3)),
        "in": normalize_scores(np.random.rand(5))
    }

    session_preference = generate_session_preference(item_i, item_neighbors, user_neighbors, scores)
    recommended_items = recommend_items(session_preference, item_neighbors)

    targets = np.random.randint(2, size=recommended_items.shape)
    recall, precision, mrr = evaluate(recommended_items, targets, args.k)
    
    print(f"Recall@{args.k}: {recall:.4f}")
    print(f"Precision@{args.k}: {precision:.4f}")
    print(f"MRR@{args.k}: {mrr:.4f}")

if __name__ == "__main__":
    main()
