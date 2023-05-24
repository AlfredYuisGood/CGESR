# This file includes the demonstration of aggregation over edge type

import numpy as np

# Compute similarity scores between adjacent items
def compute_similarity(item_i, item_j):
    return np.dot(item_i, item_j)

# Normalize similarity scores for outgoing neighbors
def normalize_scores(scores):
    return scores / np.sum(scores)

# Aggregation step for item representation
def aggregate_items(item_i, item_neighbors, scores):
    aggregated_item = np.mean(item_neighbors, axis=0)
    weighted_aggregation = scores[:, np.newaxis] * np.concatenate((aggregated_item, item_i), axis=1)
    return np.maximum(0, weighted_aggregation)

# Aggregation step for user representation
def aggregate_user(user_neighbors, item_i, scores):
    aggregated_user = np.mean(user_neighbors, axis=0)
    weighted_aggregation = scores[:, np.newaxis] * np.concatenate((aggregated_user, item_i), axis=1)
    return np.maximum(0, weighted_aggregation)

# Generate session preference representation
def generate_session_preference(item_i, item_neighbors, user_neighbors, scores):
    item_aggregation = aggregate_items(item_i, item_neighbors, scores["out"])
    user_aggregation = aggregate_user(user_neighbors, item_i, scores["user"])
    input_aggregation = aggregate_items(item_i, item_neighbors, scores["in"])
    aggregated_representation = np.mean([item_aggregation, user_aggregation, input_aggregation], axis=0)
    return aggregated_representation

# Make recommendations
def recommend_items(session_preference, items):
    probabilities = softmax(np.dot(session_preference, items.T))
    return probabilities

# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Example usage
item_i = np.random.rand(10)  # Example attribute vector for item i
item_neighbors = np.random.rand(5, 10)  # Example attribute vectors for item neighbors
user_neighbors = np.random.rand(3, 10)  # Example attribute vectors for user neighbors
scores = {
    "out": normalize_scores(np.random.rand(5)),  # Example similarity scores for outgoing neighbors
    "user": normalize_scores(np.random.rand(3)),  # Example similarity scores for user neighbors
    "in": normalize_scores(np.random.rand(5))  # Example similarity scores for input item neighbors
}

session_preference = generate_session_preference(item_i, item_neighbors, user_neighbors, scores)
recommended_items = recommend_items(session_preference, item_neighbors)


import numpy as np

def evaluate(predictions, targets, k):
    # Sort predictions and get top-k recommended items for each session
    top_k_items = np.argsort(predictions, axis=1)[:, -k:]

    num_sessions = targets.shape[0]
    recall_sum = 0
    precision_sum = 0
    mrr_sum = 0

    for i in range(num_sessions):
        # Get the true positive items for the session
        true_positives = np.where(targets[i] == 1)[0]

        # Compute Recall@K
        recall = len(set(true_positives) & set(top_k_items[i])) / len(true_positives)
        recall_sum += recall

        # Compute Precision@K
        precision = len(set(true_positives) & set(top_k_items[i])) / k
        precision_sum += precision

        # Compute MRR@K
        mrr = 0
        for j, item in enumerate(top_k_items[i]):
            if item in true_positives:
                mrr = 1 / (j + 1)
                break
        mrr_sum += mrr

    # Calculate average metrics
    recall_at_k = recall_sum / num_sessions
    precision_at_k = precision_sum / num_sessions
    mrr_at_k = mrr_sum / num_sessions

    return recall_at_k, precision_at_k, mrr_at_k
