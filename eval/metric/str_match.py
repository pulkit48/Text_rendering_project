import Levenshtein as lev

def exact_match_accuracy(ground_truths, predictions):
    """Calculates the exact match accuracy."""
    correct = sum(gt == pred for gt, pred in zip(ground_truths, predictions))
    return correct / len(ground_truths) if ground_truths else 0

def exact_match_accuracy_case_insensitive(ground_truths, predictions):
    """Calculates exact match accuracy without case sensitivity."""
    correct = sum(gt.lower() == pred.lower() for gt, pred in zip(ground_truths, predictions))
    return correct / len(ground_truths) if ground_truths else 0

def normalized_edit_distance(ground_truths, predictions):
    distance = lev.distance(ground_truths, predictions)
    max_length = max(len(ground_truths), len(predictions))
    return distance / max_length if max_length > 0 else 0

def longest_common_subsequence_length(s1, s2):
    # Create a 2D table to store lengths of LCS
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:  # Characters match
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:  # Take the max from previous subsequences
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]

def normalized_longest_common_subsequence(original, compared):
    # Calculate LCS length
    lcs_length = longest_common_subsequence_length(original, compared)

    # Calculate similarity as a ratio of LCS length to the length of the original string
    similarity = lcs_length / len(original)
    return similarity
