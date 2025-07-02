import pickle

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def calculate_win_rate(sft_dict, dpo_dict):
    common_keys = set(sft_dict.keys()) & set(dpo_dict.keys())
    
    if not common_keys:
        raise ValueError("No common image keys found in both dictionaries.")
    
    wins = 0
    total = 0

    for key in common_keys:
        sft_score = sft_dict[key]
        dpo_score = dpo_dict[key]
        
        if dpo_score > sft_score:
            wins += 1
        total += 1

    win_rate = wins / total if total > 0 else 0
    return win_rate, wins, total

# Replace with your file paths
sft_path = 'sft_scores.pkl'
dpo_path = 'dpo_scores.pkl'

sft_scores = load_pickle(sft_path)
dpo_scores = load_pickle(dpo_path)

win_rate, wins, total = calculate_win_rate(sft_scores, dpo_scores)
print(f"DPO win rate: {win_rate:.2%} ({wins}/{total})")
