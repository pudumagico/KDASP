import os
import json
import random

def split_dataset(path, array, prefix=""):
    """Save a JSON array to a file with a given prefix (train/test) in the same directory as path."""
    folder = os.path.dirname(path)
    filename = os.path.join(folder, f"{prefix}_suite.json")
    with open(filename, 'w') as file:
        json.dump(array, file, indent=2)

# Path to your input file
original_path = 'dataset/GQA_count/confirmed_connected_examples.json'

with open(original_path) as f:
    questions = json.load(f)

random.shuffle(questions)
l = len(questions)
ltrain = int(l * 0.5)

train_suite = questions[:ltrain]
test_suite = questions[ltrain:]

print("Train size:", len(train_suite))
print("Test size :", len(test_suite))

# Save splits to the same directory as the original
split_dataset(original_path, train_suite, prefix="train")
split_dataset(original_path, test_suite, prefix="test")
