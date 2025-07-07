import json
import random
import os
import clingo

def sample_datasets(dataset_paths, sample_size=500):
    """
    Samples examples from the datasets.

    Args:
        dataset_paths (dict): A dictionary where keys are dataset names and values are paths to their training suites.
        sample_size (int): Number of examples to sample from each dataset.

    Returns:
        dict: A dictionary containing the sampled examples for each dataset.
    """
    all_sampled_examples = {}

    for dataset, path in dataset_paths.items():
        print(f"Sampling dataset: {dataset}")
        with open(path, 'r') as file:
            training_suite = json.load(file)
        sampled_examples = random.sample(training_suite, min(sample_size, len(training_suite)))
        all_sampled_examples[dataset] = sampled_examples

    return all_sampled_examples

def process_samples(sampled_examples, theory_paths):
    """
    Processes the sampled examples by running Clingo on them.

    Args:
        sampled_examples (dict): A dictionary containing the sampled examples for each dataset.
        theory_paths (dict): A dictionary where keys are dataset names and values are paths to their ASP theory files.
    """
    for dataset, examples in sampled_examples.items():
        print(f"Processing dataset: {dataset}")
        
        # Load the corresponding theory file
        theory_path = theory_paths.get(dataset)
        if not theory_path or not os.path.exists(theory_path):
            print(f"Error: Theory file for dataset {dataset} not found at {theory_path}")
            continue
        
        with open(theory_path, 'r') as theory_file:
            theory = theory_file.read()
        
        # Run Clingo on the sampled examples
        print(f"Running Clingo on {dataset} samples...")
        correct_count = 0
        
        for example in examples:
            ctl = clingo.Control(["--warn=none"])
            ctl.add("base", [], theory)
            ctl.add("base", [], example["scene"])
            ctl.add("base", [], example["question"])
            ctl.add("base", [], "#show ans/1.")
            ctl.ground([("base", [])])

            # Solve and check if the answer matches
            with ctl.solve(yield_=True) as handle:
                derived_answers = [str(atom.arguments[0]) for model in handle for atom in model.symbols(shown=True)]
                
                if not type(example['answer']) == list:
                    example['answer'] = sorted([str(example['answer']).lower()])
                else:
                    example['answer'] = sorted([x.lower() for x in example['answer']])
                
                if derived_answers == ['to_the_right_of']:
                    derived_answers = ['right']
                if derived_answers == ['to_the_left_of']:
                    derived_answers = ['left']
                if '_' in derived_answers[0]:
                    derived_answers[0] = derived_answers[0].replace('_', ' ')
                    
                # print(derived_answers, example['answer'])
                if derived_answers == example['answer']:
                    correct_count += 1

        print(f"Dataset: {dataset} - Correct Answers: {correct_count}/{len(examples)}\n")

# Define paths to the training suites for each dataset
dataset_paths = {
    "GQA": "./dataset/GQA/train_suite.json"
}

# Define paths to the ASP theory files for each dataset
theory_paths = {
    "GQA": "./preprompt/theory/GQA/perfect_theory.lp"
}

# Sample datasets
sampled_examples = sample_datasets(dataset_paths, sample_size=500)
with open('data.json', 'w') as json_file:
    json.dump(sampled_examples, json_file, indent=4)
# Process the sampled examples
process_samples(sampled_examples, theory_paths)

