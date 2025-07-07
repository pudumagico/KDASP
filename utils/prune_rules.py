import os
import re
import numpy as np
import json
import clingo
import random
import statistics

    # for dataset, examples in sampled_examples.items():
def apply_theory(theory, dataset):
    # print(f"Processing dataset: {dataset}")
    
    # for examples in dataset:
        # Run Clingo on the sampled examples
    # print(f"Running Clingo on {dataset} samples...")
    correct_count = 0
    
    for example in dataset:
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
            if derived_answers:
                if '_' in derived_answers[0]:
                    derived_answers[0] = derived_answers[0].replace('_', ' ')
                
            # print(derived_answers, example['answer'])
            if derived_answers == example['answer']:
                correct_count += 1

        # print(f"Dataset Correct Answers: {correct_count}/{len(dataset)}\n")
    return correct_count/len(dataset)

def prune_theory(root_dir):
    dataset_path = 'dataset/GQA/train_suite.json'
    with open(dataset_path, 'r') as dataset_file:
        dataset = json.load(dataset_file)
    dataset = random.sample(dataset, 1000)
    
    total_redundant_rules = []

    for subdir, _, files in os.walk(root_dir):
        if 'config_log.txt' in files:
            with open(os.path.join(subdir, 'theories/current10_examples10.txt'), 'r') as theory_file:
                theory = theory_file.read()

            # Load theory (implement your own parser as needed)
            baseline_accuracy = apply_theory(theory, dataset)
            # Split the text into lines
            theory_lines = [line for line in theory.splitlines() if line.strip() != ""]

            # Find the index where the marker line appears
            for i, line in enumerate(theory_lines):
                if '% Added rules to handle new instances' in line:
                    # Collect all lines after this one
                    new_rules = theory_lines[i + 1:]
                    break
            else:
                # If the marker wasn't found
                new_rules = []
            
            redundant_rules = []
            for new_rule in new_rules:
                incumbent_theory = "\n".join([line for line in theory_lines if line != new_rule])
                accuracy = apply_theory(incumbent_theory, dataset)
                if accuracy == baseline_accuracy:
                    redundant_rules.append(new_rule)

            total_redundant_rules.append(len(redundant_rules))
            pruned_theory = "\n".join([line for line in theory_lines if not line in redundant_rules])
            with open(os.path.join(subdir, 'theories/pruned_theory.txt'), "w") as f:
                json.dump(pruned_theory, f)
                
            pruning_stats = {
                'pruned_rules': redundant_rules,
                'n_pruned_rules': len(redundant_rules)
            }
            
            with open(os.path.join(subdir, 'pruning_stats.txt'), "w") as f:
                json.dump(pruning_stats, f)
            
            
            
    print(f'Total redundant rules: {total_redundant_rules}')
    print(f'Total redundant average: {sum(total_redundant_rules)/len(total_redundant_rules)}')
    print(f'Minimum: {min(total_redundant_rules)}')
    print(f'Maximum: {max(total_redundant_rules)}')

# Define datasets and predicates for deepseek experiments
datasets = [
    'GQA', 
    'CLEVR', 
    'CLEGR'
            ]
predicates = {
    'CLEGR': [
        'adjacent', 'adjacentTo', 'countNodesBetween', 'cycle', 'paths', 'shortestPath',
        'commonStation', 'exist', 'linesOnCount', 'linesOnNames', 'sameLine'
    ],
    'CLEVR': [
        'and', 'count', 'equal_integer', 'exist', 'filter_large', 'query_shape',
        'relate_left', 'same_color', 'unique'
    ],
    'GQA': [
        # 'choose_attr', 
        'exist', 'filter', 'negate', 'or', 'query', 'relate', 'select',
        'two_different', 'two_same', 'verify_rel'
    ]
}

# datasets = ['GQA_inbetween', 'GQA_connected', 'GQA_isolated', 'GQA_count']

# predicates = {
#     'GQA_inbetween': [
#         'inbetween'
#     ],
#     'GQA_connected': [
#         'connected'
#     ],
#     'GQA_isolated': [
#         'isolated'
#     ],
#     'GQA_count': [
#         'count_class'
#     ],
# }

models = ['gpt-4o', 'deepseek/deepseek-chat', 'mistral/mistral-large-latest', 'llama']
# models = []

logs_folder = './'

# Iterate through datasets and predicates
for model in models:
    print(f"Model: {model}")
    for dataset in datasets:
        print(f"Dataset: {dataset}")
        for predicate in predicates[dataset]:
            root_dir = f'./{logs_folder}/{dataset}/{model}/{predicate}'
            print(f"Predicate: {predicate}")
            prune_theory(root_dir)
            
            
