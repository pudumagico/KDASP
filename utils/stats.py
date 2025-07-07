import os
import re
import numpy as np
import json

def find_statistics(root_dir):
    averages = []
    syntax_mendings = []
    semantic_mendings = []
    examples_used = []
    config_files_count = 0

    for subdir, _, files in os.walk(root_dir):
        if 'config_log.txt' in files:
            config_files_count += 1
            with open(os.path.join(subdir, 'config_log.txt'), 'r') as file:
                lines = file.readlines()
                if len(lines) >= 3:
                    average_line = lines[-3]
                    syntax_line = lines[-2]
                    semantic_line = lines[-1]
                    
                    avg_match = re.search(r'Average: (\d+(\.\d+)?)', average_line)
                    syntax_match = re.search(r'Syntax Mendings: (\d+(\.\d+)?)', syntax_line)
                    semantic_match = re.search(r'Semantic Mendings: (\d+(\.\d+)?)', semantic_line)
                    
                    if avg_match:
                        averages.append(float(avg_match.group(1)))
                    if syntax_match:
                        syntax_mendings.append(float(syntax_match.group(1)))
                    if semantic_match:
                        semantic_mendings.append(float(semantic_match.group(1)))



        # Check for examples_used.txt and calculate the number of examples used
        txt_files = [f for f in os.listdir(subdir) if f.endswith('.txt')]
        count = len(txt_files)
        # print(len(data))
        examples_used.append(count)

        # print("Number of .txt files:", count)
        # examples_file = os.path.join(subdir, 'examples_used.txt')
        # if os.path.exists(examples_file):
        #     with open(examples_file, 'r') as file:
        #         # examples = file.readlines()
        #         # examples_used.append(len(examples))
        #         data = json.load(file)
        #         print(data)
        #         print(len(data))
        #         examples_used.append(len(data))
    stats = {}
    
    if averages:
        stats['Average'] = {
            'mean': round(np.mean(averages), 2),
            'std': round(np.std(averages), 2),
            'max': round(np.max(averages), 2),
            'min': round(np.min(averages), 2)
        }
    
    # if syntax_mendings:
    #     stats['Syntax Mendings'] = {
    #         'mean': round(np.mean(syntax_mendings), 2),
    #         'std': round(np.std(syntax_mendings), 2),
    #         'max': round(np.max(syntax_mendings), 2),
    #         'min': round(np.min(syntax_mendings), 2)
    #     }
    
    # if semantic_mendings:
    #     stats['Semantic Mendings'] = {
    #         'mean': round(np.mean(semantic_mendings), 2),
    #         'std': round(np.std(semantic_mendings), 2),
    #         'max': round(np.max(semantic_mendings), 2),
    #         'min': round(np.min(semantic_mendings), 2)
    #     }
    
    if examples_used:
        stats['Examples Used'] = {
            'mean': round(np.mean(examples_used), 2),
            'std': round(np.std(examples_used), 2),
            'max': round(np.max(examples_used), 2),
            'min': round(np.min(examples_used), 2)
        }
    
    return stats, config_files_count

# Define datasets and predicates for deepseek experiments
datasets = ['CLEGR']
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
        'choose_attr', 'exist', 'filter', 'negate', 'or', 'query', 'relate', 'select',
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

# models = [ 'gpt-4o']
# models = [ 'deepseek/deepseek-chat']
# models = ['mistral/mistral-large-latest']
models = ['llama']

logs_folder = 'logs/no_multiple'

# Iterate through datasets and predicates
for model in models:
    print(f"Model: {model}")

    for dataset in datasets:
        print(f"Dataset: {dataset}")
        for predicate in predicates[dataset]:
            root_dir = f'./{logs_folder}/{dataset}/{model}/{predicate}'
            statistics, config_files_opened = find_statistics(root_dir)
            print(f"Predicate: {predicate}")
            print(f"Number of config files opened: {config_files_opened}")

            if statistics:
                for key, value in statistics.items():
                    print(f"{key} statistics:")
                    mean = value['mean']
                    std_dev = value['std']
                    maximum = value['max']
                    minimum = value['min']
                    print(f"${mean:.2f} \pm {std_dev:.2f}$ & $({minimum:.2f}, {maximum:.2f})$")
                
                # Print the average number of examples used
                if 'Examples Used' in statistics:
                    examples_mean = statistics['Examples Used']['mean']
                    print(f"Average examples used: {examples_mean:.2f}")
            else:
                print("No valid entries found in config_log.txt files.")
            
            print()