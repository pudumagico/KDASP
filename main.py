import argparse
import random
import os

from prompt.prompt_builder import PromptBuilder
from utils import *
from tqdm import tqdm
import datetime

seed = os.urandom(16)
random.seed(seed)

def main(args):

    syntax_correct_count = 0
    solution_correct_count = 0

    current_example = 1
    max_retries = args.max_retries
    max_mend_retries = args.mend_retries
    learning_examples = args.learning_examples
    batch_examples = args.batch_examples
    model = args.model
    strat = args.strategy
    k = args.sample_sz
    representation = args.representation
    remove_random_percentage = args.remove_random
    remove_predicate = args.remove_predicate
    state_mending = args.state_mending
    batch_theory = args.batch_theory
    dataset = args.dataset
    chosen_theory = args.chosen_theory
    multiple = not args.no_multiple
    cot = not args.no_cot
    mending = not args.no_mending
    clegr_flag = True if dataset == 'CLEGR' else False

    regression_examples = []

    # Strategy selection based on ablation flags
    strategies = []
    if multiple:
        strategies.append('multiple')
    if cot:
        strategies.extend(['simple_cot', 'complex_cot'])

    pb_multiple = PromptBuilder(dataset, True, False)
    pb_cot_simple = PromptBuilder(dataset, False, 'simple')
    pb_cot_complex = PromptBuilder(dataset, False, 'complex')
    examples = pb_multiple.generate_examples(strat, k, representation, remove_predicate)
    examples = sorted(examples, key=lambda x: len(x['question']))

    print("Learning examples:", len(examples))
    syntax_mending_success = 0
    semantic_mending_success = 0

    if chosen_theory:
        incumbent_theory = ''.join(open(chosen_theory).readlines())
    else:
        incumbent_theory = ''.join(
            open(f'./preprompt/theory/{dataset}/perfect_theory.lp').readlines())

    if batch_theory:
        if batch_theory == 'light':
            incumbent_theory = ''.join(
                open(f'./preprompt/theory/{dataset}/light.lp').readlines())
        if batch_theory == 'medium':
            incumbent_theory = ''.join(
                open(f'./preprompt/theory/{dataset}/medium.lp').readlines())
        if batch_theory == 'heavy':
            incumbent_theory = ''.join(
                open(f'./preprompt/theory/{dataset}/heavy.lp').readlines())

    if remove_random_percentage:
        incumbent_theory = remove_random_lines(
            incumbent_theory, remove_random_percentage)

    if remove_predicate:
        incumbent_theory = remove_lines_with_predicates(
            incumbent_theory, predicates=[remove_predicate])

    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_incumbent_theory(date_str,
                         incumbent_theory, model, max_retries, learning_examples, current_example, batch_examples, remove_predicate, dataset, cot, mending, multiple)

    if not learning_examples:
        learning_examples = len(examples)

    preprompt_multiple = pb_multiple.generate_preprompt([incumbent_theory])
    preprompt_simple = pb_cot_simple.generate_preprompt([incumbent_theory])
    preprompt_complex = pb_cot_complex.generate_preprompt([incumbent_theory])
    preprompts = []
    if multiple:
        preprompts.append(preprompt_multiple)
    if cot:
        preprompts.extend([preprompt_simple, preprompt_complex])

    while current_example < learning_examples:
        print(current_example, learning_examples)

        if batch_examples:
            prompt = ''
            incumbent_examples = examples[current_example:min(
                current_example+batch_examples, len(examples))]
            for example in incumbent_examples:
                prompt += example['question'] + '\n#########\n'
        else:
            incumbent_examples = [examples[current_example]]
            prompt = '%Encoded Question\n' +\
                examples[current_example]['question'] + '\n' +\
                '%Encoded Scene\n' +\
                examples[current_example]['scene'] + '\n' +\
                '%Expected Answer\n' +\
                'ans({}).'.format(examples[current_example]['answer'])

        semantic_check, semantic_error = run_asp_code(
            incumbent_theory, incumbent_examples, clegr_flag)

        if semantic_check:
            if batch_examples:
                current_example += batch_examples
            else:
                current_example += 1
            continue

        retries = 0
        done = False

        while retries < max_retries:
            for i, strategy in enumerate(strategies):

                if not done:
                    print('Trying strategy:', strategy)
                    if strategy == 'multiple':
                        response_multiple = ask_LLM(prompt, preprompt_multiple, model)
                        print(response_multiple)
                        responses = response_multiple.split("***")
                        print(responses, len(responses))
                    elif strategy == 'simple_cot':
                        response_cot_simple = ask_LLM(prompt, preprompt_simple, model)
                        responses = [response_cot_simple.split("***")[0]]
                    elif strategy == 'complex_cot':
                        response_cot_complex = ask_LLM(prompt, preprompt_complex, model)
                        responses = [response_cot_complex.split("***")[0]]
                    else:
                        continue

                    for response in responses:
                        print('Trying rules:')
                        response = response.replace('\\_', '_')
                        print(response)
                        extended_theory = incumbent_theory + '\n' + response
                        syntax_check, syntax_error = check_asp_syntax(extended_theory)
                        print('syntax_check', syntax_check)
                        if syntax_check:
                            semantic_check, semantic_error = run_asp_code(
                                extended_theory, incumbent_examples, clegr_flag)
                            print('semantic_check', semantic_check)
                            if semantic_check:
                                if args.regressive_test:
                                    for example in incumbent_examples:
                                        regression_examples.append(example)
                                    semantic_check, semantic_errors = run_asp_code(
                                        extended_theory, regression_examples, clegr_flag)
                                    if semantic_check:
                                        print('Regressive Test Success')
                                        incumbent_theory = extended_theory
                                        syntax_correct_count += batch_examples
                                        solution_correct_count += batch_examples
                                        done = True
                                        break

                            elif mending and max_mend_retries:
                                mend_retries = 0
                                while mend_retries < max_mend_retries:

                                    if state_mending:
                                        state_atoms = run_asp_code_with_states(extended_theory, incumbent_examples, clegr_flag)
                                        mended_rule = mend_semantics_with_states(
                                            response, semantic_error, examples[current_example]['answer'], incumbent_theory, preprompts[i], model, state_atoms)
                                    else:
                                        mended_rule = mend_semantics(
                                            response, semantic_error, examples[current_example]['answer'], incumbent_theory, preprompts[i], model)

                                    mended_rule = mended_rule.replace('\\_', '_')

                                    extended_theory = incumbent_theory + '\n' + mended_rule
                                    syntax_check, error = check_asp_syntax(extended_theory)
                                    if syntax_check:
                                        semantic_check, error = run_asp_code(
                                            extended_theory, incumbent_examples, clegr_flag)
                                        if semantic_check:
                                            semantic_mending_success+=1
                                            if args.regressive_test:
                                                for example in incumbent_examples:
                                                    regression_examples.append(example)
                                                semantic_check, semantic_errors = run_asp_code(
                                                    extended_theory, regression_examples, clegr_flag)
                                                if semantic_check:
                                                    print('Regressive Test Success')
                                                    incumbent_theory = extended_theory
                                                    syntax_correct_count += batch_examples
                                                    solution_correct_count += batch_examples
                                                    done = True
                                                    break

                                    mend_retries += 1

                        elif mending and max_mend_retries:
                            mend_retries = 0
                            while mend_retries < max_mend_retries:
                                mended_rule = mend_syntax(
                                    response, syntax_error, preprompts[i], model)
                                mended_rule.replace('\\_', '_')
                                extended_theory = incumbent_theory + '\n' + mended_rule
                                syntax_check, error = check_asp_syntax(extended_theory)
                                if syntax_check:
                                    syntax_mending_success+=1
                                    semantic_check, error = run_asp_code(
                                        extended_theory, incumbent_examples, clegr_flag)
                                    if semantic_check:

                                        if args.regressive_test:
                                            for example in incumbent_examples:
                                                regression_examples.append(example)
                                            semantic_check, semantic_errors = run_asp_code(
                                                extended_theory, regression_examples, clegr_flag)
                                            if semantic_check:
                                                print('Regressive Test Sucess')
                                                incumbent_theory = extended_theory
                                                done = True
                                                break

                                        incumbent_theory = extended_theory
                                        syntax_correct_count += batch_examples
                                        solution_correct_count += batch_examples
                                        done = True
                                        break
                                mend_retries += 1

            retries += 1

        if batch_examples:
            current_example += batch_examples
        else:
            current_example += 1

        log_incumbent_theory(date_str,
                             incumbent_theory, model, max_retries, learning_examples, current_example, batch_examples, remove_predicate, dataset, cot, multiple)

    log_incumbent_theory(date_str,
                         incumbent_theory, model, max_retries, learning_examples, current_example, batch_examples, remove_predicate, dataset, cot, mending, multiple)

    final_correct_train = 0
    current_example = 0

    print('Testing Theory on Learning Examples')
    while current_example < learning_examples:
        semantic_check, semantic_error = run_asp_code(
            incumbent_theory, [examples[current_example]], clegr_flag)
        if semantic_check:
            final_correct_train += 1
        current_example += 1

    print(f"Final Training Correct Solutions: {final_correct_train}")
    print(f"Total Training Examples: {learning_examples}")

    test_suite = open(f'./dataset/{dataset}/test_suite.json')
    test_suite = json.load(test_suite)
    learning_examples = len(test_suite)
    final_correct_test = 0

    print('Validating Theory')

    for i in tqdm(range(learning_examples)):
        semantic_check, semantic_error = run_asp_code(
            incumbent_theory, [test_suite[i]], clegr_flag)
        if semantic_check:
            final_correct_test += 1

    print(f"Final Correct Solutions: {final_correct_test}")
    print(f"Total Solutions: {learning_examples}")
    print(f"Percentage: {final_correct_test/learning_examples*100}")
    print(f"Syntax Mendings: {syntax_mending_success}")
    print(f"Semantic Mendings: {semantic_mending_success}")


    if cot:
        base_log_dir = f"./logs/no_cot/{dataset}/{model}/{remove_predicate}"
    elif mending:
        base_log_dir = f"./logs/no_mending/{dataset}/{model}/{remove_predicate}"
    elif multiple:
        base_log_dir = f"./logs/no_multiple/{dataset}/{model}/{remove_predicate}"
    else:
        base_log_dir = f"./logs/{dataset}/{model}/{remove_predicate}"      

    write_to_log(f'{base_log_dir}/{date_str}/config_log.txt', seed, args, f"Final Training Correct Solutions: {final_correct_train}", f"Final Correct Solutions: {final_correct_test}", f"Average: {round(final_correct_test/learning_examples*100,2)}", f"Syntax Mendings: {syntax_mending_success}", f"Semantic Mendings: {semantic_mending_success}")
    save_examples(f'{base_log_dir}/{date_str}/examples_used.txt', examples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ASP Encoding and LLM Interaction")
    parser.add_argument("--max_retries", type=int, default=1,
                        help="Maximum number of retries")
    parser.add_argument("--mend_retries", type=int, default=1,
                        help="Maximum number of mending retries")
    parser.add_argument("--learning_examples", type=int,
                        default=0, help="Number of learning examples")
    parser.add_argument("--batch_examples", type=int,
                        default=0, help="Number of batch examples")
    parser.add_argument("--model", type=str,
                        default="gpt-4-1106-preview", help="LLM model to be used")
    parser.add_argument("--strategy", type=str, default="len",
                        help="Strategy used to sample examples, len for length or pred for predicates.")
    parser.add_argument("--sample_sz", type=int, default=10,
                        help="Sample size for the strategy selected")
    parser.add_argument("--regressive_test", default=True, type=bool,
                        help="Use regressive testing of previous examples")
    parser.add_argument("--representation", type=str, default="flat",
                        help="Representation to be used")
    parser.add_argument("--remove_random", type=int, default=0,
                        help="Remove a percentage of random lines from the perfect theory to use as initial theory.")
    parser.add_argument("--remove_predicate", type=str, default="color",
                        help="Remove any rule where the predicated selected appears in the perfect theory and use this as initial theory.")
    parser.add_argument("--state_mending", type=bool, default=False,
                        help="Use semantic mending with states of the program.")
    parser.add_argument("--batch_theory", type=str, default="",
                        help="Which batch theory to use.")
    parser.add_argument("--dataset", type=str, default="GQA",
                        help="Which dataset to use, GQA, CLEVR or GLEGR.")   
    parser.add_argument("--chosen_theory", type=str, default="",
                        help="Path to a handchosen theory to fill.")   
    # Ablation flags
    parser.add_argument("--no_multiple", default="store_true",
                        help="Deactivate strategy to generate multiple sets of rules.")
    parser.add_argument("--no_cot", action="store_true",
                        help="Deactivate chain of thought to explanations using the predicates.")
    parser.add_argument("--no_mending", action="store_true",
                        help="Deactivate syntax and semantic mending processes.")

    args = parser.parse_args()
    main(args)
