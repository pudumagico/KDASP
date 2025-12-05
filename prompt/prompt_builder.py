from typing import Iterator, Tuple
import json
import os
import random
import re

# from perfect_information_encoding import encode_sample

actions = {
    "scene": "scene({T},{T1}).",
    "unique": "unique({T},{T1}).",
    "relate": "relate_{val}({T},{T1}).",
    "count": "count({T},{T1}).",
    "exist": "exist({T},{T1}).",
    "filter_size": "filter_{val}({T},{T1}).",
    "filter_color": "filter_{val}({T},{T1}).",
    "filter_material": "filter_{val}({T},{T1}).",
    "filter_shape": "filter_{val}({T},{T1}).",
    "query_size": "query_size({T},{T1}).",
    "query_color": "query_color({T},{T1}).",
    "query_material": "query_material({T},{T1}).",
    "query_shape": "query_shape({T},{T1}).",
    "same_size": "same_size({T},{T1}).",
    "same_color": "same_color({T},{T1}).",
    "same_material": "same_material({T},{T1}).",
    "same_shape": "same_shape({T},{T1}).",
    "equal_integer": "equal_integer({T},{T1},{T2}).",
    "less_than": "less_than({T},{T1},{T2}).",
    "greater_than": "greater_than({T},{T1},{T2}).",
    "equal_size": "equal_size({T},{T1},{T2}).",
    "equal_color": "equal_color({T},{T1},{T2}).",
    "equal_material": "equal_material({T},{T1},{T2}).",
    "equal_shape": "equal_shape({T},{T1},{T2}).",
    "union": "or({T},{T1},{T2}).",
    "intersect": "and({T},{T1},{T2})."
}

func_type = {
    "unary": ["scene", "unique", "count", "exist", "query_size", "query_color", "query_material",
              "query_shape", "same_size", "same_color", "same_material", "same_shape"],
    "binary_val": ["relate", "filter_size", "filter_color", "filter_material", "filter_shape"],
    "binary_in": ["equal_integer", "less_than", "greater_than", "equal_size", "equal_color", "equal_shape",
                  "equal_material", "union", "intersect"]
}

def find_predicates(text):
    pattern = r'(\w+)\('
    predicates = re.findall(pattern, text)
    return set(predicates)

def encode_CLEVR_scene(scene):
    encoded_scene = []
    obj_index = 0
    for obj in scene['objects']:
        posx = obj['pixel_coords'][0]
        posy = obj['pixel_coords'][1]
        material = obj['material']
        color = obj['color']
        shape = obj['shape']
        size = obj['size']    
        encoded_obj = 'obj({},{},{},{},{},{},{}).'.format(
            obj_index, posx, posy, material, color, shape, size)
        encoded_scene.append(encoded_obj)
        obj_index += 1

    return "\n".join(encoded_scene)

def encode_CLEVR_question(program):
    # Holds action sequence
    action_atoms = []
    # Time
    t = 0

    # Iterate over functional program and translate every basic function into an action atom
    for i, func in enumerate(program):
        t = i
        func_name = func["function"]
        if func_name in func_type["unary"]:
            if func_name == "scene":
                action_atoms.append(actions[func_name].format(T=t, T1=0))
            else:
                action_atoms.append(actions[func_name].format(
                    T=t, T1=func["inputs"][0]))
        elif func_name in func_type["binary_val"]:
            val = func["value_inputs"][0]
            action_atoms.append(actions[func_name].format(
                T=t, T1=func["inputs"][0], val=val))
        elif func_name in func_type["binary_in"]:
            t1 = func["inputs"][0]
            t2 = func["inputs"][1]
            if func_name in ["union", "intersect"]:
                action_atoms.append(actions[func_name].format(
                    T=t, T1=t1, T2=t2))
            else:
                action_atoms.append(
                    actions[func_name].format(T=t, T1=t1, T2=t2))
        else:
            print("Unknown function name: " + func_name)

    # Add end atom
    action_atoms.append(f"end({t}).")

    # Return action sequence as string
    return "\n".join(action_atoms)


class PromptBuilder:
    def __init__(self, dataset: str, multiple: bool, cot: bool, log_path: str = "../logs"):
        # self._model_name = model_name + ("_survey" if use_survey_data else "")
        self.dataset = dataset
        self.multiple = multiple
        self.cot = cot
        # self.preprompt = preprompt
        # self. strategy = strategy
        self.__logfile = None
        # self.setup_logfile(log_path)
        # self.__get_prompts = PromptBuilder().generated_prompts

    def generate_preprompt(self, theory):
        intro = open('./preprompt/introduction.txt').readlines()
        asp_syntax = open('./preprompt/asp/asp_syntax.txt').readlines()
        ans_ex = open('./preprompt/answer.txt').readlines()
        theory_ex = open('./preprompt/theory.txt').readlines()
        task_ex = open('./preprompt/task/task2.txt').readlines()

        if self.dataset == 'CLEVR':
            scene_ex = open('./preprompt/scene/CLEVR.txt').readlines() 
            question_ex = open('./preprompt/question/flat/CLEVR.txt').readlines()
            initial_theory = theory
            preprompt = intro + asp_syntax + scene_ex + question_ex + ans_ex + theory_ex + initial_theory + task_ex
            
        # elif self.dataset == 'GQA':
        elif 'GQA' in self.dataset:
            scene_ex = open('./preprompt/scene/GQA.txt').readlines() 
            question_ex = open('./preprompt/question/flat/GQA4.txt').readlines()
            initial_theory = theory
            preprompt = intro + asp_syntax + scene_ex + question_ex + ans_ex + theory_ex + initial_theory + task_ex

        elif self.dataset == 'CLEGR':
            scene_ex = open('./preprompt/scene/GQA.txt').readlines() 
            question_ex = open('./preprompt/question/flat/GQA4.txt').readlines()
            initial_theory = theory
            preprompt = intro + asp_syntax + scene_ex + question_ex + ans_ex + theory_ex + initial_theory + task_ex

        if self.multiple:
            multiple_answers = open('./preprompt/multiple_answers.txt').readlines() 
            preprompt += multiple_answers

        if self.cot == 'complex':
            cot = open('./preprompt/chain_of_thought_complex.txt').readlines() 
            preprompt += cot

        elif self.cot == 'simple':
            cot = open('./preprompt/chain_of_thought.txt').readlines() 
            preprompt += cot

        return ''.join(preprompt)

    def generate_examples(self, strategy = None, k = 2, representation = 'flat', special_predicate= None):
        dataset_path = './dataset/' + self.dataset

        examples = []
        # if self.dataset == 'CLEVR':
        #     scenes_file = open(
        #         dataset_path + '/scenes/CLEVR_train_scenes.json')
        #     scenes = json.load(scenes_file)
        #     questions_file = open(
        #         dataset_path + '/questions/CLEVR_train_questions.json')
        #     questions = json.load(questions_file)
            
        #     selected_questions = []
        #     # while not selected_questions:
        #     for scene in scenes['scene']:
        #         selected_scene = scenes

        #         # Then we get the questions an answers, from training again
        #         image_index = selected_scene['image_index']
        #         # print(image_index)

        #         for question in questions['questions']:
        #             if image_index == question['image_index']:
        #                 # if not type:
        #                 selected_questions.append(question)
        #                 # if type == 'spatial':
        #                 #     program = encode_CLEVR_question(
        #                 #         question['program'])
        #                 #     if 'filter' in program and not 'and' in program and not 'or' in program and not 'count' in program and not 'same' in program and not 'relate' in program and not 'equal' in program:
        #                 #         selected_questions.append(question)
        
        #     # selected_question = random.choice(selected_questions)


        #     encoded_scene = encode_CLEVR_scene(selected_scene)
        #     for question in selected_questions:
        #         encoded_question = encode_CLEVR_question(
        #             question['program'])

        #         example = {
        #             'scene': encoded_scene,
        #             'question': encoded_question,
        #             'answer': question['answer']
        #         }
        #         print(example)

        #         examples.append(example)
            

        # elif self.dataset == 'GQA':
        question_scene_file = open(dataset_path + '/train_suite.json')

        question_scene = json.load(question_scene_file)
        # print(question_scene.values())
        # question_scene = sorted(question_scene, key=lambda d: d['question']) 

        for instance in question_scene:
            if representation == 'nested':
                num_of_lines = instance['question'].count("\n") + 1
                instance['question'] = ''.join(instance['question'].split('\n')[:num_of_lines-1])
                instance['question'] = flat_to_nested(instance['question'])
                examples.append(instance)
            else:
                examples.append(instance)
        if strategy == 'len':
            len_dict = {}
            for example in examples:
                lq = example['question'].count('.')
                if lq not in len_dict:
                    len_dict[lq] = [example]
                else:
                    len_dict[lq].append(example)

            ordered_examples = []
            for key in sorted(len_dict.keys()):
                samples = random.sample(len_dict[key], min(k, len(len_dict[key])))
                for sample in samples:
                    ordered_examples.append(sample)

            return ordered_examples
        
        elif strategy == 'pred':
            pred_dict = {}

            for example in examples:
                preds = find_predicates(example['question'])
                for pred in preds:
                    if pred not in pred_dict:
                        pred_dict[pred] = [example]
                    else:
                        pred_dict[pred].append(example)
            
            if special_predicate:
                return random.sample(pred_dict[special_predicate], min(k,len(pred_dict[special_predicate])))

            for key in pred_dict:
                samples = random.sample(pred_dict[key], min(k, len(pred_dict[key])))
                for sample in samples:
                    ordered_examples.append(sample)
            return ordered_examples

        return examples

    def translate_questions(self, in_path, out_path):
        dataset_path = './dataset/' + self.dataset
        question_scene_file = open(dataset_path + '/questions/train_balanced_questions.json')

        question_scene = json.load(question_scene_file)
        question_scene = sorted(question_scene, key=lambda d: d['question']) 
        # num_questions = len(question_scene)

        # for example in question_scene:
        #     # examples.append(example)
        #     print(example)
        cnt = 0

        examples = []

        for key in question_scene:
            if question_scene[key]['semantic'][0]['operation'] == 'select' and question_scene[key]['semantic'][0]['argument'] == 'scene':
                continue
            # print(qid)
            # print(cnt)
            try:
                scene_encoding, question_encoding = encode_sample(question_scene[key])
            # question_len = len(question_encoding.replace('\n', '').split('.')) - 1
            # lens.append(len(question_encoding.replace('\n', '').split('.')))
            # exit()
                cnt += 1
                scene_encoding = scene_encoding.replace('has_relation', 'has_rel').replace('has_attribute', 'has_attr')
                examples.append({
                    'scene': scene_encoding,
                    'question': question_encoding,
                    'answer': question_scene[key]['answer']
                })
                # if cnt == 15000:
                #     break
            except:
                continue

        with open(out_path, 'w') as f:
            json.dump(examples, f)
        
# pb = PromptBuilder('GQA')
# pb.translate_questions(1, 'ASPALL_new.json')
# # examples = pb.generate_examples(2, 'spatial')
# # print(examples)

# for i in range(10):
#     print(examples[i]['question'])
#     print(examples[i]['scene'])
            
def extract_predicates_detailed(flat_question):
    """
    Extracts predicates with their names, output step, input steps, and remaining arguments 
    from a flat ASP representation.

    Args:
    flat_question (str): A string representation of the question in flat ASP format.

    Returns:
    List[Tuple[str, str, List[str], List[str]]]: A list of tuples, each containing the predicate name, 
    output step, list of input steps, and a list of remaining arguments.
    """

    # Improved regular expression pattern to match predicates with all details
    pattern = r'(\w+)\((.*?)\)\.'  # Matches 'predicate_name(args).'

    # Find all matches in the flat_question string
    matches = re.findall(pattern, flat_question)

    # Process each match to format the output
    extracted_predicates = []
    for match in matches:
        predicate_name = match[0]
        args = match[1].split(',')
        args = [arg.strip() for arg in args if arg.strip()]  # Remove any extra spaces and empty strings

        # Splitting the arguments into output step, input steps, and remaining arguments
        output_step = args[0]
        input_steps = [arg for arg in args[1:] if arg.isdigit()]
        remaining_args = [arg for arg in args[1:] if not arg.isdigit()]

        extracted_predicates.append((predicate_name, output_step, input_steps, remaining_args))

    return extracted_predicates


def flat_to_nested(flat_question):
    """
    Transforms a question from the flat ASP representation to a nested representation, excluding the 'end' predicate.

    Args:
    flat_question (str): A string representation of the question in flat ASP format.

    Returns:
    str: The nested representation of the question.
    """

    # Extract predicates and their detailed information
    predicates = extract_predicates_detailed(flat_question)

    # Dictionary to store the predicates with their output step number as key
    step_predicates = {output_step: (name, input_steps, remaining_args) for name, output_step, input_steps, remaining_args in predicates}

    # Recursive function to build nested representation
    def build_nested(output_step):
        if output_step not in step_predicates:
            # If it's a scene with no arguments, return 'scene()'
            return "scene()" if output_step == '0' else output_step

        name, input_steps, remaining_args = step_predicates[output_step]

        # Build nested structure for input steps
        nested_inputs = [build_nested(input_step) for input_step in input_steps]

        # Combine nested inputs with remaining arguments
        all_args = nested_inputs + remaining_args

        # Form the predicate string
        return f"{name}({', '.join(all_args)})"

    # Find the last step (which is not 'end')
    last_step = max(step_predicates.keys(), key=int)
    nested_question = build_nested(last_step) + '.'

    return nested_question