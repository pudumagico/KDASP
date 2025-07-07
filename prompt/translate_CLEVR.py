from clingo.control import Control 
import json 
from utils import sanitize_asp
from itertools import islice
from collections import defaultdict 

# from prompt_builder import *

# actions_old = {
#     "scene": "scene({T}).",
#     "unique": "unique({T}).",
#     "relate": "relate_{val}({T}).",
#     "count": "count({T}).",
#     "exist": "exist({T}).",
#     "filter_size": "filter_{val}({T}).",
#     "filter_color": "filter_{val}({T}).",
#     "filter_material": "filter_{val}({T}).",
#     "filter_shape": "filter_{val}({T}).",
#     "query_size": "query_size({T}).",
#     "query_color": "query_color({T}).",
#     "query_material": "query_material({T}).",
#     "query_shape": "query_shape({T}).",
#     "same_size": "same_size({T}).",
#     "same_color": "same_color({T}).",
#     "same_material": "same_material({T}).",
#     "same_shape": "same_shape({T}).",
#     "equal_integer": "equal_integer({T},{T1}).",
#     "less_than": "less_than({T},{T1}).",
#     "greater_than": "greater_than({T},{T1}).",
#     "equal_size": "equal_size({T},{T1}).",
#     "equal_color": "equal_color({T},{T1}).",
#     "equal_material": "equal_material({T},{T1}).",
#     "equal_shape": "equal_shape({T},{T1}).",
#     "union": "or({T},{T1}).",
#     "intersect": "and({T},{T1})."
# }

actions = {
    "scene": "scene({T}).",
    "unique": "unique({T}).",
    "relate": "relate_{val}({T}).",
    "count": "count({T}).",
    "exist": "exist({T}).",
    "filter_size": "filter_{val}({T}).",
    "filter_color": "filter_{val}({T}).",
    "filter_material": "filter_{val}({T}).",
    "filter_shape": "filter_{val}({T}).",
    "query_size": "query_size({T}).",
    "query_color": "query_color({T}).",
    "query_material": "query_material({T}).",
    "query_shape": "query_shape({T}).",
    "same_size": "same_size({T}).",
    "same_color": "same_color({T}).",
    "same_material": "same_material({T}).",
    "same_shape": "same_shape({T}).",
    "equal_integer": "equal_integer({T},{T1}).",
    "less_than": "less_than({T},{T1}).",
    "greater_than": "greater_than({T},{T1}).",
    "equal_size": "equal_size({T},{T1}).",
    "equal_color": "equal_color({T},{T1}).",
    "equal_material": "equal_material({T},{T1}).",
    "equal_shape": "equal_shape({T},{T1}).",
    "union": "or({T},{T1}).",
    "intersect": "and({T},{T1})."
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

# def encode_CLEVR_question_old(program):
#     # Holds action sequence
#     action_atoms = []
#     # Time
#     t = 0

#     # Iterate over functional program and translate every basic function into an action atom
#     for i, func in enumerate(program):
#         t = i
#         func_name = func["function"]
#         if func_name in func_type["unary"]:
#             if func_name == "scene":
#                 action_atoms.append(actions[func_name].format(T=t, T1=0))
#             else:
#                 action_atoms.append(actions[func_name].format(
#                     T=t, T1=func["inputs"][0]))
#         elif func_name in func_type["binary_val"]:
#             val = func["value_inputs"][0]
#             action_atoms.append(actions[func_name].format(
#                 T=t, T1=func["inputs"][0], val=val))
#         elif func_name in func_type["binary_in"]:
#             t1 = func["inputs"][0]
#             t2 = func["inputs"][1]
#             if func_name in ["union", "intersect"]:
#                 action_atoms.append(actions[func_name].format(
#                     T=t, T1=t1, T2=t2))
#             else:
#                 action_atoms.append(
#                     actions[func_name].format(T=t, T1=t1, T2=t2))
#         else:
#             print("Unknown function name: " + func_name)

#     # Add end atom
#     action_atoms.append(f"end({t+1}).")

#     # Return action sequence as string
#     return "\n".join(action_atoms)

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
                action_atoms.append(actions[func_name].format(T=t))
            else:
                action_atoms.append(actions[func_name].format(
                    T=func["inputs"][0] + 1 ))
        elif func_name in func_type["binary_val"]:
            val = func["value_inputs"][0]
            action_atoms.append(actions[func_name].format(
                T=func["inputs"][0] + 1, val=val))
        elif func_name in func_type["binary_in"]:
            t1 = func["inputs"][0]
            t2 = func["inputs"][1]
            if func_name in ["union", "intersect"]:
                action_atoms.append(actions[func_name].format(
                    T=t2+1, T1=t1+1))
            else:
                action_atoms.append(
                    actions[func_name].format(T=t2+1, T1=t1+1))
        else:
            print("Unknown function name: " + func_name)

    # Add end atom
    action_atoms.append(f"end({t + 1}).")

    # print("\n".join(action_atoms))
    # exit()

    # Return action sequence as string
    return "\n".join(action_atoms)


with open('/home/nelson/Research/IJCAI24/dataset/CLEVR/questions/CLEVR_val_questions.json') as q:
    questions = json.load(q)

with open('/home/nelson/Research/IJCAI24/dataset/CLEVR/scenes/CLEVR_val_scenes.json') as s:
    scenes = json.load(s)

with open('./preprompt/theory/CLEVR/theory.lp') as tf:
    theory = tf.read()

L = len(questions['questions'])
print(L)
import random

num_questions = len(questions)
correct = 0
incorrect = 0
unsat = 0
lens = []
correct_examples = []
examples = []

x = scenes['scenes']
random.shuffle(x)
# while not selected_questions:
for scene in x:
    selected_questions = []
    selected_scene = scene
    image_index = selected_scene['image_index']
    for question in questions['questions']:
        if image_index == question['image_index']:
            selected_questions.append(question)

    encoded_scene = encode_CLEVR_scene(selected_scene)
    for question in selected_questions:
        encoded_question = encode_CLEVR_question(
            question['program'])

        example = {
            'scene': encoded_scene,
            'question': encoded_question,
            'answer': question['answer']
        }
    # import pprint
    # pprint.pprint(example)
    # print(example['scene'])
    # print(example['question'])
    # print()
    # exit()
    examples.append(example)
    # print(len(examples)/L*100)
            
with open("examples_CLEVR.json", "w") as final:
    json.dump(examples, final)

# def answer_is_correct(answers, correct_answer):
#     correct = False 

#     for answer in answers:
#         if answer == sanitize_asp(correct_answer): 
#             correct = True
#         elif (answer == 'to_the_right_of' and correct_answer == 'right') or \
#             (answer == 'to_the_left_of' and correct_answer == 'left') or \
#             (answer == 'in_front_of' and correct_answer == 'front'):
#             correct = True
#     return correct 




# for qid, question in islice(questions.items(), 0, num_questions):
#     if question['semantic'][0]['operation'] == 'select' and question['semantic'][0]['argument'] == 'scene':
#         num_questions = num_questions - 1
#         continue

#     ctl = Control()
#     ctl.add(theory)

#     try:
#         scene_encoding, question_encoding = encode_sample(question)
#     except:
#         continue

#     question_len = len(question_encoding.replace('\n', '').split('.')) - 1
#     scene_encoding = scene_encoding.replace('has_relation', 'has_rel').replace('has_attribute', 'has_attr')

#     ctl.add(scene_encoding)
#     ctl.add(question_encoding)

#     answers = [[]]
#     def on_model(model):
#         answers[0] = [s.arguments[0].name for s in model.symbols(shown=True)]
#     try:
#         ctl.ground()
#         result = ctl.solve(on_model=on_model)

#         if result.satisfiable:
#             if(answer_is_correct(answers[0], question['answer'])):
#                 correct = correct + 1
#                 correct_example = {'question': question_encoding, 'scene': scene_encoding, 'answer': question['answer']}
#                 correct_examples.append(correct_example)
#             else: 
#                 incorrect = incorrect + 1
#         else: 
#             unsat = unsat + 1
#     except:
#         continue
#     # print("===============================")
#     # print(f"Total questions: {num_questions}")
#     # print(f"Correct: {correct}")
#     # print(f"Incorrect: {incorrect}")
#     # print(f"UNSAT: {unsat}")
#     if correct:
#         print(f"Percentage: {correct/num_questions*100}%")

# with open("correct_examples_CLEVR.json", "w") as final:
#     json.dump(correct_examples, final)