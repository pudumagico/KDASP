from clingo.control import Control 
import json 
from perfect_information_encoding import encode_sample
from utils import sanitize_asp
from itertools import islice
from collections import defaultdict 


with open('./dataset/GQA/50000.json') as f:
    questions = json.load(f)

with open('./preprompt/theory/GQA/perfect_theory.lp') as tf:
    theory = tf.read()

num_questions = len(questions)
correct = 0
incorrect = 0
unsat = 0
lens = []
correct_examples = []

def answer_is_correct(answers, correct_answer):
    correct = False 

    for answer in answers:
        if answer == sanitize_asp(correct_answer): 
            correct = True
        elif (answer == 'to_the_right_of' and correct_answer == 'right') or \
            (answer == 'to_the_left_of' and correct_answer == 'left') or \
            (answer == 'in_front_of' and correct_answer == 'front'):
            correct = True
    return correct 

for qid, question in islice(questions.items(), 0, num_questions):
    # print(question['question'].keys())
    if question['semantic'][0]['operation'] == 'select' and question['semantic'][0]['argument'] == 'scene':
        num_questions = num_questions - 1
        continue

    ctl = Control()
    ctl.add(theory)

    try:
        scene_encoding, question_encoding = encode_sample(question)
    except:
        continue
    question_len = len(question_encoding.replace('\n', '').split('.')) - 1
    scene_encoding = scene_encoding.replace('has_relation', 'has_rel').replace('has_attribute', 'has_attr')

    ctl.add(scene_encoding)
    ctl.add(question_encoding)

    answers = [[]]
    def on_model(model):
        answers[0] = [s.arguments[0].name for s in model.symbols(shown=True)]
    try:
        ctl.ground()
        result = ctl.solve(on_model=on_model)

        if result.satisfiable:
            if(answer_is_correct(answers[0], question['answer'])):
                correct = correct + 1
                correct_example = {'question': question_encoding, 'scene': scene_encoding, 'answer': question['answer']}
                print(correct_example)
                correct_examples.append(correct_example)
            else: 
                incorrect = incorrect + 1
        else: 
            unsat = unsat + 1
    except:
        continue
    # print("===============================")
    # print(f"Total questions: {num_questions}")
    # print(f"Correct: {correct}")
    # print(f"Incorrect: {incorrect}")
    # print(f"UNSAT: {unsat}")
    if correct:
        print(f"Percentage: {correct/num_questions*100}%")

with open("correct_examples_GQA.json", "w") as final:
    json.dump(correct_examples, final)