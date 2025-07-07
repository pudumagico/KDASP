from clingo.control import Control 
import json 
from question_encoding import encode_question
from utils import sanitize_asp
from itertools import islice

with open(r'/home/nelson/Research/IJCAI24/dataset/GQA/questions/testdev_all_questions.json') as f:
    questions = json.load(f)

with open(r'/home/nelson/Research/IJCAI24/preprompt/theory/flat/perfect_theory.lp') as tf:
    theory = tf.read()

num_questions = len(questions)
correct = 0
incorrect = 0

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
    if question['semantic'][0]['operation'] == 'select' and question['semantic'][0]['argument'] == 'scene':
        num_questions = num_questions - 1
        continue

    # ctl = Control()
    # ctl.add(theory)

    question_encoding = encode_question(question)
    # print(question, question_encoding)
    ctl.add(scene_encoding)
    ctl.add(question_encoding)

    answers = [[]]
    def on_model(model):
        answers[0] = [s.arguments[0].name for s in model.symbols(shown=True)]

    ctl.ground()
    result = ctl.solve(on_model=on_model)

    if result.satisfiable:
        if(answer_is_correct(answers[0], question['answer'])):
            # print(f"Correct answer: {answer[0]}")
            correct = correct + 1
        else: 
            print(f"Question {qid}: incorrect answer: {answers[0]} (correct: {question['answer']})")
            incorrect = incorrect + 1
    else: 
        print(f"Question {qid}: UNSAT")
        incorrect = incorrect + 1

print("===============================")
print(f"Total questions: {num_questions}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Percentage: {correct/num_questions*100}%")