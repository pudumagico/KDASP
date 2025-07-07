from clingo.control import Control 
import json, random


with open('/home/nelson/Research/IJCAI24/dataset/CLEVR/test_suite.json') as q:
    questions = json.load(q)

with open('./preprompt/theory/CLEVR/theory.lp') as tf:
    theory = tf.read()

num_questions = len(questions)
print(num_questions)

# random.shuffle(questions)
correct = 0
incorrect = 0
unsat = 0
lens = []
correct_examples = []

# def answer_is_correct(answers, correct_answer):
#     print(answers, answer)
#     correct = False 

#     for answer in answers:
#         if answer == correct_answer: 
#             print('yes')
#             correct = True
#             break
#         # elif (answer == 'to_the_right_of' and correct_answer == 'right') or \
#         #     (answer == 'to_the_left_of' and correct_answer == 'left') or \
#         #     (answer == 'in_front_of' and correct_answer == 'front'):
#             # correct = True
#     return correct 


for question_scene in questions:

    # print(question_scene['question'])
    # print(question_scene['scene'])
    # # print(theory)
    # exit()

    ctl = Control(["--warn=none"])

    ctl.add(theory)

    ctl.add(question_scene['question'])
    ctl.add(question_scene['scene'])
    print(question_scene['question'])
    print('##############')

    answers = [[]]
    def on_model(model):
        # print(question_scene['answer'])
        answers[0] = [s.arguments[0].name for s in model.symbols(shown=True)]
        # print(answers)

    try:
        ctl.ground()
        result = ctl.solve(on_model=on_model)
        # exit()
        if result.satisfiable:
            # if question_scene['answer'] == 'no':
            #     question_scene['answer'] = 'false'
            # if question_scene['answer'] == 'yes':
            #     question_scene['answer'] = 'true'
            # print(answers[0], question_scene['answer'])
            if question_scene['answer'] in answers[0]:
                correct = correct + 1
                correct_example = {'question': question_scene['question'], 'scene': question_scene['scene'], 'answer': question_scene['answer']}
                correct_examples.append(correct_example)
            else: 
                incorrect = incorrect + 1

        else: 
            print(question_scene['question'])
            print(question_scene['scene'])
            print(question_scene['answer'])
            print(answers)
            exit()
            unsat = unsat + 1
    except:   
        print(question_scene['question'])
        print(question_scene['scene'])
        print(question_scene['answer'])
        print(answers)
        exit()
        # exit()
        continue
    # print("===============================")
    # print(f"Total questions: {num_questions}")
    # print(f"Correct: {correct}")
    # print(f"Incorrect: {incorrect}")
    # print(f"UNSAT: {unsat}")
    # if correct:
    #     print(f"Percentage: {correct/num_questions*100}%")

with open("correct_examples_CLEVR.json", "w") as final:
    json.dump(correct_examples, final)