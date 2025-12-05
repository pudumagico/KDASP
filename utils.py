import re
from pattern.text.en import singularize
import random
import os
import json
from openai import OpenAI
from litellm import completion
import clingo
import subprocess

import requests
from dotenv import load_dotenv
import os
import torch
load_dotenv()  # Loads variables from .env into environment


BEARER = ""

API_URL = "https://router.huggingface.co/hyperbolic/v1/chat/completions"
headers = {"Authorization": "Bearer " + BEARER}

os.environ["OPENAI_API_KEY"] =  ""
os.environ["MISTRAL_API_KEY"] = ""
os.environ["HUGGINGFACE_API_KEY"] = ""
os.environ['DEEPSEEK_API_KEY'] = ""
os.environ['GEMINI_API_KEY'] = ""

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def build_prompt(messages):
    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += f"System: {message['content']}\n"
        elif message["role"] == "user":
            prompt += f"User: {message['content']}\n"
        elif message["role"] == "assistant":
            prompt += f"Assistant: {message['content']}\n"
    return prompt

def sanitize(name):
    # source: DFOL-VQA
    plurale_tantum = [
        "this",
        "yes",
        "pants",
        "shorts",
        "glasses",
        "scissors",
        "panties",
        "trousers",
        "binoculars",
        "pliers",
        "tongs",
        "tweezers",
        "forceps",
        "goggles",
        "jeans",
        "tights",
        "leggings",
        "chaps",
        "boxers",
        "indoors",
        "outdoors",
        "bus",
        "octapus",
        "waitress",
        "pasta",
        "pita",
        "glass",
        "asparagus",
        "hummus",
        "dress",
        "cafeteria",
        "grass",
        "class",
    ]

    irregulars = {
        "shelves": "shelf",
        "bookshelves": "bookshelf",
        "olives": "olive",
        "brownies": "brownie",
        "cookies": "cookie",
    }

    temp = name.strip().lower()
    if temp in irregulars:
        return irregulars[temp]
    elif not (temp.split(" ")[-1] in plurale_tantum or temp[-2:] == "ss"):
        return singularize(temp)
    else:
        return temp


def cleanup_whitespace(name):
    cleanup_regex = r"[^\w]"
    return re.sub(cleanup_regex, "_", name)


def sanitize_asp(name):
    return cleanup_whitespace(sanitize(name))


def remove_random_lines(theory, percentage):
    # Read the content of the file
    lines = theory.split("\n")

    # Calculate the number of lines to remove
    num_lines_to_remove = int(len(lines) * (percentage / 100))

    # Randomly select lines to remove
    lines_to_remove = set(random.sample(
        range(len(lines)), num_lines_to_remove))

    # Create a new list of lines that does not include the lines to remove
    new_lines = [
        line + "\n" for i, line in enumerate(lines) if i not in lines_to_remove
    ]

    # Return the modified content
    return "".join(new_lines)


def remove_lines_with_predicates(input_string, predicates):
    # Split the string into lines
    lines = input_string.split("\n")

    # Filter out lines that contain any of the predicates followed by a '(' symbol
    filtered_lines = [
        line for line in lines if not any(f"{predicate}(" in line for predicate in predicates)
    ]

    # Join the remaining lines back into a single string
    return "\n".join(filtered_lines)

def write_to_log(output_file_name, *args):
    # Open the file in append mode
    with open(output_file_name, "a") as file:
        # Write each argument to the file, except for the first one
        for arg in args:
            file.write(str(arg) + "\n")


def save_examples(filename, array):
    # Open the file in write mode
    with open(filename, "w") as file:
        # Use json.dump to write the array to the file
        json.dump(array, file)


def run_clingo(clingo_script):

    text_file = open("clingo_test.lp", "w")
    text_file.write(clingo_script)
    text_file.close()

    try:
        process = subprocess.Popen(
            ["clingo", './clingo_test.lp'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()

        return stderr
    except Exception as e:
        print(f"Exception: {str(e)}")
        return -1

def parse_error_callback(self, error, message):
    """
    Callback function to handle errors from Clingo.

    Parameters:
    error (Exception): The error object received from Clingo.

    Returns:
    str: A parsed error message.
    """

    self._data.get_logger().add_log("TEST", message)

def find_last_state(states_list):

    # Initialize variables to track the state with the highest first argument
    max_first_arg = -1
    max_state = ""

    # Iterate through each state and extract the arguments
    for state in states_list:
        # Extract the arguments from the state string
        args = state[state.find("(") + 1:state.find(")")].split(",")
        first_arg = int(args[0])

        # Check if this state has the highest first argument so far
        if first_arg > max_first_arg:
            max_first_arg = first_arg
            max_state = state

    return max_state

def log_incumbent_theory(
    date_str,
    theory,
    model,
    max_retries,
    learning_examples,
    current_example,
    batch_examples,
    remove_predicate,
    dataset,
    cot,
    mending,
    multiple
):
    # Format the current date for the folder name
    # Create a folder name for this specific run
    run_folder_name = "theories"
    # Base directory for logs
    if cot:
        base_log_dir = f"./logs/no_cot/{dataset}/{model}/{remove_predicate}"
    elif mending:
        base_log_dir = f"./logs/no_mending/{dataset}/{model}/{remove_predicate}"
    elif multiple:
        base_log_dir = f"./logs/no_multiple/{dataset}/{model}/{remove_predicate}"
    else:
        base_log_dir = f"./logs/{dataset}/{model}/{remove_predicate}"        
    # Create the date-named directory
    date_dir = os.path.join(base_log_dir, date_str)
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)
    # Create the run-specific directory
    run_dir = os.path.join(date_dir, run_folder_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    # Filename for the theory
    filename = f"current{current_example}_examples{learning_examples}.txt"
    # Full path for the file
    full_path = os.path.join(run_dir, filename)
    with open(full_path, "w") as log_file:
        log_file.write(theory)

def log_explanation(
    date_str,
    theory,
    model,
    max_retries,
    learning_examples,
    current_example,
    batch_examples,
    remove_predicate,
    dataset,
    cot,
    multiple
):
    # Format the current date for the folder name
    # Create a folder name for this specific run
    run_folder_name = "explanations"
    # Base directory for logs
    if cot:
        base_log_dir = f"./logs/cot/{dataset}/{model}/{remove_predicate}"
    elif multiple:
        base_log_dir = f"./logs/multiple/{dataset}/{model}/{remove_predicate}"
    else:
        base_log_dir = f"./logs/{dataset}/{model}/{remove_predicate}"        
    # Create the date-named directory
    date_dir = os.path.join(base_log_dir, date_str)
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)
    # Create the run-specific directory
    run_dir = os.path.join(date_dir, run_folder_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    # Filename for the theory
    filename = f"current{current_example}_examples{learning_examples}.txt"
    # Full path for the file
    full_path = os.path.join(run_dir, filename)
    with open(full_path, "w") as log_file:
        log_file.write(theory)

def truncate_after_asterisks(input_string):
    # Find the index of the first occurrence of three consecutive asterisks
    index = input_string.find('***')
    
    # If three asterisks are found, truncate the string at that point
    if index != -1:
        return input_string[:index]
    
    # If three asterisks are not found, return the original string
    return input_string

def is_program_only_facts(program):
    """
    Check if the given ASP program only contains facts.
    """
    ctl = clingo.Control()
    ctl.add("base", [], program)

    try:
        ctl.ground([("base", [])])
    except clingo.ClingoError as e:
        print(f"Error in grounding: {e}")
        return False

    for atom in ctl.symbolic_atoms:
        if atom.is_fact:
            # Check if there are non-fact elements
            if not atom.symbol.arguments:
                return False
    return True


# Function wrapper with timeout
# def run_asp_code(*args, **kwargs):
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future = executor.submit(run_asp_code_inner, *args, **kwargs)
#         try:
#             r1, r2 = future.result(timeout=10)
#             return r1, r2
#         except concurrent.futures.TimeoutError:
#             return False, []


def run_asp_code(theory, examples, clegr):
    correct = 0
    incorrect_answers = []
    # print(theory)
    # print(examples)
    incorrect = 0

    for example in examples:
        if clegr:
            ctl = clingo.Control(["--warn=none", "--opt-strategy=usc", "-n 0"])

            ctl.add("base", [], theory)
            ctl.add("base", [], example["scene"])
            ctl.add("base", [], example["question"])
            ctl.add("base", [], theory)
            ctl.add("base", [], "#show ans/1.")

            ctl.ground([("base", [])])
            models = []
            with ctl.solve(yield_=True) as handle:
                for m in handle:
                    models.append(m)
            gt_answer = example["answer"]

            if type(gt_answer) == list:
                gt_answer.sort()
                current_ans = [str(x).replace(' ', '').replace(
                    '-', '').lower() for x in gt_answer]
            else:
                current_ans = str(gt_answer).replace(
                    ' ', '').replace('-', '').lower()

            ans_found = False
            model_ans = []
            for model in models:
                for atom in model.symbols(shown=True):
                    if not ans_found:
                        if atom.name == 'ans':
                            model_ans.append(str(atom.arguments[0]))
                            if str(atom.arguments[0]) == current_ans:
                                ans_found = True

            model_ans.sort()

            if not ans_found and model_ans == current_ans:
                ans_found = True

            len_models = len(models)
            if not ans_found and str(len_models) == current_ans:
                ans_found = True

            if not ans_found and not models and current_ans == 'false':
                ans_found = True

            if not ans_found:
                incorrect += 1
                incorrect_answers.append(model_ans)

        else:
            ctl = clingo.Control(["--warn=none"])
            ctl.add("base", [], theory)
            ctl.add("base", [], example["scene"])
            ctl.add("base", [], example["question"])
            ctl.add("base", [], "#show ans/1.")

            ctl.ground([("base", [])])

            with ctl.solve(yield_=True) as handle:
                for model in handle:
                    if answer_is_correct(
                        [str(s.arguments[0]) for s in model.symbols(shown=True)],
                        example["answer"],
                    ):
                        correct += 1
                    else:
                        incorrect_answers.append(
                            [str(s.arguments[0]) for s in model.symbols(shown=True)],
                        )

            incorrect = len(examples) - correct

    if not incorrect:
        return True, None
    else:
        return False, incorrect_answers

def run_asp_code_with_states(theory, examples):
    correct = 0
    states = []

    for example in examples:

        ctl = clingo.Control(["--warn=none"])
        ctl.add("base", [], theory)
        ctl.add("base", [], example["scene"])
        ctl.add("base", [], example["question"])
        ctl.add("base", [], "#show state/2.")

        ctl.ground([("base", [])])

        with ctl.solve(yield_=True) as handle:
            for model in handle:
                # print(symbol.arguments[0], example['answer'], str(symbol.arguments[0]) == str(example['answer']))
                # exit()
                # if str(symbol.arguments[0]) == str(example['answer']):

                states = [str(symbol) for symbol in model.symbols(shown=True)]

    return states

def mend_syntax(broken_rule, syntax_error, preprompt, model="gpt-4-turbo"):
    syntax_error = syntax_error.replace('[#inc_base];', '')
    preprompt = f"You must repair the syntax of the prompted Answer Set Programming rule(s). Additionally, Clingo outputed the following error: {syntax_error}. You must only output the fixed ASP rule(s) and any other rules included in the prompt that have correct syntax. Do not output any natural language. The output must be in plain text only! Do not output the response as a code block!"
    messages = [
        {"role": "system", "content": preprompt},
        {"role": "user", "content": broken_rule},
    ]

    if model == 'llama':
        prompt = build_prompt(messages)
#         import transformers
#         model_id = "meta-llama/Meta-Llama-3-70B"

#         pipeline = transformers.pipeline(
#             "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
        # return pipeline(prompt)
# )
    #     data = query({"inputs": prompt, "parameters": {"return_full_text": False, "do_sample": True, "max_length": 250}})[0]['generated_text']
    #     response = query({
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": "What is the capital of France?"
    #         }
    #     ],
    #     "model": "meta-llama/Meta-Llama-3-70B-Instruct"
    # })


        response = query({
            "messages": messages,
            "model": "meta-llama/Meta-Llama-3-70B-Instruct"
        })
        

        return response["choices"][0]["message"]['content']
    else:
        response = completion(model=model, messages=messages, temperature=0)

        return response.choices[0].message.content


def mend_semantics(
    broken_rule, infered_answer, expected_answer, theory, preprompt, model="gpt-4-turbo"
):
    preprompt = f"""You must repair the semantics of the prompted Answer Set Programming rule(s).
    The original theory is: {theory}. The prompted rules are added to this theory to calculate the answer. 
    They do not calculate the correct answer, which is: {expected_answer}, 
    but instead result in the following incorrect answer: {infered_answer} (can me empty).
    You must only output the fixed ASP rule(s). 
    The rule(s) must be semantically different from the ones prompted and they should now calculate the correct answer.
    Do not output any natural language. The output must be in plain text only! Do not output the response as a code block!"""

    messages = [
        {"role": "system", "content": preprompt},
        {"role": "user", "content": broken_rule},
    ]
    if model == 'llama':
        # prompt = build_prompt(messages)
        # data = query({"inputs": prompt, "parameters": {"return_full_text": False, "do_sample": True, "max_length": 250}})[0]['generated_text']
        
        response = query({
            "messages": messages,
            "model": "meta-llama/Meta-Llama-3-70B-Instruct"
        })
        

        return response["choices"][0]["message"]['content']
        return data
    else:
        response = completion(model=model, messages=messages, temperature=0)

        return response.choices[0].message.content

def mend_semantics_with_states(
    broken_rule, infered_answer, expected_answer, theory, preprompt, model, state_atoms
):
    last_state = find_last_state(state_atoms)
    preprompt = f"""You must repair the semantics of the prompted Answer Set Programming rule(s).
    The original theory is: {theory}. The prompted rules are added to this theory to calculate the answer. 
    They do not calculate the correct answer, which is: {expected_answer}, 
    but instead result in the following incorrect answer: {infered_answer} (can me empty).
    Additionally, we give you the states that could be computed by the program: {state_atoms}.
    The last state is: {last_state}, after this the computation fails.
    You must only output the fixed ASP rule(s). 
    The rule(s) must be semantically different from the ones prompted and they should now calculate the correct answer.
    Do not output any natural language. The output must be in plain text only! Do not output the response as a code block!"""

    messages = [
        {"role": "system", "content": preprompt},
        {"role": "user", "content": broken_rule},
    ]

    if model == 'llama':
        # prompt = build_prompt(messages)
        # data = query({"inputs": prompt, "parameters": {"return_full_text": False, "do_sample": True, "max_length": 250}})[0]['generated_text']
        
        response = query({
            "messages": messages,
            "model": "meta-llama/Meta-Llama-3-70B-Instruct"
        })
        

        return response["choices"][0]["message"]['content']
        return data
    else:
        response = completion(model=model, messages=messages, temperature=0)

        return response.choices[0].message.content

def ask_LLM(question, preprompt, model="gpt-4-turbo"):
    messages = [
        {"role": "system", "content": preprompt},
        {"role": "user", "content": question},
    ]
    
    if model == 'llama':
        # prompt = build_prompt(messages)
        # data = query({"inputs": prompt, "parameters": {"return_full_text": False, "do_sample": True, "max_length": 250}})[0]['generated_text']
        
        response = query({
            "messages": messages,
            "model": "meta-llama/Meta-Llama-3-70B-Instruct"
        })
        

        return response["choices"][0]["message"]['content']
        return data
    
    else:
        response = completion(model=model, messages=messages, temperature=0)

        return response.choices[0].message.content




# def sanitize(llm_output):
#     if llm_output.choices:
#         sanitized = llm_output.choices[0].message.content

#     return sanitized

# def ask_LLM(question, preprompt, model = "gpt-3.5-turbo"):
#     chat_completion = client.chat.completions.create(
#         messages=[
#             {"role": "system", "content":preprompt},
#             {"role": "user", "content": question},
#         ],
#         model="gpt-3.5-turbo"
#     )
#     return chat_completion.choices[0].message.content.strip('```').strip('{').strip('}').strip('asp').strip('ASP')


def merge_asp_encodings(encoding1, encoding2, model="gpt-3.5-turbo"):
    # Construct the prompt for GPT
    prompt = (
        "Here are two ASP encodings:\n\n"
        "Encoding 1:\n" + encoding1 + "\n\n"
        "Encoding 2:\n" + encoding2 + "\n\n"
        "Merge these two ASP encodings into a single coherent encoding."
    )

    # Send the prompt to GPT
    messages = [
        {
            "role": "system",
            "content": "Your task is to merge these two Answer Set Programming encodings. \
                    Remove any rule with constants in the head.\
                    Remove any redundant rules or non general rules (rules with constants in the head). \
                    RETURN ONLY ONE ASP ENCODING. NO NATURAL LANGUAGE EXPLANATIONS. JUST OUTPUT ASP.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    response = completion(model=model, messages=messages)

    # Return the merged encoding
    return (
        response.choices[0]
        .message.content.strip("```")
        .strip("{")
        .strip("}")
        .strip("asp")
        .strip("ASP")
        .strip("prolog")
    )


def answer_is_correct(answers, correct_answer):
    correct = False

    for answer in answers:
        if answer == sanitize_asp(correct_answer):
            correct = True
        elif (
            (answer == "to_the_right_of" and correct_answer == "right")
            or (answer == "to_the_left_of" and correct_answer == "left")
            or (answer == "in_front_of" and correct_answer == "front")
        ):
            correct = True
    return correct


def check_asp_syntax(code):
    try:
        ctl = clingo.Control(["--warn=none"])
        ctl.add("base", [], code)
        ctl.ground([("base", [])])
        return True, None
    except Exception as e:
        errors = run_clingo(code)
        return False, errors

def check_asp_semantics(theory, examples):

    for example in examples:
        ctl = clingo.Control(["--warn=none"])
        ctl.add("base", [], theory)
        ctl.add("base", [], example["scene"])
        ctl.add("base", [], example["question"])
        ctl.add("base", [], "#show ans/1.")
        ctl.ground([("base", [])])
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                if not answer_is_correct(
                    [s.arguments[0].name for s in model.symbols(shown=True)],
                    example["answer"],
                ):
                    return example
    return "Pass"

