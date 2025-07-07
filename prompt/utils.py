from pattern.text.en import singularize
import re

def sanitize(name):
    # source: DFOL-VQA
    plurale_tantum = ['this', 'yes', 'pants', 'shorts', 'glasses', 'scissors', 'panties', 'trousers', 'binoculars', 'pliers', 'tongs',\
        'tweezers', 'forceps', 'goggles', 'jeans', 'tights', 'leggings', 'chaps', 'boxers', 'indoors', 'outdoors', 'bus', 'octapus', 'waitress',\
        'pasta', 'pita', 'glass', 'asparagus', 'hummus', 'dress', 'cafeteria', 'grass', 'class']

    irregulars = {'shelves': 'shelf', 'bookshelves': 'bookshelf', 'olives': 'olive', 'brownies': 'brownie', 'cookies': 'cookie'}
    
    temp = name.strip().lower()
    if temp in irregulars:
        return irregulars[temp]
    elif not (temp.split(' ')[-1] in plurale_tantum or temp[-2:] == 'ss'):
        return singularize(temp)
    else:
        return temp


def cleanup_whitespace(name):
    cleanup_regex = r'[^\w]'
    return re.sub(cleanup_regex, '_', name)


def sanitize_asp(name):
    return cleanup_whitespace(sanitize(name))

def remove_lines_with_predicates(input_string, predicates):
    # Split the string into lines
    lines = input_string.split("\n")

    # Filter out lines that contain any of the predicates
    filtered_lines = [
        line for line in lines if not any(predicate in line for predicate in predicates)
    ]

    # Join the remaining lines back into a single string
    return "\n".join(filtered_lines)
