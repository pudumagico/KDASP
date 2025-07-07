from utils import sanitize_asp

def encode_question(question):
    question_encoding = ""
    step_padding = 0
    ops_map = {}
    for i, operation in enumerate(question['semantic']):
        if len(operation['dependencies']) == 0:
            question_encoding += f"scene({i+step_padding}).\n"
            dependencies = [i + step_padding]
            step_padding = step_padding + 1
        else:
            dependencies = [ops_map[op] for op in operation['dependencies']]
        
        if operation['operation'] == 'select':
            target_class = sanitize_asp(operation['argument'].split('(')[0])
            question_encoding += f"select({i+step_padding}, {dependencies[0]}, {target_class}).\n"

        elif operation['operation'] == 'relate':
            target_class = sanitize_asp(operation['argument'].split(',')[0])
            relation_type = sanitize_asp(operation['argument'].split(',')[1])
            if relation_type.startswith('same_'):
                question_encoding += f"relate_attr({i+step_padding}, {dependencies[0]}, {target_class}, {relation_type[5:]}).\n"
            else:
                position = 'subject' if operation['argument'].split(',')[2].startswith('s') else 'object'
                if target_class == '_':
                    question_encoding += f"relate_any({i+step_padding}, {dependencies[0]}, {relation_type}, {position}).\n"
                else:
                    question_encoding += f"relate({i+step_padding}, {dependencies[0]}, {target_class}, {relation_type}, {position}).\n"

        elif operation['operation'] == 'query':
            question_encoding += f"unique({i+step_padding}, {dependencies[0]}).\n"
            question_encoding += f"query({i+step_padding+1}, {i+step_padding}, {operation['argument']}).\n"
            step_padding += 1

        elif operation['operation'] == 'exist':
            question_encoding += f"exist({i+step_padding}, {dependencies[0]}).\n"

        elif operation['operation'] == 'and':
            question_encoding += f"and({i+step_padding}, {dependencies[0]}, {dependencies[1]}).\n"

        elif operation['operation'] == 'or':
            question_encoding += f"or({i+step_padding}, {dependencies[0]}, {dependencies[1]}).\n"

        elif operation['operation'] == 'common':
            question_encoding += f"unique({i+step_padding}, {dependencies[0]}).\n"
            question_encoding += f"unique({i+step_padding+1}, {dependencies[1]}).\n"
            question_encoding += f"common({i+step_padding+2}, {i+step_padding}, {i+step_padding+1}).\n"
            step_padding += 2

        elif operation['operation'] == 'filter':
            if operation['argument'].startswith('not('):
                value = sanitize_asp(operation['argument'][4:-1])
                question_encoding += f"filter_any({i+step_padding}, {dependencies[0]}, {value}).\n"
                question_encoding += f"negate({i+step_padding+1}, {i+step_padding}, {dependencies[0]}).\n"
                step_padding = step_padding + 1
            else:
                value = sanitize_asp(operation['argument'])
                question_encoding += f"filter_any({i+step_padding}, {dependencies[0]}, {value}).\n"

        elif operation['operation'] == 'choose':
            option0 = sanitize_asp(operation['argument'].split('|')[0])
            option1 = sanitize_asp(operation['argument'].split('|')[1])
            question_encoding += f"unique({i+step_padding}, {dependencies[0]}).\n"
            question_encoding += f"choose_attr({i+step_padding+1}, {i+step_padding}, any, {option0}, {option1}).\n"
            step_padding += 1

        elif operation['operation'] == 'choose rel':
            target_class = sanitize_asp(operation['argument'].split(',')[0])
            option0 = sanitize_asp(operation['argument'].split(',')[1].split('|')[0])
            option1 = sanitize_asp(operation['argument'].split(',')[1].split('|')[1])
            position = 'subject' if operation['argument'].split(',')[2].startswith('s') else 'object'
            question_encoding += f"unique({i+step_padding}, {dependencies[0]}).\n"
            question_encoding += f"choose_rel({i+step_padding+1}, {i+step_padding}, {target_class}, {option0}, {option1}, {position}).\n"
            step_padding += 1

        elif operation['operation'] == 'same':
            if operation['argument'] == 'type':
                attr = 'class'
            else:
                attr = sanitize_asp(operation['argument'])
            question_encoding += f"all_same({i+step_padding}, {dependencies[0]}, {attr}).\n"

        elif operation['operation'] == 'different':
            if operation['argument'] == 'type':
                attr = 'class'
            else:
                attr = sanitize_asp(operation['argument'])
            question_encoding += f"all_different({i+step_padding}, {dependencies[0]}, {attr}).\n"

        elif operation['operation'].startswith('filter'):
            attr = sanitize_asp(' '.join(operation['operation'].split(' ')[1:]))
            if operation['argument'].startswith('not('):
                value = sanitize_asp(operation['argument'][4:-1])
                question_encoding += f"filter({i+step_padding}, {dependencies[0]}, {attr}, {value}).\n"
                question_encoding += f"negate({i+step_padding+1}, {i+step_padding}, {dependencies[0]}).\n"
                step_padding = step_padding + 1
            else:
                value = sanitize_asp(operation['argument'])
                question_encoding += f"filter({i+step_padding}, {dependencies[0]}, {attr}, {value}).\n"

        elif operation['operation'] == 'verify':
            value = sanitize_asp(operation['argument'])
            question_encoding += f"unique({i+step_padding}, {dependencies[0]}).\n"
            question_encoding += f"verify_attr({i+step_padding+1}, {i+step_padding}, any, {value}).\n"
            step_padding += 1
        
        elif operation['operation'] == 'verify rel':
            target_class = sanitize_asp(operation['argument'].split(',')[0])
            relation_type = sanitize_asp(operation['argument'].split(',')[1])
            position = 'subject' if operation['argument'].split(',')[2].startswith('s') else 'object'
            question_encoding += f"unique({i+step_padding}, {dependencies[0]}).\n"
            question_encoding += f"verify_rel({i+step_padding+1}, {i+step_padding}, {target_class}, {relation_type}, {position}).\n"
            step_padding += 1
        
        elif operation['operation'].startswith('verify'):
            attr = sanitize_asp(' '.join(operation['operation'].split(' ')[1:]))
            value = sanitize_asp(operation['argument'])
            question_encoding += f"unique({i+step_padding}, {dependencies[0]}).\n"
            question_encoding += f"verify_attr({i+step_padding+1}, {i+step_padding}, {attr}, {value}).\n"
            step_padding += 1

        elif operation['operation'].startswith('choose'):
            if operation['argument'] == '':
                op_tokens = operation['operation'].split(' ')
                question_encoding += f"unique({i+step_padding}, {dependencies[0]}).\n"
                question_encoding += f"unique({i+step_padding+1}, {dependencies[1]}).\n"

                if len(op_tokens) >= 3:
                    if sanitize_asp(op_tokens[1]) == 'more':
                        question_encoding += f"compare({i+step_padding+2}, {i+step_padding}, {i+step_padding+1}, {sanitize_asp(op_tokens[2])}, true).\n"
                    elif sanitize_asp(op_tokens[1]) == 'less':
                        question_encoding += f"compare({i+step_padding+2}, {i+step_padding}, {i+step_padding+1}, {sanitize_asp(op_tokens[2])}, false).\n"
                else:
                    token = sanitize_asp(op_tokens[1])
                    if token.endswith('er'):
                        token = token[:-2]
                        if token.endswith('i'):
                            token = token[:-1] + 'y'

                    question_encoding += f"compare({i+step_padding+2}, {i+step_padding}, {i+step_padding+1}, {token}, true).\n"
                    question_encoding += f"query({i+step_padding+3}, {i+step_padding+2}, name).\n"
                step_padding += 3
            else:
                attr = sanitize_asp(' '.join(operation['operation'].split(' ')[1:]))
                option0 = sanitize_asp(operation['argument'].split('|')[0])
                option1 = sanitize_asp(operation['argument'].split('|')[1])
                question_encoding += f"unique({i+step_padding}, {dependencies[0]}).\n"
                question_encoding += f"choose_attr({i+step_padding+1}, {i+step_padding}, {attr}, {option0}, {option1}).\n"
                step_padding += 1

        elif operation['operation'].startswith('same'):
            attr = sanitize_asp(' '.join(operation['operation'].split(' ')[1:]))
            question_encoding += f"unique({i+step_padding}, {dependencies[0]}).\n"
            question_encoding += f"unique({i+step_padding+1}, {dependencies[1]}).\n"
            question_encoding += f"two_same({i+step_padding+2}, {i+step_padding}, {i+step_padding+1}, {attr}).\n"
            step_padding += 2

        elif operation['operation'].startswith('different'):
            attr = sanitize_asp(' '.join(operation['operation'].split(' ')[1:]))
            question_encoding += f"unique({i+step_padding}, {dependencies[0]}).\n"
            question_encoding += f"unique({i+step_padding+1}, {dependencies[1]}).\n"
            question_encoding += f"two_different({i+step_padding+2}, {i+step_padding}, {i+step_padding+1}, {attr}).\n"
            step_padding += 2 
        
        ops_map[i] = i + step_padding

    question_encoding += f"end({len(question['semantic'])+step_padding-1})."
    return question_encoding