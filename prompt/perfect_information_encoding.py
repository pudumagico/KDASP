import json 
from question_encoding import encode_question
from utils import sanitize_asp

with open('./dataset/GQA/metadata/gqa_all_class.json') as f:
    categories = json.load(f)
class_to_category = {}
for category, classes in categories.items():
    for c in classes:
        if c not in class_to_category:
            class_to_category[c] = [category]
        else:
            class_to_category[c].append(category)

with open('./dataset/GQA/metadata/gqa_all_attribute.json') as f:
    attributes = json.load(f)
value_to_attribute = {}
for attribute, values in attributes.items():
    for v in values:
        if v not in value_to_attribute:
            value_to_attribute[v] = [attribute]
        else:
            value_to_attribute[v].append(attribute)


def encode_sample(question):
    return (encode_scene(question['sceneGraph']), encode_question(question))


def encode_scene(scene_graph):
    scene_encoding = ""
    for oid, object in scene_graph['objects'].items():
        scene_encoding += f"object({oid}).\n"

        scene_encoding += f"has_attribute({oid}, class, {sanitize_asp(object['name'])}).\n"
        scene_encoding += f"has_attribute({oid}, name, {sanitize_asp(object['name'])}).\n"
       
        for category in class_to_category.get(sanitize_asp(object['name']), []):
            scene_encoding += f"has_attribute({oid}, class, {category}).\n"    

        for value in object['attributes']:
            if value in value_to_attribute:
                for att in value_to_attribute[value]:
                    scene_encoding += f"has_attribute({oid}, {sanitize_asp(att)}, {sanitize_asp(value)}).\n"
            else:
                scene_encoding += f"has_attribute({oid}, any, {sanitize_asp(value)}).\n"

        if (object['x'] + object['w']/2) > scene_graph['width']/3*2:
            scene_encoding += f"has_attribute({oid}, hposition, right).\n"
        elif (object['x'] + object['w']/2) > scene_graph['width']/3:
            scene_encoding += f"has_attribute({oid}, hposition, middle).\n"
        else:
            scene_encoding += f"has_attribute({oid}, hposition, left).\n"

        if (object['y'] + object['h']/2) > scene_graph['height']/3*2:
            scene_encoding += f"has_attribute({oid}, vposition, bottom).\n"
        if (object['y'] + object['h']/2) > scene_graph['height']/3:
            scene_encoding += f"has_attribute({oid}, vposition, middle).\n"
        else:
            scene_encoding += f"has_attribute({oid}, vposition, top).\n"

        for rel in object['relations']:
            scene_encoding += f"has_relation({oid}, {sanitize_asp(rel['name'])}, {sanitize_asp(rel['object'])}).\n"

    return scene_encoding


if __name__ == '__main__':
    question_and_scene_graph_file = './scene_question_balanced.json'
    target_folder = './dataset/encoded_questions'

    with open(question_and_scene_graph_file) as f:
        questions = json.load(f)
    
    for qid, question in questions.items():
        print(f"encoding question {qid}")
        scene_encoding, question_encoding = encode_sample(question)
        print(scene_encoding, question_encoding)
        # with open(f"{target_folder}/{qid}.lp", "w") as f:
        #     f.write("% ------ scene encoding ------\n")
        #     f.write(scene_encoding)
        #     f.write("\n% ------ question encoding ------\n")
        #     f.write(question_encoding)