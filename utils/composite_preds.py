import os
import json
import random
import re

DATASET_DIR = "dataset"

def load_train_json(dataset_name):
    """Load the train_suite.json for a given dataset."""
    path = os.path.join(DATASET_DIR, dataset_name, "train_suite.json")
    with open(path, "r") as f:
        data = json.load(f)
    return data

def save_examples(dataset_name, examples, filename="modified_train_suite.json"):
    """Save modified examples to a new JSON file in the dataset folder."""
    path = os.path.join(DATASET_DIR, dataset_name, filename)
    with open(path, "w") as f:
        json.dump(examples, f, indent=2)

def extract_attrs(scene):
    """Returns a mapping from object id to its attributes."""
    objects = {}
    for line in scene.splitlines():
        match = re.match(r'has_attr\((\d+), ([^,]+), ([^)]+)\)', line)
        if match:
            obj_id, attr, val = match.groups()
            if obj_id not in objects:
                objects[obj_id] = {}
            objects[obj_id][attr] = val
    return objects

def extract_rels(scene, rel_type):
    """Returns a list of (subject, object) for the given relation."""
    return re.findall(rf'has_rel\((\d+), {rel_type}, (\d+)\)', scene)

def extract_relations(scene):
    """Returns a list of (subject, relation, object) triples."""
    return re.findall(r'has_rel\((\d+), ([^,]+), (\d+)\)', scene)

def create_connected_question(example, positive_probability=0.5):
    """Create a modified example using a recursive 'connected' predicate.
       Requires path length ≥ 2 for positive examples."""
    scene = example["scene"]
    objects = extract_attrs(scene)
    relations = extract_relations(scene)

    # Build undirected graph
    graph = {}
    for a, rel, b in relations:
        if rel == "connects":
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)

    all_ids = list(objects.keys())
    if len(all_ids) < 2:
        return None

    # Compute all reachable paths with length
    def find_reachable_with_distance(start):
        visited = {}
        queue = [(start, 0)]
        while queue:
            node, dist = queue.pop(0)
            if node in visited:
                continue
            visited[node] = dist
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return visited  # map: node -> distance

    random.shuffle(all_ids)
    for i in range(len(all_ids)):
        for j in range(i + 1, len(all_ids)):
            src = all_ids[i]
            tgt = all_ids[j]
            if src == tgt:
                continue

            src_attrs = objects.get(src, {})
            tgt_attrs = objects.get(tgt, {})
            if "class" not in src_attrs or "class" not in tgt_attrs:
                continue

            class1 = src_attrs["class"]
            class2 = tgt_attrs["class"]

            # check reachability and path length
            reachable = find_reachable_with_distance(src)
            path_length = reachable.get(tgt, None)

            make_positive = random.random() < positive_probability

            if make_positive and path_length is not None and path_length >= 2:
                # long enough path, no need to inject
                answer = "yes"
                new_scene = scene

            elif make_positive and (path_length is None or path_length < 2):
                # forcibly inject path: src -- mid -- tgt
                # pick an intermediate object
                unused_ids = [x for x in all_ids if x != src and x != tgt]
                if not unused_ids:
                    continue
                mid = random.choice(unused_ids)
                lines = scene.splitlines()
                lines.append(f"has_rel({src}, connects, {mid}).")
                lines.append(f"has_rel({mid}, connects, {tgt}).")
                new_scene = "\n".join(lines)
                answer = "yes"

            elif not make_positive and (path_length is None):
                answer = "no"
                new_scene = scene
            else:
                # skip if we can't make a valid negative example (e.g., already connected)
                continue

            question = "\n".join([
                "scene(0).",
                f"select(1, 0, {class1}).",
                f"select(2, 0, {class2}).",
                "connected(3, 1, 2).",
                "end(3)."
            ])

            return {
                "scene": new_scene,
                "question": question,
                "answer": answer,
                "modified": True,
                "connect_type": "positive" if answer == "yes" else "negative",
                "path_length": path_length if path_length is not None else ("injected" if answer == "yes" else "N/A")
            }

    return None

def create_isolated_question(example, positive_probability=0.5):
    """Create a modified example using the 'isolated/2' predicate with negation.
       If generating a positive example, ensures the object has no relations."""
    scene = example["scene"]
    objects = extract_attrs(scene)
    relations = extract_relations(scene)

    all_ids = list(objects.keys())
    if not all_ids:
        return None

    related_ids = set()
    for subj, rel, obj in relations:
        related_ids.add(subj)
        related_ids.add(obj)

    isolated_ids = [oid for oid in all_ids if oid not in related_ids]
    connected_ids = [oid for oid in all_ids if oid in related_ids]

    make_positive = random.random() < positive_probability
    candidates = isolated_ids if make_positive else connected_ids
    random.shuffle(candidates)

    for oid in candidates:
        attrs = objects.get(oid, {})
        if "class" not in attrs:
            continue
        obj_class = attrs["class"]

        lines = scene.splitlines()

        if make_positive:
            # remove all relations involving this object
            lines = [
                line for line in lines
                if not re.match(rf"has_rel\({oid},", line)
                and not re.match(rf"has_rel\([^,]+, [^,]+, {oid}\)", line)
            ]

        new_scene = "\n".join(lines)

        question = "\n".join([
            "scene(0).",
            f"select(1, 0, {obj_class}).",
            "isolated(2, 1).",
            "end(2)."
        ])

        return {
            "scene": new_scene,
            "question": question,
            "answer": "yes" if make_positive else "no",
            "modified": True,
            "isolation_type": "positive" if make_positive else "negative",
            "object_id": oid
        }

    return None

def create_count_class_question(example):
    """Create a modified example that asks how many objects of a given class exist."""
    scene = example["scene"]
    objects = extract_attrs(scene)

    class_counts = {}
    for obj_id, attrs in objects.items():
        if "class" in attrs:
            cls = attrs["class"]
            class_counts.setdefault(cls, []).append(obj_id)

    # Filter classes with at least two objects
    candidates = {cls: ids for cls, ids in class_counts.items() if len(ids) >= 2}
    if not candidates:
        return None

    chosen_class = random.choice(list(candidates.keys()))
    count = len(candidates[chosen_class])

    question = "\n".join([
        "scene(0).",
        f"select(1, 0, {chosen_class}).",
        f"count_class(2, 1, {chosen_class}).",
        "end(2)."
    ])

    return {
        "scene": scene,
        "question": question,
        "answer": str(count),
        "modified": True,
        "counted_class": chosen_class,
        "count_value": count
    }


def create_inbetween_question(example):
    """Try to create a modified example with an inbetween question for any attribute."""
    scene = example["scene"]
    objects = extract_attrs(scene)
    # if len(objects.keys()) != 3:
    #     return None
    left_of = extract_rels(scene, "to_the_left_of")
    right_of = extract_rels(scene, "to_the_right_of")

    left_of_set = set(left_of)
    right_of_set = set(right_of)

    all_ids = set(objects.keys())

    for x in all_ids:
        for a in all_ids:
            for b in all_ids:
                if x != a and x != b and a != b:
                    if (x, b) in left_of_set and (x, a) in right_of_set:
                        attrs_x = objects.get(x, {})
                        attrs_a = objects.get(a, {})
                        attrs_b = objects.get(b, {})

                        if "class" in attrs_a and "class" in attrs_b:
                            if a == b or attrs_a["class"] == attrs_b["class"] and attrs_a.get("name") == attrs_b.get("name"):
                                continue  # Avoid ambiguous same-class/name cases

                            queryable_attrs = [
                                attr for attr in attrs_x
                                if attr not in {"class", "hposition", "vposition"}
                            ]
                            if not queryable_attrs:
                                continue

                            attr_to_query = random.choice(queryable_attrs)
                            value = attrs_x[attr_to_query]
                            class_a = attrs_a["class"]
                            class_b = attrs_b["class"]

                            new_question = "\n".join([
                                "scene(0).",
                                f"select(1, 0, {class_a}).",
                                f"select(2, 0, {class_b}).",
                                f"inbetween(3, 1, 0, 2).",
                                f"query(4, 3, {attr_to_query}).",
                                "end(4)."
                            ])

                            return {
                                "scene": scene,
                                "question": new_question,
                                "answer": value,
                                "modified": True
                            }
    return None

def collect_modified_examples(dataset_name, required_count=100):
    """Collect a given number of modified examples with inbetween questions."""
    all_data = load_train_json(dataset_name)
    modified = []

    for example in all_data:
        # od_example = create_inbetween_question(example)
        od_example = create_count_class_question(example)
        if od_example:
            modified.append(od_example)
        if len(modified) >= required_count:
            break

    print(f"Collected {len(modified)} modified examples.")
    return modified

# Main runner
if __name__ == "__main__":
    dataset = "GQA"  # Change to "CLEVR" or "CLEGR" as needed
    modified_examples = collect_modified_examples(dataset, required_count=1000)
    save_examples(dataset, modified_examples, filename="modified_train_suite_count.json")
