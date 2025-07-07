import os
import json
import clingo

ENCODING_PATH = "preprompt/theory/GQA_count/perfect_theory.lp"  # your base ASP encoding file
DATASET_PATH = "dataset/GQA_count/modified_train_suite_count.json"
CONFIRMED_OUTPUT = "dataset/GQA_count/confirmed_connected_examples.json"

def run_clingo_with_input(theory_path, facts_str):
    """Run Clingo on the theory and input facts, return answer atoms."""
    ctl = clingo.Control()
    ctl.load(theory_path)
    ctl.add("base", [], facts_str)
    ctl.ground([("base", [])])

    result = []

    def on_model(model):
        atoms = model.symbols(shown=True)
        result.extend(str(atom) for atom in atoms)

    ctl.solve(on_model=on_model)
    return result

def convert_to_facts(scene_str, question_str):
    """Combine scene and question into a single string of facts."""
    return scene_str + "\n" + question_str + "\n"

def evaluate_and_confirm(dataset_path, theory_path, output_path):
    with open(dataset_path, "r") as f:
        data = json.load(f)

    confirmed = []

    for i, example in enumerate(data):
        input_facts = convert_to_facts(example["scene"], example["question"])
        answer_atoms = run_clingo_with_input(theory_path, input_facts)

        # Extract all ans(X) atoms
        predicted = [atom[4:-1] for atom in answer_atoms if atom.startswith("ans(")]

        ground_truth = example["answer"]

        print(f"Example {i+1}:")
        print("  Ground truth:", ground_truth)
        print("  Predicted   :", predicted)

        if ground_truth in predicted:
            confirmed.append(example)

    # Save confirmed examples
    with open(output_path, "w") as f:
        json.dump(confirmed, f, indent=2)

    print(f"\n✅ Confirmed {len(confirmed)} out of {len(data)} examples as satisfiable.")
    print(f"Saved to: {output_path}")

# Run evaluator
if __name__ == "__main__":
    evaluate_and_confirm(DATASET_PATH, ENCODING_PATH, CONFIRMED_OUTPUT)
