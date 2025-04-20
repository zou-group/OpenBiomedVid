import argparse
import json
import os

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def calculate_accuracy(data):
    correct = []
    for item in data:
        if item["answer"].lower() in ["yes", "no"]:
            correct.append(item["pred"].lower().startswith(item["answer"].lower()))
        else:
            correct_score = item["pred"].startswith(item["correct_option"])
            if correct_score == 0:
                correct_score = item["pred"].lower().startswith(item["answer"].lower())
            correct.append(correct_score)
    total = len(correct)
    accuracy = sum(correct) / total if total > 0 else 0
    return accuracy

def save_results_with_score(data, output_path):
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def save_accuracy(accuracy, output_path):
    with open(output_path, 'w') as f:
        json.dump({"accuracy": accuracy}, f, indent=4)

def main(args):
    # Load data
    data = load_jsonl(args.input_file)

    # Calculate accuracy
    accuracy = calculate_accuracy(data)

    # Prepare output paths
    base_path = os.path.dirname(args.input_file)
    results_with_score_path = os.path.join(base_path, 'results_with_score.jsonl')
    accuracy_path = os.path.join(base_path, 'accuracy.json')

    # Save results with score
    save_results_with_score(data, results_with_score_path)

    # Save accuracy
    save_accuracy(accuracy, accuracy_path)
    print(f"Accuracy: {accuracy}")

    print(f"Results with score saved to: {results_with_score_path}")
    print(f"Accuracy saved to: {accuracy_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate accuracy from a JSONL file.")
    parser.add_argument('--input_file', required=True, type=str, help='Path to the input JSONL file')
    args = parser.parse_args()

    main(args)