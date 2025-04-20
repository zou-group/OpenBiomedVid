import os
import json
import time
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel
import concurrent.futures  # For parallel processing

# Setup OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

class EvaluationScore(BaseModel):
    score: int

class GPTScorer:
    """
    Class to evaluate model predictions using GPT.
    """
    @staticmethod
    def generate_evaluation_prompt(question, gold_answer, pred_answer):
        return [
            {
                "role": "system",
                "content": """
        You are an expert evaluator for medical image and video understanding. Your task is to compare a gold standard answer to a predicted answer and determine the similarity score.
        Ignore minor differences in formatting and verbosity. Focus on whether the predicted answer conveys the same essential meaning as the gold answer.
        Instructions:   
        Assign a score of 0 or 1 based on how similar the prediction is to the gold answer:
        1: Correct - The prediction is mostly correct and captures the essential meaning, with minor errors being tolerable.
        0: Incorrect - The prediction is largely incorrect or unrelated.

        You must respond with ONLY a single integer number: 0 or 1.
        """
            },
            {
                "role": "user",
                "content": f"""
        Question: {question}
        Gold Answer: {gold_answer}
        Predicted Answer: {pred_answer}
        """
            }
        ]

    @staticmethod
    def fetch_gpt_score(prompt, model="gpt-4o"):
        while True:
            try:
                completion = client.beta.chat.completions.parse(
                    model=model,
                    messages=prompt,
                    response_format=EvaluationScore,
                )
                score = completion.choices[0].message.parsed.score

                if not (1 <= score <= 5):
                    raise ValueError(f"Score {score} is out of range (1-5)")

                return score
            except Exception as e:
                print(f"Error: {e}. Retrying in 10 seconds...")
                time.sleep(10)

def process_single_sample(sample):
    """
    Processes a single sample to get GPT score.
    """
    if not all(k in sample for k in ["question", "answer", "pred"]):
        print(f"Skipping sample: missing required fields")
        return None

    try:
        prompt = GPTScorer.generate_evaluation_prompt(
            sample["question"],
            sample["answer"],
            sample["pred"]
        )
        score = GPTScorer.fetch_gpt_score(prompt)
        sample["gpt_score"] = score
        return sample
    except Exception as e:
        print(f"Error processing sample: {str(e)}")
        return None

def evaluate_predictions_parallel(input_path, output_file, max_workers=64):
    """
    Evaluates predictions using GPT in parallel and saves the results.
    """
    # Read all lines first to get total count
    with open(input_path, 'r') as f:
        lines = list(f)

    # Clear output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Keep track of total score for final average
    total_score = 0
    count = 0

    # Process samples in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        with open(output_file, 'w') as out_f:
            futures = {
                executor.submit(process_single_sample, json.loads(line)): line
                for line in lines
            }
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(lines), desc="Evaluating predictions"):
                result = future.result()
                if result:
                    total_score += result.get("gpt_score", 0)
                    count += 1
                    json.dump(result, out_f)
                    out_f.write('\n')
                    out_f.flush()  # Ensure it's written to disk


    # Print final summary
    print("\nEvaluation Summary:")
    print(f"Total processed samples: {count}")
    print(f"Final average score: {(total_score / count) * 100:.2f}%")
    print(f"Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save evaluation results")
    parser.add_argument("--max_workers", type=int, default=32, help="Maximum number of parallel threads")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    evaluate_predictions_parallel(args.input_path, args.output_path, args.max_workers)

if __name__ == "__main__":
    main()