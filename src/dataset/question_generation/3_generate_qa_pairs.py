import argparse
import os
import json
from tqdm import tqdm
from pydantic import BaseModel
from typing import List
from openai import OpenAI
import concurrent.futures

client = OpenAI()

SYSTEM_PROMPT = """
You are an expert in biomedical video analysis and medical diagnosis. Your task is to generate high-quality multiple-choice question-answer (Q/A) pairs for a biomedical video benchmark dataset.

### Instructions:
1. **Visual Evidence Required:**
   - The answer must require watching the video and be based on visual details observed in the video, not general medical knowledge.
   - Avoid questions that could be answered without analyzing the video.

2. **Focus Areas:**
   - Diagnoses: Identifying visible pathologies or abnormalities.
   - Procedures: Describing surgical techniques or medical interventions.
   - Medical Findings: Observations like tissue or organ abnormalities.
   - Anatomical Features: Describing structural changes or specific parts visible in the video.

3. **Multiple-Choice Format:**
   - Generate closed-ended questions with 4 distinct options (labeled A, B, C, D).
   - Ensure one of the options is the correct answer and clearly indicate which one it is.
   - Make sure the other 3 options are plausible but incorrect.

4. **Clarity and Precision:**
   - Ensure questions are clear, direct, and focused on a single aspect.
   - Avoid compound or multi-part questions.
   - Keep answers concise (few words), avoiding unnecessary elaboration.

5. **Strict Visual Grounding:**
   - The questions must be strictly based on the visual content of the video.
   - Do not introduce hallucinated or unrelated information.

6. **Avoid Triviality:**
   - Avoid generic or overly simple questions (e.g., "Is there a heart in the video?").
   - Ensure questions are medically relevant and non-obvious.

### Schema:
[
    {
        "question": <question>,
        "option_A": <option A>,
        "option_B": <option B>,
        "option_C": <option C>,
        "option_D": <option D>,
        "correct_option": <correct option label>,  # One of "A", "B", "C", "D"
        "correct_answer": <correct answer>
    },
    ...
]

### Example Output:

Caption: "The ejection fraction of the heart is 55% as measured during the echocardiogram."
Output: [
    {
        "question": "What is the ejection fraction of the heart as shown in the video?",
        "option_A": "45%",
        "option_B": "55%",
        "option_C": "65%",
        "option_D": "75%",
        "correct_option": "B",
        "correct_answer": "55%"
    }
]

Caption: "Lung abnormalities are detected in the CT scan, including areas of fibrosis and fluid accumulation."
Output: [
    {
        "question": "What abnormalities are detected in the lungs as shown in the video?",
        "option_A": "Emphysema and scarring.",
        "option_B": "Fibrosis and fluid accumulation.",
        "option_C": "Tumor and pleural effusion.",
        "option_D": "Collapsed lung and abscess.",
        "correct_option": "B",
        "correct_answer": "Fibrosis and fluid accumulation."
    }
]

Caption: "An endoscopic view shows an ulcer in the stomach lining with visible inflammation."
Output: [
    {
        "question": "What medical condition is shown in the video?",
        "option_A": "Gastric polyp.",
        "option_B": "Ulcer with inflammation.",
        "option_C": "Cancerous growth.",
        "option_D": "Diverticulitis.",
        "correct_option": "B",
        "correct_answer": "Ulcer with inflammation."
    }
]

Caption: "A biopsy image shows a malignant tumor with irregular borders and increased vascularization."
Output: [
    {
        "question": "What medical condition is shown in the video?",
        "option_A": "Benign cyst.",
        "option_B": "Inflamed polyp.",
        "option_C": "Malignant tumor.",
        "option_D": "Fibrotic scar.",
        "correct_option": "C",
        "correct_answer": "Malignant tumor."
    }
]
"""

class QAPair(BaseModel):
    question: str
    option_A: str
    option_B: str
    option_C: str
    option_D: str
    correct_option: str
    correct_answer: str

class QAResponse(BaseModel):
    qa_pairs: List[QAPair]

# Function to process a single caption and generate Q/A pairs
def process_single_qa_pairs(sample):
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sample['cleaned_caption']}
            ],
            response_format=QAResponse,
        )

        qa_data = completion.choices[0].message.parsed
        # Create output data structure - retain all original keys and add qa_pairs
        output_data = {
            **sample,  # Include all original keys
            "qa_pairs": [qa_pair.dict() for qa_pair in qa_data.qa_pairs]
        }
        return output_data

    except Exception as e:
        print(f"Error processing video {sample['video_id']}: {str(e)}")
        return None  # Return None if an error occurs

# Parallel processing function
def process_qa_pairs_parallel(input_path, output_path, max_workers=8):
    # Read input JSONL file
    with open(input_path, 'r') as f:
        samples = [json.loads(line) for line in f]

    # Ensure output path exists and clear existing file if necessary
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # if os.path.exists(output_path):
    #     os.remove(output_path)

    # Process captions in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        with open(output_path, 'a') as out_f:
            futures = {
                executor.submit(process_single_qa_pairs, sample): sample
                for sample in samples
            }
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(samples), desc="Processing captions"):
                result = future.result()
                json.dump(result, out_f)
                out_f.write('\n')
                out_f.flush()  # Ensure it's written to disk

# Main function
def main(args):
    input_path = args.input_path
    output_path = args.output_path
    max_workers = args.max_workers

    print("Input path:", input_path)
    print("Output path:", output_path)
    print("Max workers:", max_workers)

    # Run parallel processing
    process_qa_pairs_parallel(input_path, output_path, max_workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input JSONL file containing cleaned captions')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to file to save Q/A pairs (not just directory)')
    parser.add_argument('--max_workers', type=int, default=16,
                        help='Maximum number of parallel threads')
    args = parser.parse_args()

    main(args)