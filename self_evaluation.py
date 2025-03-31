import json
import os
import faiss
import openai
import pymupdf
import numpy as np

import embeddingsIE

# === Configuration ===
studyarea = embeddingsIE.studyarea
REPORTS_FOLDER = os.path.join(studyarea, 'reports')
QUESTIONS_FILE = "aquiferie_insight_prompts.txt"  # Text file containing questions (one per line)
EMBEDDING_MODEL = "text-embedding-3-large"

client = openai.OpenAI(api_key='sk-proj-UsTt8Y55T1eFBUvULb-lazHiCRdX'
                               '-9VUNJa3UAxObyJYREltqifJPK3btItM7bLqm40AfQ1JQfT3BlbkFJNZza_UXf'
                               '-xYhSg0xrR5WxF1ZsYnDz44irH3Sxyfm9kh9a-vrj3c4PcwymFfw7Ivi1SO26mVZMA')


def build_prompt(report_text, question, answer):
    """Creates a consistent evaluation prompt for both models."""
    return f"""
You are evaluating whether an AI-generated answer is accurate based on a scientific report.

Report:
{report_text}

Question: {question}
Original Answer: {answer}

Evaluate the answer:
1. Is the answer correct based on the report? (Yes/No)
2. If yes, concisely reference any specific quotes, content, or sections from the report that support the answer.
3. If not, what is wrong or missing? Be concise.
4. Provide a corrected answer if possible, and reference any specific quotes or content from the report. Be concise.

Respond in this format:
Correct?: <Yes/No>
Explanation: <...>
Corrected Answer (if needed): <...>
"""


def evaluate_answer(model_name, relevant_text, question, answer):
    prompt = build_prompt(relevant_text, question, answer)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def run_side_by_side(pdf_filename, question, answer):
    print(f"\nEvaluating: {pdf_filename}\nQuestion: {question}\nAnswer: {answer}\n")

    relevant_text = embeddingsIE.search_relevant_section(question, pdf_filename)[0]
    o1_eval = evaluate_answer("o1", relevant_text, question, answer)
    o3mini_eval = evaluate_answer("o3-mini", relevant_text, question, answer)

    print("\n================== O1 Evaluation ==================\n")
    print(o1_eval)

    print("\n================ O3-mini Evaluation =================\n")
    print(o3mini_eval)

    return '\n\n'.join(['o1 Evaluation:', o1_eval, 'o3-mini Evaluation:', o3mini_eval])


def main():
    # Replace this with a real test case from your data
    pdf_filename = "58_p0195_p0208.pdf"
    question = "What is the name of the aquifer(s) focused on in the study?"
    original_answer = "From the formations described (Arroyo Ojito, Cerro Conejo, Zia, etc.), all of which are " \
                      "subdivisions of the Santa Fe Group in this area, it appears the study primarily focuses on the " \
                      "Santa Fe Group aquifer system."

    run_side_by_side(pdf_filename, question, original_answer)


# === EXAMPLE INPUT ===
if __name__ == "__main__":
    main()




