import json
import os
import faiss
import openai
import pymupdf
import numpy as np

import embeddingsIE

# === Configuration ===
studyarea = 'AlbuquerqueBasin'
REPORTS_FOLDER = os.path.join(studyarea, 'reports_test')
QUESTIONS_FILE = "aquiferie_insight_prompts.txt"  # Text file containing questions (one per line)
EMBEDDING_MODEL = "text-embedding-3-large"

client = openai.OpenAI(api_key='sk-proj-UsTt8Y55T1eFBUvULb-lazHiCRdX'
                               '-9VUNJa3UAxObyJYREltqifJPK3btItM7bLqm40AfQ1JQfT3BlbkFJNZza_UXf'
                               '-xYhSg0xrR5WxF1ZsYnDz44irH3Sxyfm9kh9a-vrj3c4PcwymFfw7Ivi1SO26mVZMA')


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file using PyMuPDF (same as your main script)."""
    pdf_path = os.path.join(REPORTS_FOLDER, pdf_file)
    text = ""
    try:
        with pymupdf.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return "Error extracting text from PDF."
    return text


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
2. If yes, reference any specific quotes, content, or sections from the report that support the answer.
3. If not, what is wrong or missing?
4. Provide a corrected answer if possible, and reference any specific quotes or content from the report.

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


def get_embedding(text):
    """Generate an embedding for a given text."""

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    embedding = np.array(response.data[0].embedding)

    # Debug: Print a sample embedding for verification
    # print(f"Sample embedding (first 5 values): {embedding[:5]}")
    return embedding


# def search_relevant_section(query, report_url, top_k=5):
#     """Find the most relevant report section based on a query."""
#     index = faiss.read_index(f"{studyarea}/{report_url.split('/')[-1]}_embeddings.index")
#     with open(f"{studyarea}/{report_url.split('/')[-1]}_metadata.json", "r", encoding="utf-8") as f:
#         metadata = json.load(f)
#
#     query_embedding = get_embedding(query).astype('float32').reshape(1, -1)
#
#     # distances, indices = index.search(query_embedding, top_k)
#     distances, indices = index.search(query_embedding, top_k)
#
#     results = [metadata[i] for i in indices[0]]
#     return results  # Return top_k sections instead of just one


def run_side_by_side(pdf_filename, question, answer):
    print(f"\nEvaluating: {pdf_filename}\nQuestion: {question}\nAnswer: {answer}\n")

    relevant_text = embeddingsIE.search_relevant_section(question, pdf_filename)[0]
    o1_eval = evaluate_answer("o1", relevant_text, question, answer)
    o3mini_eval = evaluate_answer("o3-mini", relevant_text, question, answer)

    print("\n================== O1 Evaluation ==================\n")
    print(o1_eval)

    print("\n================ O3-mini Evaluation =================\n")
    print(o3mini_eval)

    return '\n\n'.join([o1_eval, o3mini_eval])


def main():
    # Replace this with a real test case from your data
    pdf_filename = "58_p0195_p0208.pdf"  # Must exist in AlbuquerqueBasin/reports_test
    question = "What is the name of the aquifer(s) focused on in the study?"
    original_answer = "From the formations described (Arroyo Ojito, Cerro Conejo, Zia, etc.), all of which are " \
                      "subdivisions of the Santa Fe Group in this area, it appears the study primarily focuses on the " \
                      "Santa Fe Group aquifer system."

    run_side_by_side(pdf_filename, question, original_answer)


# === EXAMPLE INPUT ===
if __name__ == "__main__":
    main()




