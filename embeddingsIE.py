import csv
import os
import requests
import pymupdf
import openai
import numpy as np
import faiss
import json
import pandas as pd
import re
import self_evaluation
import time
from openai_api_client import client






def setup_model(study_area):
    config = {'study_area': study_area,

              'questions_file': 'aquiferie_insight_prompts2.txt',
              'download_dir': 'reports',
              'output_csv': os.path.join(study_area, f'{study_area}_test.csv'),
              'embedding_model': 'text-embedding-3-large'

              }
    # Configuration
    # studyarea = ''
    #QUESTIONS_FILE = "aquiferie_insight_prompts.txt"  # Text file containing questions (one per line)
    # QUESTIONS_FILE = "bbox_question_only.txt"  # Text file containing questions (one per line)
    # DOWNLOAD_DIR = "reports_2"
    # OUTPUT_CSV = f"{studyarea}/{studyarea}_aquiferinsights_selfeval.csv"
    # EMBEDDING_MODEL = "text-embedding-3-large"
    # Ensure download directory exists
    os.makedirs(config['download_dir'], exist_ok=True)

    OUTPUT_CSV = get_versioned_filename(config['output_csv'])
    print(f"Saving results to: {OUTPUT_CSV}")

    return config


def get_versioned_filename(base_path):
    """Auto-version the output CSV file to avoid overwriting."""
    if not os.path.exists(base_path):
        return base_path

    base, ext = os.path.splitext(base_path)
    version = 1
    while True:
        new_path = f"{base}_v{version}{ext}"
        if not os.path.exists(new_path):
            return new_path
        version += 1



def download_pdf(url, filename):
    """Download PDF from a given URL."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f"Downloaded: {filename}")
        return filename
    else:
        print(f"Failed to download: {url}")
        return None


def extract_text_from_pdf(pdf_file, config):
    """Extract text from a PDF file using PyMuPDF (better handling for various PDF types)."""

    pdf_path = os.path.join(config['study_area'], config['download_dir'], pdf_file)
    text = ""
    try:
        with pymupdf.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return "Error extracting text from PDF."
    return text


def get_embedding(text, config):
    """Generate an embedding for a given text."""
    response = client.embeddings.create(model=config['embedding_model'], input=text)
    embedding = np.array(response.data[0].embedding)

    # Debug: Print a sample embedding for verification
    # print(f"Sample embedding (first 5 values): {embedding[:5]}")
    return embedding


def split_text(text, chunk_size=1200, overlap=300):
    """Split text into overlapping sections to improve retrieval."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


def process_reports(config, limit=1):
    # df = pd.read_csv(REPORTS_CSV).head(limit)  # Limit to first <limit> reports
    # df = pd.read_csv(REPORTS_CSV)


    folder_path = os.path.join(config['study_area'],config['download_dir'])
    pdfs = [file for file in os.listdir(folder_path) if file.lower().endswith(".pdf")]
    # pdfs = pdfs[0:limit]
    extracts = []

    # for idx, row in df.iterrows():
    #     url = row["report_url"]
    #     region = row['SimpleRegion']
    #     pdf_filename = os.path.join(DOWNLOAD_DIR, f"{region}_{idx}.pdf")
    #     if not os.path.exists(pdf_filename):
    #         pdf_filename = download_pdf(url, pdf_filename)
    for pdf in pdfs:
        if pdf:
            text = extract_text_from_pdf(pdf, config)
            extracts.append({"pdf": pdf, "text": text})
    print(f"Extracted text from {len(extracts)} reports.")
    return extracts


def generate_embeddings(report, config):
    studyarea = config['study_area']

    embedding_exist = f"{studyarea}/{report['pdf'].split('/')[-1]}_embeddings.index"
    if os.path.exists(embedding_exist):
        print(f'Embeddings for report {studyarea}/{report['pdf']} already exists.\nSkipping generating embeddings for {report['pdf']}.')
        return

    embeddings, metadata = [], []
    # for report in reports:
    url = report["pdf"]
    sections = split_text(report["text"])
    for section in sections:
        embedding = get_embedding(section, config)
        embeddings.append(embedding)
        metadata.append({"URL": url, "Text": section})
    embeddings_array = np.array(embeddings).astype('float32')
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    faiss.write_index(index, f"{studyarea}/{report['pdf'].split('/')[-1]}_embeddings.index")
    with open(f"{studyarea}/{report['pdf'].split('/')[-1]}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print("Embeddings generated and stored.")


def search_relevant_section(config, query, report_url, top_k=5):
    """Find the most relevant report section based on a query."""

    studyarea = config['study_area']

    index = faiss.read_index(f"{studyarea}/{report_url.split('/')[-1]}_embeddings.index")
    with open(f"{studyarea}/{report_url.split('/')[-1]}_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    query_embedding = get_embedding(query,config).astype('float32').reshape(1, -1)

    # distances, indices = index.search(query_embedding, top_k)
    distances, indices = index.search(query_embedding, top_k)

    results = [metadata[i] for i in indices[0]]
    return results  # Return top_k sections instead of just one


def ask_openai(query, relevant_text):
    """Extract insights using OpenAI from the most relevant section."""
    prompt = f"""
    You are analyzing the following aquifer report sections:

    {relevant_text}

    Answer the following query clearly and concisely. If the section does not 
    explicitly contain the answer, infer the best possible response from the available data, but keep the 
    answer concise: {query}
    """
    response = client.chat.completions.create(model="o1", messages=[{"role": "user", "content": prompt}],
                                              )

    # Log token usage
    # usage = response.usage
    # print(
    #     f"Tokens used - Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")

    results = response.choices[0].message.content
    result = re.sub(r"[*_`]", "", results)  # Remove *, _, and ` used in markdown formatting

    return result


def load_questions(config):
    """Load questions from a text file."""
    with open(config['questions_file'], "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if line.strip()]


def extract_bboxes(report, config, df_results):
    question = load_questions(config)[0]
    report_url = report['pdf']

    best_match = search_relevant_section(config, question, report_url)[0]
    answer = ask_openai(question, best_match["Text"])

    df_results.loc[df_results['Report'] == report_url, f'{question}'] = answer

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


def extract_answers(report, config):
    questions = load_questions(config)
    results = []
    # for report in reports:
    report_url = report['pdf']
    seval_results = []
    for q in questions:
        best_match = search_relevant_section(config, q, report_url)[0]
        answer = ask_openai(q, best_match["Text"])
        # results.append(answer)
        # seval = self_evaluation.run_side_by_side(config, pdf_filename=report_url, question=q, answer=answer)
        # seval_results.append(seval)
        # results.append('\n\n'.join([answer, seval]))
        print(f"\nEvaluating: {report_url}\nQuestion: {q}\nAnswer: {answer}\n")

        relevant_text = search_relevant_section(config, q, report_url)[0]
        # o1_eval = evaluate_answer("o1", relevant_text, question, answer)
        o3mini_eval = evaluate_answer("o3-mini", relevant_text, q, answer)

        # print("\n================== O1 Evaluation ==================\n")
        # print(o1_eval)

        # print("\n================ O3-mini Evaluation =================\n")
        # print(o3mini_eval)

        # return '\n\n'.join(['o1 Evaluation:', o1_eval, 'o3-mini Evaluation:', o3mini_eval])
        seval = f'o3-mini Evaluation:\n{o3mini_eval}'


        results.append(f'{answer}\n\n{seval}')

    df_results = pd.DataFrame([results], columns=questions)
    df_results['Report'] = report_url
    out = config['output_csv']
    df_results.to_csv(out, mode='a', header=not os.path.exists(out), index=False)

    print(f"Extraction complete. Data saved to {out}.")


def save_script(output_script_folder=None, script_names=None):
    if script_names is None:
        script_names = ['embeddingsIE.txt', 'self_evaluation.txt']

    for script_name in script_names:
        base_name, ext = os.path.splitext(script_name)
        output_script_path = os.path.join(output_script_folder, script_name)

        # If the file exists, append an incrementing number to avoid overwrite
        counter = 1
        while os.path.exists(output_script_path):
            new_name = f"{base_name}_{counter}{ext}"
            output_script_path = os.path.join(output_script_folder, new_name)
            counter += 1

        # Get the path of the currently running script
        # current_script = os.path.abspath(__file__)
        current_script = f'{base_name}.py'

        # Read the current script's contents
        with open(current_script, 'r', encoding='utf-8') as f:
            script_content = f.read()

        # Write the content to the new file
        with open(output_script_path, 'w') as out_f:
            out_f.write(script_content)

        print(f"Script saved as: {output_script_path}")

def do_model(config):
    timing_data = []
    total_start = time.perf_counter()

    reports = process_reports(config)
    # df_results = pd.read_csv(OUTPUT_CSV)
    for i, report in enumerate(reports):
        print(f'REPORT {i + 1} of {len(reports)}')

        start = time.perf_counter()
        generate_embeddings(report, config)
        # extract_bboxes(report, df_results)
        extract_answers(report, config)
        end = time.perf_counter()

        elapsed = end - start

        print(f"Time for report {i + 1}, {report['pdf']}: {elapsed:.2f} seconds\n")

        # Save per-report timing info
        timing_data.append({
            "Report Index": i + 1,
            "Report Name": report['pdf'],
            "Time (seconds)": round(elapsed, 2)
        })

    # df_results.to_csv(OUTPUT_CSV, index=False)

    total_end = time.perf_counter()
    total_time = total_end - total_start
    average_time = total_time / len(reports) if reports else 0
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per report: {average_time:.2f} seconds")

    # Save per-report times to CSV
    df = pd.DataFrame(timing_data)
    df.to_csv(os.path.join(config['study_area'], "report_times.csv"), index=False)

    save_script(config['study_area'])

def main():
    cfg = setup_model('test_study_area')
    do_model(cfg)


if __name__ == "__main__":
    main()

