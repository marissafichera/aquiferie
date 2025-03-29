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

# Configuration
studyarea = 'AlbuquerqueBasin'
REPORTS_CSV = f"{studyarea}/{studyarea}_aquiferie_report_links.csv"  # CSV file containing report URLs
QUESTIONS_FILE = "aquiferie_insight_prompts.txt"  # Text file containing questions (one per line)
DOWNLOAD_DIR = "reports_test"
OUTPUT_CSV = f"{studyarea}/{studyarea}_aquiferinsights_selfeval_test.csv"
EMBEDDING_MODEL = "text-embedding-3-large"

# Ensure download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
client = openai.OpenAI(api_key='sk-proj-UsTt8Y55T1eFBUvULb-lazHiCRdX'
                               '-9VUNJa3UAxObyJYREltqifJPK3btItM7bLqm40AfQ1JQfT3BlbkFJNZza_UXf'
                               '-xYhSg0xrR5WxF1ZsYnDz44irH3Sxyfm9kh9a-vrj3c4PcwymFfw7Ivi1SO26mVZMA')

# check if output file exists and warn of overwrite
if os.path.exists(OUTPUT_CSV):
    print('WARNING, OUTPUT_CSV EXISTS AND WILL BE OVERWRITTEN - END SCRIPT NOW TO AVOID LOSING $$ OF AI WORK')


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


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file using PyMuPDF (better handling for various PDF types)."""
    pdf_path = os.path.join(studyarea, 'reports_test', pdf_file)
    text = ""
    try:
        with pymupdf.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return "Error extracting text from PDF."
    return text


def get_embedding(text):
    """Generate an embedding for a given text."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
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


def process_reports(limit=1):
    # df = pd.read_csv(REPORTS_CSV).head(limit)  # Limit to first <limit> reports
    # df = pd.read_csv(REPORTS_CSV)
    folder_path = os.path.join(studyarea, 'reports_test')
    pdfs = [file for file in os.listdir(folder_path) if file.lower().endswith(".pdf")]
    pdfs = pdfs[0:limit]
    extracts = []

    # for idx, row in df.iterrows():
    #     url = row["report_url"]
    #     region = row['SimpleRegion']
    #     pdf_filename = os.path.join(DOWNLOAD_DIR, f"{region}_{idx}.pdf")
    #     if not os.path.exists(pdf_filename):
    #         pdf_filename = download_pdf(url, pdf_filename)
    for pdf in pdfs:
        if pdf:
            text = extract_text_from_pdf(pdf)
            extracts.append({"pdf": pdf, "text": text})
    print(f"Extracted text from {len(extracts)} reports.")
    return extracts


def generate_embeddings(report):
    embeddings, metadata = [], []
    # for report in reports:
    url = report["pdf"]
    sections = split_text(report["text"])
    for section in sections:
        embedding = get_embedding(section)
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


def search_relevant_section(query, report_url, top_k=5):
    """Find the most relevant report section based on a query."""
    index = faiss.read_index(f"{studyarea}/{report_url.split('/')[-1]}_embeddings.index")
    with open(f"{studyarea}/{report_url.split('/')[-1]}_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    query_embedding = get_embedding(query).astype('float32').reshape(1, -1)

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


def load_questions():
    """Load questions from a text file."""
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if line.strip()]


def extract_answers(report):
    questions = load_questions()
    results = []
    # for report in reports:
    report_url = report['pdf']
    seval_results = []
    for q in questions:
        best_match = search_relevant_section(q, report_url)[0]
        answer = ask_openai(q, best_match["Text"])
        results.append(answer)
        seval = self_evaluation.run_side_by_side(pdf_filename=report_url, question=q, answer=answer)
        seval_results.append(seval)


    df_results = pd.DataFrame([results, seval_results], columns=questions)
    df_results['Report'] = report_url

    df_results.to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)

    print(f"Extraction complete. Data saved to {OUTPUT_CSV}.")


if __name__ == "__main__":
    reports = process_reports()
    for report in reports:
        generate_embeddings(report)
        extract_answers(report)