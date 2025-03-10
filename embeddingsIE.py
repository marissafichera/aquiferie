import csv
import os
import requests
import pdfplumber
import pymupdf
import openai
import numpy as np
import faiss
import json
import pandas as pd
import re

# Configuration
studyarea = 'AlbuquerqueBasin'
REPORTS_CSV = f"{studyarea}aquiferie_report_links.csv"  # CSV file containing report URLs
QUESTIONS_FILE = "aquiferie_insight_prompts.txt"  # Text file containing questions (one per line)
DOWNLOAD_DIR = "reports"
OUTPUT_CSV = f"{studyarea}_aquifer_insights_embeddings.csv"
EMBEDDING_MODEL = "text-embedding-3-large"

# Ensure download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
client = openai.OpenAI(api_key='sk-proj-UsTt8Y55T1eFBUvULb-lazHiCRdX'
                               '-9VUNJa3UAxObyJYREltqifJPK3btItM7bLqm40AfQ1JQfT3BlbkFJNZza_UXf'
                               '-xYhSg0xrR5WxF1ZsYnDz44irH3Sxyfm9kh9a-vrj3c4PcwymFfw7Ivi1SO26mVZMA')


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


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF (better handling for various PDF types)."""
    text = ""
    try:
        with pymupdf.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return "Error extracting text from PDF."
    return text


# def get_embedding(text):
#     """Generate an embedding for a given text."""
#     response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
#     return np.array(response.data[0].embedding)

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
    # df = pd.read_csv(REPORTS_CSV).head(limit)  # Limit to first 3 reports
    df = pd.read_csv(REPORTS_CSV)

    reports = []
    for idx, row in df.iterrows():
        url = row["report_url"]
        region = row['SimpleRegion']
        pdf_filename = os.path.join(DOWNLOAD_DIR, f"{region}_{idx}.pdf")
        if not os.path.exists(pdf_filename):
            pdf_filename = download_pdf(url, pdf_filename)
        if pdf_filename:
            text = extract_text_from_pdf(pdf_filename)
            reports.append({"url": url, "text": text})
    print(f"Extracted text from {len(reports)} reports.")
    return reports


def generate_embeddings(report):
    embeddings, metadata = [], []
    # for report in reports:
    url = report["url"]
    sections = split_text(report["text"])
    for section in sections:
        embedding = get_embedding(section)
        embeddings.append(embedding)
        metadata.append({"URL": url, "Text": section})
    embeddings_array = np.array(embeddings).astype('float32')
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    faiss.write_index(index, f"{report['url'].split('/')[-1]}_embeddings.index")
    with open(f"{report['url'].split('/')[-1]}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # faiss.write_index(index, "report_embeddings.index")
    # with open("metadata.json", "w", encoding="utf-8") as f:
    #     json.dump(metadata, f, indent=4)
    print("Embeddings generated and stored.")


def search_relevant_section(query, report_url, top_k=5):
    """Find the most relevant report section based on a query."""
    index = faiss.read_index(f"{report_url.split('/')[-1]}_embeddings.index")
    with open(f"{report_url.split('/')[-1]}_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # """Find the most relevant report sections for a query."""
    # index = faiss.read_index("report_embeddings.index")
    #
    # with open("metadata.json", "r", encoding="utf-8") as f:
    #     metadata = json.load(f)

    query_embedding = get_embedding(query).astype('float32').reshape(1, -1)

    # distances, indices = index.search(query_embedding, top_k)
    distances, indices = index.search(query_embedding, top_k)

    # Debug: Print distances for each retrieved section
    # for i, dist in enumerate(distances[0]):
    #     print(f"Match {i + 1}: Distance={dist:.4f} | Text={metadata[indices[0][i]]['Text'][:300]}...")

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
    response = client.chat.completions.create(model="gpt-4-turbo", messages=[{"role": "user", "content": prompt}],
                                              temperature=0)

    # Log token usage
    # usage = response.usage
    # print(
    #     f"Tokens used - Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")

    results = response.choices[0].message.content
    result = re.sub(r"[*_`]", "", results)  # Remove *, _, and ` used in markdown formatting

    return result
    # except json.JSONDecodeError:
    #     return {"Query": query, "Answer": "Invalid response from OpenAI"}


def load_questions():
    """Load questions from a text file."""
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if line.strip()]


# def extract_answers():
#     questions = load_questions()
#     results = []
#     for q in questions:
#         best_match = search_relevant_section(q)[0]
#         json_response = ask_openai(q, best_match["Text"])
#         results.append(json_response)
#
#     df_results = pd.DataFrame(results)
#     df_results.T.to_csv(OUTPUT_CSV, index=False)
#     print(f"Extraction complete. Data saved to {OUTPUT_CSV}.")


# def extract_answers(reports):
#     questions = load_questions()
#     answers = []
#     for q in questions:
#         best_matches = search_relevant_section(q)
#
#         # Print retrieved sections for debugging
#         print(f"\nQuery: {q}")
#         # for match in best_matches:
#         #     print(f"Matched Section:\n{match['Text'][:500]}...\n")  # Show first 500 chars
#
#         answer = ask_openai(q, best_matches[0]["Text"])
#         answers.append(answer)
#
#     df_results = pd.DataFrame({'Question': questions, 'Answer': answers}, index=None)
#     df_r = df_results.T
#     df_r['Report_URL'] = reports[0]["url"]
#     df_r.to_csv(OUTPUT_CSV)
#     print(f"Extraction complete. Data saved to {OUTPUT_CSV}.")

def extract_answers(report):
    questions = load_questions()
    results = []
    # for report in reports:
    report_url = report['url']
    for q in questions:
        best_match = search_relevant_section(q, report_url)[0]
        results.append(ask_openai(q, best_match["Text"]))
    df_results = pd.DataFrame({'Question': questions, 'Answer': results}, index=None)
    df_r = df_results.T
    df_r['Report_URL'] = report_url


    # Check if the file exists
    file_exists = os.path.exists(OUTPUT_CSV)

    if not file_exists:
        df_r.to_csv(OUTPUT_CSV, mode='a', index=False)
    else:
        df_r.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

    # # Write the DataFrame to CSV
    # with open(OUTPUT_CSV, mode='a', newline='') as f:
    #     df_r.to_csv(f, header=not file_exists, index=False)
    #
    df_r.to_csv(f'backup{OUTPUT_CSV}', mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
    # # df_results.T.to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
    print(f"Extraction complete. Data saved to {OUTPUT_CSV}.")


# def extract_answers(reports):
#     """Extract answers for all reports and save in a CSV file."""
#     questions = load_questions()
#     all_results = []
#
#     for report in reports:
#         report_url = report["url"]
#         report_answers = {"Report_URL": report_url}
#
#         for q in questions:
#             best_matches = search_relevant_section(q)
#             answer = ask_openai(q, best_matches[0]["Text"])
#             report_answers[q] = answer  # Store answers per question
#
#         all_results.append(report_answers)
#
#     # Convert results to DataFrame
#     df_results = pd.DataFrame(all_results)
#
#     # Save results to CSV
#     df_results.to_csv(OUTPUT_CSV, index=False)
#     print(f"Extraction complete. Data saved to {OUTPUT_CSV}.")


if __name__ == "__main__":
    reports = process_reports()
    for report in reports:
        generate_embeddings(report)
        extract_answers(report)
