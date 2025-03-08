import os
import requests
import pdfplumber
import openai
import numpy as np
import faiss
import json
import pandas as pd

# Configuration
REPORTS_CSV = "report_links.csv"  # CSV file containing report URLs
DOWNLOAD_DIR = "reports"
OUTPUT_CSV = "aquifer_extracted_insights.csv"
EMBEDDING_MODEL = "text-embedding-3-small"

# Ensure download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
client = openai.OpenAI()

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
    """Extract text from a PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text

def get_embedding(text):
    """Generate an embedding for a given text."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return np.array(response.data[0].embedding)

def split_text(text, chunk_size=500):
    """Split text into sections of approximately chunk_size words."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def process_reports():
    df = pd.read_csv(REPORTS_CSV)
    reports = []
    for idx, row in df.iterrows():
        url = row["report_url"]
        pdf_filename = os.path.join(DOWNLOAD_DIR, f"report_{idx}.pdf")
        if not os.path.exists(pdf_filename):
            pdf_filename = download_pdf(url, pdf_filename)
        if pdf_filename:
            text = extract_text_from_pdf(pdf_filename)
            reports.append({"url": url, "text": text})
    print(f"Extracted text from {len(reports)} reports.")
    return reports

def generate_embeddings(reports):
    embeddings, metadata = [], []
    for report in reports:
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
    faiss.write_index(index, "report_embeddings.index")
    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print("Embeddings generated and stored.")

def search_relevant_section(query, top_k=1):
    """Find the most relevant report section based on a query."""
    index = faiss.read_index("report_embeddings.index")
    with open("metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    query_embedding = get_embedding(query).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    results = [metadata[i] for i in indices[0]]
    return results

def ask_openai(query, relevant_text, url):
    """Extract insights using OpenAI from the most relevant section."""
    prompt = f"""
    You are analyzing the following aquifer report section:

    {relevant_text}

    Answer the following query concisely: {query}

    Provide a structured JSON response like this:
    {{
      "Query": "{query}",
      "Report URL": "{url}",
      "Answer": "..."
    }}
    """
    response = client.chat.completions.create(model="gpt-4-turbo", messages=[{"role": "user", "content": prompt}], temperature=0)
    try:
        json_response = json.loads(response.choices[0].message.content)
        return json_response
    except json.JSONDecodeError:
        return {"Query": query, "Report URL": url, "Answer": "Invalid response from OpenAI"}

def extract_answers():
    questions = [
        "What is the aquifer's depth?",
        "What is the water quality?",
        "What are the recharge rates?",
        "Are there contamination risks?",
        # Add remaining 22 questions...
    ]
    results = []
    for q in questions:
        best_match = search_relevant_section(q)[0]
        results.append(ask_openai(q, best_match["Text"], best_match["URL"]))
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"Extraction complete. Data saved to {OUTPUT_CSV}.")

if __name__ == "__main__":
    reports = process_reports()
    generate_embeddings(reports)
    extract_answers()
