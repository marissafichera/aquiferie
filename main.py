import os
import requests
import pdfplumber
import pandas as pd
import openai
import time
import sys
import pymupdf  # PyMuPDF

# Configuration
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STUDYAREA = 'SanLuisBasin'
REPORTS_CSV = f"{STUDYAREA}_aquiferie_report_links.csv"  # CSV file containing report URLs
DOWNLOAD_DIR = "reports"
OUTPUT_CSV = f"{STUDYAREA}_aquifer_insights.csv"

# Ensure download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_pdf(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f"Downloaded: {filename}")
        if not os.path.getsize(filename):  # Check if file is empty
            print(f"Warning: {filename} is empty.")
    else:
        print(f"Failed to download: {url}")


# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text += page.extract_text() + "\n"
#     return text


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pymupdf.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None
    return text



def ask_openai(text, question):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4o",  # Use "gpt-3.5-turbo" if needed
        messages=[
            {"role": "system", "content": "You are an expert in analyzing aquifer reports. Please answer the following questions as concisely as possible."},
            {"role": "user", "content": f"Here is a report:\n\n{text}\n\n{question}"}
        ],
        temperature=0.5
    )

    return response.choices[0].message.content


def load_questions(question_file):
    """Load descriptors and questions from a CSV file."""
    df = pd.read_csv(question_file)
    return {row["Description"]: row["Question"] for _, row in df.iterrows()}


def process_reports():
    df = pd.read_csv(REPORTS_CSV)
    questions = load_questions('aquiferie_insight_prompts.csv')
    results = []

    for index, row in df.iterrows():
        url = row["report_url"]
        report_name = os.path.join(DOWNLOAD_DIR, f"report_{index}.pdf")

        download_pdf(url, report_name)
        text = extract_text_from_pdf(report_name)

        if text is not None:
            insights = {"Report Name": report_name}

            for descriptor, question in questions.items():
                print(f'{report_name}, Asking OpenAI: {question}')
                insights[descriptor] = ask_openai(text, question)
                time.sleep(30)

            results.append(insights)
            pd.DataFrame(results).T.to_csv(OUTPUT_CSV)

    pd.DataFrame(results).T.to_csv(OUTPUT_CSV)

    print("Processing complete. Data saved to", OUTPUT_CSV)


if __name__ == "__main__":
    process_reports()
