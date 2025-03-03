import os
import requests
import pdfplumber
import pandas as pd
import openai

# Configuration
OPENAI_API_KEY = "your_openai_api_key"  # Replace with your API key
REPORTS_CSV = "report_links.csv"  # CSV file containing report URLs
DOWNLOAD_DIR = "reports"
OUTPUT_CSV = "aquifer_data.csv"

# Ensure download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_pdf(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download: {url}")


def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


def ask_openai(text, question):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in hydrology analyzing aquifer reports."},
            {"role": "user", "content": f"Extract the following information: {question}\n\n{text}"}
        ],
        api_key=OPENAI_API_KEY,
    )
    return response["choices"][0]["message"]["content"].strip()


def process_reports():
    df = pd.read_csv(REPORTS_CSV)
    results = []

    for index, row in df.iterrows():
        url = row["report_url"]
        report_name = os.path.join(DOWNLOAD_DIR, f"report_{index}.pdf")

        download_pdf(url, report_name)
        text = extract_text_from_pdf(report_name)

        insights = {
            "Report Name": report_name,
            "Water Table Depth": ask_openai(text, "What is the water table depth in this report?"),
            "Recharge Rate": ask_openai(text, "What is the recharge rate of the aquifer?"),
            "Contaminants": ask_openai(text, "What contaminants are present in the aquifer?"),
        }
        results.append(insights)

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print("Processing complete. Data saved to", OUTPUT_CSV)


if __name__ == "__main__":
    process_reports()
