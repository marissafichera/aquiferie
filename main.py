import csv
import io
import os
import pickle
import pprint

import requests
import pdfplumber
import pandas as pd
import openai
import time
import sys
import pymupdf  # PyMuPDF
import json
import os

# Configuration
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# STUDYAREA = 'TularosaBasin'
# REPORTS_CSV = os.path.join(STUDYAREA, f"{STUDYAREA}_aquiferie_report_links.csv")  # CSV file containing report URLs
DOWNLOAD_DIR = "reports"


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


def get_pickled_response(path):
    if os.path.isfile(path):
        with open(path, 'rb') as rf:
            return pickle.load(rf)

def ask_openai(report_text, report_url):
    # path = 'response.pickle'
    # response = get_pickled_response(path)
    response = None

    if response is None:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        prompt = f"""
            You are analyzing the following aquifer report:
    
            {report_text}
    
            Answer the following questions concisely:
            1.	What is the web hyperlink to the study?
            2.	What is the name of the aquifer(s) focused on in the study?
            3.	Did the study define the aquifer’s boundaries in 2D, 3D, or neither?
            4.	What data was used to define the aquifer’s boundaries?
            5.	Did the study publish a digital product that defined the aquifer’s boundaries?
            6.	Did the study define the hydrostratigraphy of the subsurface?
            7.	What data was used to define the hydrostratigraphy?
            8.	What were the hydrostratigraphy data sources?
            9.	Did the study define subsurface structural controls on groundwater flow?
            10.	What data was used to define the structural controls?
            11.	What were the associated data sources?
            12.	Did the study define recharge and discharge zones?
            13.	What data was used to define recharge and discharge zones?
            14.	Did the study quantify recharge and discharge rates or amounts?
            15.	What data was used to quantify recharge and discharge rates or amounts?
            16.	Did the study produce a water table elevation map or a depth to water map?
            17.	What data was used to create the water table elevation map or depth to water map?
            18.	Was the data collected for the study or compiled from other sources? 
            19.	What were the data sources used to create the water table elevation map or depth to water map?
            20.	Does the study include water quality data?
            21.	Was the data collected for the study or compiled from other sources? 
            22.	What were the water quality data sources?
            23.	Does the study report hydrogeologic properties such as porosity, permeability, storage coefficients, 
            specific storage, specific yield, hydraulic conductivity, and/or transmissivity?
            24.	Were properties determined for the study or compiled from other sources?
            25.	What were the hydrogeologic properties data sources?
            26.	What are the future work recommendations for this study?
    
    
            Provide the answers in a structured JSON format like this:
            {{
                'Report URL': f'{report_url},
                'Aquifer(s) name': '...',
                'Aquifer boundary definition': '...',
                'Data used to define aquifer boundary': '...',
                'Digital product of aquifer boundary': '...',
                'Hydrostratigraphy': '...',
                'Data used to define hydrostratigraphy': '...',
                'Data sources - hydrostratigraphy': '...',
                'Structural controls': '...',
                'Data used to define structural controls': '...',
                'Data sources - structural controls': '...',
                'Recharge and discharge zones': '...',
                'Data used to define recharge and discharge zones': '...',
                'Recharge and discharge rates': '...',
                'Data used to define recharge and discharge rates': '...',
                'Water table elevation or depth to water map': '...',
                'Data used to create WTE or DTW map': '...',
                'WTE DTW map data collected or compiled': '...',
                'WTE DTW data sources': '...',
                'Water quality data': '...',
                'Water quality data collected or compiled': '...',
                'Water quality data sources': '...',
                'Hydrogeologic properties': '...',
                'Hydrogeo properties collected or compiled': '...',
                'Hydrogeo properties data sources': '...',
                'Future work recommendations': '...'
            }}
            """

        response = client.chat.completions.create(
            model="gpt-4.5-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0  # Keep responses consistent
        )

        #with open(path, 'wb') as wf:
        #    pickle.dump(response, wf)

    content = response.choices[0].message.content

    return json.loads(content)


    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",  # Use "gpt-3.5-turbo" if needed
    #     messages=[
    #         {"role": "system", "content": "You are an expert in analyzing aquifer reports. Please answer the following questions as concisely as possible."},
    #         {"role": "user", "content": f"Here is a report:\n\n{text}\n\n{question}"}
    #     ],
    #     temperature=0.5
    # )

    # return response.choices[0].message.content


def load_questions(question_file):
    """Load descriptors and questions from a CSV file."""
    df = pd.read_csv(question_file)
    return {row["Description"]: row["Question"] for _, row in df.iterrows()}


def process_reports(csvwriter, df, OUTPUT_CSV):
    print(f'Processing report URLs from {df}')
    # questions = load_questions('aquiferie_insight_prompts.csv')
    results = []
    header = None
    for index, row in df.iterrows():
        report_url = row["report_url"]
        print(f'{report_url=}')
        report_name = os.path.join(DOWNLOAD_DIR, f"report_{index}.pdf")

        download_pdf(report_url, report_name)
        report_text = extract_text_from_pdf(report_name)

        if report_text is not None:
            insights = {"Report Name": report_name}

            results = ask_openai(report_text, report_url)

            # json_results = json.loads(results)

            # # Save as JSON
            # with open(f"{STUDYAREA}report_{index}_aquifer_insights.json", "w", encoding="utf-8") as f:
            #     json.dump(results, f, indent=4)

            # print(f'Extraction complete. Data saved to {STUDYAREA}_aquifer_insights.json')

            # # Convert to DataFrame
            #print(results)
            # df = pd.DataFrame(results)
            # df = pd.read_json(results)
            # Save as CSV
            #df.to_csv(f'{OUTPUT_CSV}', index=False)

            # pprint.pprint(results)
            if header is None:
                header = results.keys()
                csvwriter.writerow(header)

            row = [results[k] for k in header]
            csvwriter.writerow(row)
            print(f"Extraction complete. Data saved to {OUTPUT_CSV}")

            #!!!!! WARNING REMOVE ME !!!!
            #break
            # for descriptor, question in questions.items():
            #     print(f'{report_name}, Asking OpenAI: {question}')
            #     insights[descriptor] = ask_openai(text, question)
            #     time.sleep(30)

            # results.append(insights)
            # pd.DataFrame(results).T.to_csv(OUTPUT_CSV)

    # pd.DataFrame(results).T.to_csv(OUTPUT_CSV)

    # print("Processing complete. Data saved to", OUTPUT_CSV)


def configure():
    df = pd.read_csv('aquiferie_report_links.csv')
    studyareas = df['SimpleRegion'].unique()

    return studyareas, df



def main():
    studyareas, df = configure()
    for studyarea in studyareas:
        df2 = df[df['SimpleRegion'] == studyarea]
        print(f'Generating report links csv for {studyarea}')

        os.makedirs(DOWNLOAD_DIR, exist_ok=True)

        if not os.path.exists(studyarea):
            os.makedirs(studyarea)
            df2.to_csv(os.path.join(studyarea, f'{studyarea}_aquiferie_report_links.csv'))
            OUTPUT_CSV = os.path.join(studyarea, f'{studyarea}_aquifer_insights.csv')
            with open(OUTPUT_CSV, 'w', newline='') as wf:
                csvwriter = csv.writer(wf)
                process_reports(csvwriter, df, OUTPUT_CSV)
        if os.path.exists(studyarea):
            OUTPUT_CSV = os.path.join(studyarea, f'{studyarea}_aquifer_insights.csv')
            if os.path.exists(OUTPUT_CSV):
                df_links = pd.read_csv(os.path.join(studyarea, f'{studyarea}_aquiferie_report_links.csv'))
                df_insights = pd.read_csv(OUTPUT_CSV)
                for index, r1 in df_links.iterrows():
                    for i2, r2 in df_insights.iterrows():
                        if r1['report_url'] == r2['Report URL']:
                            print('report already analzyed, moving on to next report')
                    else:
                        with open(OUTPUT_CSV, 'w', newline='') as wf:
                            csvwriter = csv.writer(wf)
                            process_reports(csvwriter, df, OUTPUT_CSV)



            print(f'{studyarea} directory already exists, moving on to next region because AI is expensive')


if __name__ == "__main__":
    main()

