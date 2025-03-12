import pandas as pd
import geopandas as gpd
import os
import openai
import sys

client = openai.OpenAI(api_key='sk-proj-UsTt8Y55T1eFBUvULb-lazHiCRdX'
                               '-9VUNJa3UAxObyJYREltqifJPK3btItM7bLqm40AfQ1JQfT3BlbkFJNZza_UXf'
                               '-xYhSg0xrR5WxF1ZsYnDz44irH3Sxyfm9kh9a-vrj3c4PcwymFfw7Ivi1SO26mVZMA')
QUESTIONS_FILE = "aquiferie_insight_prompts.txt"  # Text file containing questions (one per line)
HEADERS_FILE = 'categories.txt'


def load_questions(file):
    """Load questions from a text file."""
    with open(file, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if line.strip()]



def summarize_text(studyarea):
    df = pd.read_csv(f'{studyarea}_aquifer_insights_embeddings.csv')
    # df = df.drop([1, 3, 5, 7, 9, 11, 13, 15, 17])
    print(df)

    text_columns = 'What are the future work recommendations for this study? Please be detailed.'
    text_columns = load_questions(file=QUESTIONS_FILE)
    headers = load_questions(file=HEADERS_FILE)

    for h, c in zip(headers, text_columns[:-1]):
        # Combine all text from the specified column into a single string
        combined_text = ' '.join(df[c].dropna().astype(str).tolist())

        response = client.chat.completions.create(
            model="o1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": f"Please summarize the following text:\n\n{combined_text}"}
            ],
        )
        summary = response.choices[0].message.content

        # Define the output text file path
        output_txt = f'{studyarea}_{h}_summary.txt'

        # Write the summary to the text file
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(summary)

        print(f"Summary has been written to {output_txt}")



def count_reports_per_studyarea(reports, areas):
    n_sas = reports['FID_webmap'].value_counts().reset_index()
    n_sas.columns = ['FID_webmap', 'Count']
    print(n_sas)

    merged_df = areas.merge(n_sas, left_on='FID', right_on='FID_webmap', how="left")
    merged_df['Count'] = merged_df['Count'].fillna(0)
    print(merged_df)
    merged_df.to_file(r'C:\Users\mfichera\OneDrive - nmt.edu\Documents\ArcGIS\Projects\NMAquifersResearch\AMP_studyareas_2025_ranks.shp')



def main():
    aquifer_reports = pd.read_csv('aquiferie_report_links.csv')
    # print(aquifer_reports['SimpleRegion'].unique())

    summarize_text('TularosaBasin')

    # amp_study_areas = gpd.read_file(r'W:\gis_base\amp_coverages\studyareas\AMP_Study_Areas_Winter2025.shp')
    # amp_study_areas = amp_study_areas.reset_index(names='FID')
    # print(amp_study_areas)
    #
    # count_reports_per_studyarea(aquifer_reports, amp_study_areas)





if __name__ == '__main__':
    main()