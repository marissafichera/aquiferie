import os
import re
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from transformers import pipeline
import openai

# === CONFIGURATION ===
studyarea = 'RoswellArtesianBasin'
CSV_PATH = rf"{studyarea}\{studyarea}_aquiferinsights_selfeval.csv"
QUESTIONS_TXT_PATH = "rank_insights_questions.txt"
CACHE_FILE = fr"{studyarea}\qa_cache.json"
BBOX_CACHE_FILE = fr"{studyarea}\bbox_cache.json"
OUTPUT_CSV = fr"{studyarea}\simplified_binary_answers.csv"
OUTPUT_SHP = fr"{studyarea}\simplified_reports_with_bbox.shp"
BBOX_COL = "What is the geographic bounding box of the study area, in decimal degrees?"
client = openai.OpenAI(api_key='sk-proj-UsTt8Y55T1eFBUvULb-lazHiCRdX'
                               '-9VUNJa3UAxObyJYREltqifJPK3btItM7bLqm40AfQ1JQfT3BlbkFJNZza_UXf'
                               '-xYhSg0xrR5WxF1ZsYnDz44irH3Sxyfm9kh9a-vrj3c4PcwymFfw7Ivi1SO26mVZMA')
# === LOAD ===
def load_data(csv_path, questions_txt_path):
    with open(questions_txt_path, 'r') as f:
        selected_questions = [line.strip() for line in f if line.strip()]
    df = pd.read_csv(csv_path)
    return df, selected_questions

# === QA PROCESSING ===
def trim_answer(answer):
    if not isinstance(answer, str):
        return ""
    return answer.split("o3-mini Evaluation")[0].strip()

def classify_answer(question, answer, classifier, cache):
    cache_key = f"{question}|||{answer}"
    if cache_key in cache:
        return cache[cache_key]
    if not isinstance(answer, str) or not answer.strip():
        result = "no"
    else:
        try:
            result = classifier(
                sequences=answer,
                candidate_labels=["yes", "no"]
            )["labels"][0].lower()
        except:
            result = "no"
    cache[cache_key] = result
    return result

# === BOUNDING BOX EXTRACTION ===
def extract_bbox_with_openai(text, client, bbox_cache):
    if not isinstance(text, str) or not text.strip():
        return None
    if text in bbox_cache:
        print('HERE')
        print(bbox_cache[text])
        return bbox_cache[text]

    prompt = (
        "Extract the geographic bounding box from the following answer.\n"
        "Respond in the format: West, East, South, North. Only return the four float values separated by commas.\n\n"
        f"Answer: {text}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        print(f"📦 BBox Response: {result}")
        numbers = [float(x.strip()) for x in result.split(",")]

        if len(numbers) == 4:
            bbox_cache[text] = numbers
            return numbers

    except Exception as e:
        print(f"❌ Error extracting bbox with OpenAI: {e}")

    # bbox_cache[text] = None
    return None

# === GEOMETRY CREATION ===
def row_to_geometry(row):
    try:
        return Polygon([
            (row["West"], row["South"]),
            (row["West"], row["North"]),
            (row["East"], row["North"]),
            (row["East"], row["South"]),
            (row["West"], row["South"]),
        ])
    except:
        return None

# === MAIN ===
def main():
    df, selected_questions = load_data(CSV_PATH, QUESTIONS_TXT_PATH)
    df_selected = df[["Report"] + selected_questions].copy()
    df_selected[selected_questions] = df_selected[selected_questions].applymap(trim_answer)

    # Load QA cache and classifier
    answer_cache = json.load(open(CACHE_FILE)) if os.path.exists(CACHE_FILE) else {}
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Convert answers to binary
    binary_df = pd.DataFrame()
    binary_df["Report"] = df["Report"].values
    for q in selected_questions:
        if q.lower() != "report":
            print(f"Processing question: {q}")
            binary_df[q] = df_selected[q].apply(lambda ans: 1 if classify_answer(q, ans, classifier, answer_cache) == "yes" else 0)

    binary_df["total"] = binary_df.drop(columns=["Report"]).sum(axis=1)
    with open(CACHE_FILE, 'w') as f:
        json.dump(answer_cache, f, indent=2)
    binary_df.to_csv(OUTPUT_CSV, index=False)

    # Bounding box extraction
    binary_df["bounding_box_raw"] = df[BBOX_COL]
    openai_client = client
    bbox_cache = json.load(open(BBOX_CACHE_FILE)) if os.path.exists(BBOX_CACHE_FILE) else {}

    test_text = binary_df["bounding_box_raw"].dropna().iloc[0]
    print("Testing extract_bbox_with_openai on one value:\n", test_text)

    result = extract_bbox_with_openai(test_text, openai_client, bbox_cache)
    print("Result:", result)

    bbox_parsed = binary_df["bounding_box_raw"].apply(lambda x: extract_bbox_with_openai(x, openai_client, bbox_cache))

    binary_df["West"] = bbox_parsed.apply(lambda x: x[0] if x else None)
    binary_df["East"] = bbox_parsed.apply(lambda x: x[1] if x else None)
    binary_df["South"] = bbox_parsed.apply(lambda x: x[2] if x else None)
    binary_df["North"] = bbox_parsed.apply(lambda x: x[3] if x else None)

    # Ensure longitudes are negative (western hemisphere)
    binary_df["West"] = binary_df["West"].apply(lambda x: -abs(x) if pd.notnull(x) else None)
    binary_df["East"] = binary_df["East"].apply(lambda x: -abs(x) if pd.notnull(x) else None)

    with open(BBOX_CACHE_FILE, 'w') as f:
        json.dump(bbox_cache, f, indent=2)

    # Create geometry and save shapefile
    binary_df["geometry"] = binary_df.apply(row_to_geometry, axis=1)
    gdf = gpd.GeoDataFrame(binary_df, geometry="geometry", crs="EPSG:4326")
    # gdf = gdf[gdf["geometry"].notnull()]
    gdf.to_file(OUTPUT_SHP)
    print(f"\n✅ Shapefile exported: {OUTPUT_SHP}")

if __name__ == "__main__":
    main()



