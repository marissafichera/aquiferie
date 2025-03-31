import pandas as pd
import os

studyarea = 'AlbuquerqueBasin'
ROOT = studyarea

def view_incorrect_evals(csv):
    # Load your CSV
    df = pd.read_csv(csv)

    # List to collect rows with "Correct? No"
    incorrect_answers = []

    # Iterate through each row
    for idx, row in df.iterrows():
        report_name = row["Report"]
        for question in df.columns:
            if question == "Report":
                continue
            answer = str(row[question])
            if "Correct?: No" in answer:
                incorrect_answers.append({
                    "Report": report_name,
                    "Question": question,
                    "Answer": answer
                })

    # Create a new DataFrame and save to CSV
    output_df = pd.DataFrame(incorrect_answers)
    output_df.to_csv(os.path.join(ROOT, f"ai_incorrect_answers_{studyarea}.csv"), index=False)


def main():
    csv = os.path.join(ROOT, f'{studyarea}_aquiferinsights_selfeval.csv')

    view_incorrect_evals(csv)


if __name__ == '__main__':
    main()
