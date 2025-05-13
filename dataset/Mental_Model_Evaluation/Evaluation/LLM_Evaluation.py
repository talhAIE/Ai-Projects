import json
import os
import csv
import random
import matplotlib.pyplot as plt
import numpy as np
import re
import logging
import pandas as pd
import time
from tqdm import tqdm
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from openai import OpenAI
import google.generativeai as genai

# === API KEYS ===
os.environ["OPENAI_API_KEY"] = ""
DEEPSEEK_API_KEY = ""
GEMINI_API_KEY = ""
genai.configure(api_key=GEMINI_API_KEY)

# === Setup ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
openai_client = OpenAI()

# === Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === MODEL FUNCTIONS ===
def ask_gpt(model, prompt):
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in ask_gpt with model {model}: {e}", exc_info=True)
        raise

def ask_deepseek(prompt):
    import requests
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error in ask_deepseek: {e}", exc_info=True)
        raise

def ask_gemini(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error in ask_gemini: {e}", exc_info=True)
        raise

models = {
    "gpt-3.5-turbo": lambda p: retry_request(lambda: ask_gpt("gpt-3.5-turbo", p)),
    "gpt-4o": lambda p: retry_request(lambda: ask_gpt("gpt-4o", p)),
    "deepseek-chat": lambda p: retry_request(ask_deepseek, p),
    "gemini-1.5-pro-latest": lambda p: retry_request(ask_gemini, p),
}

def retry_request(func, *args, max_retries=3, delay=5, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning(f"Retry {attempt+1}/{max_retries} failed: {e}")
            time.sleep(delay)
    logging.error(f"All retries failed for {func.__name__}.")
    raise Exception(f"All retries failed for {func.__name__}.")

# === METRICS ===
geval_metric = GEval(
    name="Answer Quality",
    criteria="Is the answer accurate, relevant, and clear?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

def extract_answer(output, q_type):
    lines = output.strip().splitlines()
    if not lines:
        return output.strip()
    first_line = lines[0].strip()

    if q_type == "MC":
        #  Safely strip unwanted symbols but allow "A" or "A."
        first_line = re.sub(r"[#¤%&/*^]", "", first_line).strip()
        match = re.match(r"^([A-Da-d])\)?", first_line)
        if match:
            return match.group(1).upper()
        return first_line.upper()

    if q_type == "YN":
        # Safely strip unwanted symbols
        first_line = re.sub(r"[#¤%&/*^]", "", first_line).strip()
        if first_line.lower() in ["yes", "no"]:
            return first_line.capitalize()
        return first_line.capitalize()

    return output.strip()

def evaluate_question(question_data, models, geval_metric):
    results = []
    question = question_data["question"]
    q_type = question_data["id"][:2]
    base_prompt = f"Answer this {q_type} question:\n{question}"

    if "options" in question_data:
        base_prompt += "\nOptions:\n" + "\n".join([f"{k}) {v}" for k, v in question_data["options"].items()])

    if q_type == "MC":
        base_prompt += "\n\nPlease answer with the correct option letter (A, B, C, or D) on the first line. Please don't use any of this symboles to answer # ¤ % & / * ^  Then explain your reasoning."
    elif q_type == "YN":
        base_prompt += "\n\nPlease answer with Yes or No on the first line. Please don't use any of this symboles to answer # ¤ % & / * ^  Then explain your reasoning."

    expected_answer = question_data.get("expected_answer", question_data.get("correct_option", ""))
    mental_model = question_data.get("mental_model", "")

    for model_name, func in models.items():
        for with_mm in [False, True]:
            prompt = base_prompt
            if with_mm and mental_model:
                prompt += f"\n\nUse this mental model to help you answer better: {mental_model}"

            try:
                output = func(prompt)
                score, reason = calculate_score(output, expected_answer, q_type, prompt, geval_metric)

                results.append({
                    "model": model_name,
                    "question_id": question_data["id"],
                    "question_type": q_type,
                    "with_mental_model": with_mm,
                    "score": score,
                    "reason": reason,
                    "output": output,
                })
            except Exception as e:
                logging.error(f"Error with model {model_name}: {e}", exc_info=True)
    return results

def calculate_score(output, expected_answer, q_type, prompt, geval_metric):
    try:
        if q_type == "OE":
            test_case = LLMTestCase(
                input=prompt,
                actual_output=output,
                expected_output=expected_answer,
            )
            geval_metric.measure(test_case)
            return round(geval_metric.score, 2), geval_metric.reason
        else:
            predicted = extract_answer(output, q_type)
            is_correct = predicted.lower() == expected_answer.strip().lower()
            score = 1.0 if is_correct else 0.0
            reason = "Matched" if is_correct else f"Expected: {expected_answer}, Got: {predicted}"
            return score, reason
    except Exception as e:
        logging.error(f"Error calculating score: {e}", exc_info=True)
        return 0.0, str(e)

def generate_llm_analysis_plots(df):
    output_dir = "plots/llm_only"
    os.makedirs(output_dir, exist_ok=True)

    for q_type in df["question_type"].unique():
        subset = df[df["question_type"] == q_type]
        if subset.empty:
            continue

        no_mm = subset[subset["with_mental_model"] == False]
        scores_no_mm = no_mm.groupby("model")["score"].mean()

        plt.figure(figsize=(8, 6))
        scores_no_mm.sort_values().plot(kind="bar", color="skyblue")
        plt.title(f"LLM Scores without Mental Model ({q_type})")
        plt.ylabel("Average Score")
        plt.xlabel("LLMs")
        plt.xticks(rotation=0)
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"scores_without_mental_model_{q_type}.png"))
        plt.close()

        with_mm = subset[subset["with_mental_model"] == True].groupby("model")["score"].mean()
        delta = (with_mm - scores_no_mm).fillna(0)

        plt.figure(figsize=(8, 6))
        delta.sort_values().plot(kind="bar", color="coral")
        plt.title(f"Mental Model Effect (\u0394 Score) on LLMs ({q_type})")
        plt.ylabel("\u0394 Score (With MM − Without MM)")
        plt.xlabel("LLMs")
        plt.axhline(0, color="gray", linestyle="--")
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"MM_delta_{q_type}.png"))
        plt.close()

        comparison = (
            subset.groupby(["model", "with_mental_model"])
            .agg({"score": "mean"})
            .reset_index()
            .pivot(index="model", columns="with_mental_model", values="score")
        )

        comparison = comparison.rename(columns={True: "With MM", False: "No MM"})

        plt.figure(figsize=(8, 6))
        comparison.plot(kind="bar", ax=plt.gca())
        plt.title(f"Score Comparison by Mental Model Usage ({q_type})")
        plt.ylabel("Average Score")
        plt.xlabel("LLMs")
        plt.xticks(rotation=0)
        plt.ylim(0, 1.05)
        plt.legend(title="Condition")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"score_comparison_mm_{q_type}.png"))
        plt.close()

# === LOAD QUESTIONS ===
try:
    with open("C:/Users/Diana/Bachelor/Code/questions.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        data = raw_data["dataset"] if isinstance(raw_data, dict) and "dataset" in raw_data else raw_data
except FileNotFoundError:
    logging.error("questions.json not found.")
    raise
except json.JSONDecodeError:
    logging.error("Invalid JSON format in questions.json")
    raise

# === EVALUATE ===
results = []
for item in tqdm(data, desc="Evaluating Questions"):
    results.extend(evaluate_question(item, models, geval_metric))

df = pd.DataFrame(results)

# === SAVE RESULTS ===
df.to_csv("evaluation_results.csv", index=False)

# === SAVE MARKDOWN ===
md_lines = ["# Evaluation Results\n"]
df['order'] = df.apply(lambda row: int(row['question_id'][2:]), axis=1)  # Extract numeric part for sorting
grouped = df.sort_values('order').groupby(["question_id", "model", "with_mental_model"])

try:
    for (qid, model, with_mm), group in grouped:
        row = group.iloc[0]
        mm_label = "✅ With Mental Model" if with_mm else "❌ No Mental Model"
        q_type = row["question_type"]
        score = row["score"]

        if q_type in ["MC", "YN"]:
            score_display = "✅ Correct" if score == 1.0 else "❌ Incorrect"
        else:  # OE
            score_display = f"{score:.2f}"

        md_lines.append(f"## QID: {qid} — {model} — {mm_label}")
        md_lines.append(f"**Score**: {score_display}")
        md_lines.append(f"**Output**:\n```text\n{row['output']}\n```")
        md_lines.append(f"**Evaluation Reason**:\n> {row['reason']}")
        md_lines.append("")
except Exception as e:
    logging.error(f"Error saving results to Markdown: {e}", exc_info=True)

# ✅ Add final good summary table ONLY ONCE
md_lines.append("\n# Final Summary Table\n")
md_lines.append("| QID | Type | Model | Expected | Output | Correct/Score | Mental Model |")
md_lines.append("| --- | ---- | ----- | -------- | ------ | ------------- | ------------ |")

for idx, row in df.iterrows():
    qid = row['question_id']
    q_type = row['question_type']
    model = row['model']
    mm_used = "Yes" if row['with_mental_model'] else "No"

    if q_type == "OE":
        expected = ""
        output = "—"
        correct_score = f"{row['score']:.2f}"
    else:
        expected = row['reason'].split('Expected: ')[-1].split(', Got')[0] if 'Expected:' in row['reason'] else ''
        output = extract_answer(row['output'], q_type)
        correct_score = "✅" if row['score'] == 1.0 else "❌"

    md_lines.append(f"| {qid} | {q_type} | {model} | {expected} | {output} | {correct_score} | {mm_used} |")

# ✅ Save only ONCE
os.makedirs("results/LLM_Evaluation", exist_ok=True)
with open("results/LLM_Evaluation/evaluation_results.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))


# === GENERATE PLOTS ===
generate_llm_analysis_plots(df)

