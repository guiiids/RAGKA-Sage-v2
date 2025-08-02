# Enhanced Sentiment and Aspect Analysis Script for User Feedback
#
# Reads comments (with timestamp) from sage_feedback_with_comments.xlsx,
# analyzes using Azure Text Analytics "opinion mining" (only Azure APIs used),
# outputs a per-row CSV plus summary aspect table weighted by confidence,
# with crude trend ("first half"/"second half") analysis.
#
# Prerequisites:
# - AZURE_LANGUAGE_ENDPOINT and AZURE_LANGUAGE_KEY set in env
# - pip install azure-ai-textanalytics pandas openpyxl tqdm python-dotenv

import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from tqdm import tqdm
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

INPUT_FILE = "sage_feedback_with_comments.xlsx"
OUTPUT_FILE = "feedback_sentiment_detailed.csv"
SUMMARY_FILE = "aspect_summary.csv"
COMMENT_COL = "comment"
TIMESTAMP_COL = "timestamp"

def get_text_analytics_client():
    endpoint = os.environ["AZURE_LANGUAGE_SERVICE_ENDPOINT"]
    key = os.environ["AZURE_LANGUAGE_SERVICE_KEY"]
    return TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

def load_feedback_df():
    # Read Excel with correct header row
    df = pd.read_excel(INPUT_FILE, engine="openpyxl")
    # Lowercase headers just in case
    df.columns = df.columns.str.lower()
    return df

def extract_documents(df):
    # Only use rows where comment is a non-empty string
    docs = df[df[COMMENT_COL].notnull()].copy()
    docs = docs[docs[COMMENT_COL].astype(str).str.strip() != ""]
    return docs

def analyze_opinion_mining(client, comments):
    # Azure processes a batch of up to 10 documents at once
    results = []
    batch_size = 10
    for start in tqdm(range(0, len(comments), batch_size), desc="Analyzing..."):
        batch = comments[start:start+batch_size]
        response = client.analyze_sentiment(batch, show_opinion_mining=True)
        results.extend(list(response))
    return results

def get_half_split_index(length):
    return (length // 2) + (length % 2)

def main():
    client = get_text_analytics_client()
    df = load_feedback_df()
    docs = extract_documents(df)
    print(f"Processing {len(docs)} comments from {INPUT_FILE}")

    comments = docs[COMMENT_COL].astype(str).tolist()
    timestamps = docs[TIMESTAMP_COL].astype(str).tolist() if TIMESTAMP_COL in docs.columns else ["" for _ in comments]

    # Split indices for basic trend analysis
    split_idx = get_half_split_index(len(comments))
    first_half_idx = set(range(split_idx))
    second_half_idx = set(range(split_idx, len(comments)))

    # Analyze all comments via Azure
    results = analyze_opinion_mining(client, comments)

    # Per-row output: comment, timestamp, doc sentiment, doc conf, aspects/targets, target sent/conf, opinions/conf
    output_rows = []
    aspect_weighted_counts = {}
    aspect_trend_first = {}
    aspect_trend_second = {}

    for idx, (res, comment, ts) in enumerate(zip(results, comments, timestamps)):
        if res.is_error:
            output_rows.append({
                "timestamp": ts,
                "comment": comment,
                "sentiment": "error",
                "sentiment_confidence": "",
                "aspects": "",
                "aspect_sentiments": "",
                "aspect_confidences": "",
            })
            continue

        # Document level
        overall_sent = res.sentiment
        conf = max(res.confidence_scores.positive, res.confidence_scores.negative, res.confidence_scores.neutral)

        # Aspects/opinion (if any)
        aspects, sentiments, aspect_confid = [], [], []
        if hasattr(res, "sentences"):
            for sent in res.sentences:
                for mined_op in getattr(sent, "mined_opinions", []):
                    target = mined_op.target
                    aspects.append(target.text)
                    sentiments.append(target.sentiment)
                    tc = max(target.confidence_scores.positive, target.confidence_scores.negative, target.confidence_scores.neutral)
                    aspect_confid.append(tc)
                    # Weighted aspect count/trend
                    d = aspect_weighted_counts.setdefault(target.text, 0.0)
                    w = aspect_trend_first if idx in first_half_idx else aspect_trend_second
                    w.setdefault(target.text, 0.0)
                    aspect_weighted_counts[target.text] = d + tc
                    w[target.text] += tc

        output_rows.append({
            "timestamp": ts,
            "comment": comment,
            "sentiment": overall_sent,
            "sentiment_confidence": conf,
            "aspects": ";".join(aspects),
            "aspect_sentiments": ";".join(sentiments),
            "aspect_confidences": ";".join(map(str, aspect_confid)),
        })

    pd.DataFrame(output_rows).to_csv(OUTPUT_FILE, index=False)

    # Summary: aspect, weighted total, first-half, second-half, crude trend
    summary = []
    all_aspects = set(list(aspect_weighted_counts) + list(aspect_trend_first) + list(aspect_trend_second))
    for aspect in all_aspects:
        total = aspect_weighted_counts.get(aspect, 0.0)
        f = aspect_trend_first.get(aspect, 0.0)
        s = aspect_trend_second.get(aspect, 0.0)
        trend = ("up" if s > f else "down" if s < f else "flat") if (f > 0 or s > 0) else ""
        summary.append({
            "aspect": aspect,
            "total_weighted": round(total, 3),
            "first_half": round(f, 3),
            "second_half": round(s, 3),
            "trend": trend,
        })
    pd.DataFrame(summary).sort_values("total_weighted", ascending=False).to_csv(SUMMARY_FILE, index=False)

    print(f"Saved row-by-row results to: {OUTPUT_FILE}")
    # Query summary: identify questions/topics with low ratings with ratio
    qs = docs.groupby('user_query').agg(
        total_ratings=('rating', 'size'),
        negative_ratings=('rating', lambda x: sum(x == 'negative'))
    )
    qs['negative_ratio'] = qs['negative_ratings'] / qs['total_ratings']
    qs_sort = qs.sort_values('negative_ratio', ascending=False)
    qs_sort.to_csv("query_summary.csv", index=True)
    print("Saved query summary (sorted by negative_ratio) to: query_summary.csv")
    # Detailed verification: export raw rows for queries with multiple votes
    multi_queries = qs[qs['total_ratings'] > 1].index.tolist()
    if multi_queries:
        details_df = docs[docs['user_query'].isin(multi_queries)][[
            'user_query', 'rating', COMMENT_COL, 'feedback_tags'
        ]]
        details_df.to_csv("query_details.csv", index=False)
        print("Saved detailed rows for multi-vote queries to: query_details.csv")
    else:
        print("No queries with more than one vote to verify.")

    # Feedback tag summary: semantic issue categories (as before)
    tag_summary = {}
    for tags in docs['feedback_tags']:
        for tag in tags:
            tag_summary[tag] = tag_summary.get(tag, 0) + 1
    tag_df = pd.DataFrame([{"tag": k, "count": v} for k, v in tag_summary.items()])
    tag_df.to_csv("tag_summary.csv", index=False)
    print("Saved feedback tag summary to: tag_summary.csv")
    print(f"Saved aspect summary/trends to: {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
