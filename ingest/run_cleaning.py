import os
from ingest.load_tweets import load_tweets
from ingest.clean_tweets import clean_tweets

RAW_PATH = "data/raw/trump_tweets_dataset.csv"
OUTPUT_PATH = "data/processed/trump_tweets_clean.csv"

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    df = load_tweets(RAW_PATH)
    print("Loaded rows:", len(df))

    df_clean = clean_tweets(df)
    print("Clean rows:", len(df_clean))

    df_clean.to_csv(OUTPUT_PATH, index=False)
    print("Saved cleaned dataset to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()