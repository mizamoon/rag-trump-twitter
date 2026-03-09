import pandas as pd

def clean_tweets(df):
    df = df[["date", "platform", "text", "quote_flag", "repost_flag", "post_url"]].copy()
    df = df[(df["quote_flag"] == False) & (df["repost_flag"] == False)]

    df["text"] = (
        df["text"]
        .astype(str)
        .str.replace(r"\\n|\n|\r", " ", regex=True)
        .str.replace(r"\[(?:Image|Video|QuickTime Video)\]", " ", regex=True)
        .str.replace(r"\bRT:\s*", " ", regex=True)
        .str.replace(
            r"https?://(?:[^\s]+|\s+(?=[A-Za-z0-9/@._%#?&=:+-]))+",
            " ",
            regex=True
        )
        .str.replace(
            r"\bwww\.(?:[^\s]+|\s+(?=[A-Za-z0-9/_%#?&=:+.-]))+",
            " ",
            regex=True
        )
        .str.replace(
            r"(?:^|\s)\S*/(?:statuses|posts)/\d+[A-Za-z0-9._-]*",
            " ",
            regex=True
        )
        .str.replace(r"\bpic\.twitter\.com/\S+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip() != ""]
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df[["date", "platform", "text", "post_url"]]
    df["text"] = df["text"].astype(str)

    return df.reset_index(drop=True)