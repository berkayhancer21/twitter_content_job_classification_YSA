import tweepy
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

BEARER_TOKEN = os.getenv('BEARER_TOKEN_2')

if not BEARER_TOKEN:
    raise ValueError("Bearer Token bulunamadı. Lütfen .env dosyanızı kontrol edin.")

client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

def fetch_tweets(terim, max_tweets=100):
    tweets = []
    search_query = f"{terim} lang:tr -is:retweet"
    try:
        paginator = tweepy.Paginator(
            client.search_recent_tweets,
            query=search_query,
            tweet_fields=['id', 'text', 'author_id', 'created_at'],
            max_results=100
        )
        for tweet in paginator.flatten(limit=max_tweets):
            tweets.append({
                'Tweet_ID': tweet.id,
                'Tweet': tweet.text,
                'Author_ID': tweet.author_id,
                'Created_At': tweet.created_at,
                'Search_Term': terim
            })
    except tweepy.TweepyException as e:
        print(f"Hata oluştu: {e}")
    return tweets

def get_trending_terms():
    # Popüler konuları belirleyerek tweetlerimizi çekelim ve bu tweetlerin hangi meslek grupları tarafından atıldığını belirlemeye çalışalım
    trending_terms = ["psikoloji","ekonomi", "eğitim", "ziraat", "sanat"]
    return trending_terms

if __name__ == "__main__":
    arama_terimleri = get_trending_terms()
    tweet_sayisi = 50  # Her arama terimi için çekilecek tweet sayısı

    tum_tweets = []

    for terim in arama_terimleri:
        print(f"'{terim}' terimi için tweetler çekiliyor...")
        fetched_tweets = fetch_tweets(terim, tweet_sayisi)
        if fetched_tweets:
            tum_tweets.extend(fetched_tweets)
            print(f"{len(fetched_tweets)} tweet '{terim}' terimi için çekildi.")
        else:
            print(f"'{terim}' terimi için hiç tweet çekilemedi.")

    if tum_tweets:
        new_df = pd.DataFrame(tum_tweets)

        # Eğer 'tweets_pull_with_api.csv' dosyası varsa, append modunda yaz
        csv_path = 'data/unprocessed/tweets_pull_with_api.csv'
        if os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path)

                combined_df = pd.concat([existing_df, new_df], ignore_index=True)

                before_dedup = combined_df.shape[0]
                combined_df.drop_duplicates(subset=['Tweet_ID'], inplace=True)
                after_dedup = combined_df.shape[0]

                combined_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"{len(fetched_tweets)} yeni tweet '{csv_path}' dosyasına eklendi. "
                      f"Toplam {before_dedup - after_dedup} duplicate tweet kaldırıldı.")
            except Exception as e:
                print(f"CSV dosyası işlenirken hata oluştu: {e}")
        else:
            try:
                new_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"{len(fetched_tweets)} tweet '{csv_path}' dosyasına kaydedildi.")
            except Exception as e:
                print(f"CSV dosyası oluşturulurken hata oluştu: {e}")
    else:
        print("Hiç tweet çekilemedi.")
