import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os

def main():
    nltk.download('stopwords')
    nltk.download('punkt')

    #input_path = 'data/unprocessed/tweets_pull_with_api.csv'
    input_path = 'data/processed/preprocessed_tweets.csv'
    output_path = 'data/processed/preprocessed_tweets_cleaned.csv'

    try:
        data = pd.read_csv(input_path)
        print(f"Veri yüklendi: {input_path}")
    except FileNotFoundError:
        print(f"Hata: Dosya bulunamadı - {input_path}")
        return
    except Exception as e:
        print(f"Veri yüklenirken hata oluştu: {e}")
        return

    stop_words = set(stopwords.words('turkish'))

    def clean_text(text):
        """Tweet metnini temizler."""
        try:
            # URL'leri temizleme
            text = re.sub(r'http\S+', '', text)
            # Mention'ları temizleme
            text = re.sub(r'@\w+', '', text)
            # Hashtag'leri temizleme
            text = re.sub(r'#\w+', '', text)
            # Özel karakterleri temizleme (Türkçe karakterleri dahil et)
            text = re.sub(r'[^a-zA-ZçğıöşüÇĞİÖŞÜ\s]', '', text)
            return text.lower()
        except Exception as e:
            print(f"Metin temizlenirken hata oluştu: {e}")
            return ""

    def preprocess_text(text):
        """Tweet metnini temizler ve stop words kaldırır."""
        cleaned = clean_text(text)
        tokens = nltk.word_tokenize(cleaned)
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    if 'tweet' not in data.columns:
        print("Hata: 'tweet' sütunu bulunamadı.")
        return

    data['Cleaned_Tweet'] = data['tweet'].astype(str).apply(preprocess_text)
    print("Metin temizleme ve stop words kaldırma tamamlandı.")

    if 'label' not in data.columns:
        print("Hata: 'label' sütunu bulunamadı.")
        return

    data = data[['Cleaned_Tweet', 'label']]
    print("Orijinal 'tweet' sütunu kaldırıldı ve sadece 'Cleaned_Tweet' ile 'label' sütunları bırakıldı.")

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Temizlenmiş veri kaydedildi: {output_path}")
    except Exception as e:
        print(f"Temizlenmiş veri kaydedilirken hata oluştu: {e}")

if __name__ == "__main__":
    main()
