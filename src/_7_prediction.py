import pandas as pd
import pickle
from keras.src.saving import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def load_new_tweets(filepath):
    """Temizlenmiş tweet'leri yükler."""
    try:
        data = pd.read_csv(filepath)
        print(f"Yeni tweet'ler yüklendi: {filepath}")
        return data
    except FileNotFoundError:
        print(f"Hata: Dosya bulunamadı - {filepath}")
        raise
    except Exception as e:
        print(f"Veri yüklenirken hata oluştu: {e}")
        raise


def load_objects(vectorizer_path, label_encoder_path, model_path):
    """TF-IDF vectorizer, label encoder ve modeli yükler."""
    try:
        with open(vectorizer_path, 'rb') as file:
            tfidf_vectorizer = pickle.load(file)
        print(f"TF-IDF Vectorizer yüklendi: {vectorizer_path}")
    except Exception as e:
        print(f"TF-IDF Vectorizer yüklenirken hata oluştu: {e}")
        raise

    try:
        with open(label_encoder_path, 'rb') as file:
            label_encoder = pickle.load(file)
        print(f"Label Encoder yüklendi: {label_encoder_path}")
    except Exception as e:
        print(f"Label Encoder yüklenirken hata oluştu: {e}")
        raise

    try:
        model = load_model(model_path)
        print(f"Model yüklendi: {model_path}")
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        raise

    return tfidf_vectorizer, label_encoder, model


def vectorize_tweets(data, text_column, vectorizer):
    """Tweet'leri TF-IDF ile vektörleştirir."""
    try:
        X = vectorizer.transform(data[text_column])
        print("Metin TF-IDF ile vektörleştirildi.")
        return X
    except Exception as e:
        print(f"TF-IDF vektörleştirme sırasında hata oluştu: {e}")
        raise


def predict_professions(X, model):
    """Meslek gruplarını tahmin eder."""
    try:
        predictions = model.predict(X)
        y_pred = predictions.argmax(axis=1)
        return y_pred
    except Exception as e:
        print(f"Tahmin sırasında hata oluştu: {e}")
        raise


def create_label_mapping():
    """Sayısal etiketleri string etiketlerle eşleştirir."""
    label_mapping = {
        0: 'avukat',
        1: 'diyetisyen',
        2: 'doktor',
        3: 'ekonomist',
        4: 'ogretmen',
        5: 'psikolog',
        6: 'sporyorumcusu',
        7: 'tarihci',
        8: 'yazilimci',
        9: 'ziraatmuhendisi'
    }
    return label_mapping


def main():
    new_tweets_path = 'data/processed/tweets_pull_with_api_cleaned.csv'
    model_path = 'models/baseline_model.h5'
    vectorizer_path = 'data/prepared_data/tfidf_vectorizer.pkl'
    label_encoder_path = 'data/prepared_data/label_encoder.pkl'
    output_path = 'data/processed/tweets_job_prediction.csv'

    try:
        data = load_new_tweets(new_tweets_path)
        tfidf_vectorizer, label_encoder, model = load_objects(vectorizer_path, label_encoder_path, model_path)
    except Exception as e:
        print(f"Yükleme sırasında hata oluştu: {e}")
        return

    if 'Cleaned_Tweet' not in data.columns:
        print("Hata: 'Cleaned_Tweet' sütunu bulunamadı. Lütfen ön işleme adımlarınızı kontrol edin.")
        return

    before_drop = data.shape[0]
    data = data.dropna(subset=['Cleaned_Tweet'])
    after_drop = data.shape[0]
    if before_drop != after_drop:
        print(f"{before_drop - after_drop} satır 'Cleaned_Tweet' sütununda eksik değerler nedeniyle kaldırıldı.")

    before_empty_drop = data.shape[0]
    # 'Cleaned_Tweet' sütunundaki boş veya sadece boşluk içeren metinleri kaldırma
    data = data[data['Cleaned_Tweet'].str.strip().astype(bool)]
    after_empty_drop = data.shape[0]
    if before_empty_drop != after_empty_drop:
        print(f"{before_empty_drop - after_empty_drop} satır boş veya sadece boşluk içeriyor ve kaldırıldı.")

    try:
        X = vectorize_tweets(data, 'Cleaned_Tweet', tfidf_vectorizer)
    except Exception as e:
        print(f"Vektörleştirme sırasında hata oluştu: {e}")
        return

    try:
        y_pred = predict_professions(X, model)

        # Sayısal etiketleri string etiketlere dönüştürme
        label_mapping = create_label_mapping()
        labels = [label_mapping.get(label, 'bilinmiyor') for label in y_pred]
        print("Sayısal etiketler string etiketlere dönüştürüldü.")
    except Exception as e:
        print(f"Tahmin sırasında hata oluştu: {e}")
        return

    data['Label'] = labels

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_data = data[['Cleaned_Tweet', 'Label']]
        output_data.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Tahminler kaydedildi: {output_path}")
    except Exception as e:
        print(f"Tahminler kaydedilirken hata oluştu: {e}")

if __name__ == "__main__":
    main()
