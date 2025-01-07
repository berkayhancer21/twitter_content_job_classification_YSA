import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import os

def load_data(filepath):
    """Temizlenmiş veri setini yükler."""
    try:
        data = pd.read_csv(filepath)
        print(f"Veri yüklendi: {filepath}")
        return data
    except FileNotFoundError:
        print(f"Hata: Dosya bulunamadı - {filepath}")
        raise
    except Exception as e:
        print(f"Veri yüklenirken hata oluştu: {e}")
        raise


def encode_labels(data, label_column):
    """Etiketleri sayısal değerlere dönüştürür."""
    le = LabelEncoder()
    data[label_column] = le.fit_transform(data[label_column])
    return data, le


def vectorize_text(data, text_column, max_features=5000):
    """Metni TF-IDF ile vektörleştirir."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(data[text_column])
    return X, vectorizer


def save_objects(obj, filepath):
    """Python objelerini pickle ile kaydeder."""
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(obj, file)
        print(f"Objeler kaydedildi: {filepath}")
    except Exception as e:
        print(f"Objeler kaydedilirken hata oluştu: {e}")
        raise


def main():
    input_path = 'data/processed/preprocessed_tweets_labels_cleaned.csv'
    output_prepared_data_path = 'data/prepared_data/'

    try:
        data = load_data(input_path)
    except Exception as e:
        print(f"Veri yükleme başarısız: {e}")
        return

    if 'tweet' not in data.columns:
        print("Hata: 'tweet' sütunu bulunamadı. Lütfen ön işleme adımlarınızı kontrol edin.")
        return

    before_drop = data.shape[0]
    data = data.dropna(subset=['tweet'])
    after_drop = data.shape[0]
    if before_drop != after_drop:
        print(f"{before_drop - after_drop} satır 'tweet' sütununda eksik değerler nedeniyle kaldırıldı.")

    if 'label' not in data.columns:
        print("Hata: 'label' sütunu bulunamadı. Lütfen etiketlerinizi kontrol edin.")
        return

    # Etiket Kodlama
    try:
        data, label_encoder = encode_labels(data, 'label')
        print("Etiket kodlama tamamlandı.")
    except Exception as e:
        print(f"Etiket kodlama sırasında hata oluştu: {e}")
        return

    try:
        os.makedirs(output_prepared_data_path, exist_ok=True)
        save_objects(label_encoder, os.path.join(output_prepared_data_path, 'label_encoder.pkl'))
    except Exception as e:
        print(f"Label encoder kaydedilirken hata oluştu: {e}")
        return

    # TF-IDF Vektörleştirme
    try:
        X, tfidf_vectorizer = vectorize_text(data, 'tweet')
        print("TF-IDF vektörleştirme tamamlandı.")
    except Exception as e:
        print(f"TF-IDF vektörleştirme sırasında hata oluştu: {e}")
        return

    try:
        save_objects(tfidf_vectorizer, os.path.join(output_prepared_data_path, 'tfidf_vectorizer.pkl'))
    except Exception as e:
        print(f"TF-IDF vectorizer kaydedilirken hata oluştu: {e}")
        return

    y = data['label']

    # Veri Setini Eğitim ve Test Setlerine Bölme
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("Veri seti eğitim ve test setlerine bölündü.")
    except Exception as e:
        print(f"Eğitim ve test setlerine bölme sırasında hata oluştu: {e}")
        return

    # Eğitim ve Test Setlerini Kaydetme
    try:
        sparse.save_npz(os.path.join(output_prepared_data_path, 'X_train.npz'), X_train)
        sparse.save_npz(os.path.join(output_prepared_data_path, 'X_test.npz'), X_test)
        pd.to_pickle(y_train, os.path.join(output_prepared_data_path, 'y_train.pkl'))
        pd.to_pickle(y_test, os.path.join(output_prepared_data_path, 'y_test.pkl'))
        print("Eğitim ve test setleri kaydedildi.")
    except Exception as e:
        print(f"Eğitim ve test setleri kaydedilirken hata oluştu: {e}")
        return


if __name__ == "__main__":
    main()
