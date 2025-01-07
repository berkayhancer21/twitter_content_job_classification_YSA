import pickle
import tensorflow as tf
from sklearn.metrics import classification_report
from src._4_model_variations import (
    create_baseline_model,
    create_model_variation_1,
    create_model_variation_2,
    create_model_variation_3
)
from scipy import sparse
import pandas as pd
import os

def load_objects(vectorizer_path, label_encoder_path):
    """Kaydedilmiş objeleri yükler."""
    with open(vectorizer_path, 'rb') as file:
        tfidf_vectorizer = pickle.load(file)
    with open(label_encoder_path, 'rb') as file:
        label_encoder = pickle.load(file)
    return tfidf_vectorizer, label_encoder


def load_data(X_train_path, X_test_path, y_train_path, y_test_path):
    """Eğitim ve test verilerini yükler."""
    X_train = sparse.load_npz(X_train_path)
    X_test = sparse.load_npz(X_test_path)
    y_train = pd.read_pickle(y_train_path)
    y_test = pd.read_pickle(y_test_path)
    return X_train, X_test, y_train, y_test

def load_label_encoder(label_encoder_path):
    """Etiket encoder'ını yükler."""
    with open(label_encoder_path, 'rb') as file:
        label_encoder = pickle.load(file)
    return label_encoder

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, label_encoder, epochs=10, batch_size=64):
    """Modeli eğitir ve değerlendirir."""
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )

    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)

    # Classification Report
    report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_filepath = f"models/classification_reports/{model_name.replace(' ', '_').lower()}_classification_report.csv"
    report_df.to_csv(report_filepath)
    print(f"Classification report saved to {report_filepath}")

    return model, history


if __name__ == "__main__":
    vectorizer_path = 'data/prepared_data/tfidf_vectorizer.pkl'
    label_encoder_path = 'data/prepared_data/label_encoder.pkl'
    X_train_path = 'data/prepared_data/X_train.npz'
    X_test_path = 'data/prepared_data/X_test.npz'
    y_train_path = 'data/prepared_data/y_train.pkl'
    y_test_path = 'data/prepared_data/y_test.pkl'

    tfidf_vectorizer, label_encoder = load_objects(vectorizer_path, label_encoder_path)
    X_train, X_test, y_train, y_test = load_data(X_train_path, X_test_path, y_train_path, y_test_path)

    input_dim = X_train.shape[1]
    output_dim = len(label_encoder.classes_)

    models = {
        "Baseline Model": create_baseline_model(input_dim, output_dim),
        "Model Variation 1": create_model_variation_1(input_dim, output_dim),
        "Model Variation 2": create_model_variation_2(input_dim, output_dim),
        "Model Variation 3": create_model_variation_3(input_dim, output_dim),
    }

    os.makedirs('models/classification_reports', exist_ok=True)

    for model_name, model in models.items():
        trained_model, history = train_and_evaluate_model(
            model, X_train, y_train, X_test, y_test, model_name, label_encoder, epochs=10, batch_size=64
        )
        model_filename = f"models/{model_name.replace(' ', '_').lower()}.h5"
        trained_model.save(model_filename)
        print(f"{model_name} saved to {model_filename}\n")
