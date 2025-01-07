import tensorflow as tf
from keras.src.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

def create_baseline_model(input_dim, output_dim):
    """Basit bir YSA modeli."""
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='softmax'))
    return model

def create_model_variation_1(input_dim, output_dim):
    """Daha fazla katman ve Batch Normalization eklenmiş YSA modeli."""
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim, activation='softmax'))
    return model

def create_model_variation_2(input_dim, output_dim):
    """Daha fazla nöron ve Dropout oranını artırılmış YSA modeli."""
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(output_dim, activation='softmax'))
    return model

def create_model_variation_3(input_dim, output_dim):
    """Farklı aktivasyon fonksiyonları ve daha düşük Dropout oranı ile YSA modeli."""
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(output_dim, activation='softmax'))
    return model

