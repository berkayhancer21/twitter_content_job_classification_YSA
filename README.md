# Social_Media_Profession_Classification_YSA

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [Technologies](#technologies)
- [Results](#results)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Introduction

Social media platforms generate vast amounts of data daily. Classifying these posts into predefined profession categories (e.g., Lawyer, Engineer, Doctor) can provide valuable insights for businesses, researchers, and marketers. This project utilizes Artificial Neural Networks (YSA) to automate and enhance the accuracy of profession classification based on Turkish tweets.

By leveraging Natural Language Processing (NLP) techniques and deep learning architectures, this project aims to develop a robust model capable of accurately predicting user professions from their tweet content. The classification process involves data collection, preprocessing, feature engineering, model training, evaluation, and deployment for real-time predictions.

## Features

- **Data Collection:** Automated extraction of tweets using Twitter API.
- **Data Cleaning:** Removal of URLs, mentions, hashtags, and special characters; conversion to lowercase; stop words elimination.
- **Feature Engineering:** TF-IDF vectorization to convert text data into numerical representations.
- **Model Development:** Implementation of multiple Neural Network architectures, including a Baseline Model and variations with LSTM layers.
- **Evaluation:** Comprehensive performance metrics and visualizations to assess model efficacy.
- **Prediction:** Application of the trained model to predict professions of new Twitter users.

## Project Structure
```plaintext
Social_Media_Profession_Classification_YSA/
│
├── data/
│   ├── unprocessed/
│       ├── Tweets.txt
│       └── tweets_pull_with_api.csv
│   ├── processed/
│       ├── preprocessed_tweets.csv
│       ├── preprocessed_tweets_cleaned.csv
│       ├── preprocessed_tweets_labels_cleaned.csv
│       ├── tweets_job_prediction.csv
│       └── tweets_pull_with_api_cleaned.csv
│   └── prepared_data/
│       ├── tfidf_vectorizer.pkl
│       ├── tfidf_vectorizer.pkl
│       ├── X_test.npz
│       ├── X_train.npz
│       ├── y_test.pkl
│       └── y_train.pkl
│
├── models/
│   ├── baseline_model.h5
│   ├── variation1_model.h5
│   ├── variation2_model.h5
│   ├── variation3_model.h5
│   └── variation4_model.h5
│   └── classification_reports/
│        ├── baseline_model_classification_report.csv
│        ├── model_variation_1_classification_report.csv
│        ├── model_variation_2_classification_report.csv
│        └── model_variation_3_classification_report.csv       
│
├── scripts/
│   ├── __init__.py
│   ├── _0_data_collection_twitter_api.py
│   ├── _1_data_preprocessing.py
│   ├── _2_txt_to_csv_converter.py
│   ├── _3_feature_engineering.py
│   ├── _4_model_variations.py
│   ├── _5_model_training.py
│   ├── _6_evaluation.py
│   └── _7_prediction.py
│
├── README.md
└── requirements.txt
```

## Installation

### Prerequisites

- **Python 3.8 or higher**
- **pip** (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/your_username/Social_Media_Profession_Classification_YSA.git
cd Social_Media_Profession_Classification_YSA

