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

## Installation

### Prerequisites

- **Python 3.8 or higher**
- **pip** (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/your_username/Social_Media_Profession_Classification_YSA.git
cd Social_Media_Profession_Classification_YSA

