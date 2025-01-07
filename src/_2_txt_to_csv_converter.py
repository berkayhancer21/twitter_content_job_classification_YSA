import pandas as pd
import os

def txt_to_csv(input_filepath, output_filepath):
    """
    .txt dosyasını .csv formatına dönüştürür.
    Her satırda bir tweet ve onun meslek etiketi bulunmalıdır
    Aralarında tab boşluk olduğundan dolayı bu csv dönüşümünü
    yapabiliriz.
    """
    tweets = []
    labels = []

    with open(input_filepath, 'r', encoding='utf-8') as file:
        for line in file:
            # Her satırın formatı: tweet\tmeslek
            parts = line.strip().split('\t')
            if len(parts) == 2:
                tweet, label = parts
                tweets.append(tweet)
                labels.append(label)
            else:
                print(f"Skipping malformed line: {line}")

    df = pd.DataFrame({
        'tweet': tweets,
        'label': labels
    })

    df.to_csv(output_filepath, index=False, encoding='utf-8')
    print(f"Converted {len(df)} entries to {output_filepath}")


def main():
    input_path = 'data/unprocessed/Tweets.txt'
    output_path = 'data/processed/preprocessed_tweets.csv'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    txt_to_csv(input_path, output_path)


if __name__ == "__main__":
    main()
