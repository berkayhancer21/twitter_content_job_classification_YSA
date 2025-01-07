import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_classification_reports(models_dir):
    """
    Her modelin classification report'unu yükler.
    """
    performans = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }

    classification_reports_dir = os.path.join(models_dir, 'classification_reports')

    for report_file in os.listdir(classification_reports_dir):
        if report_file.endswith('_classification_report.csv'):
            model_name = report_file.replace('_classification_report.csv', '').replace('_', ' ').title()
            report_path = os.path.join(classification_reports_dir, report_file)
            report_df = pd.read_csv(report_path, index_col=0)

            try:
                accuracy = report_df.loc['accuracy', 'precision']
                precision = report_df.loc['weighted avg', 'precision']
                recall = report_df.loc['weighted avg', 'recall']
                f1_score = report_df.loc['weighted avg', 'f1-score']
            except KeyError as e:
                print(f"KeyError: {e} in {report_file}. Skipping this report.")
                continue

            performans['Model'].append(model_name)
            performans['Accuracy'].append(accuracy)
            performans['Precision'].append(precision)
            performans['Recall'].append(recall)
            performans['F1-Score'].append(f1_score)

    df_performans = pd.DataFrame(performans)
    return df_performans

def plot_performance(df):
    """Performans metriklerini görselleştirir."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    plt.figure(figsize=(16, 8))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        sns.barplot(
            x='Model',
            y=metric,
            data=df,
            hue='Model',
            palette='viridis'
        )
        plt.title(f'Model {metric} Comparison')
        plt.ylabel(metric)
        plt.xlabel('Model')
        plt.ylim(0, 1)
        plt.legend([], [], frameon=False)

    plt.tight_layout()
    plt.show()


def main():
    models_dir = 'models'

    df_performans = load_classification_reports(models_dir)

    print(df_performans)

    plot_performance(df_performans)


if __name__ == "__main__":
    main()
