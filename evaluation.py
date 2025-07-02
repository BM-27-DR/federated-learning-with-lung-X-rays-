import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

output_dir = "federated_metrics_plots/resnet18"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("metrics_log1.csv")

metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
for metric in metrics:
    plt.figure(figsize=(8, 5))
    plt.plot(df['Round'], df[metric], marker='o', linestyle='-')
    plt.title(f"{metric} over Rounds")
    plt.xlabel("Round")
    plt.ylabel(metric)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric.lower()}_per_round.png"))
    plt.close()

for idx, row in df.iterrows():
    round_num = row['Round']
    cm = ast.literal_eval(row['ConfusionMatrix']) 

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["COVID-19", "Pneumonia", "Normal"],
                yticklabels=["COVID-19", "Pneumonia", "Normal"])
    plt.title(f"Confusion Matrix - Round {round_num}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_round_{round_num}.png"))
    plt.close()

print(f" All plots saved to: {output_dir}/")
