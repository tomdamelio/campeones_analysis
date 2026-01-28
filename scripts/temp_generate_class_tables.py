import pandas as pd
import os

results_path = r'results/modeling/sub-27_combined_filtered_trend_classification_results.csv'
if not os.path.exists(results_path):
    print(f"File not found: {results_path}")
    exit(1)

df = pd.read_csv(results_path)

with open('scripts/class_tables_output.txt', 'w') as f:
    for dim in ['arousal', 'valence', 'luminance']:
        f.write(f'#### {dim.capitalize()}\\n')
        # Sort by Balanced Accuracy Descending
        sub = df[df['Dimension'] == dim].sort_values('BalancedAccuracy', ascending=False)
        
        f.write('| Fold (Test Video) | Train/Test N | Pos Rate (Test) | Acc | Bal Acc | AUC |\\n')
        f.write('| :--- | :---: | :---: | :---: | :---: | :---: |\\n')
        for _, row in sub.iterrows():
            auc_str = f"{row['AUC']:.3f}" if pd.notna(row['AUC']) else "NaN"
            f.write(f"| {row['TestVideo']} | {row['TrainSize']}/{row['TestSize']} | {row['UpRate_Test']:.2%} | {row['Accuracy']:.3f} | {row['BalancedAccuracy']:.3f} | {auc_str} |\\n")
        
        f.write(f"| **Promedio** | - | - | **{sub['Accuracy'].mean():.3f}** | **{sub['BalancedAccuracy'].mean():.3f}** | **{sub['AUC'].mean():.3f}** |\\n")
        f.write('\\n')
