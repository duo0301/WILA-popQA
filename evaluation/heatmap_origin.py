import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the data
df_avg_language = pd.read_csv('evaluation_results_by_language_SameLanguage.csv')

# Create output directory if it doesn't exist
output_dir = 'heatmaps'
os.makedirs(output_dir, exist_ok=True)

# List of metrics to visualize
metrics = ['Precision', 'Recall', 'F1']

# Loop through each unique ModelName and RunID
for model_name in df_avg_language['ModelName'].unique():
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    for run_id in df_avg_language['RunID'].unique():
        run_dir = os.path.join(model_dir, f'RunID{run_id}')
        os.makedirs(run_dir, exist_ok=True)
        
        for metric in metrics:
            # Filter the DataFrame for the current ModelName and RunID
            filtered_df = df_avg_language[(df_avg_language['ModelName'] == model_name) &
                                          (df_avg_language['RunID'] == run_id)]
            
            # Pivot the DataFrame to get the mean metric score for each (PromptLang, EntityLang) pair
            heatmap_data = filtered_df.pivot_table(values=metric, index='PromptLang', columns='EntityLang', aggfunc='mean')
            
            # Create the heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar_kws={'label': f'{metric} Score'})
            plt.title(f'Heatmap of {metric} Scores by Prompt Language and Entity Language\n(Model: {model_name}, RunID: {run_id})')
            plt.xlabel('Entity Language')
            plt.ylabel('Prompt Language')
            
            # Save the heatmap to a file
            file_name = f'heatmap_{metric}.png'
            file_path = os.path.join(run_dir, file_name)
            plt.savefig(file_path)
            plt.close()

print(f"Heatmaps saved in '{output_dir}' directory.")
