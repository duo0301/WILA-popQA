import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

# List of file names
files = [
    "Arabic.txt",
    "Chinese.txt",
    "English.txt",
    "French.txt",
    "German.txt",
    "Hindi.txt",
    "Italian.txt",
    "Polish.txt",
    "Russian.txt",
]

root_dir = 'data/dataset_v2/entities_ids'

# Read identifiers from each file into a dictionary
identifiers = {}
for file in files:
    language = file.split('_')[-1].split('.')[0]  # Extract language from file name
    with open(os.path.join(root_dir, file), 'r') as f:
        identifiers[language] = set(f.read().splitlines())

# Create an empty DataFrame to store intersection counts
languages = list(identifiers.keys())
intersection_matrix = pd.DataFrame(index=languages, columns=languages, dtype=float)

# Calculate intersections between each pair of languages
for lang1, lang2 in combinations(languages, 2):
    for lang3 in languages:
        if lang3 != lang1 and lang3 != lang2:
            intersection_count = len(identifiers[lang1].intersection(identifiers[lang2], identifiers[lang3]))
            intersection_matrix.loc[lang1 + ', ' + lang2, lang3] = intersection_count

# replace NaN values with 0
intersection_matrix.fillna(0, inplace=True)

# drop row with no combinations
intersection_matrix = intersection_matrix[intersection_matrix.index.str.contains(',')]

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(intersection_matrix, annot=True, fmt=".0f", cmap="Blues", cbar=True, linewidths=0.5)
plt.title("Intersection of Identifiers Between Languages")
plt.xlabel("Language")
plt.ylabel("Language")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
