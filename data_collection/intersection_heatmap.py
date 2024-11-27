import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

# List of file names
files = [
    "filtered_creator_ids_Arabic.txt",
    "filtered_creator_ids_Chinese.txt",
    "filtered_creator_ids_English.txt",
    "filtered_creator_ids_French.txt",
    "filtered_creator_ids_German.txt",
    "filtered_creator_ids_Hindi.txt",
    "filtered_creator_ids_Italian.txt",
    "filtered_creator_ids_Polish.txt",
    "filtered_creator_ids_Russian.txt",
]

# Read identifiers from each file into a dictionary
identifiers = {}
for file in files:
    language = file.split('_')[-1].split('.')[0]  # Extract language from file name
    with open(file, 'r') as f:
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
