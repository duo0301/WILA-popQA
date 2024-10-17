import pandas as pd
import numpy as np
from scipy import stats

#find z-score for given list of entities 
entities_to_process = ['Q178995', 'Q866', 'Q355', 'Q30', "Q12219365", "Q258541", "Q27942633", "Q214115"]
#the first entities are from the first ~10 entities to check the outliers 
#here I normalized QRank by dividing /1000, but there are probably better strategies. Still WIP 

# Read the CSV file
csv_file = 'qrank.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(csv_file, dtype={'Entity': 'object', 'QRank': 'float64'})

# Sort the data by QRank (should already be so, just in case)
df = df.sort_values('QRank', ascending=False)

# Compute mean, standard deviation of normalized QRank
df['NormalizedQRank'] = df['QRank'] / 1000
total_count = len(df)
qrank_mean = df['NormalizedQRank'].mean()
qrank_std = df['NormalizedQRank'].std()

print(f"Total entities: {total_count}")
print(f"Normalized QRank Mean: {qrank_mean}")
print(f"Normalized QRank Std Dev: {qrank_std}")

def get_entity_info(wikidata_id):
    result = df[df['Entity'] == wikidata_id]
    if not result.empty:
        qrank = result['QRank'].values[0]
        normalized_qrank = qrank / 1000
        rank = (df['QRank'] > qrank).sum() + 1
        percentile = 100 * (1 - (rank - 1) / total_count)
        z_score = (normalized_qrank - qrank_mean) / qrank_std
        return {
            'QRank': qrank,
            'NormalizedQRank': normalized_qrank,
            'Rank': rank,
            'Percentile': percentile,
            'Z_Score': z_score
        }
    return None

# Example usage
entity_id = 'Q27942633'
info = get_entity_info(entity_id)

if info:
    print(f"\nEntity: {entity_id}")
    print(f"QRank: {info['QRank']}")
    print(f"Normalized QRank: {info['NormalizedQRank']:.6f}")
    print(f"Rank: {info['Rank']}")
    print(f"Percentile: {info['Percentile']:.2f}")
    print(f"Z-Score: {info['Z_Score']:.2f}")
else:
    print(f"Entity {entity_id} not found in the dataset.")

# Function to process entities in batches
def process_entities_batch(entity_ids):
    results = []
    batch_df = df[df['Entity'].isin(entity_ids)]
    for _, row in batch_df.iterrows():
        entity_id = row['Entity']
        qrank = row['QRank']
        normalized_qrank = qrank / 1000
        rank = (df['QRank'] > qrank).sum() + 1
        percentile = 100 * (1 - (rank - 1) / total_count)
        z_score = (normalized_qrank - qrank_mean) / qrank_std
        results.append({
            'Entity': entity_id,
            'QRank': qrank,
            'NormalizedQRank': normalized_qrank,
            'Rank': rank,
            'Percentile': percentile,
            'Z_Score': z_score
        })
    return results

least_popular = df.loc[df['QRank'].idxmin()]

print("Least popular entity:")
print(f"Entity ID: {least_popular['Entity']}")
print(f"QRank: {least_popular['QRank']}")
print(f"Normalized QRank: {least_popular['NormalizedQRank']:.6f}")

# Example of processing multiple entities
batch_results = process_entities_batch(entities_to_process)

print("\nBatch processing results:")
for result in batch_results:
    print(f"Entity: {result['Entity']}")
    print(f"QRank: {result['QRank']}")
    print(f"Normalized QRank: {result['NormalizedQRank']:.6f}")
    print(f"Rank: {result['Rank']}")
    print(f"Percentile: {result['Percentile']:.2f}")
    print(f"Z-Score: {result['Z_Score']:.2f}")
    print()