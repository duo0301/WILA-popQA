import pandas as pd
import itertools
from itertools import chain, combinations


def powerset(iterable):
    "powerset([p1, p2, p3]) --> () (p1,) (p2,) (p3,) (p1,p2) (p1,p3) (p2,p3) (p1,p2,p3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

Languages = ["English", "Arabic", "German", "French", "Italian", "Polish", "Hindi", "Russian", "Chinese"]

for lang in Languages:

    # file paths
    input_file = 'entities_properties_' + lang + '.csv'
    output_file = 'coverage_results_' + lang + '.csv'

    df = pd.read_csv(input_file)
    # List of property columns (excluding 'creator_id')
    properties = [col for col in df.columns if col != 'creator_id']

    # Create a mapping of properties to the set of creators who have them
    property_creator_map = {}
    for prop in properties:
        # Get creators who have the property
        creators_with_prop = set(df[df[prop] == 1]['creator_id'])
        property_creator_map[prop] = creators_with_prop

    # Generate all non-empty subsets of properties
    property_subsets = list(powerset(properties))

    coverage_results = []

    for subset in property_subsets:
        # Initialize with the set of creators who have the first property
        subset_creators = property_creator_map[subset[0]]
        # Take the intersection with the creators of each subsequent property
        for prop in subset[1:]:
            subset_creators = subset_creators.intersection(property_creator_map[prop])
        # Record the coverage
        coverage = len(subset_creators)
        coverage_results.append({
            'properties': subset,
            'num_properties': len(subset),
            'coverage': coverage,
            'creators': subset_creators
        })

    # Convert results to a DataFrame
    coverage_df = pd.DataFrame([
        {
            'properties': ', '.join(subset),
            'num_properties': len(subset),
            'coverage': result['coverage']
        }
        for result in coverage_results
        for subset in [result['properties']]
    ])

    # Sort the DataFrame by coverage (descending) and number of properties (ascending)
    coverage_df['score'] = coverage_df['coverage'] * coverage_df['num_properties']
    coverage_df = coverage_df.sort_values(by=['score'], ascending=False).reset_index(drop=True)

    # drop score == 0
    coverage_df = coverage_df[coverage_df['score'] > 0]

    # coverage_df = coverage_df.sort_values(by=['num_properties','coverage'], ascending=[True, False]).reset_index(drop=True)

    # Display 
    print(coverage_df.head(20))
    coverage_df.to_csv(output_file, index=False)
    print(f"\nCoverage results have been written to '{output_file}'.")


#from mlxtend.frequent_patterns import apriori
# Prepare the data for the Apriori algorithm
# transactions = df[properties].astype(bool)
# frequent_itemsets = apriori(transactions, min_support=0.1, use_colnames=True)
# frequent_itemsets['coverage'] = frequent_itemsets['support'] * len(df)
# Sort by coverage
# frequent_itemsets = frequent_itemsets.sort_values(by='coverage', ascending=False)
# Display the results
# print(frequent_itemsets[['itemsets', 'coverage']])
