import os
import pandas as pd
import itertools
from itertools import chain, combinations
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def powerset(iterable):
    "powerset([p1, p2, p3]) --> () (p1,) (p2,) (p3,) (p1,p2) (p1,p3) (p2,p3) (p1,p2,p3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

Languages = ["English", "Arabic", "German", "French", "Italian", "Polish", "Hindi", "Russian", "Chinese"]
input_dir = 'data/dataset_v2/entities_properties_matrix'
output_dir = 'data/dataset_v2/coverage_results'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_language(lang, position=0):
    input_file_path = os.path.join(input_dir, lang+'.csv')
    output_file_path = os.path.join(output_dir, lang+'.csv')
    
    df = pd.read_csv(input_file_path)
    properties = [col for col in df.columns if col != 'creator_id']

    # Create a mapping of properties to the set of creators who have them
    property_creator_map = {}
    for prop in properties:
        property_creator_map[prop] = set(df[df[prop] == 1]['creator_id'])

    # Generate all non-empty subsets of properties
    property_subsets = list(powerset(properties))

    coverage_results = []

    # Create a progress bar for property subsets processing
    with tqdm(property_subsets, 
              desc=f"{lang} (processing subsets of properties)", 
              position=position + 1,  # Position after main progress bar
              leave=True) as subset_pbar:
        for subset in subset_pbar:
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
    coverage_df.to_csv(output_file_path, index=False)
    print(f"\nCoverage results have been written to '{output_file_path}'.")
    
    return coverage_df

# Main execution
if __name__ == '__main__':
    # Use number of CPU cores minus 1 to avoid system overload
    num_cores = multiprocessing.cpu_count() - 1
    
    # Create main progress bar for languages
    main_pbar = tqdm(total=len(Languages), 
                    desc="Processed languages", 
                    position=0,
                    leave=True)
    
    def process_with_progress(language):
        result = process_language(language, position=Languages.index(language))
        main_pbar.update(1)
        return result
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(process_with_progress, Languages))
    
    # Close main progress bar
    main_pbar.close()
