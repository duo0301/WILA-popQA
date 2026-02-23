import os
import pandas as pd


Languages = ["English", "Arabic", "German", "French", "Italian", "Polish", "Hindi", "Russian", "Chinese"]

# filter_properties = ["has_place_of_birth", "has_occupation", "has_country_of_citizenship", "has_date_of_birth"]

filter_properties = ['has_date_of_death', 'has_place_of_death', 'has_place_of_birth', 'has_date_of_birth', 'has_occupation', 'has_country_of_citizenship']

input_dir = 'data/dataset_v2/entities_properties_matrix'
output_dir = 'data/dataset_v2/filtered_entities_ids'

for lang in Languages:
    # file paths
    input_file = os.path.join(input_dir, lang + '.csv')
    output_file = os.path.join(output_dir,  lang + '.csv')
    
    entities_properties_df = pd.read_csv(input_file)

    # create a mask for filtering
    mask = entities_properties_df[filter_properties[0]] == 1
    for prop in filter_properties[1:]:
        mask = mask & (entities_properties_df[prop] == 1)

    created_entities_df = entities_properties_df[mask]

    # keep only the ids
    created_entities_df = created_entities_df['creator_id']
    created_entities_df.to_csv(output_file, index=False, header=False)