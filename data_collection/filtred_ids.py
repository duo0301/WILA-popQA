import pandas as pd


Languages = ["English", "Arabic", "German", "French", "Italian", "Polish", "Hindi", "Russian", "Chinese"]

filter_properties = ["has_place_of_birth", "has_occupation", "has_country_of_citizenship", "has_date_of_birth"]

for lang in Languages:
    # file paths
    input_file = 'entities_properties_' + lang + '.csv'
    output_file = 'filtered_entities_ids_' + lang + '.csv'
    
    entities_properties_df = pd.read_csv(input_file)

    # create a mask for filtering
    mask = entities_properties_df[filter_properties[0]] == 1
    for prop in filter_properties[1:]:
        mask = mask & (entities_properties_df[prop] == 1)
    
    created_entities_df = entities_properties_df[mask]
    created_entities_df.to_csv(output_file, index=False)