import pandas as pd

Languages = ["English", "Arabic", "German", "French", "Italian", "Polish", "Hindi", "Russian", "Chinese"]

data = pd.DataFrame(columns=["PropertiesSet", "English", "Arabic", "German", "French", "Italian", "Polish", "Hindi", "Russian", "Chinese"])

# get coverage of each language
set_properties = []

for lang in Languages:
    coverage_sets_df = pd.read_csv("coverage_results_" + lang + ".csv").head(10000)
    '''
    properties,num_properties,coverage,score 
    "has_place_of_birth, has_country_of_citizenship, has_occupation, has_date_of_birth",4,8643,34572
    ... 
    '''    
    for index, row in coverage_sets_df.iterrows():
        properties = set([prop for prop in row["properties"].split(", ")])
        coverage = row["coverage"]
        if coverage < 100 :
            continue
        if properties in set_properties: 
            data.loc[data["PropertiesSet"] == str(properties), lang] = int(coverage)
        else:
            new_row = pd.Series({"PropertiesSet": str(properties), 
                                 "English": 0, "Arabic": 0, "German": 0 , 
                                 "French": 0, "Italian": 0, "Polish": 0, "Hindi": 0, "Russian": 0, "Chinese":0 })   
            new_row[lang] = int(coverage)
            data = pd.concat([data, new_row.to_frame().T], ignore_index=True)
            set_properties.append(properties)

# Calculate the average coverage for each set of properties
data["Average"] = data[["English", "Arabic", "German", "French", "Italian", "Polish", "Hindi", "Russian", "Chinese"]].mean(axis=1)
# Calculate a score for each set of properties
data["Score"] = [ x*y for x, y in zip(data["Average"], data["PropertiesSet"].apply(lambda x: len(x)))]

# the more languages covered the better

data["Score"] = data["Score"] * (
    (data["English"] > 0).astype(int)
    + (data["Arabic"] > 0).astype(int)
    + (data["German"] > 0).astype(int)
    + (data["French"] > 0).astype(int)
    + (data["Italian"] > 0).astype(int)
    + (data["Polish"] > 0).astype(int)
    + (data["Hindi"] > 0).astype(int)
    + (data["Russian"] > 0).astype(int)
    + (data["Chinese"] > 0).astype(int)
)

# Eliminate lines with at least one language with 0 coverage


data = data.sort_values(by=["Score"], ascending=False)

data = data[
    (data[["English", "Arabic", "German", "French", "Italian", "Polish", "Hindi", "Russian", "Chinese"]] >= 1).all(axis=1)
]

data.to_csv("coverage_results.csv", index=False)

print(data)