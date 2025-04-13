import pandas as pd
import os

Languages = ["English", "Arabic", "German", "French", "Italian", "Polish", "Hindi", "Russian"]# , "Chinese"]

data = pd.DataFrame(columns=["PropertiesSet", *Languages])

# get coverage of each language
set_properties = []

data_dir = os.path.join(os.path.dirname(__file__), "data", "dataset_v2", "coverage_results_001")

for lang in Languages:
    coverage_file = os.path.join(data_dir, lang +".csv")
    coverage_sets_df = pd.read_csv(coverage_file)

    for index, row in coverage_sets_df.iterrows():
        properties = set([prop for prop in row["properties"].split(", ")])
        coverage = row["coverage"]
        if coverage < 100 :
            continue
        if properties in set_properties: 
            data.loc[data["PropertiesSet"] == str(properties), lang] = int(coverage)
        else:
            new_row = pd.Series({"PropertiesSet": str(properties), 
                                 **{lang: 0 for lang in Languages} })   
            new_row[lang] = int(coverage)
            data = pd.concat([data, new_row.to_frame().T], ignore_index=True)
            set_properties.append(properties)

# Calculate the average coverage for each set of properties
data["Average"] = data[Languages].mean(axis=1)
# Calculate a score for each set of properties
data["Score"] = [ x*y for x, y in zip(data["Average"], data["PropertiesSet"].apply(lambda x: len(x)))]

# the more languages covered the better

sum_lang = [ (data[lang] > 0).astype(int) for lang in Languages]
sum_lang = sum(sum_lang)

data["Score"] = data["Score"] * sum_lang

# Eliminate lines with at least one language with 0 coverage


data = data.sort_values(by=["Score"], ascending=False)

data = data[
    (data[Languages] >= 1).all(axis=1)
]

data.to_csv("coverage_results.csv", index=False)

print(data)