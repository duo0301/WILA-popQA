import pandas as pd
import os

data_dir = os.path.join(os.path.dirname(__file__), "data", "dataset_v2", "coverage_results")
# file names are the languages, remove the .csv extension
languages = [os.path.splitext(file)[0] for file in os.listdir(data_dir)]

data = pd.DataFrame(columns=["PropertiesSet", "size", *languages])

excluded_properties = ["has_gender"]

# get coverage of each language
set_properties = []
for lang in languages:
    coverage_file = os.path.join(data_dir, lang +".csv")
    coverage_sets_df = pd.read_csv(coverage_file)

    # exclude row that has a property set that contains an excluded property
    coverage_sets_df = coverage_sets_df[~coverage_sets_df["properties"].str.contains("|".join(excluded_properties))]

    for index, row in coverage_sets_df.iterrows():
        properties = set([prop for prop in row["properties"].split(", ")])
        coverage = row["coverage"]
        if coverage < 100 :
            continue
        if properties in set_properties:
            data.loc[data["PropertiesSet"] == str(properties), lang] = int(coverage)
        else:
            new_row = pd.Series({"PropertiesSet": str(properties),
                                 "size": len(properties),
                                 **{lang: 0 for lang in languages} })
            new_row[lang] = int(coverage)

            data = pd.concat([data, new_row.to_frame().T], ignore_index=True)
            set_properties.append(properties)

# Calculate the average coverage for each set of properties
data["Average"] = data[languages].mean(axis=1)
# Calculate a score for each set of properties
data["Score"] = [ x*y for x, y in zip(data["Average"], data["size"])]

# the more languages covered the better

sum_lang = [ (data[lang] > 0).astype(int) for lang in languages]
sum_lang = sum(sum_lang)

data["Score"] = data["Score"] * sum_lang

# Score =  Average * size * sum_lang : the average coverage of the properties set * the number of properties in the set * the number of languages covered by the set

# Eliminate lines with at least one language with 0 coverage

data = data.sort_values(by=["Score"], ascending=False)

data = data[
    (data[languages] >= 1).all(axis=1)
]

data.to_csv(
    os.path.join(os.path.dirname(__file__), "data", "dataset_v2", "coverage_results.csv"), 
    index=False)

print(data)
