import pandas as pd
import os
import argparse

def main(data_dir=None, output_file=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "data", "dataset_v4", "author", "coverage_results")
    if output_file is None:
        output_file = os.path.join(os.path.dirname(__file__), "data", "dataset_v4", "coverage_results.csv")
    
    languages = [os.path.splitext(file)[0] for file in os.listdir(data_dir)]
    data = pd.DataFrame(columns=["PropertiesSet", "size", *languages])
    excluded_properties = ["has_gender"]
    
    set_properties = []
    for lang in languages:
        coverage_file = os.path.join(data_dir, lang + ".csv")
        coverage_sets_df = pd.read_csv(coverage_file)
        coverage_sets_df = coverage_sets_df[~coverage_sets_df["properties"].str.contains("|".join(excluded_properties))]
        
        for index, row in coverage_sets_df.iterrows():
            properties = set([prop for prop in row["properties"].split(", ")])
            coverage = row["coverage"]
            if coverage < 100:
                continue
            if properties in set_properties:
                data.loc[data["PropertiesSet"] == str(properties), lang] = int(coverage)
            else:
                new_row = pd.Series({"PropertiesSet": str(properties), "size": len(properties), **{lang: 0 for lang in languages}})
                new_row[lang] = int(coverage)
                data = pd.concat([data, new_row.to_frame().T], ignore_index=True)
                set_properties.append(properties)
    
    data["Average"] = data[languages].mean(axis=1)
    data["Score"] = [x * y for x, y in zip(data["Average"], data["size"])]
    sum_lang = sum([(data[lang] > 0).astype(int) for lang in languages])
    data["Score"] = data["Score"] * sum_lang
    data = data.sort_values(by=["Score"], ascending=False)
    data = data[(data[languages] >= 1).all(axis=1)]
    data.to_csv(output_file, index=False)
    print(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate coverage results per language")
    parser.add_argument("--data-dir", help="Input data directory")
    parser.add_argument("--output", help="Output CSV file path")
    args = parser.parse_args()
    main(args.data_dir, args.output)
