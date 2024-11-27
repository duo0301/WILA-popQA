import requests
import os
import json

# Directory containing your filtered files
filtered_files_dir = '.'

# Wikidata SPARQL endpoint
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# Define language sets
language_sets = {
    "English": 'en',
    "Arabic": 'ar',
    "German": 'de',
    "French": 'fr',
    "Kannada": 'kn',
    "Italian": 'it',
    "Polish": 'pl',
    "Hindi": 'hi',
    "Russian": 'ru',
    "Chinese": 'zh'
}

# Function to query Wikidata SPARQL endpoint for sitelinks
def get_sitelinks(entity_id):
    query = f"""
        SELECT distinct ?sitelink 
        WHERE {{
            ?sitelink   schema:about wd:{entity_id};
                        schema:isPartOf ?wiki .
            FILTER(REGEX(STR(?wiki), "^https://[a-z-]+\\.wikipedia\\.org/$"))
        }}
    """
    headers = {
        "Accept": "application/sparql-results+json"
    }
    response = requests.get(SPARQL_ENDPOINT, params={"query": query}, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error: ", response.status_code, response.text)
        return None

# Function to query Wikidata SPARQL endpoint for sitelink count
def get_sitelink_count(entity_ids):
    ids_str = " ".join([f"wd:{entity_id}" for entity_id in entity_ids])
    query = f"""
        SELECT distinct ?entity (count(?sitelink) as ?sitelink_count) 
        WHERE {{
            VALUES ?entity {{{ids_str}}}
            ?sitelink   schema:about ?entity;
                        schema:isPartOf ?wiki .
            FILTER(REGEX(STR(?wiki), "^https://[a-z-]+\\\\.wikipedia\\\\.org/$"))
        }}
        Group by ?entity
    """
    headers = {
        "Accept": "application/sparql-results+json"
    }
    response = requests.get(SPARQL_ENDPOINT, params={"query": query}, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error: ", response.status_code, response.text)
        return None

# find filtred_creator_ids_{language}.txt
file_names = [file_name for file_name in os.listdir(filtered_files_dir) if file_name.startswith("filtered_entities_ids")]

for file_name in file_names:
    file_path = os.path.join(filtered_files_dir, file_name)
    print(f"Reading file: {file_name}")

    with open(file_path, 'r') as f:
        entity_ids = f.readlines()
        entity_ids = [entity_id.strip() for entity_id in entity_ids]
        print(f"Retrieved {len(entity_ids)} entity IDs")
        if entity_ids:
            result = get_sitelink_count(entity_ids)
            if result:
                result_to_save = dict()
                for binding in result['results']['bindings']:
                    entity = binding['entity']['value']
                    entity = entity.split('/')[-1]
                    sitelink_count = binding['sitelink_count']['value']
                    result_to_save[entity] = int(sitelink_count)

                # save result to file
                language_group = file_name.split('_')[-1].split('.')[0] 
                result_file_name = f"sitelink_count_{language_group}.txt"
                # order the result by sitelink count
                result_to_save = {k: v for k, v in sorted(result_to_save.items(), key=lambda item: item[1], reverse=True)}
                with open(result_file_name, 'w') as file:
                    for entity, sitelink_count in result_to_save.items():
                        file.write(f"{entity}, {sitelink_count}\n")
            
        
