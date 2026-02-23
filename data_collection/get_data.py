'''
This script queries a SPARQL endpoint to retrieve properties and sitelinks for a list of Wikidata QIDs.
It formats the results into a structured dictionary and saves them as JSON files.

Here's a detailed breakdown of the inputs and outputs:

+ Inputs:
  - qids: A list of QIDs (Wikidata entity IDs) to be queried.
  - url: The URL endpoint for querying Wikidata.
  - label_languages: A list of languages for which labels should be retrieved.
  - properties: A list of properties to be retrieved for each QID.
  - batch_size: The number of QIDs to process in each batch.
  - data_output_path: The directory path where the results JSON file will be saved.
  - lang: The language code of the labels to be retrieved.

+ Outputs:
  - A JSON file: The results are saved to a JSON file named {lang}.json in the specified data_output_path. 
  It contains a dictionary where each key is a QID and the value is another dictionary of properties and their corresponding values labled in different languages.

+ Functions:
  - get_properties_query(qids, languages, properties): Generates a SPARQL query to retrieve properties for given QIDs.
  - get_properties(url, qids, languages, properties): Executes the SPARQL query to get properties for given QIDs.
  - get_sitelinks(SPARQL_ENDPOINT, entity_id): Queries the SPARQL endpoint for sitelinks of a given entity.
  - get_sitelink_count(url, entity_ids): Queries the SPARQL endpoint for sitelink counts of given entities.
  - format_results(results, sitelinks_result, labelLanguages): Formats the SPARQL query results into a structured dictionary.
'''

import requests, os
import json
from tqdm import tqdm

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
LOCAL_ENDPOINT = "http://localhost:1234/api/endpoint/sparql"

def get_properties_query(qids, languages, properties):
    
    valueLabels = " ".join([f'?valueLabel_{lang}' for lang in languages])
    properties = " ".join([f'wdt:{prop}' for prop in properties])
    optional_langs = " ".join([f'OPTIONAL {{ ?value rdfs:label ?valueLabel_{lang} FILTER(LANG(?valueLabel_{lang}) = "{lang}") }}' for lang in languages])
    qids = " ".join([f'wd:{qid}' for qid in qids])

    qidLabels = " ".join([f'?qidLabel_{lang}' for lang in languages])
    optional_qid_langs = " ".join([f'OPTIONAL {{ ?qid rdfs:label ?qidLabel_{lang} FILTER(LANG(?qidLabel_{lang}) = "{lang}") }}' for lang in languages])

    query = f'''
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?qid ?property ?value {valueLabels} {qidLabels}
        WHERE {{
          VALUES ?qid {{ {qids} }}     
          VALUES ?property {{ {properties} }}
          ?qid ?property ?value .
          {optional_langs}
          {optional_qid_langs}
        }}
    '''

    return query

def get_properties(url, qids, languages, properties):

    query = get_properties_query(qids, languages, properties)
    headers = {
        'Accept': 'application/sparql-results+json'
    }
    response = requests.get(url, params={'query': query}, headers=headers)
    results = response.json()

    return results

# Function to query Wikidata SPARQL endpoint for sitelinks
def get_sitelinks(SPARQL_ENDPOINT, entity_id):
    query = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

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
def get_sitelink_count(url, entity_ids):
    ids_str = " ".join([f"wd:{entity_id}" for entity_id in entity_ids])
    query = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX schema: <http://schema.org/>

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
    response = requests.get(url, params={"query": query}, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error: ", response.status_code, response.text)
        return None
    
def format_results(results, sitelinks_result, labelLanguages):
    """
    Formats the results from a SPARQL query into a structured dictionary.
    Args:
        results (dict): The main results from the SPARQL query containing QIDs, properties, and their values.
        sitelinks_result (dict or None): Additional results containing sitelink counts for each QID.
        labelLanguages (list): List of language codes to extract labels for QIDs and values.
    Returns:
        dict: A dictionary where each key is a QID and the value is another dictionary containing:
            - 'labels': A dictionary of labels for the QID in different languages.
            - 'labels_count': The number of labels for the QID.
            - Properties (e.g., "P19", "P106"): Each property contains:
                - 'values': A list of dictionaries, each containing:
                    - 'qid': The QID of the value.
                    - 'labels': A dictionary of labels for the value in different languages.
                    - 'labels_count': The number of labels for the value.
                - 'count': The number of values for the property.
            - 'sitelinks': The number of sitelinks for the QID (if sitelinks_result is provided).
    Example:
        formatted_results = format_results(results, sitelinks_result, ['en', 'fr'])
    """


    formatted_results = {}
    # Iterate through all results

    # Add QID labels to the results
    for result in results['results']['bindings']:
        qid = result['qid']['value'].split('/')[-1]
        for lang in labelLanguages:
            qid_label_key = f"qidLabel_{lang}"
            if qid_label_key in result:
                if qid not in formatted_results:
                    formatted_results[qid] = {}
                if 'labels' not in formatted_results[qid]:
                    formatted_results[qid]['labels'] = {}
                formatted_results[qid]['labels'][lang] = result[qid_label_key]['value']
                # Count the number of labels for the current QID
                formatted_results[qid]['labels_count'] = len(formatted_results[qid]['labels'])

    for result in results['results']['bindings']:
        # Extract QID and Property ID
        qid = result['qid']['value'].split('/')[-1]
        property_id = result['property']['value'].split('/')[-1]  # e.g., "P19", "P106"

        # Initialize QID entry if it doesn't exist
        if qid not in formatted_results:
            formatted_results[qid] = {}

        # Prepare value entry
        value_entry = {
            "qid": result['value']['value'],
            "labels": {}
        }

        # Add language-specific labels if they exist
        label_count = 0
        for lang in labelLanguages:
            label_key = f"valueLabel_{lang}"
            if label_key in result:
                value_entry['labels'][lang] = result[label_key]['value']
                label_count += 1

        # Add label count to the value entry
        value_entry['labels_count'] = label_count

        # Handle multiple values for the same property
        if property_id in formatted_results[qid]:
            if isinstance(formatted_results[qid][property_id]['values'], list):
                formatted_results[qid][property_id]['values'].append(value_entry)
            else:
                formatted_results[qid][property_id]['values'] = [formatted_results[qid][property_id]['values'], value_entry]
        else:
            formatted_results[qid][property_id] = {}
            # First entry for the property
            formatted_results[qid][property_id]['values'] = [value_entry] 
            # Count the number of values for the current property
        
        formatted_results[qid][property_id]['count'] = len(formatted_results[qid][property_id])



    # Add sitelinks count to the results
    if sitelinks_result is not None:
        for result in sitelinks_result['results']['bindings']:
            qid = result['entity']['value'].split('/')[-1]
            sitelink_count = int(result['sitelink_count']['value'])
            if qid not in formatted_results:
                continue
            else:
                formatted_results[qid]['sitelinks'] = sitelink_count

    return formatted_results


# Selected properties after calculate property set coverage 
#properties = ["P19","P569", "P106", "P27"]
properties = ['P570', 'P20', 'P19', 'P569', 'P106', 'P27']
              
# Languages to get the value labels
label_languages = ["en", "de", "fr", "ru", "hi", "zh", "it", "pl", "ar"]
languages = ["German", "Italian", "Polish", "French", "English"]

# Read filtred_ids to get the qids
data_path = "data/dataset_v2/filtered_entities_ids/"
data_output_path = "data/dataset_v2/entities_data/"

batch_size = 200

for lang in tqdm(languages, desc="Processing languages"):
    list_qids_path = os.path.join(data_path, f'{lang}.csv') 
    with open(list_qids_path, 'r') as f:
        qids = f.readlines()
        qids = [qid.strip() for qid in qids]

    print(f"Querying {len(qids)} QIDs for properties, for {lang} language speaking entities")
    # Get the results

    all_results = []
    
    for i in tqdm(range(0, len(qids), batch_size), desc="Processing batches"):
        batch_qids = qids[i:i + batch_size]
        batch_results = get_properties(LOCAL_ENDPOINT, batch_qids, label_languages, properties)
        sitelinks_result = get_sitelink_count(WIKIDATA_ENDPOINT, batch_qids)
        # sitelinks_result = None # we can use it when we use wikidata_full dump (truthy dump, don't contain sitelinks)
        formatted_batch_results = format_results(batch_results, sitelinks_result, label_languages)
        all_results.append(formatted_batch_results)

    # Combine all batch results into a single dictionary
    combined_results = {}
    for batch_results in all_results:

        # if sitelinks_result is < 9, we remove the qid from the batch_results
        for qid, properties in batch_results.items():
            
            if qid not in combined_results:
                combined_results[qid] = properties
            else:
                for prop, values in properties.items():
                    if prop not in combined_results[qid]:
                        combined_results[qid][prop] = values
                    else:
                        combined_results[qid][prop].extend(values)


    filtered_results = {}
    for qid in combined_results:
        if 'sitelinks' in combined_results[qid]:
            if combined_results[qid]['sitelinks']>= 9:
                filtered_results[qid] = combined_results[qid]

    print("filtered_results :", len(filtered_results), "combined_results:", len(combined_results))

    # Save the results
    results_path = os.path.join(data_output_path, f"{lang}.json")
    if not os.path.exists(data_output_path):
        os.makedirs(data_output_path)
    with open(results_path, 'w') as f:
        json.dump(filtered_results, f, indent=4, ensure_ascii=False)