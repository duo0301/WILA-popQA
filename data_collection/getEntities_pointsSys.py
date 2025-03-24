import os
from SPARQLWrapper import SPARQLWrapper, JSON
import time
from tqdm import tqdm
import random
import pandas as pd

def query_wikidata(sparql_query, add_prefix=True):
    # endpoint_url = "https://query.wikidata.org/sparql"
    endpoint_url = "http://localhost:1234/api/endpoint/sparql"
    
    sparql = SPARQLWrapper(endpoint_url)
    if add_prefix:
        sparql.setQuery(f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        {sparql_query}
        """)
    else:
        sparql.setQuery(sparql_query)

    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results

def build_select_query(values_clause):
    return f"""
        SELECT DISTINCT ?creator
            (IF(BOUND(?place_of_birth), 1, 0) AS ?has_place_of_birth)
            (IF(BOUND(?place_of_death), 1, 0) AS ?has_place_of_death)
            (IF(BOUND(?gender), 1, 0) AS ?has_gender) 
            (IF(BOUND(?father), 1, 0) AS ?has_father) 
            (IF(BOUND(?mother), 1, 0) AS ?has_mother)
            (IF(BOUND(?spouse), 1, 0) AS ?has_spouse)
            (IF(BOUND(?country_of_citizenship), 1, 0) AS ?has_country_of_citizenship)
            (IF(BOUND(?position_held), 1, 0) AS ?has_position_held)
            (IF(BOUND(?child), 1, 0) AS ?has_child)
            (IF(BOUND(?educated_at), 1, 0) AS ?has_educated_at)
            (IF(BOUND(?field_of_work), 1, 0) AS ?has_field_of_work)
            (IF(BOUND(?native_language), 1, 0) AS ?has_native_language)
            (IF(BOUND(?occupation), 1, 0) AS ?has_occupation)
            (IF(BOUND(?employer), 1, 0) AS ?has_employer)
            (IF(BOUND(?award_received), 1, 0) AS ?has_award_received)
            (IF(BOUND(?date_of_death), 1, 0) AS ?has_date_of_death)
            (IF(BOUND(?date_of_birth), 1, 0) AS ?has_date_of_birth)
            (IF(BOUND(?cause_of_death), 1, 0) AS ?has_cause_of_death)
            (IF(BOUND(?academic_degree), 1, 0) AS ?has_academic_degree)
            (IF(BOUND(?notable_work), 1, 0) AS ?has_notable_work)
            (IF(BOUND(?religion), 1, 0) AS ?has_religion)
            (IF(BOUND(?ethnic_group), 1, 0) AS ?has_ethnic_group)
            
        WHERE {{
            VALUES ?creator {{ {values_clause} }}
            OPTIONAL {{ ?creator wdt:P19  ?place_of_birth }}
            OPTIONAL {{ ?creator wdt:P20  ?place_of_death }}
            OPTIONAL {{ ?creator wdt:P21  ?sex_or_gender}}
            OPTIONAL {{ ?creator wdt:22 ?father }}
            OPTIONAL {{ ?creator wdt:25 ?mother }}
            OPTIONAL {{ ?creator wdt:P26 ?spouse }}
            OPTIONAL {{ ?creator wdt:P27  ?country_of_citizenship }}
            OPTIONAL {{ ?creator wdt:P39  ?position_held }}
            OPTIONAL {{ ?creator wdt:40 ?child }}
            OPTIONAL {{ ?creator wdt:P69  ?educated_at }}
            OPTIONAL {{ ?creator wdt:P101 ?field_of_work }} 
            OPTIONAL {{ ?creator wdt:P103 ?native_language }}
            OPTIONAL {{ ?creator wdt:P106 ?occupation }}
            OPTIONAL {{ ?creator wdt:P108 ?employer }}
            OPTIONAL {{ ?creator wdt:P166 ?award_received }}
            OPTIONAL {{ ?creator wdt:P570 ?date_of_death }}
            OPTIONAL {{ ?creator wdt:P569 ?date_of_birth }}
            OPTIONAL {{ ?creator wdt:P509 ?cause_of_death }}
            OPTIONAL {{ ?creator wdt:P512 ?academic_degree }}
            OPTIONAL {{ ?creator wdt:P800 ?notable_work }}
            OPTIONAL {{ ?creator wdt:P140 ?religion }}
            OPTIONAL {{ ?creator wdt:P172 ?ethnic_group }}
        }}
    """

def construct_sparql_query_get_creators(languages_clause, subclasses_clause, limit=1000, offset=0):
    sparql_query = f"""
    SELECT distinct ?creator 
    WHERE {{
        VALUES ?languages_spoken {{ {languages_clause} }}
        VALUES ?occupation {{ {subclasses_clause} }}
        ?creator wdt:P31 wd:Q5; # Instance of human
            wdt:P106 ?occupation; # Occupation
            wdt:P1412 ?languages_spoken .
    }}
    LIMIT {limit} OFFSET {offset}
    """
    return sparql_query

if __name__ == '__main__':

    # Step 1: Set the list of occupations interested in
    # Creator (Q2500638), politician (Q82955), actor (Q33999), writer (Q36180)
    occupation_classes = ["Q2500638", "Q82955", "Q33999", "Q36180"]   

    # Define language sets
    language_sets = {
        # "English": ["Q1860", "Q7976", "Q7979", "Q44676", "Q44679", "Q7053766", "Q48767245"],  # English variants
        "Arabic": ["Q13955", "Q29919", "Q56499","Q1194795", "Q1654327","Q5329979"],  # Arabic and Egyptian Arabic
        # "German": ["Q188", "Q248682", "Q306626","Q106937689", "Q26721", "Q387066"],  # German and its variants
        # "French": ["Q150", "Q1450506","Q214086", "Q3083193", "Q979914","Q83503"],  # French and Canadian French
        # "Kannada": ["Q33673", "Q6363888","Q6478506"],
        # "Italian": ["Q652"],
        # "Polish": ["Q809"],
        # "Hindi": ["Q1568"],
        # "Russian": ["Q7737","Q608923"],
        # "Chinese": ["Q7850"]
    }


    # Step 2: Retrieve creator IDs for each language group
    output_dir = os.path.join(os.path.dirname(__file__), 'data','dataset_v2')
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "entities_ids"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "entities_properties"), exist_ok=True)

    for language_group, language_ids in language_sets.items():
        print(f"Step 2: Retrieving creator IDs for language group: {language_group}")
        entity_ids_file_path = os.path.join(output_dir, "entities_ids", f'entity_ids_{language_group}.txt')
        entity_properties_file_path = os.path.join(output_dir, "entities_properties", f'entities_properties_{language_group}.csv')
        entity_properties_matrix_file_path = os.path.join(output_dir, "entities_properties_matrix", f'{language_group}.csv')
        if os.path.exists(entity_ids_file_path):
            print(f"File with Entities IDs already exists for language group {language_group}: {entity_ids_file_path}")
            with open(entity_ids_file_path, 'r') as file:
                set_creators_ids = set(line.strip() for line in file)
                print(f"Total entities IDs retrieved for language group {language_group}: {len(set_creators_ids)}")
        else:
            # Build the SPARQL query
            languages_clause = " ".join([f"wd:{lang_id}" for lang_id in language_ids])
            subclasses_clause = " ".join([f"wd:{class_id}" for class_id in occupation_classes])
            limit = 10000
            total = 200000
            set_creators_ids = set()

            for offset in tqdm(range(0, total, limit), desc=f"Retrieving creators for {language_group}"):
                
                try:
                    sparql_query = construct_sparql_query_get_creators(languages_clause, subclasses_clause, limit, offset)
                    query_result = query_wikidata(sparql_query, add_prefix=True)
                    if not query_result['results']['bindings']:
                        print(f"No creators found for language group {language_group}")
                        print(f"Total creators retrieved for language group {language_group}: {len(set_creators_ids)}")
                        break
                    else : 
                        creator_ids_batch = [result['creator']['value'].split('/')[-1] for result in query_result['results']['bindings']]
                        set_creators_ids.update(set(creator_ids_batch))
                        print(f"Retrieved {len(creator_ids_batch)} creators for language group {language_group}")
                        print(f"Total creators retrieved for language group {language_group}: {len(set_creators_ids)}")

                except Exception as e:
                    print(f"Error retrieving creators for language group {language_group}: {e}")
                    print(f"Total creators retrieved for language group {language_group}: {len(set_creators_ids)}")
                    break
            
            with open(entity_ids_file_path, 'w') as file:
                for creator_id in list(set_creators_ids):
                    file.write(f"{creator_id}\n")

        # if os.path.exists(entity_properties_file_path):
        #     # Read QIDs from the existing file
        #     data = pd.read_csv(entity_properties_file_path)
        #     existing_creator_ids = set(data['creator_id'])
        #     print(f"Found {len(existing_creator_ids)} existing entity IDs for language group {language_group} in {filename_prop}")
        #     print(f"Query returned {len(creator_ids)} new entity IDs for language group {language_group}")
        #     # Remove the existing creator IDs from the list
        #     creator_ids = list(set(creator_ids) - existing_creator_ids)
        #     print(f"Adding {len(creator_ids)} new creator IDs for language group {language_group}")
        # else:
        #     existing_creator_ids = []

        # if os.path.exists(entity_ids_file_path):
        #     full_creator_ids = list(set(creator_ids).union(existing_creator_ids))
        # else : 
        #     full_creator_ids = creator_ids

        # Step 3: Filtering IDs for additional properties

        filtered_creator_ids = [] 
        data = []
        
        batch_size = 10  # Number of creators to check per batch

        i = 0
        creator_ids = list (set_creators_ids)
        nb_creator_ids = len(creator_ids)
        with tqdm(total=nb_creator_ids, 
                  desc=f"Filtering creators in batches for {language_group}",
                  unit="entity") as pbar:

            while i < nb_creator_ids :
                batch = creator_ids[i:i + batch_size]
                values_clause = " ".join([f"wd:{creator_id}" for creator_id in batch])
                select_query = build_select_query(values_clause)
                
                retries = 4
                for attempt in range(retries):

                    try:
                        batch_filtered_creators = query_wikidata(select_query, add_prefix=True)
                        for result in batch_filtered_creators['results']['bindings']:
                            creator_id = result['creator']['value'].split('/')[-1]
                            data.append({
                                'creator_id': creator_id,
                                'has_place_of_birth': int(result['has_place_of_birth']['value']),
                                'has_place_of_death': int(result['has_place_of_death']['value']),
                                'has_gender': int(result['has_gender']['value']),
                                'has_father': int(result['has_father']['value']),
                                'has_mother': int(result['has_mother']['value']),
                                'has_spouse': int(result['has_spouse']['value']),
                                'has_country_of_citizenship': int(result['has_country_of_citizenship']['value']),
                                'has_position_held': int(result['has_position_held']['value']),
                                'has_child': int(result['has_child']['value']),
                                'has_educated_at': int(result['has_educated_at']['value']),
                                'has_field_of_work': int(result['has_field_of_work']['value']),
                                'has_native_language': int(result['has_native_language']['value']),
                                'has_occupation': int(result['has_occupation']['value']),
                                'has_employer': int(result['has_employer']['value']),
                                'has_award_received': int(result['has_award_received']['value']),
                                'has_date_of_death': int(result['has_date_of_death']['value']),
                                'has_date_of_birth': int(result['has_date_of_birth']['value']),
                                'has_cause_of_death': int(result['has_cause_of_death']['value']),
                                'has_academic_degree': int(result['has_academic_degree']['value']),
                                'has_notable_work': int(result['has_notable_work']['value']),
                                'has_religion': int(result['has_religion']['value']),
                                'has_ethnic_group': int(result['has_ethnic_group']['value'])
                            })
                        # time.sleep(1 + random.uniform(0, 2))  # Add delay to avoid overwhelming the server
                        i += batch_size
                        pbar.update(batch_size)
                        break
                    except Exception as e:
                        print(f"Error querying batch starting with creator {batch[0]} for language group {language_group}: {e}")
                        if "429" in str(e):
                            print("HTTP 429 Error: Too Many Requests. Retrying after a delay...")
                            time.sleep(10)
                            if i > 0:
                                i -= batch_size
                        if "500" in str(e):
                            print("HTTP 500 Error: Server Error. Retrying after a delay...")
                            time.sleep(10)
                            if i > 0:
                                i -= batch_size
                    if attempt == retries - 1:
                        print(f"Skipping batch {batch[0]} after {retries} attempts")
            
        # Save the properties data
        properties_matrix = pd.DataFrame(data)
        properties_matrix.to_csv(entity_properties_matrix_file_path, index=False)
            
            # if os.path.exists(filename_prop):
            #     # Read the existing file into a DataFrame
            #     df = pd.read_csv(filename_prop)
            #     # Append the new data
            #     df = pd.concat([df, pd.DataFrame(data)]).drop_duplicates().reset_index(drop=True)
            # else :
            #     # Convert the data list to a pandas DataFrame
            #     df = pd.DataFrame(data)
            # # Save the DataFrame to a CSV file for further analysis
            # df.to_csv(f'entities_properties_{language_group}.csv', index=False)