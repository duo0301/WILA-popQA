import os, time,random
from tqdm import tqdm
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
LOCAL_ENDPOINT = "http://localhost:1234/api/endpoint/sparql"

def query_wikidata(sparql_query, add_prefix=True, timeout=300):

    endpoint_url = LOCAL_ENDPOINT
    sparql = SPARQLWrapper(endpoint=endpoint_url)
    if add_prefix:
        sparql.setQuery(f"""
        PREFIX wd: <http://www.wikidata.org/entity/> 
        PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
        {sparql_query}
        """)
    else:
        sparql.setQuery(sparql_query)

    sparql.setReturnFormat(JSON)
    sparql.setTimeout(timeout)
    try:
        results = sparql.query().convert()
        return results
    except Exception as e:
        print(f"Error executing SPARQL query: {str(e)}")
        return None

def build_select_query(entities_ids_batch, properties_to_check, property_to_id):
    entity_id_clause = " ".join([f"wd:{entity_id}" for entity_id in entities_ids_batch])
    properties_ids_to_check = [property_to_id[property] for property in properties_to_check]
    select_clause = "DISTINCT ?creator "+ " ".join([f"(IF(BOUND(?{property}), 1, 0) AS ?has_{property})" for property in properties_to_check])
    where_clause = f'VALUES ?creator {{{entity_id_clause}}} '+" ".join([f"OPTIONAL {{ ?creator wdt:{property_id} ?{property} }}" 
                                                                       for property_id, property 
                                                                       in zip(properties_ids_to_check, properties_to_check)])
    query = f"""
        SELECT {select_clause}
        WHERE {{ {where_clause} }}
    """
    return query

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

    # Step 1 - Define 3 sets : 
    # 1. Occupations of interest
    # 2. Languages of interest
    # 3. Properties of interest

    # Set the list of occupations interested in
    occupation_classes = ["Q2500638", "Q82955", "Q33999", "Q36180"]  # Creator (Q2500638), politician (Q82955), actor (Q33999), writer (Q36180)

    # Define language sets
    language_sets = {
        "English": ["Q1860", "Q7976", "Q7979", "Q44676", "Q44679", "Q7053766", "Q48767245"],  # English variants
        "Arabic": ["Q13955", "Q29919", "Q56499","Q1194795", "Q1654327","Q5329979"],  # Arabic and Egyptian Arabic
        "German": ["Q188", "Q248682", "Q306626","Q106937689", "Q26721", "Q387066"],  # German and its variants
        "French": ["Q150", "Q1450506","Q214086", "Q3083193", "Q979914","Q83503"],  # French and Canadian French
        "Italian": ["Q652"],
        "Polish": ["Q809"],
        "Hindi": ["Q1568"],
        "Russian": ["Q7737","Q608923"],
        ## Chinese : Sino-Tibetan , zh-cn, zh-hans, zh-hant, zh-hant
        "Chinese": ["Q7850", "Q24841726", "Q13414913", "Q18130932", "Q100148307"]
        # "Kannada": ["Q33673", "Q6363888","Q6478506"] # We ignore Kannada as it contains 282 entities (few entities)
    }

    # Define the properties of interest
    property_to_id = {
        "place_of_birth": "P19",
        "place_of_death": "P20",
        "gender": "P21",
        "father": "P22",
        "mother": "P25",
        "spouse": "P26",
        "country_of_citizenship": "P27",
        "position_held": "P39",
        "child": "P40",
        "educated_at": "P69",
        "field_of_work": "P101",
        "native_language": "P103",
        "occupation": "P106",
        "employer": "P108",
        "award_received": "P166",
        "date_of_death": "P570",
        "date_of_birth": "P569",
        "cause_of_death": "P509",
        "academic_degree": "P512",
        "notable_work": "P800",
        "religion": "P140",
        "ethnic_group": "P172"
    }

    # Step 2 - Retrieve entities IDs for each language group
    output_dir_name = 'dataset_v2'
    output_dir = os.path.join(os.path.dirname(__file__), 'data', output_dir_name)
    
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
            set_creators_ids = set()


        # Build the SPARQL query
        languages_clause = " ".join([f"wd:{lang_id}" for lang_id in language_ids])
        subclasses_clause = " ".join([f"wd:{class_id}" for class_id in occupation_classes])
        limit = 100
        total = 100000
        start = len(set_creators_ids)
        for offset in tqdm(range(start, total, limit), desc=f"Retrieving creators for {language_group}"):    
            try:
                sparql_query = construct_sparql_query_get_creators(languages_clause, subclasses_clause, limit, offset)
                query_result = query_wikidata(sparql_query, add_prefix=True, timeout=300)  
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

        # Step 3 - Filtering IDs - 
        # Check if the entity has the properties of interest

        filtered_creator_ids = [] 
        data = []
        batch_size = 100  # Number of entities to check per batch
        i = 0
        creator_ids = list (set_creators_ids)
        nb_creator_ids = len(creator_ids)

        with tqdm(total=nb_creator_ids, 
                  desc=f"Filtering creators in batches for {language_group}", 
                  initial=i,
                  unit="entity") as pbar:

            # Define the properties to check
            properties_to_check = list(property_to_id.keys())
            properties_batch_size = 4 # batch the properties to check in order to avoid overwhelming the server
            properties_batches = [properties_to_check[i:i + properties_batch_size] for i in range(0, len(properties_to_check), properties_batch_size)]

            while i < nb_creator_ids :
                entities_ids_batch = creator_ids[i:i + batch_size]
                
                batch_results = {}

                for properties_batch in properties_batches:
                    batch_select_query = build_select_query(entities_ids_batch, properties_batch, property_to_id)
                    batch_filtered_entities = query_wikidata(batch_select_query, add_prefix=True)

                    # Fill the batch_results dictionary
                    for result in batch_filtered_entities['results']['bindings']:
                        creator_id = result['creator']['value'].split('/')[-1]
                        if creator_id not in batch_results:
                            batch_results[creator_id] = {}
                        for property in properties_batch:
                            property_name = "has_" + property
                            batch_results[creator_id][property_name] = result[property_name]['value']

                # Once done with all properties, we can append the results that concern the entities_ids_batch
                for creator_id in batch_results.keys():
                    data.append({
                        'creator_id': creator_id,
                        **batch_results[creator_id]
                    })

                # for result in batch_filtered_entities['results']['bindings']:
                #         creator_id = result['creator']['value'].split('/')[-1]
                #         batch_data.append({
                #                     'creator_id': creator_id,
                #                     'has_place_of_birth': int(result['has_place_of_birth']['value']),
                #                     'has_place_of_death': int(result['has_place_of_death']['value']),
                #                     'has_gender': int(result['has_gender']['value']),
                #                     'has_father': int(result['has_father']['value']),
                #                     'has_mother': int(result['has_mother']['value']),
                #                     'has_spouse': int(result['has_spouse']['value']),
                #                     'has_country_of_citizenship': int(result['has_country_of_citizenship']['value']),
                #                     'has_position_held': int(result['has_position_held']['value']),
                #                     'has_child': int(result['has_child']['value']),
                #                     'has_educated_at': int(result['has_educated_at']['value']),
                #                     'has_field_of_work': int(result['has_field_of_work']['value']),
                #                     'has_native_language': int(result['has_native_language']['value']),
                #                     'has_occupation': int(result['has_occupation']['value']),
                #                     'has_employer': int(result['has_employer']['value']),
                #                     'has_award_received': int(result['has_award_received']['value']),
                #                     'has_date_of_death': int(result['has_date_of_death']['value']),
                #                     'has_date_of_birth': int(result['has_date_of_birth']['value']),
                #                     'has_cause_of_death': int(result['has_cause_of_death']['value']),
                #                     'has_academic_degree': int(result['has_academic_degree']['value']),
                #                     'has_notable_work': int(result['has_notable_work']['value']),
                #                     'has_religion': int(result['has_religion']['value']),
                #                     'has_ethnic_group': int(result['has_ethnic_group']['value'])
                #                 })
                            # time.sleep(1 + random.uniform(0, 2))  # Add delay to avoid overwhelming the server
                i += batch_size
                pbar.update(batch_size)
            #     except Exception as e:
            #         print(f"Error querying batch starting with creator {entities_ids_batch[0]} for language group {language_group}: {e}")
            #         if "429" in str(e):
            #             print("HTTP 429 Error: Too Many Requests. Retrying after a delay...")
            #             if i > 0:
            #                 i -= batch_size
            #         if "500" in str(e):
            #             print("HTTP 500 Error: Server Error. Retrying after a delay...")
            #             if i > 0:
            #                 i -= batch_size
            # if attempt == retries - 1:
            #     print(f"Skipping batch {entities_ids_batch[0]} after {retries} attempts")

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