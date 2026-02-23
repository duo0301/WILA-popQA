import os, time,random
from tqdm import tqdm
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import requests

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
LOCAL_ENDPOINT = "http://localhost:1234/api/endpoint/sparql"

# We excluded the following properties :  sex or gender (P21), languages spoken, written, or signed (wdt:P1412), and writing language (wdt:P6886) 
# to avoid confounding effects arising from the entity’s name or explicit language metadata that could bias the model’s responses.
EXCLUDED_PIDs  = ["P21", "P1412", "P6886"]

# Creator (Q2500638), politician (Q82955), actor (Q33999), writer (Q36180)
ENTITY_CLASS_TO_QID = {
    "creator": "Q2500638",
    "politician": "Q82955",
    "actor": "Q33999",
    "writer": "Q36180",
    "artist": "Q483501",
    "athlete": "Q2066131",
    "researcher": "Q1650915",
    "university teacher": "Q1622272",
    "professor": "Q121594",
    "author": "Q482980",
}

### most 10 occupation per language group :
# most10occupations_per_language_group = {
#     "Kannada": ["writer", "actor", "poet", "politician", "film director", "film producer", "journalist", "researcher", "screenwriter", "singer"],
#     "Arabic": ["writer", "politician", "journalist", "poet", "actor", "singer", "researcher", "film director", "professor", "screenwriter"],
#     "Chinese": ["poet", "badminton player", "university teacher", "writer", "politician", "actor", "translator", "historian", "sinologist", "researcher"],
#     "Hindi": ["politician", "actor", "writer", "television actor", "film actor", "film director", "model", "poet", "screenswriter", "film producer"],
#     "Russian": ["writer", "association football player", "politician", "university teacher", "actor", "poet", "journalist", "scientist", "translator", "historian"],
#     "Polish": ["researcher", "politician", "university teacher", "actor", "writer", "historian", "journalist", "military officer", "association football player", "poet"],
#     "Italian": ["politician", "university teacher", "association football player", "writer", "actor", "journalist", "painter", "catholic priest", "historian", "poet"],
#     "French": [],
#     "German": [],
#     "English": [],
#     "Spanish": []
# }

PROPERTY_TO_PID = {
    # Core Neutral Biographical Properties
    "place_of_birth": "P19",
    "place_of_death": "P20",
    "place_of_burial": "P119",
    "cause_of_death": "P509",
    "residence": "P551",
    "date_of_birth": "P569",
    "date_of_death": "P570",
    # Relationships 
    "father": "P22",
    "mother": "P25",
    "spouse": "P26",
    "child": "P40",
    "sibling": "P3373",
    "student_of": "P1066",
    "doctoral_advisor": "P184",
    # Career & Work-related Properties
    "position_held": "P39",
    "field_of_work": "P101",
    "occupation": "P106",
    "employer": "P108",
    "movement": "P135",
    "award_received": "P166",
    "member_of": "P463",
    "academic_degree": "P512",
    "notable_work": "P800",
    "work_location": "P937",
    
    # For Actors
    "cast_member_of": "P161",
    "character_role": "P453",
    "voice_actor": "P725",
    
    # miscellaneous
    "country_of_citizenship": "P27",
    "educated_at": "P69",
    "native_language": "P103",
    "religion": "P140",
    "ethnic_group": "P172",
    "member_of_political_party":"P102",
}

LANGUAGE_TO_QIDS = {
    "English": ["Q1860", "Q7976", "Q7979", "Q44676", "Q44679", "Q7053766", "Q48767245"],  # English variants
    "Arabic": ["Q13955", "Q29919", "Q56499","Q1194795", "Q1654327","Q5329979"],  # Arabic and Egyptian Arabic
    "German": ["Q188", "Q248682", "Q306626","Q106937689", "Q26721", "Q387066"],  # German and its variants
    "French": ["Q150", "Q1450506","Q214086", "Q3083193", "Q979914","Q83503"],  # French and Canadian French
    "Italian": ["Q652"],
    "Polish": ["Q809"],
    "Hindi": ["Q1568"],
    "Russian": ["Q7737","Q608923"],
    "Spanish": ["Q1321"],
    # Chinese : Sino-Tibetan , zh-cn, zh-hans, zh-hant, zh-hant
    "Chinese": ["Q7850", "Q24841726", "Q13414913", "Q18130932", "Q100148307"],
    #"Kannada": ["Q33673", "Q6363888","Q6478506"] # We ignore Kannada as it contains few entities
}

def query_wikidata(sparql_query, endpoint_url, add_prefix=True, timeout=300):

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
        if endpoint_url == WIKIDATA_ENDPOINT:
            # To avoid hitting the rate limit of the Wikidata endpoint, we can add a random sleep between requests
            time.sleep(random.uniform(1, 5))  # Sleep for a random time between 1 and 3 seconds
        results = sparql.query().convert()
        return results
    except Exception as e:
        print(f"Error executing SPARQL query: {str(e)}")
        return None

def build_select_query(entities_ids_batch, properties_to_check, pid_to_property):
    entity_id_clause = " ".join([f"wd:{entity_id}" for entity_id in entities_ids_batch])
    properties_ids_to_check = properties_to_check
    select_clause = "DISTINCT ?entity "+ " ".join([f"(IF(BOUND(?{property}), 1, 0) AS ?has_{pid_to_property[property]})" for property in properties_to_check])
    where_clause = f'VALUES ?entity {{{entity_id_clause}}} '+" ".join([f"OPTIONAL {{ ?entity wdt:{property_id} ?{property} }}" 
                                                                       for property_id, property 
                                                                       in zip(properties_ids_to_check, properties_to_check)])
    query = f"""
        SELECT {select_clause}
        WHERE {{ {where_clause} }}
    """
    return query

def construct_sparql_query_get_entities(languages_clause, subclasses_clause, limit=1000, offset=0):
    sparql_query = f"""
    SELECT distinct ?entity 
    WHERE {{
        VALUES ?languages_spoken {{ {languages_clause} }}
        VALUES ?occupation {{ {subclasses_clause} }}
        ?entity wdt:P31 wd:Q5;    
            wdt:P106 ?occupation;
            wdt:P1412 ?languages_spoken .
    }}
    LIMIT {limit} OFFSET {offset}
    """
    return sparql_query

# Function to query Wikidata SPARQL endpoint for sitelink count
def build_sitelinks_counts_query(entity_ids):
    ids_str = " ".join([f"wd:{entity_id}" for entity_id in entity_ids])
    query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX schema: <http://schema.org/>

        SELECT distinct ?entity (count(?sitelink) as ?sitelinks_count) 
        WHERE {{
            VALUES ?entity {{{ids_str}}}
            ?sitelink   schema:about ?entity;
                        schema:isPartOf ?wiki .
            FILTER(REGEX(STR(?wiki), "^https://[a-z-]+\\\\.wikipedia\\\\.org/$"))
        }}
        Group by ?entity
    """
    return query

def construct_sparql_query_get_entities_count(languages_clause, subclasses_clause):
    sparql_query = f"""
    SELECT (COUNT(DISTINCT ?entity) AS ?count)
    WHERE {{
        VALUES ?languages_spoken {{ {languages_clause} }}
        VALUES ?occupation {{ {subclasses_clause} }}
        ?entity wdt:P31 wd:Q5;    
            wdt:P106 ?occupation;
            wdt:P1412 ?languages_spoken .
    }}
    """
    return sparql_query
    
def construct_sparql_query_get_subclasses(entity_classes_of_interest, depth=1):
    subclasses_clause = " ".join([f"wd:{class_id}" for class_id in entity_classes_of_interest])

    def path_of_len(n: int) -> str:
        # n hops of wdt:P279
        return "/".join(["wdt:P279"] * n)
    
    query = f"""
    SELECT DISTINCT ?subclass
    WHERE {{
        VALUES ?occupation {{ {subclasses_clause} }}
        ?subclass {path_of_len(depth)} ?occupation .
    }}
    """
    return query

def construct_sparql_query_get_num_entities_per_occupation(languages_clause, occupation_classes, depth=1):

    def subclassof(n: int) -> str:
        # n hops of wdt:P279
        return "/".join(["wdt:P279"] * n)
    
    query = f"""
    SELECT (COUNT(DISTINCT ?entity) AS ?count)
    WHERE {{
        VALUES ?languages_spoken {{ {languages_clause} }}
        VALUES ?occupation {{ {occupation_classes} }}
        {{
            ?entity wdt:P31 wd:Q5;    
            wdt:P106 ?occ;
            wdt:P1412 ?languages_spoken .
            ?occ {subclassof(depth)} ?occupation .
        }}
        UNION
        {{
            ?entity wdt:P31 wd:Q5;
            wdt:P106 ?occupation;
            wdt:P1412 ?languages_spoken .
        }}
    }}
    """
    return query

if __name__ == '__main__':

    PID_TO_PROPERTY = {v: k for k, v in PROPERTY_TO_PID.items()}

    output_dir_name = 'dataset_v4/author'
    output_dir = os.path.join(os.path.dirname(__file__), 'data', output_dir_name)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "entities_ids"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "entities_properties_matrix"), exist_ok=True)

    # Step 1 - User define 3 sets of interest : 
    # 1. Occupations / entity classes
    # 2. Languages 
    # 3. Properties
    # entity_classes_of_interest = ["creator", "politician", "actor", "writer"]
    entity_classes_of_interest = ["author"]

    languages_of_interest = ["Polish"] # "French","English", "Arabic", "French", "German", "Italian", "Spanish", "Russian", "Chinese", "Hindi", "Polish"]
    
    properties_of_interest = ["place_of_birth", "place_of_death", "date_of_birth", "date_of_death", 
                              "cause_of_death", "place_of_burial", "residence", "father", "mother", "spouse", 
                              "child", "sibling", "student_of", "doctoral_advisor", "position_held", "field_of_work", 
                              "occupation", "employer", "movement", "award_received", "member_of", "academic_degree", 
                              "notable_work", "work_location", "cast_member_of", "character_role", "voice_actor", 
                              "country_of_citizenship", "educated_at", "native_language", "religion", "ethnic_group", 
                              "member_of_political_party"]
    
    # Convert to QID and PIDs
    entity_classes_of_interest = [ENTITY_CLASS_TO_QID[entity_class] for entity_class in entity_classes_of_interest]
    # for properties, we need to remove the excluded properties
    properties_of_interest = [PROPERTY_TO_PID[property] for property in properties_of_interest if property not in EXCLUDED_PIDs]
    languages_of_interest = {language_grp: LANGUAGE_TO_QIDS[language_grp] for language_grp in languages_of_interest}

    # Step 2 - Retrieve entities IDs for each language group
    for language_group, language_ids in languages_of_interest.items():

        entity_ids_file_path = os.path.join(output_dir, "entities_ids", f'{language_group}.txt')
        entity_properties_matrix_file_path = os.path.join(output_dir, "entities_properties_matrix", f'{language_group}.csv')
        print(f"Step 2: Retrieving QID of entities for language group: {language_group}")

        if os.path.exists(entity_ids_file_path):
            print(f"File with Entities IDs already exists for language group {language_group}: {entity_ids_file_path}")
            with open(entity_ids_file_path, 'r') as file:
                set_entity_ids = set(line.strip() for line in file)
                print(f"Total entities IDs retrieved for language group {language_group}: {len(set_entity_ids)}")
        else:
            set_entity_ids = set()

        # Build the SPARQL query
        languages_clause = " ".join([f"wd:{lang_id}" for lang_id in language_ids])

        # Get the subclasses of the entity classes of interest, let "depth" be the number of levels of subclasses we want to retrieve

        subclasses_query = construct_sparql_query_get_subclasses(entity_classes_of_interest, depth=1)
        subclasses_result = query_wikidata(subclasses_query, LOCAL_ENDPOINT, add_prefix=True, timeout=10)  
        subclasses_ids = [result['subclass']['value'].split('/')[-1] for result in subclasses_result['results']['bindings']]
        all_classes_of_interest = list(set(entity_classes_of_interest + subclasses_ids))

        print(f"Total number of occupation subclasses retrieved for language group {language_group}: {len(all_classes_of_interest)}")

        
        # Query to get the total number of entities for the language group 

        # entity_classes_of_interest_formatted = " ".join([f"wd:{class_id}" for class_id in entity_classes_of_interest])
        # nb_entities_per_occupation_query = construct_sparql_query_get_num_entities_per_occupation(languages_clause, entity_classes_of_interest_formatted, depth=1)
        # nb_entities_per_occupation_result = query_wikidata(nb_entities_per_occupation_query, LOCAL_ENDPOINT, add_prefix=True, timeout=300)  
        # total = int(nb_entities_per_occupation_result['results']['bindings'][0]['count']['value']) if nb_entities_per_occupation_result['results']['bindings'] else 0
        # print(f"Total entities for the occupation classes of interest for language group {language_group}: {total}")

        total = 10000
        limit = 100
        start = len(set_entity_ids)

        subclasses_clause = " ".join([f"wd:{class_id}" for class_id in all_classes_of_interest])
        
        print(f"Starting from offset: {start}")
        for offset in tqdm(range(start, total, limit), desc=f"Retrieving entities for {language_group}"):   
            size_subclasses_batch = 4
            for subclass_batch in [all_classes_of_interest[i:i+size_subclasses_batch] for i in range(0, len(all_classes_of_interest), size_subclasses_batch)]:
                subclasses_clause = " ".join([f"wd:{class_id}" for class_id in subclass_batch])
                
                try:
                    sparql_query = construct_sparql_query_get_entities(languages_clause, subclasses_clause, limit, offset)
                    query_result = query_wikidata(sparql_query, LOCAL_ENDPOINT, add_prefix=True, timeout=300)  
                    

                    if not query_result['results']['bindings']:
                        print(f"No entities found for language group {language_group}")
                        print(f"Total entities retrieved for language group {language_group}: {len(set_entity_ids)}")
                        break
                    else : 
                        nb_entity_ids_retrieved = len(query_result['results']['bindings'])
                        print(f"Found {nb_entity_ids_retrieved} entities for language group {language_group} with the initial query")

                        entity_ids_batch = [result['entity']['value'].split('/')[-1] for result in query_result['results']['bindings']]

                        # Filter by SiteLinks counts (>=9, at least)
                        # nb: use WIKIDATA_ENDPOINT if you're using the truthy version of Wikidata 
                        sitelinks_query = build_sitelinks_counts_query(entity_ids_batch)
                        sl_query_result = query_wikidata(sitelinks_query, WIKIDATA_ENDPOINT, add_prefix=True, timeout=300)  

                        entity_ids_batch = [result['entity']['value'].split('/')[-1] for result in sl_query_result['results']['bindings'] if int(result['sitelinks_count']['value']) >= 9]

                        set_entity_ids.update(set(entity_ids_batch))
                        print(f"Filtered {len(entity_ids_batch)} entities for language group {language_group}")
                        print(f"Total entities retrieved for language group {language_group}: {len(set_entity_ids)}")

                except Exception as e:
                    print(f"Error retrieving entities for language group {language_group}: {e}")
                    print(f"Total entities retrieved for language group {language_group}: {len(set_entity_ids)}")
                    break
                
            with open(entity_ids_file_path, 'w') as file:
                for qid in list(set_entity_ids):
                    file.write(f"{qid}\n")

        

        # filter by sitelinks : 
        # - Number of sitelinks
        # - Languages
        # - Adapt lisa code in here
        # - Store in a separate file the number of site links 

        # list of entities -- filter -- narrowed list 


        # Step 3 - Filtering IDs - 
        # Check if the entity has the properties of interest

        filtered_creator_ids = [] 
        data = []
        batch_size = 100  # Number of entities to check per batch
        i = 0
        entity_ids = list (set_entity_ids)
        nb_entity_ids = len(entity_ids)

        with tqdm(total=nb_entity_ids, 
                  desc=f"Filtering entities in batches for {language_group}", 
                  initial=i,
                  unit="entity") as pbar:

            # Define the properties to check
            properties_to_check = properties_of_interest
            properties_batch_size = 4 
            properties_batches = [properties_to_check[i:i + properties_batch_size] for i in range(0, len(properties_to_check), properties_batch_size)]

            while i < nb_entity_ids :
                entities_ids_batch = entity_ids[i:i + batch_size]
                
                batch_results = {}

                for properties_batch in properties_batches:
                    batch_select_query = build_select_query(entities_ids_batch, properties_batch, PID_TO_PROPERTY)
                    batch_filtered_entities = query_wikidata(batch_select_query, endpoint_url=LOCAL_ENDPOINT, add_prefix=True, timeout=300)

                    # Fill the batch_results dictionary
                    if batch_filtered_entities and 'results' in batch_filtered_entities and 'bindings' in batch_filtered_entities['results']:
                        for result in batch_filtered_entities['results']['bindings']:
                            entity_id = result['entity']['value'].split('/')[-1]
                            if entity_id not in batch_results:
                                batch_results[entity_id] = {}
                            for property in properties_batch:
                                property_name = "has_" + PID_TO_PROPERTY[property]
                                batch_results[entity_id][property_name] = result[property_name]['value']
                    else:
                        print(f"No results found for batch query")

                # Once done with all properties, we can append the results that concern the entities_ids_batch
                for entity_id in batch_results.keys():
                    data.append({
                        'entity_id': entity_id,
                        **batch_results[entity_id]
                    })

                i += batch_size
                pbar.update(batch_size)

        # Save the properties data
        properties_matrix = pd.DataFrame(data)
        properties_matrix.to_csv(entity_properties_matrix_file_path, index=False)