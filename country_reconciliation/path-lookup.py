import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import logging
import sys
import time
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('semantic_paths.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SemanticPathFinder:
    def __init__(self):
        self.endpoint = "https://query.wikidata.org/sparql"
        self.cache = {}
        self.user_agent = "SemanticPathFinder/1.0 (andschimmenti@gmail.com)"
        
    def find_semantic_path(self, entity1_id: str, entity2_id: str) -> str:
        """Find semantic paths between entities focusing on geographic and other relationships."""
        if not entity1_id or not entity2_id:
            return None

        # Check cache first
        cache_key = tuple(sorted([entity1_id, entity2_id]))
        if cache_key in self.cache:
            return self.cache[cache_key]

        sparql = SPARQLWrapper(self.endpoint, agent=self.user_agent)
        
        # Using the same properties as the original code
        query = f"""
        SELECT DISTINCT ?path WHERE {{
          VALUES (?e1 ?e2) {{ (wd:{entity1_id} wd:{entity2_id}) (wd:{entity2_id} wd:{entity1_id}) }}
          {{
            ?e1 wdt:P36 ?e2 .  BIND("P36" AS ?path)  # Capital
          }} UNION {{
            ?e2 wdt:P36 ?e1 .  BIND("P36" AS ?path)
          }} UNION {{
            ?e1 wdt:P131/wdt:P131* ?e2 .  BIND("P131" AS ?path)  # Located in administrative territory
          }} UNION {{
            ?e2 wdt:P131/wdt:P131* ?e1 .  BIND("P131" AS ?path)
          }} UNION {{
            ?e1 wdt:P361/wdt:P361* ?e2 .  BIND("P361" AS ?path)  # Part of
          }} UNION {{
            ?e2 wdt:P361/wdt:P361* ?e1 .  BIND("P361" AS ?path)
          }} UNION {{
            ?e1 wdt:P17 ?e2 .  BIND("P17" AS ?path)  # Country
          }} UNION {{
            ?e2 wdt:P17 ?e1 .  BIND("P17" AS ?path)
          }} UNION {{
            ?e1 wdt:P36 ?intermediate . ?intermediate wdt:P17 ?e2 . BIND("P36/P17" AS ?path)  # Capital-Country
          }} UNION {{
            ?e2 wdt:P36 ?intermediate . ?intermediate wdt:P17 ?e1 . BIND("P36/P17" AS ?path)
          }} UNION {{
            ?e1 wdt:P276 ?e2 .  BIND("P276" AS ?path)  # Location
          }} UNION {{
            ?e2 wdt:P276 ?e1 .  BIND("P276" AS ?path)
          }} UNION {{
            ?e1 wdt:P3842 ?e2 . BIND("P3842" AS ?path)  # Located in protected area
          }} UNION {{
            ?e2 wdt:P3842 ?e1 . BIND("P3842" AS ?path)
          }} UNION {{
            ?e1 wdt:P156 ?e2 . BIND("P156" AS ?path)  # Follows
          }} UNION {{
            ?e2 wdt:P156 ?e1 . BIND("P156" AS ?path)
          }} UNION {{
            ?e1 wdt:P155 ?e2 . BIND("P155" AS ?path)  # Followed by
          }} UNION {{
            ?e2 wdt:P155 ?e1 . BIND("P155" AS ?path)
          }} UNION {{
            ?e1 wdt:P144 ?e2 . BIND("P144" AS ?path)  # Based on
          }} UNION {{
            ?e2 wdt:P144 ?e1 . BIND("P144" AS ?path)
          }}
        }}
        LIMIT 1
        """

        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)

        max_retries = 3
        retry_delay = 5  # Initial delay in seconds

        for attempt in range(max_retries):
            try:
                results = sparql.query().convert()
                if results["results"]["bindings"]:
                    path = results["results"]["bindings"][0]["path"]["value"]
                    self.cache[cache_key] = path
                    return path
                return None
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached. Giving up.")
                    return None

def process_data(file_path: str):
    """Process the CSV file and find semantic paths between entities."""
    logger.info(f"Starting processing of file: {file_path}")
    
    # Initialize path finder
    path_finder = SemanticPathFinder()
    
    # Read the CSV file
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    logger.info(f"Loaded CSV with {len(df)} rows")
    
    # Add new column for semantic paths if it doesn't exist
    if 'semantic_path' not in df.columns:
        df['semantic_path'] = None
    
    for idx, row in df.iterrows():
        try:
            gt_entity = row['gt_entity']
            llm_entity = row['llm_entity']
            
            if pd.isna(gt_entity) or pd.isna(llm_entity):
                logger.info(f"Row {idx}: Missing entity IDs")
                continue
            
            # Find semantic path
            path = path_finder.find_semantic_path(gt_entity, llm_entity)
            if path:
                logger.info(f"Row {idx}: Found path {path}")
                df.at[idx, 'semantic_path'] = path
            else:
                logger.info(f"Row {idx}: No path found")
            
            # Add delay for rate limiting
            time.sleep(1)
            
            if idx % 20 == 0:
                logger.info(f"Processed {idx} rows")
                # Save progress
                output_path = file_path.replace('.csv', '_with_paths.csv')
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                
        except Exception as e:
            logger.error(f"Error processing row {idx}: {str(e)}")
            continue
    
    # Save final results
    output_path = file_path.replace('.csv', '_with_paths.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"Results saved to {output_path}")
    return df

if __name__ == "__main__":
    try:
        input_file = "property_country_qwen2.5_7b_with_entities.csv"
        result_df = process_data(input_file)
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        raise