import pandas as pd
import requests
import time
import logging
import sys
import ast
from pathlib import Path
from file_utils import ensure_directory_exists, get_input_files, get_output_path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('country_matching.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class WikidataNormalizer:
    def __init__(self):
        self.base_url = "https://www.wikidata.org/w/api.php"
        self.cache = {}
        
    def _clean_text(self, text: str) -> str:
        """Clean text by removing punctuation and normalizing whitespace."""
        if not isinstance(text, str):
            return str(text) if text else ""
        return text.strip().rstrip('.').strip()

    def normalize_place(self, place_name: str, source_lang: str = None):
        """Normalize place name using Wikidata API with language-specific queries."""
        if not place_name or not source_lang:
            return None

        # Clean input
        place_name = self._clean_text(place_name)
        cache_key = f"{place_name}_{source_lang}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Search in source language
        params = {
            'action': 'wbsearchentities',
            'search': place_name,
            'language': source_lang,
            'type': 'item',
            'format': 'json'
        }

        try:
            response = requests.get(
                self.base_url,
                params=params,
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            
            # Get the first result if any
            if data.get('search'):
                entity_id = data['search'][0]['id']
                self.cache[cache_key] = entity_id
                return entity_id
                    
            return None

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 414:  # URI Too Long
                logger.warning(f"URI too long for '{place_name[:50]}...', skipping")
                return None
            else:
                raise

def clean_list_string(s):
    """Clean a string that might contain list notation using a more robust approach."""
    if pd.isna(s):
        return None
    
    if not isinstance(s, str):
        return str(s) if s else None
        
    # Remove whitespace and brackets
    s = s.strip()
    
    # Handle multi-line strings by taking only the first non-empty line
    if '\n' in s:
        lines = [line.strip() for line in s.split('\n') if line.strip()]
        if lines:
            s = lines[0]
    
    # Remove list brackets
    s = s.strip('[]')
    
    # Split by comma and clean each item
    items = [item.strip().strip("'\"") for item in s.split(',') if item.strip()]
    
    # Return the first non-empty item
    return items[0] if items else None

def process_data(file_path):
    """Process the CSV file and fetch Wikidata entities."""
    logger.info(f"Starting processing of file: {file_path}")
    
    # Initialize normalizer
    normalizer = WikidataNormalizer()
    
    # Read the CSV file
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    logger.info(f"Loaded CSV with {len(df)} rows")
    
    # Add new columns for Wikidata entities
    df['gt_entity'] = None
    df['llm_entity'] = None
    
    for idx, row in df.iterrows():
        try:
            # Clean and get ground truth country
            gt_country = clean_list_string(row['gt'])
            if gt_country:
                gt_entity = normalizer.normalize_place(gt_country, row['LoE'])
                df.at[idx, 'gt_entity'] = gt_entity
            
            # Clean and get LLM output country
            llm_country = clean_list_string(row['llm_output'])
            if llm_country:
                llm_entity = normalizer.normalize_place(llm_country, row['LoQ'])
                df.at[idx, 'llm_entity'] = llm_entity
            
            # Add a small delay to respect rate limits
            time.sleep(0.1)
            
            if idx % 100 == 0:
                logger.info(f"Processed {idx} rows")
                
        except Exception as e:
            logger.error(f"Error processing row {idx}: {str(e)}")
            continue
    
    # Generate output path
    output_path = get_output_path(Path(file_path), "results", "_with_entities")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"Results saved to {output_path}")
    return df, output_path

def process_all_files(input_dir, pattern=None):
    """Process all CSV files in the input directory."""
    input_files = get_input_files(input_dir, pattern)
    logger.info(f"Found {len(input_files)} files to process")
    
    results = {}
    for file_path in input_files:
        try:
            logger.info(f"Processing file: {file_path}")
            df, output_path = process_data(file_path)
            results[file_path] = output_path
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {str(e)}")
    
    return results

if __name__ == "__main__":
    try:
        input_dir = "Inputs"  # Directory containing model subdirectories
        pattern = "*_country_*.csv"  # Pattern to match country evaluation files
        
        # Ensure output directory exists
        ensure_directory_exists("results")
        
        # Process all matching files
        processed_files = process_all_files(input_dir, pattern)
        
        logger.info(f"Successfully processed {len(processed_files)} files")
        for input_file, output_file in processed_files.items():
            logger.info(f"Processed {input_file} -> {output_file}")
            
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        raise