import logging
import sys
import argparse
from pathlib import Path
from match_countries import process_all_files as process_entities
from path_lookup import process_all_files as process_paths
from file_utils import ensure_directory_exists

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation_pipeline.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_pipeline(input_dir, data_type):
    """Run the complete evaluation pipeline."""
    logger.info(f"Starting evaluation pipeline for {data_type} data")
    
    # Step 1: Ensure directories exist
    ensure_directory_exists("results")
    
    # Step 2: Process entities
    pattern = f"*_{data_type}_*.csv"
    logger.info(f"Processing entities for files matching: {pattern}")
    entity_results = process_entities(input_dir, pattern)
    logger.info(f"Processed entities for {len(entity_results)} files")
    
    # Step 3: Process semantic paths
    logger.info("Processing semantic paths for files with entities")
    path_results = process_paths("results", f"**/*_{data_type}_*_with_entities.csv")
    logger.info(f"Processed semantic paths for {len(path_results)} files")
    
    logger.info("Evaluation pipeline completed successfully")
    return entity_results, path_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation pipeline for location data")
    parser.add_argument("--input-dir", default="Inputs", help="Directory containing model subdirectories with input CSV files")
    parser.add_argument("--data-type", choices=["country", "dob"], required=True, 
                        help="Type of data to process (country or date of birth)")
    
    args = parser.parse_args()
    
    try:
        entity_results, path_results = run_pipeline(args.input_dir, args.data_type)
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise 