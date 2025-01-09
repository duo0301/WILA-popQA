import pandas as pd
from typing import Dict, Optional, Tuple, List
import csv
import json
import os
import time
from wikidata_normalizer import WikidataNormalizer
import requests
import re 

class LocationComparator:
    def __init__(self, cache_file: str = "location_cache.json", retry_delay: int = 300):
        self.normalizer = WikidataNormalizer()
        self.cache_file = cache_file
        self.cache = self.load_cache()
        self.queries_since_save = 0
        self.SAVE_FREQUENCY = 100
        self.retry_delay = retry_delay

    def load_cache(self) -> Dict[str, Optional[Tuple[str, int]]]:
        """Load cache from file if it exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    raw_cache = json.load(f)
                    return {
                        k: tuple(v) if v is not None else None
                        for k, v in raw_cache.items()
                    }
            except Exception as e:
                print(f"Error loading cache: {e}")
                return {}
        return {}

    def save_cache(self, force: bool = False):
        """Save cache to file if enough queries have been made or if forced."""
        if force or self.queries_since_save >= self.SAVE_FREQUENCY:
            try:
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json_cache = {
                        k: list(v) if v is not None else None
                        for k, v in self.cache.items()
                    }
                    json.dump(json_cache, f, ensure_ascii=False, indent=2)
                self.queries_since_save = 0
                print(f"Cache saved ({len(self.cache)} entries)")
            except Exception as e:
                print(f"Error saving cache: {e}")

    def save_progress(self, output_file: str, output_data: List[dict]):
        """Save current progress to CSV file with enhanced error checking."""
        if not output_data:
            print("Warning: No data to save!")
            return

        print(f"Attempting to save {len(output_data)} records to {output_file}")

        try:
            # First try to save to a temporary file
            temp_file = f"{output_file}.temp"
            with open(temp_file, "w", newline="", encoding="utf-8") as f:
                # Print first few records for debugging
                print(f"Sample of data being saved: {output_data[:2]}")

                fieldnames = [
                    "row_id",
                    "line_id",
                    "lang",
                    "gt",
                    "gt_normalized",
                    "gt_geonames_id",
                    "llm_output",
                    "llm_normalized",
                    "llm_geonames_id",
                    "match",
                ]

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for row in output_data:
                    # Ensure all fields are present
                    row_clean = {field: row.get(field, "") for field in fieldnames}
                    writer.writerow(row_clean)

            # If temporary file was created successfully, rename it to the actual output file
            if os.path.exists(temp_file):
                if os.path.exists(output_file):
                    os.replace(output_file, f"{output_file}.old")
                os.rename(temp_file, output_file)
                print(f"Successfully saved to {output_file}")

                # Verify the file was saved correctly
                try:
                    df = pd.read_csv(output_file)
                    print(f"Verification: saved file contains {len(df)} rows")
                except Exception as e:
                    print(f"Warning: Could not verify saved file: {e}")

        except Exception as e:
            print(f"Error saving to {output_file}: {e}")
            # Try to save to a backup file
            backup_file = f"{output_file}.backup"
            try:
                with open(backup_file, "w", newline="", encoding="utf-8") as f:
                    fieldnames = [
                        "row_id",
                        "line_id",
                        "lang",
                        "gt",
                        "gt_normalized",
                        "gt_geonames_id",
                        "llm_output",
                        "llm_normalized",
                        "llm_geonames_id",
                        "match",
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in output_data:
                        row_clean = {field: row.get(field, "") for field in fieldnames}
                        writer.writerow(row_clean)
                print(f"Successfully saved to backup file: {backup_file}")
            except Exception as e:
                print(f"Critical error: Could not save to backup file: {e}")
                # As a last resort, try to print the data
                print("Emergency data dump:")
                print(output_data[:5])

    def load_progress(self, output_file: str) -> List[dict]:
        """Load previous progress from CSV file."""
        try:
            if os.path.exists(output_file):
                df = pd.read_csv(output_file)
                return df.to_dict("records")
            return []
        except Exception as e:
            print(f"Error loading progress: {e}")
            # Try to load from backup file
            backup_file = f"{output_file}.backup"
            try:
                if os.path.exists(backup_file):
                    df = pd.read_csv(backup_file)
                    return df.to_dict("records")
            except Exception as e:
                print(f"Error loading from backup file: {e}")
            return []

    def get_cache_key(self, place_name: str, lang: str) -> str:
        """Generate cache key from place name and language."""
        return f"{place_name}|{lang}"

    def _check_is_location(self, wikidata_id: str) -> bool:
        """
        Check if entity is an instance of a subclass of geographical location.
        """
        try:
            query = (
                """
            SELECT ?item WHERE {
                wd:%s wdt:P31 ?item .
                ?item wdt:P279* wd:Q82794 .
            }
            LIMIT 1
            """
                % wikidata_id
            )

            response = requests.get(
                "https://query.wikidata.org/sparql",
                params={"format": "json", "query": query},
                headers={"User-Agent": "LocationComparator/1.0"},
            )

            if response.status_code == 429:
                print("Rate limit reached, waiting...")
                time.sleep(self.retry_delay)
                return self._check_is_location(wikidata_id)

            data = response.json()
            return len(data.get("results", {}).get("bindings", [])) > 0

        except Exception as e:
            print(f"Error checking location type for {wikidata_id}: {e}")
            return False

    def _select_location_entity(self, entities: List[Dict]) -> Optional[Dict]:
        """
        Select best matching entity from candidates based on geographic type.
        """
        geographic_entities = []

        for entity in entities:
            entity_id = entity.get("id")
            if entity_id and self._check_is_location(entity_id):
                geographic_entities.append(entity)

        if not geographic_entities:
            return None

        return geographic_entities[0]

    def get_geonames_info(self, place_name: str, lang: str, is_extraction: bool = False) -> Optional[Tuple[str, int]]:
        """
        Get Geonames information with improved handling of cached results and recursion prevention.
        
        Args:
            place_name (str): Place name to look up
            lang (str): Language code
            is_extraction (bool): Flag to prevent recursive extraction
            
        Returns:
            Optional[Tuple[str, int]]: Normalized name and ID if found
        """
        # Clean the input text first
        place_name = self._clean_text(place_name)
        cache_key = self.get_cache_key(place_name, lang)

        if cache_key in self.cache:
            print(f"Cache hit for: {place_name} ({lang})")
            cached_result = self.cache[cache_key]
            
            # Only attempt extraction if this isn't already part of an extraction
            if cached_result is None and not is_extraction:
                if any(word[0].isupper() for word in place_name.split()):
                    print(f"Cached null result for '{place_name}', attempting extraction...")
                    potential_locations = self.extract_potential_locations(place_name, lang)
                    if potential_locations:
                        first_location = potential_locations[0]
                        print(f"Found location in text: {first_location}")
                        return first_location[1]
            
            return cached_result

        while True:
            try:
                # Get entities from Wikidata API
                params = {
                    "action": "wbsearchentities",
                    "format": "json",
                    "language": lang,
                    "search": place_name,
                }

                response = requests.get(
                    "https://www.wikidata.org/w/api.php", params=params
                )

                if response.status_code == 429:
                    print("Rate limit reached, waiting...")
                    time.sleep(self.retry_delay)
                    continue

                data = response.json()
                candidates = data.get("search", [])

                if not candidates:
                    self.cache[cache_key] = None
                    self.queries_since_save += 1
                    self.save_cache()
                    return None

                # Get full entity data for candidates
                entity_ids = [c["id"] for c in candidates]
                params = {
                    "action": "wbgetentities",
                    "format": "json",
                    "ids": "|".join(entity_ids),
                    "languages": lang,
                }

                response = requests.get(
                    "https://www.wikidata.org/w/api.php", params=params
                )

                if response.status_code == 429:
                    print("Rate limit reached, waiting...")
                    time.sleep(self.retry_delay)
                    continue

                data = response.json()
                entities = list(data.get("entities", {}).values())

                # Select best geographic entity
                best_entity = self._select_location_entity(entities)

                if best_entity:
                    # Convert to normalized format
                    normalized_name = (
                        best_entity.get("labels", {})
                        .get(lang, {})
                        .get("value", place_name)
                    )
                    wikidata_id = best_entity["id"]
                    result = (normalized_name, wikidata_id)

                    self.cache[cache_key] = result
                    self.queries_since_save += 1
                    self.save_cache()
                    return result

                self.cache[cache_key] = None
                self.queries_since_save += 1
                self.save_cache()
                return None

            except Exception as e:
                print(f"Error getting info for {place_name}: {e}")
                print(
                    f"Network error detected. Waiting {self.retry_delay} seconds before retrying..."
                )
                time.sleep(self.retry_delay)
                print("Retrying...")
                continue
            
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted characters and normalizing whitespace.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove common punctuation marks and normalize spaces
        text = re.sub(r'[.,!?()[\]{}"\'`]', ' ', text)
        # Normalize whitespace (including newlines and tabs)
        text = ' '.join(text.split())
        return text.strip()

    def extract_potential_locations(self, text: str, lang: str) -> List[Tuple[str, Optional[Tuple[str, int]]]]:
        """
        Extract and verify potential location names from a text string.
        Handles hyphenated words and cleans text before processing.
        
        Args:
            text (str): Input text containing potential location names
            lang (str): Language code
            
        Returns:
            List[Tuple[str, Optional[Tuple[str, int]]]]: List of extracted locations with their info
        """
        # Clean the text first
        text = self._clean_text(text)
        
        # Split by spaces and hyphens
        all_parts = []
        words = text.split()
        
        for word in words:
            # Split hyphenated words and add both the full word and its parts
            if '-' in word:
                parts = word.split('-')
                all_parts.append(word)  # Add the full hyphenated word
                all_parts.extend(parts)  # Add individual parts
            else:
                all_parts.append(word)
        
        # Filter for capitalized words
        capitalized_words = [
            word for word in all_parts 
            if word and word[0].isupper()
        ]
        
        results = []
        seen = set()  # To avoid processing duplicates
        
        for word in capitalized_words:
            if word in seen:
                continue
                
            seen.add(word)
            try:
                # Add a flag parameter to prevent infinite recursion
                geonames_info = self.get_geonames_info(word, lang, is_extraction=True)
                if geonames_info:
                    results.append((word, geonames_info))
            except Exception as e:
                print(f"Error processing word '{word}': {e}")
                continue
        
        return results
    
    def process_csv(self, input_file: str, output_file: str):
        while True:
            try:
                df = pd.read_csv(input_file)
                print(f"Loaded {len(df)} rows from input file")
                break
            except Exception as e:
                print(f"Error reading input file: {e}")
                print(f"Waiting {self.retry_delay} seconds before retrying...")
                time.sleep(self.retry_delay)

        # Load previous progress
        output_data = self.load_progress(output_file)
        print(f"Loaded {len(output_data)} rows from previous progress")

        # Create a unique identifier combining line_id and language
        df["unique_id"] = df["Q_number"] + "|" + df["LoQ"]

        # Find entries needing reprocessing from output data
        processed_ids = {
            f"{row['line_id']}|{row['lang']}"
            for row in output_data
            if row.get("llm_geonames_id") is not None
            and row.get("gt_geonames_id") is not None
        }

        print(f"Found {len(processed_ids)} successfully processed entries")

        # Filter input dataframe
        df = df[~df["unique_id"].isin(processed_ids)]
        print(f"Remaining entries to process: {len(df)}")

        # Clear output_data to start fresh with only successful entries
        output_data = [
            row
            for row in output_data
            if f"{row['line_id']}|{row['lang']}" in processed_ids
        ]

        # Add row_id after filtering
        df = df.reset_index(drop=True)
        df["row_id"] = range(len(df))

        total_entries = len(df)
        processed = len(output_data)
        total_to_process = total_entries + processed

        print(f"Will process {total_entries} new/failed entries")
        print(f"Already have {processed} successful entries")

        print(f"Resuming from {processed}/{total_to_process} entries")

        for _, row in df.iterrows():
            while True:
                try:
                    row_id = row["row_id"]
                    line_id = row["Q_number"]
                    lang = row["LoQ"]
                    gt = row["gt"]
                    llm_output = row["llm_output"]

                    processed += 1
                    print(
                        f"\nProcessing row {row_id} (Q{line_id}) ({processed}/{total_to_process}):"
                    )
                    print(f"Ground Truth: {gt} (Lang: {lang})")
                    print(f"LLM Output: {llm_output}")

                    # Handle NaN or empty values in ground truth
                    if pd.isna(gt) or str(gt).strip() == '':
                        gt_info = None
                        print("GT is empty or NaN")
                    else:
                        gt_info = self.get_geonames_info(gt, lang)
                        print(f"GT match found: {gt_info}")

                    # Handle NaN or empty values in LLM output
                    if pd.isna(llm_output) or str(llm_output).strip() == '':
                        llm_info = None
                        print("LLM output is empty or NaN")
                    else:
                        llm_info = self.get_geonames_info(llm_output, lang)
                        print(f"Direct LLM match found: {llm_info}")

                        # If no direct match for LLM output, try extracting locations from the text
                        if not llm_info:
                            print("Attempting to extract locations from LLM output...")
                            potential_locations = self.extract_potential_locations(
                                str(llm_output), lang
                            )

                            if potential_locations:
                                print(f"Found potential locations: {potential_locations}")
                                # Use the first found location as the match
                                first_location = potential_locations[0]
                                llm_info = first_location[1]
                                llm_output = first_location[0]  # Update llm_output to the found location
                                print(f"Using location: {llm_output} with info: {llm_info}")
                            else:
                                print("No valid locations found in text")

                    gt_geonames_id = gt_info[1] if gt_info else None
                    llm_geonames_id = llm_info[1] if llm_info else None

                    output_data.append(
                        {
                            "row_id": row_id,
                            "line_id": line_id,
                            "lang": lang,
                            "gt": gt if not pd.isna(gt) else "",
                            "gt_geonames_id": gt_geonames_id,
                            "llm_output": llm_output if not pd.isna(llm_output) else "",
                            "llm_geonames_id": llm_geonames_id,
                            "match": (
                                gt_geonames_id == llm_geonames_id
                                if (gt_geonames_id and llm_geonames_id)
                                else False
                            ),
                            "gt_normalized": gt_info[0] if gt_info else None,
                            "llm_normalized": llm_info[0] if llm_info else None,
                        }
                    )

                    # Save progress every 50 entries
                    if processed % 50 == 0:
                        self.save_progress(output_file, output_data)
                        print(f"Progress saved: {processed}/{total_to_process}")

                    break

                except Exception as e:
                    print(f"\nError during processing row {row_id}: {e}")
                    print(f"Waiting {self.retry_delay} seconds before retrying...")
                    self.save_progress(output_file, output_data)
                    self.save_cache(force=True)
                    time.sleep(self.retry_delay)
                    print("Retrying current row...")

        # Final save
        self.save_cache(force=True)
        self.save_progress(output_file, output_data)

        # Print statistics
        matches = sum(1 for row in output_data if row["match"])
        total = len(output_data)
        print(f"\nProcessing complete!")
        print(f"Total entries processed: {total}")
        print(f"Expected total: {total_to_process}")
        if total != total_to_process:
            print(f"WARNING: Missing {total_to_process - total} entries!")
        print(f"Matches: {matches}")
        print(f"Match rate: {(matches/total)*100:.2f}%")
        print(f"Cache size: {len(self.cache)} entries")


def main():
    while True:
        try:
            comparator = LocationComparator()
            comparator.process_csv(
                input_file="property_pob.csv", output_file="aligned_locations.csv"
            )
            break
        except Exception as e:
            print(f"Critical error during processing: {e}")
            print("Waiting 5 minutes before restarting the entire process...")
            time.sleep(300)
            print("Restarting...")


if __name__ == "__main__":
    main()
