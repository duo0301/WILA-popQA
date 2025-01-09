import requests
from typing import Optional, Union, Tuple, Dict, List
import urllib.parse
import time

class WikidataNormalizer:
    def __init__(self):
        self.base_url = "https://www.wikidata.org/w/api.php"
        self.cache = {}
        
    def _get_wiki_site(self, lang: str) -> str:
        """Convert language code to wiki site code."""
        return f"{lang}wiki"
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing punctuation and normalizing whitespace."""
        return text.strip().rstrip('.').strip()

    def normalize_place(self, 
                       place_name: str, 
                       source_lang: str = None,
                       return_id: bool = False) -> Optional[Union[str, Tuple[str, int]]]:
        """
        Normalize place name using Wikidata API with language-specific queries.
        
        Args:
            place_name (str): Place name to normalize
            source_lang (str): Language code (e.g., 'en', 'ru', 'ar')
            return_id (bool): Whether to return the Wikidata QID
        """
        if not place_name or not source_lang:
            return None

        # Clean input
        place_name = self._clean_text(place_name)
        cache_key = f"{place_name}_{source_lang}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]

        # First try: search in source language
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
                
                # Now get full entity details
                entity_params = {
                    'action': 'wbgetentities',
                    'ids': entity_id,
                    'props': 'labels|claims',
                    'languages': f"{source_lang}|en",
                    'format': 'json'
                }
                
                entity_response = requests.get(
                    self.base_url,
                    params=entity_params,
                    timeout=5
                )
                entity_data = entity_response.json()
                
                entity = entity_data['entities'][entity_id]
                
                # Try to get label in source language first, fall back to English
                label = (
                    entity.get('labels', {}).get(source_lang, {}).get('value') or
                    entity.get('labels', {}).get('en', {}).get('value')
                )
                
                # Check if it's a location (more permissive check)
                claims = entity.get('claims', {})
                is_location = False
                location_types = [
                    'Q515',   # city
                    'Q5107',  # populated place
                    'Q486972',  # human settlement
                    'Q1549591',  # municipality
                    'Q484170',  # village
                    'Q15284',   # municipality
                    'Q123705',  # neighborhood
                ]
                
                if 'P31' in claims:
                    for claim in claims['P31']:
                        value_id = claim.get('mainsnak', {}).get('datavalue', {}).get('value', {}).get('id')
                        if value_id in location_types:
                            is_location = True
                            break
                
                # For debugging, accept all matches initially
                if label:
                    result = (label, int(entity_id[1:]))
                    self.cache[cache_key] = result
                    return result if return_id else result[0]
                    
            return None

        except requests.exceptions.RequestException as e:
            print(f"Error querying Wikidata API for '{place_name}': {e}")
            return None
        except Exception as e:
            print(f"Unexpected error processing '{place_name}': {e}")
            return None

    def normalize_batch(self, 
                       places: List[str], 
                       source_lang: str,
                       return_id: bool = False) -> List[Optional[Union[str, Tuple[str, int]]]]:
        """Normalize a batch of place names."""
        results = []
        for place in places:
            result = self.normalize_place(place, source_lang, return_id)
            results.append(result)
            time.sleep(0.1)  # Rate limiting
        return results
    
normalizer = WikidataNormalizer()

# Test with Arabic
result = normalizer.normalize_place("قنيطرة", "ar", return_id=True)
print(result)

# Test with Russian
result = normalizer.normalize_place("Айн-Млила", "ru", return_id=True)
print(result)